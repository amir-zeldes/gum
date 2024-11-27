from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.nn import util, InitializerApplicator, Activation
from allennlp.training.metrics import CategoricalAccuracy

from gucorpling_models.features import Feature, get_feature_modules


@Model.register("disrpt_2021_rel_baseline")
class Disrpt2021RelBaseline(Model):
    """
    A simple encoder-decoder baseline which embeds all four spans (each unit's sentence and discourse unit),
    uses a seq2vec encoder to encode each span, and decodes using a simple linear transform
    for the direction and the relation.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder1: Seq2VecEncoder,
        encoder2: Seq2VecEncoder,
        dropout: float = 0.5,
        features: Dict[str, Feature] = None,
        feedforward: FeedForward = None,
        initializer: InitializerApplicator = None
    ):
        super().__init__(vocab)
        self.embedder = embedder

        self.encoder1 = encoder1
        self.encoder2 = encoder2

        num_relations = vocab.get_vocab_size("relation_labels")
        self.dropout = torch.nn.Dropout(dropout)

        # setup handwritten feature modules
        if features is not None and len(features) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0

        # we will decode the concatenated span reprs ...
        linear_input_size = self.encoder1.get_output_dim() * 1 + self.encoder2.get_output_dim() * 1
        # plus the direction label size
        linear_input_size += 1
        self.feedforward = FeedForward(
            input_dim=encoder1.get_output_dim() * 2 + 1 + feature_dims,
            num_layers=3,
            hidden_dims=[512, 256, 128],
            activations=Activation.by_name('relu')(),
            dropout=0.1
        )
        self.relation_decoder = torch.nn.Linear(
            linear_input_size if self.feedforward is None else self.feedforward.get_output_dim(), num_relations
        )

        # these are stateful objects that keep track of accuracy across an epoch
        self.direction_accuracy = CategoricalAccuracy()
        self.relation_accuracy = CategoricalAccuracy()

        # convenience dict mapping relation indices to labels
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")

        if initializer:
            initializer(self)

    def _get_combined_feature_tensor(self, kwargs):
        output_tensors = []
        for module_key, module in self.feature_modules.items():
            output_tensor = module(kwargs[module_key])
            if len(output_tensor.shape) == 1:
                output_tensor = output_tensor.unsqueeze(-1)
            output_tensors.append(output_tensor)

        combined_feature_tensor = torch.cat(output_tensors, dim=1)
        return combined_feature_tensor

    def forward(  # type: ignore
        self,
        unit1_body: TextFieldTensors,
        unit1_sentence: TextFieldTensors,
        unit2_body: TextFieldTensors,
        unit2_sentence: TextFieldTensors,
        direction: torch.Tensor,
        relation: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Embed the text. Shape: (batch_size, num_tokens, embedding_dim)
        embedded_unit1_body = self.embedder(unit1_body)
        # embedded_unit1_sentence = self.embedder(unit1_sentence)
        embedded_unit2_body = self.embedder(unit2_body)
        # embedded_unit2_sentence = self.embedder(unit2_sentence)

        # Encode the text. Shape: (batch_size, encoder_output_dim)
        encoded_unit1_body = self.encoder1(embedded_unit1_body, util.get_text_field_mask(unit1_body))
        # encoded_unit1_sentence = self.encoder1(embedded_unit1_sentence, util.get_text_field_mask(unit1_sentence))
        encoded_unit2_body = self.encoder2(embedded_unit2_body, util.get_text_field_mask(unit2_body))
        # encoded_unit2_sentence = self.encoder2(embedded_unit2_sentence, util.get_text_field_mask(unit2_sentence))

        # Get the features
        features = self._get_combined_feature_tensor(kwargs)

        # Concatenate the vectors. Shape: (batch_size, encoder1_output_dim * 2 + encoder2_output_dim * 2 + 1)
        combined = torch.cat(
            (
                encoded_unit1_body,
                # encoded_unit1_sentence,
                encoded_unit2_body,
                # encoded_unit2_sentence,
                direction.unsqueeze(-1),
                features
            ),
            1,
        )

        # Apply dropout
        combined = self.dropout(combined)

        if self.feedforward:
            combined = self.feedforward(combined)

        # Decode the concatenated vector into relation logits
        relation_logits = self.relation_decoder(combined)

        output = {
            "relation_logits": relation_logits,
        }
        if relation is not None:
            self.relation_accuracy(relation_logits, relation)
            output["gold_relation"] = relation
            output["loss"] = F.cross_entropy(relation_logits, relation)
        return output

    # Takes output of forward() and turns tensors into strings or probabilities wherever appropriate
    # Note that the output dict, because it's just from forward(), represents a batch, not a single
    # instance: every key will have a list that's the size of the batch
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        # if we have the gold label, decode it into a string
        if "gold_relation" in output_dict:
            output_dict["gold_relation"] = [self.relation_labels[i.item()] for i in output_dict["gold_relation"]]

        # output_dict["relation_logits"] is a tensor of shape (batch_size, num_relations): argmax over the last
        # to get the most likely relation for each instance in the batch
        relation_index = output_dict["relation_logits"].argmax(-1)
        # turn each one into a label
        output_dict["pred_relation"] = [self.relation_labels[i.item()] for i in relation_index]

        # turn relation logits into relation probabilities and present them in a dict
        # where the name of the relation (a string) maps to the probability
        output_dict["relation_probs"] = []
        for relation_logits_row in output_dict["relation_logits"]:
            relation_probs = F.softmax(relation_logits_row)
            output_dict["relation_probs"].append(
                {self.relation_labels[i]: relation_probs[i] for i in range(len(relation_probs))}
            )
        # remove the logits (humans prefer probs)
        del output_dict["relation_logits"]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "relation_accuracy": self.relation_accuracy.get_metric(reset),  # type: ignore
        }
