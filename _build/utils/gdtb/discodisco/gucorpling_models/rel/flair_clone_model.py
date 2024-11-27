import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, TokenEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from gucorpling_models.features import Feature, get_feature_modules, get_combined_feature_tensor, FeatureBundle
from gucorpling_models.rel.featureful_bert_embedder import FeaturefulBertEmbedder

logger = logging.getLogger(__name__)


def truncate_sentence(s, max_len=512):
    s["tokens"]["token_ids"] = s["tokens"]["token_ids"][:, :max_len]
    s["tokens"]["mask"] = s["tokens"]["mask"][:, :max_len]
    s["tokens"]["type_ids"] = s["tokens"]["type_ids"][:, :max_len]


@Model.register("disrpt_2021_flair_clone")
class FlairCloneModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TokenEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        feature_dropout: float = 0.4,
        features: FeatureBundle = None,
        initializer: InitializerApplicator = None
    ):
        # For some reason, this label appears in test but never in train
        if features.corpus == "spa.rst.sctb":
            vocab.add_token_to_namespace("enablement", "relation_labels")
        elif features.corpus == "spa.rst.rststb":
            vocab.add_token_to_namespace("background", "relation_labels")
        elif features.corpus == "nld.rst.nldt":
            vocab.add_token_to_namespace("span", "relation_labels")
        super().__init__(vocab)
        self.embedder = embedder
        if isinstance(embedder, FeaturefulBertEmbedder) and features is not None:
            embedder.init_feature_modules(features, vocab)
        self.encoder = seq2vec_encoder
        self.feature_dropout = torch.nn.Dropout(feature_dropout)

        # setup handwritten feature modules
        if features is not None and len(features.corpus_keys) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0
        # direction
        #feature_dims += 1
        self.num_relations = vocab.get_vocab_size("relation_labels")

        self.relation_decoder = torch.nn.Linear(self.encoder.get_output_dim() + feature_dims, self.num_relations)
        # self.relation_decoder = FeedForward(
        #     input_dim=self.encoder.get_output_dim() + feature_dims,
        #     num_layers=3,
        #     hidden_dims=[512, 256, num_relations],
        #     activations=[Activation.by_name('relu')(), Activation.by_name('relu')(), Activation.by_name('linear')()],
        #     dropout=0.1
        # )

        self.relation_accuracy = CategoricalAccuracy()
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")

        if initializer:
            initializer(self)
        else:
            torch.nn.init.xavier_uniform_(self.relation_decoder.weight)

    def forward(  # type: ignore
        self,
        combined_body: TextFieldTensors,
        combined_sentence: TextFieldTensors,
        direction: torch.Tensor,
        relation: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        embedder_params = combined_body["tokens"]
        embedder_params["direction"] = direction
        embedder_params.update(kwargs)
        embedded_sentence = self.embedder(**embedder_params)

        mask = util.get_text_field_mask(combined_body)
        sentence_embedding = self.encoder(embedded_sentence, mask)

        components = [
            sentence_embedding,
            #direction.unsqueeze(-1),
        ]
        if self.feature_modules:
            components.append(self.feature_dropout(get_combined_feature_tensor(self, kwargs)))

        combined = torch.cat(components, dim=1)

        # Apply dropout
        combined = self.feature_dropout(combined)

        # Decode the concatenated vector into relation logits
        relation_logits = self.relation_decoder(combined)

        output = {
            "relation_logits": relation_logits,
        }
        if relation is not None:
            self.relation_accuracy(relation_logits, relation)
            output["gold_relation"] = relation

            #output["loss"] = sequence_cross_entropy_with_logits(
            #    relation_logits.unsqueeze(1),
            #    relation.unsqueeze(1),
            #    torch.ones(list(relation.shape[0:2])).bool().to(relation_logits.device),
            #    gamma=5
            #)
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
