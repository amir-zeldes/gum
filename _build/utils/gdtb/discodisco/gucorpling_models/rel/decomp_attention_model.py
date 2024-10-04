from typing import Dict, Optional, List, Any

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.training.metrics import CategoricalAccuracy

from gucorpling_models.features import Feature, get_feature_modules


@Model.register("disrpt_2021_rel_decomp")
class DecomposableAttention(Model):
    """
    modified from https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pair_classification/models/decomposable_attention.py
    # Parameters
    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the `premise` and `hypothesis` `TextFields` we get as input to the
        model.
    attend_feedforward : `FeedForward`
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    matrix_attention : `MatrixAttention`
        This is the attention function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : `FeedForward`
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : `FeedForward`
        This final feedforward network is applied to the concatenated, summed result of the
        `compare_feedforward` network, and its output is used as the entailment class logits.
    premise_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        After embedding the premise, we can optionally apply an encoder.  If this is `None`, we
        will do nothing.
    hypothesis_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is `None`,
        we will use the `premise_encoder` for the encoding (doing nothing if `premise_encoder`
        is also `None`).
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        attend_feedforward: FeedForward,
        matrix_attention: MatrixAttention,
        compare_feedforward: FeedForward,
        unit1_encoder: Optional[Seq2SeqEncoder] = None,
        unit2_encoder: Optional[Seq2SeqEncoder] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        features: Dict[str, Feature] = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = matrix_attention
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._unit1_encoder = unit1_encoder
        self._unit2_encoder = unit2_encoder or unit1_encoder

        self._relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")
        self._num_labels = vocab.get_vocab_size(namespace="relation_labels")

        if features is not None and len(features) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0

        self._aggregate_feedforward = FeedForward(
            input_dim=compare_feedforward.get_output_dim() * 2 + feature_dims + 1,
            num_layers=1,
            hidden_dims=compare_feedforward.get_output_dim(),
            activations=Activation.by_name("relu")(),
            dropout=0.2
        )
        self._decoder = torch.nn.Linear(self._aggregate_feedforward.get_output_dim(), self._num_labels)

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            attend_feedforward.get_input_dim(),
            "text field embedding dim",
            "attend feedforward input dim",
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

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
        embedded_unit1 = self._text_field_embedder(unit1_body)
        embedded_unit2 = self._text_field_embedder(unit2_body)
        unit1_mask = get_text_field_mask(unit1_body)
        unit2_mask = get_text_field_mask(unit2_body)

        if self._unit1_encoder:
            embedded_unit1 = self._unit1_encoder(embedded_unit1, unit1_mask)
        if self._unit2_encoder:
            embedded_unit2 = self._unit2_encoder(embedded_unit2, unit2_mask)

        projected_unit1 = self._attend_feedforward(embedded_unit1)
        projected_unit2 = self._attend_feedforward(embedded_unit2)
        # Shape: (batch_size, unit1_length, unit2_length)
        similarity_matrix = self._matrix_attention(projected_unit1, projected_unit2)

        # Shape: (batch_size, unit1_length, unit2_length)
        p2h_attention = masked_softmax(similarity_matrix, unit2_mask)
        # Shape: (batch_size, unit1_length, embedding_dim)
        attended_unit2 = weighted_sum(embedded_unit2, p2h_attention)

        # Shape: (batch_size, unit2_length, unit1_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), unit1_mask)
        # Shape: (batch_size, unit2_length, embedding_dim)
        attended_unit1 = weighted_sum(embedded_unit1, h2p_attention)

        unit1_compare_input = torch.cat([embedded_unit1, attended_unit2], dim=-1)
        unit2_compare_input = torch.cat([embedded_unit2, attended_unit1], dim=-1)

        compared_unit1 = self._compare_feedforward(unit1_compare_input)
        compared_unit1 = compared_unit1 * unit1_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_unit1 = compared_unit1.sum(dim=1)

        compared_unit2 = self._compare_feedforward(unit2_compare_input)
        compared_unit2 = compared_unit2 * unit2_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_unit2 = compared_unit2.sum(dim=1)

        features = self._get_combined_feature_tensor(kwargs)
        aggregate_input = torch.cat([compared_unit1, compared_unit2, direction.unsqueeze(-1), features], dim=-1)
        relation_logits = self._decoder(self._aggregate_feedforward(aggregate_input))
        relation_probs = torch.nn.functional.softmax(relation_logits, dim=-1)

        output_dict = {
            "relation_logits": relation_logits,
            "relation_probs": relation_probs,
            "h2p_attention": h2p_attention,
            "p2h_attention": p2h_attention,
        }

        if relation is not None:
            loss = self._loss(relation_logits, relation.long().view(-1))
            self._accuracy(relation_logits, relation)
            output_dict["gold_relation"] = relation
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self._accuracy.get_metric(reset)}

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        if "gold_relation" in output_dict:
            output_dict["gold_relation"] = [self._relation_labels[i.item()] for i in output_dict["gold_relation"]]
        predictions = output_dict["relation_probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary("relation_labels").get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["pred_relation"] = classes
        return output_dict
