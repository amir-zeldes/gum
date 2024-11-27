import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward, GatedSum
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.archival import archive_model, load_archive
from gucorpling_models.features import Feature, get_feature_modules

logger = logging.getLogger(__name__)


def weighted_sum(att, mat):
    if att.dim() == 2 and mat.dim() == 3:
        return att.unsqueeze(1).bmm(mat).squeeze(1)
    elif att.dim() == 3 and mat.dim() == 3:
        return att.bmm(mat)
    else:
        AssertionError('Incompatible attention weights and matrix.')


@Model.register("disrpt_2021_e2e")
class E2eResolver(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            context_layer: Seq2SeqEncoder,
            # feature_size: int,
            lexical_dropout: float = 0.2,
            encoder_decoder_dropout: float = 0.3,
            features: Dict[str, Feature] = None,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._global_attention = TimeDistributed(torch.nn.Linear(text_field_embedder.get_output_dim(), 1))

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        # self._distance_embedding = torch.nn.Embedding(num_embeddings=20, embedding_dim=feature_size)
        self._dir_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=1)

        # setup handwritten feature modules
        if features is not None and len(features) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0

        num_relations = vocab.get_vocab_size("relation_labels")
        self.dropout = torch.nn.Dropout(encoder_decoder_dropout)
        self.relation_decoder = torch.nn.Linear(context_layer.get_output_dim()*6+feature_dims+1, num_relations)

        # these are stateful objects that keep track of accuracy across an epoch
        self.direction_accuracy = CategoricalAccuracy()
        self.relation_accuracy = CategoricalAccuracy()

        # convenience dict mapping relation indices to labels
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")
        initializer(self)

    def _get_end_points_enmbeddings(self, contextualized_unit_sentence, unit_span_indices):
        unit_span_indices = unit_span_indices.tolist()

        out_start, out_end = [], []
        for i, span in enumerate(unit_span_indices):
            start_idx, end_idx = span[0][0], span[-1][-1]
            cur_start_embedding = torch.unsqueeze(contextualized_unit_sentence[i][start_idx], 0)
            cur_end_embedding = torch.unsqueeze(contextualized_unit_sentence[i][end_idx], 0)
            out_start.append(cur_start_embedding)
            out_end.append(cur_end_embedding)
        start_embeddings = torch.cat(out_start, dim=0)
        end_embeddings = torch.cat(out_end, dim=0)
        return start_embeddings, end_embeddings

    def _get_span_embeddings(self, contextualized_unit_sentence, unit_span_mask):
        """
        Get span representations based on attention weights
        """
        global_attention_logits = self._global_attention(contextualized_unit_sentence)  # [b, s, 1]
        concat_tensor = torch.cat([contextualized_unit_sentence, global_attention_logits], -1)  # [b, s, e+1]

        resized_span_mask = torch.unsqueeze(unit_span_mask, -1).expand(-1, -1, concat_tensor.size(-1))
        concat_output = concat_tensor * resized_span_mask
        span_embeddings = concat_output[:, :, :-1]  # [b, s, e]
        span_attention_logits = concat_output[:, :, -1] # [b, s, 1]

        span_attention_weights = F.softmax(span_attention_logits, 1)  # [b, s]
        attended_text_embeddings = weighted_sum(span_attention_weights, span_embeddings)    # [b, e]
        return attended_text_embeddings

    def _get_span_representations(self, embedded_unit_sentence, unit_span_mask, unit_span_indices, unit_sentence_mask):
        assert unit_span_mask.size(1) == embedded_unit_sentence.size(1)
        contextualized_unit_sentence = self._context_layer(embedded_unit_sentence, unit_sentence_mask)
        weighted_span_embeds = self._get_span_embeddings(embedded_unit_sentence, unit_span_mask)  # [b, e]
        start_embeddings, end_embeddings = self._get_end_points_enmbeddings(contextualized_unit_sentence, unit_span_indices)    # [b, e]
        span_representations = torch.cat([start_embeddings, weighted_span_embeds, end_embeddings], 1)   # [b, 3e]
        return span_representations

    def _get_combined_feature_tensor(self, kwargs):
        output_tensors = []
        for module_key, module in self.feature_modules.items():
            output_tensor = module(kwargs[module_key])
            if len(output_tensor.shape) == 1:
                output_tensor = output_tensor.unsqueeze(-1)
            output_tensors.append(output_tensor)

        combined_feature_tensor = torch.cat(output_tensors, dim=1)
        return combined_feature_tensor

    @overrides
    def forward(
            self,  # type: ignore
            sentence1: TextFieldTensors,
            sentence2: TextFieldTensors,
            unit1_span_mask: torch.Tensor,
            unit1_span_indices: torch.Tensor,
            unit2_span_mask: torch.Tensor,
            unit2_span_indices: torch.Tensor,
            direction: torch.Tensor,
            # distance: torch.Tensor,
            relation: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        sentence1_mask = util.get_text_field_mask(sentence1)
        sentence2_mask = util.get_text_field_mask(sentence2)

        embedded_unit1_sentence = self._lexical_dropout(self._text_field_embedder(sentence1))
        embedded_unit2_sentence = self._lexical_dropout(self._text_field_embedder(sentence2))

        unit1_span_representations = self._get_span_representations(embedded_unit1_sentence, unit1_span_mask, unit1_span_indices, sentence1_mask)   # [b, 3e]
        unit2_span_representations = self._get_span_representations(embedded_unit2_sentence, unit2_span_mask, unit2_span_indices, sentence2_mask)   # [b, 3e]

        # dist_embeds = self._distance_embedding(distance)
        dir_embeds = self._dir_embedding(direction)

        # Get the features
        # feature_embeds = torch.cat((dist_embeds, dir_embeds), -1) # [b, f]
        if self.feature_modules:
            features = self._get_combined_feature_tensor(kwargs)
            combined = torch.cat((unit1_span_representations, unit2_span_representations, dir_embeds, features), 1) # [b, e*2+f+dir]
        else:
            combined = torch.cat((unit1_span_representations, unit2_span_representations, dir_embeds), 1)  # [b, e*2+dir]
        self.dropout(combined)
        # Decode the concatenated vector into relation logits
        relation_logits = self.relation_decoder(combined)
        relation_logits = F.relu(relation_logits)

        output = {
            "relation_logits": relation_logits,
        }
        if relation is not None:
            self.relation_accuracy(relation_logits, relation)
            output["gold_relation"] = relation
            output["loss"] = F.cross_entropy(relation_logits, relation)
        return output


    @overrides
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

    def _compute_span_pair_embeddings(
            self,
            top_span_embeddings: torch.FloatTensor,
            antecedent_embeddings: torch.FloatTensor,
            antecedent_offsets: torch.FloatTensor,
    ):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        # Parameters

        top_span_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : `torch.FloatTensor`, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : `torch.IntTensor`, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (batch_size, num_spans_to_keep, max_antecedents).

        # Returns

        span_pair_embeddings : `torch.FloatTensor`
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        )

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat(
            [
                target_embeddings,
                antecedent_embeddings,
                antecedent_embeddings * target_embeddings,
                antecedent_distance_embeddings,
            ],
            -1,
        )
        return span_pair_embeddings