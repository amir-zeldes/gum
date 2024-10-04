import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from allennlp.modules.seq2seq_encoders import PytorchTransformer
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder, BertPooler
from overrides import overrides

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from gucorpling_models.features import Feature, get_feature_modules

logger = logging.getLogger(__name__)


def weighted_sum(att, mat):
    if att.dim() == 2 and mat.dim() == 3:
        return att.unsqueeze(1).bmm(mat).squeeze(1)
    elif att.dim() == 3 and mat.dim() == 3:
        return att.bmm(mat)
    else:
        AssertionError('Incompatible attention weights and matrix.')


def concat_masked_sequences(s1, s2, m1, m2, sm1, sm2, combine_using_submask):
    """
    First, combine two batches of masked sequences so that the sequences are flush against each other.
    For example, if we have:

        s1 = torch.tensor([[[0.1],[0.2],[0]]])
        s2 = torch.tensor([[[0.3],[0],[0]]])
        m1 = torch.tensor([[1,1,0]])
        m2 = torch.tensor([[1,0,0]])

    the output will be:

        combined = torch.tensor([[[0.1],[0.2],[0.3]]])
        combined_mask = torch.tensor([[1,1,1]])
        masks_1 = torch.tensor([[1,1,0]])
        masks_2 = torch.tensor([[0,0,1]])

    Note that we ASSUME that all arguments have the same dim0 (batch size), and
    that masks fit the regular expression 1*0* where 1 indicates a real value and 0
    indicates a non-value.

    sm1 and sm2 are submasks that are catted to the output.
    For sm1=torch.tensor([[[0,1,0]]]) and sm2=torch.tenssor([[[1,0,0]]]):

        catted = torch.tensor([[[0.1,0,0],[0.2,1.0,0],[0.3,0,1.0]]])
    """
    device = s1.device
    batch_size, _, emb_dim = s1.shape
    sm1 = sm1.bool()
    sm2 = sm2.bool()

    lengths_1 = util.get_lengths_from_binary_sequence_mask(m1)
    lengths_2 = util.get_lengths_from_binary_sequence_mask(m2)
    total_lengths = lengths_1 + lengths_2
    max_length = torch.max(total_lengths).item()
    if combine_using_submask:
        sublengths_1 = util.get_lengths_from_binary_sequence_mask(sm1)
        sublengths_2 = util.get_lengths_from_binary_sequence_mask(sm2)
        total_sublengths = sublengths_1 + sublengths_2
        max_length = torch.max(total_sublengths).item()

    instance_tensors = []
    combined_masks = []
    masks_1 = []
    masks_2 = []
    submasks_1 = []
    submasks_2 = []
    for i in range(batch_size):
        if combine_using_submask:
            t1 = torch.masked_select(s1[i], sm1[i].unsqueeze(-1)).reshape((-1, emb_dim))
            t2 = torch.masked_select(s2[i], sm2[i].unsqueeze(-1)).reshape((-1, emb_dim))
            length_1 = t1.shape[0]
            length_2 = t2.shape[0]
        else:
            length_1 = lengths_1[i]
            length_2 = lengths_2[i]
            t1 = s1[i][:length_1]
            t2 = s2[i][:length_2]

        padding_length = max_length - (length_1 + length_2)
        padding = torch.zeros(padding_length, t1.shape[-1]).to(device)
        combined = torch.cat((t1, t2, padding), dim=0).to(device)
        instance_tensors.append(combined)

        ones_1 = torch.ones(length_1).to(device)
        ones_2 = torch.ones(length_2).to(device)
        zeros_1 = torch.zeros(length_1).to(device)
        zeros_2 = torch.zeros(length_2).to(device)
        zeros_pad = torch.zeros(padding_length).to(device)
        combined_mask = torch.cat((ones_1, ones_2, zeros_pad)).to(device)
        mask_1 = torch.cat((ones_1, zeros_2, zeros_pad)).to(device)
        mask_2 = torch.cat((zeros_1, ones_2, zeros_pad)).to(device)
        masks_1.append(mask_1)
        masks_2.append(mask_2)
        combined_masks.append(combined_mask)

        submask_1 = torch.cat((sm1[i][:length_1], zeros_2, zeros_pad)).to(device)
        submask_2 = torch.cat((zeros_1, sm2[i][:length_2], zeros_pad)).to(device)
        submasks_1.append(submask_1)
        submasks_2.append(submask_2)

    combined = torch.stack(instance_tensors, dim=0).to(device)
    combined_masks = torch.stack(combined_masks, dim=0).to(device)
    masks_1 = torch.stack(masks_1, dim=0).to(device)
    masks_2 = torch.stack(masks_2, dim=0).to(device)
    submasks_1 = torch.stack(submasks_1, dim=0).to(device)
    submasks_2 = torch.stack(submasks_2, dim=0).to(device)

    catted = torch.cat((combined, submasks_1.unsqueeze(-1), submasks_2.unsqueeze(-1)), dim=2).to(device)

    return combined, combined_masks.bool(), masks_1.bool(), masks_2.bool(), catted


@Model.register("disrpt_2021_combined_sequence")
class CombinedSequenceModel(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder_args: dict,
            dropout: float = 0.3,
            features: Dict[str, Feature] = None,
            combine_sequences_using_submask: bool = False,
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.combine_sequences_using_submask = combine_sequences_using_submask

        self.embedder = text_field_embedder
        self.bert_pooler = BertPooler(text_field_embedder._token_embedders['tokens'].config._name_or_path)

        # setup handwritten feature modules
        if features is not None and len(features) > 0:
            feature_modules, feature_dims = get_feature_modules(features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0

        encoder_args["input_dim"] += feature_dims
        encoder_args["input_dim"] += 2  # submask dims
        encoder_args["input_dim"] += 1  # direction dims
        self.dummy_dim = 0
        while encoder_args["input_dim"] % encoder_args["num_attention_heads"] > 0:
            self.dummy_dim += 1
            encoder_args["input_dim"] += 1
        self.encoder = PytorchTransformer(**encoder_args)
        self.seq2vec = LstmSeq2VecEncoder(
            input_size=encoder_args["input_dim"],
            num_layers=1,
            bias=True,
            hidden_size=256,
            bidirectional=True,
        )

        num_relations = vocab.get_vocab_size("relation_labels")
        self.dropout = torch.nn.Dropout(dropout)

        self.relation_decoder = torch.nn.Linear(
            self.bert_pooler.get_output_dim() * 2 + self.seq2vec.get_output_dim() + 1, num_relations
        )

        # these are stateful objects that keep track of accuracy across an epoch
        self.relation_accuracy = CategoricalAccuracy()

        # convenience dict mapping relation indices to labels
        self.relation_labels = self.vocab.get_index_to_token_vocabulary("relation_labels")
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
            relation: torch.Tensor,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        # dist_embeds = self._distance_embedding(distance)

        sentence1_mask = util.get_text_field_mask(sentence1)
        sentence2_mask = util.get_text_field_mask(sentence2)

        def truncate_sentence(s):
            s["tokens"]["token_ids"] = s["tokens"]["token_ids"][:, :512]
            s["tokens"]["mask"] = s["tokens"]["mask"][:, :512]
            s["tokens"]["type_ids"] = s["tokens"]["type_ids"][:, :512]
        truncate_sentence(sentence1)
        truncate_sentence(sentence2)
        unit1_span_mask = unit1_span_mask[:, :512]
        unit2_span_mask = unit2_span_mask[:, :512]

        embedded_unit1_sentence = self.embedder(sentence1)
        embedded_unit2_sentence = self.embedder(sentence2)

        unit1_vec = self.bert_pooler(embedded_unit1_sentence)
        unit2_vec = self.bert_pooler(embedded_unit2_sentence)

        _, combined_mask, _, _, combined_sequence = concat_masked_sequences(
            embedded_unit1_sentence,
            embedded_unit2_sentence,
            sentence1_mask,
            sentence2_mask,
            unit1_span_mask,
            unit2_span_mask,
            combine_using_submask=self.combine_sequences_using_submask
        )

        dummy_shape = combined_sequence.shape[0:2] + torch.Size([self.dummy_dim])
        dummy_slice = torch.zeros(dummy_shape).to(combined_sequence.device)
        combined_sequence = torch.cat((combined_sequence, dummy_slice), dim=2).to(combined_sequence.device)

        # Get the features
        sequence_length = combined_sequence.shape[1]
        expanded_direction = direction.unsqueeze(-1).expand(-1, sequence_length).unsqueeze(-1)
        if self.feature_modules:
            features = self._get_combined_feature_tensor(kwargs).unsqueeze(1).expand(-1, sequence_length, -1)
            combined_sequence = torch.cat((combined_sequence, expanded_direction, features), 2)
        else:
            combined_sequence = torch.cat((combined_sequence, expanded_direction), 2)

        encoded_sequence = self.encoder(combined_sequence, combined_mask)
        encoded_sequence = self.dropout(encoded_sequence)

        sequence_embedding = self.seq2vec(encoded_sequence, combined_mask)
        combined_pair_embedding = torch.cat((unit1_vec, unit2_vec, sequence_embedding, direction.unsqueeze(-1)), dim=1)
        # Decode the concatenated vector into relation logits
        relation_logits = self.relation_decoder(combined_pair_embedding)
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
