from pprint import pprint
from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import (
    TextFieldEmbedder,
    Seq2VecEncoder,
    TimeDistributed,
    ConditionalRandomField,
    FeedForward,
)
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, PytorchTransformer, FeedForwardEncoder
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure

from gucorpling_models.seg.util import detect_encoding
from gucorpling_models.features import get_feature_modules, Feature


@Model.register("disrpt_2021_seg_baseline")
class Disrpt2021Baseline(Model):
    """
    A simple seq2seq baseline for discourse segmentation. It:
    - embeds the sentence and neighboring sentences
    - uses a seq2vec encoder for the neighboring sentences
    - seq2seq encodes the sentence along with categorical features such as POS tags
    - decodes using a CRF

    Based in part on https://github.com/allenai/allennlp-models/blob/main/allennlp_models/tagging/models/crf_tagger.py
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder_hidden_dim: int,
        encoder_recurrent_dropout: float,
        prev_sentence_encoder: Seq2VecEncoder,
        next_sentence_encoder: Seq2VecEncoder,
        use_crf: bool = True,
        token_features: Dict[str, Feature] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        dropout: float = 0.5,
        feature_dropout: float = 0.3,
    ):
        super().__init__(vocab)
        print(vocab)
        for k, v in vocab._index_to_token.items():
            if len(v) < 50:
                print(k)
                print(" ", v)
        self.labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(self.labels)

        # modules for features
        if token_features is not None and len(token_features) > 0:
            feature_modules, feature_dims = get_feature_modules(token_features, vocab)
            self.feature_modules = feature_modules
        else:
            self.feature_modules = None
            feature_dims = 0

        # encoding --------------------------------------------------
        self.embedder = embedder
        encoder_input_dim = embedder.get_output_dim() + next_sentence_encoder.get_output_dim() * 2 + feature_dims
        # self.encoder = PytorchTransformer(
        #    encoder_input_dim,
        #    1,
        #    feedforward_hidden_dim=512,
        #    num_attention_heads=4,
        #    positional_encoding="embedding",
        #    positional_embedding_size=256,
        #    activation="gelu",
        #    dropout_prob=0.0
        # )
        self.encoder = PytorchSeq2SeqWrapper(
            StackedBidirectionalLstm(
                encoder_input_dim, encoder_hidden_dim, 1, recurrent_dropout_probability=encoder_recurrent_dropout,
            )
        )
        self.prev_sentence_encoder = prev_sentence_encoder
        self.next_sentence_encoder = next_sentence_encoder
        self.feature_dropout = torch.nn.Dropout(feature_dropout)
        self.dropout = torch.nn.Dropout(dropout)

        # decoding --------------------------------------------------
        hidden_size = self.encoder.get_output_dim()

        # encoding scheme
        label_encoding, encoding_map = detect_encoding(self.labels)
        if use_crf:
            self.label_projection_layer = TimeDistributed(torch.nn.Linear(hidden_size, num_labels))
            self._encoding_map = encoding_map
            constraints = allowed_transitions(label_encoding, self.labels)
            self.crf = ConditionalRandomField(num_labels, constraints, include_start_end_transitions=True)
        else:
            self.crf = None
            # self.decode_feedforward = PytorchTransformer(
            #     hidden_size,
            #     1,
            #     feedforward_hidden_dim=1024,
            #     num_attention_heads=8,
            #     positional_encoding="sinusoidal",
            #     positional_embedding_size=256
            # )
            # self.decode_feedforward = FeedForwardEncoder(
            #     FeedForward(
            #         hidden_size, 2, [512, 256], Activation.by_name("tanh")(), 0.0
            #     )
            # )
            self.label_projection_layer = TimeDistributed(torch.nn.Linear(hidden_size, num_labels))

        # util --------------------------------------------------
        # these are stateful objects that keep track of accuracy across an epoch
        self.metrics = {
            "span_f1": SpanBasedF1Measure(vocab, tag_namespace="labels", label_encoding=label_encoding),
            "accuracy": CategoricalAccuracy(),
        }

        initializer(self)
        print(self)

    def _get_encoded_context(self, prev_sentence, next_sentence, num_tokens):
        """
        Represent neighboring sentences using seq2vec encoders. The shape of the vec is
        (batch_size, context_encoder_hidden_dim * 2), but we expand this to
        (batch_size, num_toks, context_encoder_hidden_dim * 2) in order to make it easy to combine with the sequence
        """
        embedded_prev_sentence = self.embedder(prev_sentence)
        embedded_next_sentence = self.embedder(next_sentence)
        encoded_prev_sentence = self.prev_sentence_encoder(embedded_prev_sentence, get_text_field_mask(prev_sentence))
        encoded_next_sentence = self.next_sentence_encoder(embedded_next_sentence, get_text_field_mask(next_sentence))
        context = torch.cat((encoded_prev_sentence, encoded_next_sentence), 1)
        # Context is of shape (batch_size, h_1 + h_2).
        # This turns it into (batch_size, num_tokens, h_1 + h_2)
        time_distributed_context = context.unsqueeze(1).expand(-1, num_tokens, -1)
        return time_distributed_context

    def _get_combined_feature_tensor(self, kwargs):
        output_tensors = []
        for module_key, module in self.feature_modules.items():
            output_tensor = module(kwargs[module_key])
            if len(output_tensor.shape) == 2:
                output_tensor = output_tensor.unsqueeze(-1)
            output_tensors.append(output_tensor)

        combined_feature_tensor = torch.cat(output_tensors, dim=2)
        return self.feature_dropout(combined_feature_tensor)

    def forward(  # type: ignore
        self,
        sentence: TextFieldTensors,
        prev_sentence: TextFieldTensors,
        next_sentence: TextFieldTensors,
        sentence_tokens: List[str],
        labels: torch.LongTensor = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)

        # Encoding --------------------------------------------------
        embedded_sentence = self.embedder(sentence)

        # Context from neighboring sentences
        context = self._get_encoded_context(prev_sentence, next_sentence, embedded_sentence.shape[1])

        # Combine everything we'll be feeding to the encoder
        encoder_inputs = [embedded_sentence, context]
        if self.feature_modules is not None:
            encoder_inputs.append(self._get_combined_feature_tensor(kwargs))
        encoder_input = torch.cat(encoder_inputs, dim=2)

        encoded_sequence = self.encoder(encoder_input, mask)
        encoded_sequence = self.dropout(encoded_sequence)

        # Decoding --------------------------------------------------
        if self.crf:
            # project into the label space and use viterbi decoding on the CRF
            label_logits = self.label_projection_layer(encoded_sequence)
            pred_labels = [
                best_label_seq
                for best_label_seq, viterbi_score in self.crf.viterbi_tags(label_logits, mask, top_k=None)
            ]
        else:
            label_logits = self.label_projection_layer(encoded_sequence)
            pred_labels = torch.argmax(label_logits, dim=2)

        output = {
            "tokens": sentence_tokens,
            "label_logits": label_logits,
            "mask": mask,
            "pred_labels": pred_labels,
        }
        if labels is not None:
            self._compute_loss_and_metrics(label_logits, pred_labels, labels, mask, output)

        return output

    def _compute_loss_and_metrics(self, label_logits, pred_labels, labels, mask, output):
        output["gold_labels"] = labels

        if self.crf:
            # Represent viterbi labels as "class probabilities" that we can feed into the metrics
            class_probabilities = label_logits * 0.0
            for i, instance_labels in enumerate(pred_labels):
                for j, label_id in enumerate(instance_labels):
                    class_probabilities[i, j, label_id] = 1
            log_likelihood = self.crf(label_logits, labels, mask)
            output["loss"] = -log_likelihood
            for metric in self.metrics.values():
                metric(class_probabilities, labels, mask)
        else:
            output["loss"] = sequence_cross_entropy_with_logits(label_logits, labels, mask)
            for metric in self.metrics.values():
                metric(label_logits, labels, mask)

    # Takes output of forward() and turns tensors into strings or probabilities wherever appropriate
    # Note that the output dict, because it's just from forward(), represents a batch, not a single
    # instance: every key will have a list that's the size of the batch
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        def decode_labels(labels):
            return [self.vocab.get_token_from_index(int(label), "labels") for label in labels]

        output_dict["pred_labels"] = [decode_labels(t) for t in output_dict["pred_labels"]]
        output_dict["gold_labels"] = [decode_labels(t) for t in output_dict["gold_labels"]]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"tag_accuracy": self.metrics["accuracy"].get_metric(reset)}
        f1_metrics = {
            "span_" + k.replace("-overall", "").replace("-measure", ""): v
            for k, v in self.metrics["span_f1"].get_metric(reset=reset).items()
            if "overall" in k
        }
        metrics.update(f1_metrics)
        return metrics  # type: ignore
