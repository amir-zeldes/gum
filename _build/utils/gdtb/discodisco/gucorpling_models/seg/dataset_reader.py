# add categorical features from below to a neural baseline:
# https://github.com/gucorpling/GumDrop2/blob/master/lib/conll_reader.py#L271

import csv
import os
import sys
from typing import Dict, Iterable, Any, List, Optional, Tuple
from pprint import pprint
import re

import torch
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from gucorpling_models.features import get_feature_field, Feature
from gucorpling_models.seg.gumdrop_reader import read_conll_conn


def group_by_sentence(token_dicts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    sentences = []

    current_s_id = None
    sentence: List[Dict[str, Any]] = []
    for token in token_dicts:
        s_id = token["s_id"]
        if s_id != current_s_id:
            if sentence:
                sentences.append(sentence)
            sentence = []
            current_s_id = s_id
        sentence.append(token)
    if sentence:
        sentences.append(sentence)
    return sentences


LABEL_TO_ENCODING = {
    "BeginSeg": "B",
    "_": "O",
    "Seg=B-Conn": "B-Conn",
    "Seg=I-Conn": "I-Conn",
}


# Corpus-specific preprocessing, currently used only for russian
def preprocess_text(file_path, tokens):
    for i in range(len(tokens)):
        token = tokens[i]
        # see 94175 in rus.rst.rrt_train.conll
        if "rus.rst.rrt" in file_path:
            # Dates explode when they're wordpiece tokenized and some sentences have a lot--replace them with "Tuesday"
            if re.match(r"\d\d.\d\d.\d\d\d\d", token):
                tokens[i] = "вторник"
            # So many weird backslashes for some reason
            elif token.endswith("\\") and len(token) > 1:
                tokens[i] = token[:-1]
            elif token.endswith("\\."):
                tokens[i] = token[:-2] + "."
            # urls
            elif token.startswith("http://") or token.startswith("https://") or token.startswith("www."):
                tokens[i] = "веб-сайт"
    return tokens


@DatasetReader.register("disrpt_2021_seg")
class Disrpt2021SegReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        document_boundary_token: str = "@@DOCUMENT_BOUNDARY@@",
        token_features: Dict[str, Feature] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens if max_tokens is not None else sys.maxsize  # useful for BERT
        self.document_boundary_token = document_boundary_token
        self.token_features = token_features

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["sentence"].token_indexers = self.token_indexers  # type: ignore
        instance.fields["prev_sentence"].token_indexers = self.token_indexers  # type: ignore
        instance.fields["next_sentence"].token_indexers = self.token_indexers  # type: ignore

    def text_to_instance(  # type: ignore
        self,
        sentence: str,
        prev_sentence: Optional[str],
        next_sentence: Optional[str],
        labels: List[str],
        features: Dict[str, Any],
    ) -> Instance:
        if prev_sentence is None:
            prev_sentence = self.document_boundary_token
        if next_sentence is None:
            next_sentence = self.document_boundary_token
        sentence_tokens = self.tokenizer.tokenize(sentence)
        prev_sentence_tokens = self.tokenizer.tokenize(prev_sentence)
        next_sentence_tokens = self.tokenizer.tokenize(next_sentence)

        if len(sentence_tokens) != len(labels):
            raise ValueError(
                f"Found {len(sentence_tokens)} tokens but {len(labels)} labels. "
                "If you are using a transformer embedding model like BERT, you should be "
                "using a whitespace tokenizer and the special PretrainedTransformerMismatchedIndexer "
                "and PretrainedTransformerMismatchedEmbedder. See: "
                "http://docs.allennlp.org/main/api/data/token_indexers/pretrained_transformer_mismatched_indexer/"
            )

        sentence_field = TextField(sentence_tokens)
        # note: if a namespace ends in _tags, it won't get an OOV token automatically. Use only
        # for fields where you're 100% certain all values will occur in train
        fields: Dict[str, Field] = {
            "sentence": sentence_field,
            "prev_sentence": TextField(prev_sentence_tokens),
            "next_sentence": TextField(next_sentence_tokens),
            "sentence_tokens": MetadataField(sentence_tokens),
        }

        # read in handcrafted features
        if self.token_features is not None:
            for feature_name, token_feature in self.token_features.items():
                if feature_name not in features.keys():
                    raise Exception(f"Feature {feature_name} not found. Sentence:\n  {sentence}")
                feature_data = features[feature_name]
                fields[feature_name] = get_feature_field(token_feature, feature_data, sentence_field)

        if labels:
            fields["labels"] = SequenceLabelField(labels, sentence_field)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".conll") or file_path.endswith(".conllu")

        conll_file_path = file_path
        # tok_file_path = rels_file_path.replace(".conll", ".tok")

        # use gumdrop's function for reading the conll
        token_dicts, _, _, _, _ = read_conll_conn(conll_file_path)
        token_dicts_by_sentence = group_by_sentence(token_dicts)
        # ks = token_dicts[0].keys()
        # for k in ks:
        #     print(k, set(x[k] for x in token_dicts))
        # assert False
        sentence_tokens = [
            preprocess_text(file_path, [td["word"] for td in sentence]) for sentence in token_dicts_by_sentence
        ]
        # Tokens shouldn't have space in them--if they do replace them with underscores
        sentence_tokens = [[token.replace(" ", "_") for token in sentence] for sentence in sentence_tokens]

        # read handcrafted features provided by gumdrop code
        features = [
            {
                feature_name: [td[fdict.source_key] for td in sentence]
                for feature_name, fdict in (self.token_features.items() if self.token_features else [])
            }
            for sentence in token_dicts_by_sentence
        ]

        for i, token_dicts in enumerate(token_dicts_by_sentence):
            prev_sentence = " ".join(sentence_tokens[i - 1]) if i > 0 else self.document_boundary_token
            sentence = " ".join(sentence_tokens[i])
            next_sentence = (
                " ".join(sentence_tokens[i + 1])
                if i < len(token_dicts_by_sentence) - 1
                else self.document_boundary_token
            )
            labels = [LABEL_TO_ENCODING[td["label"]] for td in token_dicts]
            yield self.text_to_instance(
                sentence=sentence,
                prev_sentence=prev_sentence,
                next_sentence=next_sentence,
                labels=labels,
                features=features[i],
            )
