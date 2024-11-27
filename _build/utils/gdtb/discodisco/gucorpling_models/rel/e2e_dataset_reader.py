import csv
import io, os
import math
from typing import Dict, Iterable, Optional, List, Tuple, Any
from pprint import pprint
from collections import defaultdict

from overrides import overrides

import torch
from torch import tensor
from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField, TensorField, ArrayField, ListField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from gucorpling_models.features import Feature, get_feature_field
from gucorpling_models.rel.features import process_relfile
from gucorpling_models.seg.gumdrop_reader import read_conll_conn
from gucorpling_models.seg.dataset_reader import group_by_sentence


def get_span_indices(unit_toks, s_toks, max_length: None):
    s_start, s_end = int(s_toks.split("-")[0]), int(s_toks.split("-")[-1])
    # REPLACED_FLAG = False

    if "," in unit_toks:
        splitted = unit_toks.split(",")
        if len(splitted) > 2:
            a = 1
        # cur_span = []
        span = []
        for chunk in splitted:
            s, e = int(chunk.split("-")[0]) - s_start, int(chunk.split("-")[-1]) - s_start
            span.append((s, e))
    else:
        left, right = int(unit_toks.split("-")[0])-s_start, int(unit_toks.split("-")[-1])-s_start
        span = [(left, right)]
    return span


def get_span_dist(unit1, unit2):
    unit1_start_indice = int(unit1.split(",")[0].split("-")[0])
    unit2_start_indice = int(unit2.split(",")[0].split("-")[0])
    return unit1_start_indice-unit2_start_indice


@DatasetReader.register("disrpt_2021_rel_e2e")
class Disrpt2021RelReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_length: int = None,
        features: Dict[str, Feature] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_length = max_length  # useful for BERT
        self.features = features

    def tokenize_with_subtoken_map(self, text, span):
        subtoken_map = [0]
        token_to_subtokens = {}
        count = 1

        tokenized_text = [Token('[CLS]')]
        for i, word in enumerate(text.split(' ')):
            tokenized = self.tokenizer.tokenize(word)
            # if self.max_length and i >= self.max_length:
            #     break
            tokenized_text += tokenized[1:-1]
            # if self.max_length and len(tokenized_text) >= self.max_length - 1:
            #     break
            subtoken_map += [i+1] * (len(tokenized)-2)
            token_to_subtokens[i] = (count, count+len(tokenized)-3)
            count += len(tokenized)-2
        tokenized_text += [Token('[SEP]')]
        subtoken_map.append(subtoken_map[-1] + 1)

        # if span is larger than the max length, only use the EDU text
        # span starts from 0 and subtoken_map starts from 1
        if span[-1][-1]+1 not in subtoken_map[:self.max_length]:
            tokenized_span = [Token('[CLS]')]
            for i,s in enumerate(span):
                tokenized_span += tokenized_text[token_to_subtokens[s[0]][0]:token_to_subtokens[s[1]][-1]+1]
                a = 1
            tokenized_span += [Token('[SEP]')]
            tokenized_text = tokenized_span

            span_mask = [1] * len(tokenized_text)
            span_mask[0], span_mask[-1] = 0, 0
            span = [(1, 511)] if len(tokenized_text) > self.max_length else [(1, len(tokenized_text)-2)]

        # otherwise use the sentence text
        else:
            span_mask = [0] * len(tokenized_text)

            for i, s in enumerate(span):
                s_start, s_end = s
                new_start, new_end = token_to_subtokens[s_start][0], token_to_subtokens[s_end][-1]
                span[i] = (new_start, new_end)
                for x in range(new_start, new_end+1):
                    span_mask[x] = 1
            assert len(subtoken_map) == len(tokenized_text)

        return tokenized_text[:self.max_length], span, span_mask[:self.max_length]

    @overrides
    def text_to_instance(
        self,  # type: ignore
        unit1_txt: str,
        unit1_sent: str,
        unit2_txt: str,
        unit2_sent: str,
        # span_dist: int,
        unit1_span_indices: list,
        unit2_span_indices: list,
        dir: str,
        label: str = None,
        features: Dict[str, Any] = None
    ) -> Instance:

        unit1_sent_tokens, new_unit1_span_indices, unit1_span_mask = self.tokenize_with_subtoken_map(unit1_sent, unit1_span_indices)
        unit2_sent_tokens, new_unit2_span_indices, unit2_span_mask = self.tokenize_with_subtoken_map(unit2_sent, unit2_span_indices)

        # sent_tokens = unit1_sent_tokens + unit2_sent_tokens[1:]
        # adjusted_unit1_span_mask = unit1_span_mask + [0] * (len(unit2_span_mask)-1)
        # adjusted_unit2_span_mask = [0] * len(unit1_span_mask) + unit2_span_mask[1:]

        # unit1_txt_tokens = self.tokenizer.tokenize(unit1_txt)
        # unit2_txt_tokens = self.tokenizer.tokenize(unit2_txt)

        # unit1_span_idx = [i for i, n  in enumerate(unit1_span_mask) if n != 0]
        # unit2_span_idx = [i for i, n  in enumerate(unit2_span_mask) if n != 0]

        fields: Dict[str, Field] = {
            "sentence1": TextField(unit1_sent_tokens[:self.max_length], self.token_indexers),
            "sentence2": TextField(unit2_sent_tokens[:self.max_length], self.token_indexers),
            "unit1_span_mask": TensorField(torch.tensor(unit1_span_mask[:self.max_length]) > 0),
            "unit1_span_indices": TensorField(torch.tensor(new_unit1_span_indices)),
            "unit2_span_mask": TensorField(torch.tensor(unit2_span_mask[:self.max_length]) > 0),
            "unit2_span_indices": TensorField(torch.tensor(new_unit2_span_indices)),
            "direction": LabelField(dir, label_namespace="direction_labels"),
            # "distance": TensorField(torch.tensor(span_dist))
        }

        # read in handcrafted features
        if self.features is not None:
            for feature_name, feature_config in self.features.items():
                if feature_name not in features:
                    raise Exception(f"Feature {feature_name} not found. Pair:\n  {unit1_txt}\n  {unit2_txt}")
                feature_data = features[feature_name]
                fields[feature_name] = get_feature_field(feature_config, feature_data)

        if label:
            fields["relation"] = LabelField(label, label_namespace="relation_labels")
        return Instance(fields)


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".rels")

        corpus = file_path.split(os.sep)[-1].split('_')[0]
        features = process_relfile(
            file_path,
            file_path.replace(".rels", ".conllu"),
            corpus
        )

        with io.open(file_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                # doc = row["doc"]
                unit1_toks = row["unit1_toks"]
                s1_toks = row["s1_toks"]
                unit2_toks = row["unit2_toks"]
                s2_toks = row["s2_toks"]

                # span_dist = get_span_dist(unit1_toks, unit2_toks)
                # span_dist = round(math.log(abs(span_dist)+1, 2))   # smooth the distance
                unit1_span_indices = get_span_indices(unit1_toks, s1_toks, self.max_length)
                unit2_span_indices = get_span_indices(unit2_toks, s2_toks, self.max_length)

                # try:
                yield self.text_to_instance(
                    unit1_txt=row["unit1_txt"],
                    unit1_sent=row["unit1_sent"],
                    unit2_txt=row["unit2_txt"],
                    unit2_sent=row["unit2_sent"],
                    # span_dist=span_dist,
                    unit1_span_indices=unit1_span_indices,
                    unit2_span_indices=unit2_span_indices,
                    dir=row["dir"],
                    label=row["label"],
                    features=features[i]
                )
                # except:
                #     with open('bug.txt', 'a', encoding='utf-8') as f:
                #         f.write(str(row) + '\n')
                #     continue