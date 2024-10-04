import csv
import logging
import os
from typing import Dict, Iterable, Any
from pprint import pprint

from allennlp.data import DatasetReader, Instance, Field, Token
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from gucorpling_models.features import Feature, get_feature_field, FeatureBundle
from gucorpling_models.rel.features import process_relfile

logger = logging.getLogger(__name__)


@DatasetReader.register("disrpt_2021_rel_flair_clone")
class Disrpt2021FlairCloneReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_length: int = 511,
        features: FeatureBundle = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_length = max_length
        self.features = features

    def text_to_instance(  # type: ignore
        self,
        unit1_txt: str,
        unit1_sent: str,
        unit2_txt: str,
        unit2_sent: str,
        dir: str,
        label: str = None,
        features: Dict[str, Any] = None
    ) -> Instance:
        unit1_txt_tokens = self.tokenizer.tokenize(unit1_txt)
        unit1_sent_tokens = self.tokenizer.tokenize(unit1_sent)
        unit2_txt_tokens = self.tokenizer.tokenize(unit2_txt)
        unit2_sent_tokens = self.tokenizer.tokenize(unit2_sent)
        cls_token = unit1_txt_tokens[0]
        sep_token = unit1_txt_tokens[-1]
        if cls_token.text not in ["[CLS]", '<s>']:
            raise Exception(f"Unrecognized cls token: {cls_token.text}")
        if sep_token.text not in ["[SEP]", '</s>']:
            raise Exception(f"Unrecognized sep token: {sep_token.text}")

        if dir == "1<2":
            left_tokens = []
            right_tokens = [Token("{")]
            dir_tokens = self.tokenizer.tokenize(sep_token.text + " <")[1:-1]
        else:
            left_tokens = [Token("}")]
            right_tokens = []
            dir_tokens = self.tokenizer.tokenize("> " + sep_token.text)[1:-1]
        combined_txt_tokens = (
            unit1_txt_tokens[:1]
            + left_tokens
            + unit1_txt_tokens[1:-1]
            + dir_tokens
            + unit2_txt_tokens[1:-1]
            + right_tokens
            + unit2_txt_tokens[-1:]
        )
        combined_sent_tokens = (
            unit1_sent_tokens[:1]
            + left_tokens
            + unit1_sent_tokens[1:-1]
            + dir_tokens
            + unit2_sent_tokens[1:-1]
            + right_tokens
            + unit2_sent_tokens[-1:]
        )

        fields: Dict[str, Field] = {
            "combined_body": TextField(combined_txt_tokens[:self.max_length], self.token_indexers),
            "combined_sentence": TextField(combined_sent_tokens[:self.max_length], self.token_indexers),
            "direction": LabelField(dir, label_namespace="direction_labels"),
        }

        # read in handcrafted features
        if self.features is not None:
            features_configs = self.features.features

            for feature_name, feature_config in features_configs.items():
                if feature_name not in features:
                    raise Exception(f"Feature {feature_name} not found. Pair:\n  {unit1_txt}\n  {unit2_txt}")
                if feature_name not in self.features.corpus_keys:
                    continue
                feature_data = features[feature_name]
                fields[feature_name] = get_feature_field(feature_config, feature_data)

        if label:
            fields["relation"] = LabelField(label, label_namespace="relation_labels")
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".rels")
        rels_file_path = file_path
        corpus = file_path.split(os.sep)[-1].split('_')[0]
        features = process_relfile(
            file_path,
            file_path.replace(".rels", ".conllu"),
            corpus
        )

        with open(rels_file_path, "r") as f:
            logger.debug("reading " + rels_file_path)
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

            for i, row in enumerate(reader):
                yield self.text_to_instance(
                    unit1_txt=row["unit1_txt"],
                    unit1_sent=row["unit1_sent"],
                    unit2_txt=row["unit2_txt"],
                    unit2_sent=row["unit2_sent"],
                    dir=row["dir"],
                    label=row["label"],
                    features=features[i]
                )
