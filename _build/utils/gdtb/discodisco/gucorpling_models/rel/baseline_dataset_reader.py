import csv
import logging
import os
from typing import Dict, Iterable, Any
from pprint import pprint

from allennlp.data import DatasetReader, Instance, Field
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from gucorpling_models.features import Feature, get_feature_field
from gucorpling_models.rel.features import process_relfile

logger = logging.getLogger(__name__)


@DatasetReader.register("disrpt_2021_rel")
class Disrpt2021RelReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        features: Dict[str, Feature] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
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
        combined_txt_tokens = unit1_txt_tokens + unit2_txt_tokens[1:]
        combined_sent_tokens = unit1_sent_tokens + unit2_sent_tokens[1:]

        fields: Dict[str, Field] = {
            "unit1_body": TextField(unit1_txt_tokens, self.token_indexers),
            "unit1_sentence": TextField(unit1_sent_tokens, self.token_indexers),
            "unit2_body": TextField(unit2_txt_tokens, self.token_indexers),
            "unit2_sentence": TextField(unit2_sent_tokens, self.token_indexers),
            "combined_body": TextField(combined_txt_tokens, self.token_indexers),
            "combined_sentence": TextField(combined_sent_tokens, self.token_indexers),
            "direction": LabelField(dir, label_namespace="direction_labels"),
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

    def _read(self, file_path: str) -> Iterable[Instance]:
        assert file_path.endswith(".rels")

        rels_file_path = file_path
        # conllu_file_path = rels_file_path.replace(".rels", ".conllu")
        # tok_file_path = rels_file_path.replace(".rels", ".tok")
        corpus = file_path.split(os.sep)[-1].split('_')[0]
        features = process_relfile(
            file_path,
            file_path.replace(".rels", ".conllu"),
            corpus
        )

        with open(rels_file_path, "r") as f:
            logger.debug("reading " + rels_file_path)
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            # Keys:
            # METADATA
            # doc - name of the document
            # LABELS
            # label - label of the relation between the two units in the CODI-DISRPT 2021 constrained vocabulary
            # orig_label - label of the dependency between the two units as it is in the original treebank
            # dir - 1>2 if unit 1 dominates unit 2, 1<2 otherwise
            # DATA
            # unit1_txt, unit2_txt - DISCOURSE UNIT-level, whitespace-tokenized tokens (for use with conllu)
            # unit1_toks, unit2_toks - DISCOURSE UNIT-level, document-wide token ID range
            # unit1_sent, unit2_sent - SENTENCE-level, whitespace-tokenized tokens (for use with conllu)
            # s1_toks, s2_toks - SENTENCE-level, document-wide token ID range

            # Some gotchas:
            # - A special symbol <*> is used to indicate a break in discontinuous units
            # - For discontinuous units, tok ranges will be separated by a comma, like 1-3,5-8

            # Full example:
            # {'doc': 'GUM_academic_exposure',
            #  ## labels
            #  'label': 'preparation',
            #  'orig_label': 'preparation',
            #  ## data
            #  'dir': '1>2',
            #
            #  'unit1_txt': 'Introduction',
            #  'unit1_toks': '1',
            #  'unit1_sent': 'Introduction',
            #  's1_toks': '1',
            #
            #  'unit2_txt': 'In the present study , we examine the outcomes of such a period '
            #               'of no exposure on the neurocognition of L2 grammar :',
            #  'unit2_toks': '186-208',
            #  'unit2_sent': 'In the present study , we examine the outcomes of such a '
            #                'period of no exposure on the neurocognition of L2 grammar : '
            #                'that is , whether a substantial period of no exposure leads to '
            #                'decreased proficiency and / or less native-like neural '
            #                'processes ( “ use it or lose it ” [ 20 ] ) , no such changes , '
            #                'or perhaps whether even higher proficiency and / or more '
            #                'native-like processing may be observed .',
            #  's2_toks': '186-262'}

            #    2019/deu.rst.pcc/deu.rst.pcc_dev.rels
            # => deu.rst.pcc_dev.rels
            # => deu.rst.pcc

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
