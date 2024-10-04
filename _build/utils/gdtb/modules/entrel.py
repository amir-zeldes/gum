"""
@Amir

Module to add EntRel relations to consecutive sentence pairs without a higher ranked relations type,
provided they contain a coref signal.
"""

import os
from argparse import ArgumentParser
from modules.base import ConvertBase


class EntRel(ConvertBase):
    def __init__(self, data_dir, direct_mappings_dir, disco_pred_dir):
        super().__init__(data_dir, direct_mappings_dir, disco_pred_dir)
        self.allowed_relations = ["elaboration-additional_r", "organization-preparation_r","organization-heading_r"]

    def convert(self, doc, rel):
        if rel.relname not in self.allowed_relations:  # if not a question skip
            return
        if len(rel.pdtb_rels) > 0:  # EntRel cannot be added if other relations exist
            if any([reltype != "implicit" for reltype in rel.pdtb_rels]):
                return
            else:
                for prel in rel.pdtb_rels["implicit"]:
                    if prel[0] != "norel":
                        return
        if not any([s.sigtype=="reference" or s.subtype in ["repetition","meronymy","synoymy"] for s in rel.signals]):  # Must have a coref signal
            return
        source_sent_ids, target_sent_ids = sorted(list(rel.source.sent_ids)), sorted(list(rel.target.sent_ids))

        # Check if the sentence blocks are adjacent
        if any([s in target_sent_ids for s in source_sent_ids]) or any([s in source_sent_ids for s in target_sent_ids]):
            return  # No overlap
        if rel.is_forward:
            source_para = doc.sents[source_sent_ids[-1]].par_id
            target_para = doc.sents[target_sent_ids[0]].par_id
            if source_sent_ids[-1] != target_sent_ids[0] - 1 or source_para != target_para:
                return
        else:
            source_para = doc.sents[source_sent_ids[0]].par_id
            target_para = doc.sents[target_sent_ids[-1]].par_id
            if target_sent_ids[-1] != source_sent_ids[0] - 1 or source_para != target_para:
                return

        if "implicit" in rel.pdtb_rels:
            del rel.pdtb_rels["implicit"]
        rel.pdtb_rels['entrel'].append(("entrel", "EntRel", [], "_"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", help="data dir", default='../data')
    parser.add_argument("-d", help="document name", default='GUM_interview_chomsky')
    args = parser.parse_args()

    data_dir = args.i
    docname = args.d
    conllu_dir = os.path.join(data_dir, "dep", docname+".conllu")
    rs4_dir = os.path.join(data_dir, "rst", "rstweb", docname+".rs4")
    direct_mappings_dir = os.path.join(data_dir, 'mappings.json')
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')

    entrel_module = EntRel(data_dir, direct_mappings_dir, disco_pred_dir)
    doc = entrel_module.process_files(conllu_dir, rs4_dir, docname)

    for node_id, rel in doc.rels.items():
        entrel_module.convert(doc, rel)
        entrel_module.output(docname, node_id, rel, 'entrel')
