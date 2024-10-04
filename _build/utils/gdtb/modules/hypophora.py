"""
@Amir

Module to add hypophora relations based on RST topic-question relations.
TODO: improve argspans
"""

import os
from argparse import ArgumentParser
from modules.base import ConvertBase


class Hypophora(ConvertBase):
    def __init__(self, data_dir, direct_mappings_dir, disco_pred_dir):
        super().__init__(data_dir, direct_mappings_dir, disco_pred_dir)

    def convert(self, doc, rel):
        if "question" not in rel.relname:  # if not a question skip
            return
        elif "cache" in rel.pdtb_rels:
            return
        rel.pdtb_rels['hypophora'].append(("hypophora", "hypophora", [], ""))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", help="data dir", default='../data')
    parser.add_argument("-f", help="example filename", default='GUM_interview_chomsky')
    args = parser.parse_args()

    data_dir = args.i
    docname = args.f
    conllu_dir = os.path.join(data_dir, "dep", docname+".conllu")
    rs4_dir = os.path.join(data_dir, "rst", "rstweb", docname+".rs4")
    direct_mappings_dir = os.path.join(data_dir, 'mappings.json')
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')

    hypophora_module = Hypophora(data_dir, direct_mappings_dir, disco_pred_dir)
    doc = hypophora_module.process_files(conllu_dir, rs4_dir, docname)

    for node_id, rel in doc.rels.items():
        hypophora_module.convert(doc, rel)
        hypophora_module.output(docname, node_id, rel, 'hypophora')
