"""
@Amir

Cache module to add relations from the cache file to the document state.
These relations supercede any other relations inferred for the same key IDs and are guaranteed to be
added to the final output. Allows for manual correction of data (originally seeded from Google Spreadsheet annotations)
"""

import os
from argparse import ArgumentParser
from modules.base import ConvertBase
from collections import defaultdict
from nodes import Span
from copy import deepcopy


class Cache(ConvertBase):
    def __init__(self, data_dir, direct_mappings_dir, disco_pred_dir, filter="full"):
        super().__init__(data_dir, direct_mappings_dir, disco_pred_dir)
        cache_data = open(data_dir + os.sep + "cached_rels.tab").read().strip().split("\n")[1:]
        self.mapping = defaultdict(lambda :defaultdict(tuple))
        self.additional_rels = defaultdict(list)
        for line in cache_data:
            docname,reltype,conn,sense,rst_rel,first_arg,second_arg,target_ids,source_ids,key,notes = line.split("\t")
            if filter == "filtered":
                if "background" not in rst_rel and "other" not in rst_rel:
                    continue
            if rst_rel.lower() == "none":  # This is an added relation from the cache not paralleled by an RST relation
                rel = type('',(),{})()
                rel.info = [(reltype,sense,[],conn), eval(source_ids), eval(target_ids)]
                self.additional_rels[docname].append(rel)
            self.mapping[docname][key] = (conn, sense, reltype)

    def convert(self, doc, rel):
        if rel.key in self.mapping[doc.docname]:
            conn, sense, reltype = self.mapping[doc.docname][rel.key]
            rel.pdtb_rels['cache'].append((reltype, sense, [], conn))

    def set_doc_state(self, doc_state):
        final_rels = []
        for addrel in self.additional_rels[doc_state.docname]:
            pdtb_rel, arg1_ids, arg2_ids = addrel.info
            # Clone a node with max EDU ID for the relation
            found = False
            for edu in sorted(set(arg1_ids + arg2_ids),reverse=True):
                if found:
                    break
                for rel in doc_state.rels.values():
                    if rel.head_edu == str(edu):
                        clone = deepcopy(rel)
                        clone.node = deepcopy(rel.node)
                        found = True
                        break
            clone.nid = "a" + clone.nid
            clone.relname = "NONE"
            clone.dep_parent = str(max(arg1_ids))
            clone.head_edu = str(min(arg2_ids))
            clone.signals = []
            clone.pdtb_rels[pdtb_rel[0]] = [pdtb_rel]
            clone.source = Span(doc_state, arg2_ids, [str(min(arg2_ids))], [arg2_ids], {min(arg2_ids):"root"})
            clone.target = Span(doc_state, arg1_ids, [str(min(arg1_ids))], [arg1_ids], {min(arg1_ids):"root"})
            clone.key = clone.head_edu + "-" + clone.dep_parent + "-" + clone.relname
            final_rels.append(clone)
        self.additional_rels[doc_state.docname] = final_rels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", help="data dir", default='../data')
    parser.add_argument("-d", help="example docname", default='GUM_academic_economics')
    args = parser.parse_args()

    data_dir = args.i
    docname = args.d
    conllu_dir = os.path.join(data_dir, "dep", docname+".conllu")
    rs4_dir = os.path.join(data_dir, "rst", "rstweb", docname+".rs4")
    direct_mappings_dir = os.path.join(data_dir, 'mappings.json')
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')

    cache_module = Cache(data_dir, direct_mappings_dir, disco_pred_dir)
    doc = cache_module.process_files(conllu_dir, rs4_dir, docname)

    for node_id, rel in doc.rels.items():
        cache_module.convert(doc, rel)
        cache_module.output(docname, node_id, rel, 'cache')
