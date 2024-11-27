"""
@Tatsuya @Devika
"""
import os
from argparse import ArgumentParser
from modules.base import ConvertBase


class Explicit(ConvertBase):
    def __init__(self, data_dir, direct_mappings_dir, disco_pred_dir):
        super().__init__(data_dir, direct_mappings_dir, disco_pred_dir)
        self.banned_dms = ['than']
        self.banned_rst = ['elaboration-attribute', 'purpose-attribute', 'attribution-positive'
                           ]
        # self.direct_mappings for explicit mappings

    def convert(self, doc, rel):
        def make_fluent(dm, ids):
            toks = dm.split()
            # if just 1 tok or the length of the toks is NOT a multiple of unique elements
            if len(toks) < 2 or len(toks) % len(set(toks)):
                return (dm, ids)
            return (' '.join(toks[:len(set(toks))]),
                    ids[:len(set(toks))])

        if len(rel.signals) == 0 or "question" in rel.relname:  # if no signal or is question skip
            return
        elif "cache" in rel.pdtb_rels:
            return

        for sig in rel.signals:
            if sig.subtype not in ['dm', 'orphan']:
                continue  # explicit = dm or orphan
            note = ""
            dm = ' '.join([tok.text for tok in sig.toks]).lower()
            ids = sig.tok_ids
            rst = rel.relname[:-2]
            if dm in self.banned_dms or rst in self.banned_rst:
                continue
            all_probs = self.get_rel_probs(doc.docname, rel, doc.sents)

            pdtbs = self.direct_mappings['dm@rst2pdtb'].get('@'.join([dm, rst]), [])

            if len(pdtbs) == 1:  # ideal case, only one PDTB in the intersection
                pdtb = pdtbs[0]
                rel.pdtb_rels['explicit'].append((note, pdtb, ids, dm))
                continue
            elif len(pdtbs) == 0:  # if no intersection
                # get union of allowed PDTB relations from RST & DM
                # pdtbs = self.direct_mappings['dm2pdtb'].get(dm, []) +\
                    # self.direct_mappings['rst2pdtb'].get(rst, [])
                # get PDTB rels
                note = 'unattested combo'
                pdtbs = self.direct_mappings['dm2pdtb'].get(dm, [])
                if not pdtbs:  # if dm is unattested in PDTB
                    dm, ids = make_fluent(dm, ids)
                    note = 'disfluent DM'
                    pdtbs = self.direct_mappings['dm2pdtb'].get(dm, [])
                    if not pdtbs:  # even if unattested after fixing disfluency
                        note = 'unattested DM'
                        pdtbs = self.direct_mappings['rst2pdtb'].get(rst, [])
            elif len(pdtbs) > 1:
                pass
                # simply use the multiple pdtb candidates
            # now we have a list of possible pdtb relations
            if all_probs:  # if the disco preds are available (won't need this once fixed)
                probs = [[pdtb_rel, all_probs.get(pdtb_rel, 0)] for pdtb_rel in pdtbs]  # TODO get method in case of PDTB rel name discrepancies
                pdtb = sorted(probs, key=lambda x: x[1], reverse=True)[0][0]
            else:  # if disco preds are NOT available - for yilun and janet to fix
                if not pdtbs:  # TODO figure out why some token offset / attachment are off
                    print(f"docname: {doc.docname}, dm: {dm}, rst: {rst}")
                    continue
                pdtb = pdtbs[0]  # just pick the first one in the list
            rel.pdtb_rels['explicit'].append((note, pdtb, ids, dm))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", help="data dir", default='../data')
    parser.add_argument("-d", help="example docname", default='GUM_academic_art')
    args = parser.parse_args()

    data_dir = args.i
    docname = args.d
    conllu_dir = os.path.join(data_dir, "dep", docname+".conllu")
    rs4_dir = os.path.join(data_dir, "rst", "rstweb", docname+".rs4")
    direct_mappings_dir = os.path.join(data_dir, 'mappings.json')
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')

    explicit_module = Explicit(data_dir, direct_mappings_dir, disco_pred_dir)
    doc = explicit_module.process_files(conllu_dir, rs4_dir, docname)

    for node_id, rel in doc.rels.items():
        explicit_module.convert(doc, rel)
        explicit_module.output(docname, node_id, rel, 'explicit')
