"""
@Lauren
"""
import os
import json
from argparse import ArgumentParser
from modules.base import ConvertBase
from collections import defaultdict


class Altlex(ConvertBase):
    def __init__(self, data_dir, direct_mappings_dir, probs_mappings_dir):
        super().__init__(data_dir, direct_mappings_dir, probs_mappings_dir)
        # self.direct_mappings for explicit mappings
        map = defaultdict(list)
        map.update(self.direct_mappings['rst2pdtb'])
        self.direct_mappings = map
        connsense_dir = os.path.join(data_dir, 'altlex_string_connsense.json')
        self.used_indices = set()  # Do not use the same tokens for two relations
        with open(connsense_dir, 'r') as file:
            self.string_connsense = json.load(file)

    def convert(self, doc, rel):
        edu1_sent_id = rel.source.head_edu_sent.sent_id
        edu2_sent_id = rel.target.head_edu_sent.sent_id
        if abs(edu2_sent_id - edu1_sent_id) > 1:
            return

        if doc.sents[edu1_sent_id].par_id != doc.sents[edu2_sent_id].par_id:
            return

        if len(rel.pdtb_rels["explicit"]) > 0 or len(rel.pdtb_rels["altlex"]) > 0 or len(rel.pdtb_rels["altlexc"]) > 0:  # Can't already be explicit or altlex
            return

        if any ([r[0] == "altlex" for r in rel.pdtb_rels["cache"]]):
            return  # Cache already specifies AltLex

        if not any([e.head_pos.startswith("V") or "cop" in e.child_funcs for e in rel.source.head_edu]):
            return
        if not any([e.head_pos.startswith("V") or "cop" in e.child_funcs for e in rel.target.head_edu]):
            return

        # Special constructions
        if "adversative" in rel.relname:
            neg_sig = [s for s in rel.signals if s.subtype == "negation"]
            if len(neg_sig) > 0:
                neg_tok = [t.text.lower() for t in neg_sig[0].toks if t.text.lower() in ["not","no","n't","n’t","never"]]
                neg_idx = [neg_sig[0].tok_ids[i] for i, t in enumerate(neg_sig[0].toks) if t.text.lower() in ["not","no","n't","n’t","never"]]
                rel.pdtb_rels["altlex"].append(["AltLex", "expansion.substitution.arg1-as-subst", neg_idx, " ".join(neg_tok)])
                return

        # Find candidate AltLex expressions in source and target edus:
        src_head_edus = rel.source.head_edu
        trg_head_edus = rel.target.head_edu
        altlext_candidates = self.find_altlex_candidates(src_head_edus, trg_head_edus)

        # if no candidates, return
        if len(altlext_candidates) == 0:
            return

        # Add compatible relations for candidates
        rst_rel = rel.relname[:-2]
        altlext_candidates = self.find_compatible_pdtb_relations(rst_rel, altlext_candidates)

        # if no compatible relations, return
        if len(altlext_candidates) == 0:
            return

        # select candidate
        altlext_candidates = self.re_org_candidates(altlext_candidates)
        selected_candidate = self.select_candidate(altlext_candidates)
        # print(selected_candidate)

        # select relation, if multiple
        if len(selected_candidate["possible_relations"]) > 1:
            pdtb_relation = self.select_relation(rel, selected_candidate, doc)
        else:
            pdtb_relation = selected_candidate["possible_relations"][0]

        # calculate token index
        token_index = selected_candidate["tok_index"]
        alt_lex_text = " ".join([doc.tokens[i].text.lower() for i in token_index])

        if pdtb_relation not in rel.pdtb_rels['altlex']:
            if not any([(doc.docname,t) in self.used_indices for t in token_index]):
                rel.pdtb_rels['altlex'].append(('AltLex', pdtb_relation, token_index, alt_lex_text))
                for t in token_index:
                    self.used_indices.add((doc.docname,t))

    def find_altlex_candidates(self, src_edus, trg_edus):
        altlext_candidates = []
        # check for exact matches in edus
        altlex_strings = self.string_connsense.keys()
        for altlex in altlex_strings:
            # Assume AltLex must be in the later EDUs
            if any(["same" in e.rel for e in src_edus]):  # Prefer relation source embedded inside same-unit, not the same-unit wrapper
                edus = trg_edus
            elif any(["same" in e.rel for e in src_edus]):
                edus = src_edus
            else:  # If no same unit, we pick the later EDU block
                edus = src_edus if min([e.edu_id for e in src_edus]) > min([e.edu_id for e in trg_edus]) else trg_edus
            if "position" in self.string_connsense[altlex]:  # Some altlexes only work in source or target
                position = self.string_connsense[altlex]["position"]
                if position == "source":
                    edus = src_edus
                elif position == "target":
                    edus = trg_edus
            for edu in edus:
                text = " " + edu.text + " "
                char_index = text.find(" " + altlex + " ") - 1
                if char_index != -2:
                    tok_index = text.split(" " + altlex,1)[0].count(" ")
                    pos_filter = self.string_connsense[altlex]["upos"]
                    if pos_filter != []:
                        if edu.upos[tok_index:len(altlex.split())] != pos_filter:
                            continue
                    deprel_filter = self.string_connsense[altlex]["deprel"]
                    if deprel_filter != []:
                        if edu.deprels[tok_index:len(altlex.split())] != deprel_filter:
                            continue
                    tok_index = edu.tok_ids[tok_index:tok_index+len(altlex.split())]
                    altlext_candidates.append({"text": altlex, "type": "exact_match", "source": edu.edu_id, "char_index": char_index + 1, "tok_index": tok_index})
                    break
        # check for pattern matches in edus
        # also grab pattern that is matched on, altlex text will need to be grabbed from edu_text pasted on that
        return altlext_candidates

    def find_compatible_pdtb_relations(self, rst_rel, altlext_candidates):
        new_candidates = []
        # string_connsense = load_string_connsense()
        # # load pattern map as well
        # rst2pdtb = load_rst2pdtb()
        compat_rst = self.direct_mappings[rst_rel]
        for candidate in altlext_candidates:
            if candidate["type"] == "exact_match":
                compat_cand = self.string_connsense[candidate["text"]]["senses"]
            else:  # candidate["type"] == "pattern_match"
                compat_cand = {}
            candidate["possible_relations"] = []
            for sense in compat_cand:
                if sense in compat_rst:
                    candidate["possible_relations"].append(sense)
            if len(candidate["possible_relations"]) > 0:
                new_candidates.append(candidate)
        return new_candidates

    def select_candidate(self, altlext_candidates):
        # favor exact string match
        # favor appearance in edu2
        # favor leftward appearance within edu
        candidate = {}
        if len(altlext_candidates["exact_match"]) > 0:
            edu = min(altlext_candidates["exact_match"])
            candidate = min(altlext_candidates["exact_match"][edu], key=lambda x: x["char_index"])
            candidate["source"] = edu
            candidate["type"] = "exact_match"
            candidate["tok_index"] = altlext_candidates["exact_match"][edu][0]["tok_index"]
        elif len(altlext_candidates["pattern_match"]) > 0:  # there are only pattern matches
            edu = min(altlext_candidates["pattern_match"])
            candidate = min(altlext_candidates["pattern_match"][edu], key=lambda x: x["char_index"])
            candidate["source"] = edu
            candidate["type"] = "pattern_match"
            candidate["tok_index"] = altlext_candidates["pattern_match"][edu][0]["tok_index"]
        return candidate

    def re_org_candidates(self, altlext_candidates):
        new_candidates = defaultdict(lambda : defaultdict(list))
        for candidate in altlext_candidates:
            if candidate["type"] == "exact_match":
                sub_entry = {"text": candidate["text"], "char_index": candidate["char_index"], "tok_index": candidate["tok_index"],
                             "possible_relations": candidate["possible_relations"]}
                new_candidates["exact_match"][candidate["source"]].append(sub_entry)
            else:  # candidate["type"] == "pattern_match"
                sub_entry = {"text": candidate["text"], "char_index": candidate["char_index"], "tok_index": candidate["tok_index"],
                             "possible_relations": candidate["possible_relations", "pattern": candidate["pattern"]]}
                new_candidates["pattern_match"][candidate["source"]].append(sub_entry)
        return new_candidates

    def select_relation(self, rel, selected_candidate, doc):
        # TODO
        # integrate discodisco to select most likely relation
        # for now, use frequency as an approximation
        all_probs = self.get_rel_probs(doc.docname, rel, doc.sents)
        selected_sense = ""
        if selected_candidate["type"] == "exact_match":
            options_w_freq = []
            possible_probs = {k: v for k, v in all_probs.items() if k in selected_candidate["possible_relations"]}
            if len(possible_probs) > 0:
                selected = max(possible_probs,key=possible_probs.get)
                return selected
            else:
                for sense in selected_candidate["possible_relations"]:
                    options_w_freq.append({"sense": sense, "freq": self.string_connsense[selected_candidate["text"]]["senses"][sense]})
                selected = max(options_w_freq, key=lambda x: x["freq"])
                selected_sense = selected["sense"]
        else:  # selected_candidate["type"] == "pattern_match"
            # add freq mapping for pattern match
            pass
        return selected_sense


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", help="data dir", default='../data')
    parser.add_argument("-d", help="example docname", default='GUM_court_equality')
    args = parser.parse_args()

    data_dir = args.i
    docname = args.d
    conllu_dir = os.path.join(data_dir, "dep", docname+".conllu")
    rs4_dir = os.path.join(data_dir, "rst", "rstweb", docname+".rs4")
    direct_mappings_dir = os.path.join(data_dir, 'mappings.json')
    probs_mappings_dir = os.path.join(data_dir, 'discodisco_preds')

    altlex_module = Altlex(data_dir, direct_mappings_dir, probs_mappings_dir)
    doc = altlex_module.process_files(conllu_dir, rs4_dir, docname)

    for node_id, rel in doc.rels.items():
        altlex_module.convert(doc, rel)
        altlex_module.output(docname, node_id, rel, 'altlex')
