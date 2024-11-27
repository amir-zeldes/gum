"""
@Amir

Module to add no rel objects for sentence pairs left over after all conversions.
Unlike other modules, this module should not be invoked during the main relation loop,
but only after all other modules have been run.
"""

import re

from nodes import Sentence

class NoRel:

    def __init__(self):
        pass

    @staticmethod
    def convert(docstate):
        sent_pairs = {}
        prev_sent = prev_par = -1
        for s in docstate.sents:
            this_par = s.par_id
            if this_par == prev_par:
                sent_pairs[(prev_sent.sent_id,s.sent_id)] = (prev_sent, s)
            prev_par = this_par
            prev_sent = s

        for rel in docstate.rels.values():
            rel.source.sent_ids.sort()
            rel.target.sent_ids.sort()
            if len(rel.pdtb_rels) > 0:
                if min(rel.target.sent_ids) > max(rel.source.sent_ids) or (min(rel.target.sent_ids) == max(rel.source.sent_ids) and rel.is_forward):  # forward
                    s2 = rel.target.sent_ids[-1]
                    s1 = rel.source.sent_ids[0]
                else:
                    s1 = rel.target.sent_ids[-1]
                    s2 = rel.source.sent_ids[0]
                if abs(s1 - s2) > 1:
                    continue  # Not a consecutive pair
                if any([s > rel.source.sent_ids[0] for s in rel.target.sent_ids]) and any([s < rel.source.sent_ids[0] for s in rel.target.sent_ids]):
                    continue  # Medial relation, not a consecutive pair
                if s2 < s1:
                    continue  # Target contains sentence of source
                    #raise IOError("Misordered sentences for NoRel!")
                if (s1, s2) in sent_pairs:
                    del sent_pairs[(s1,s2)]  # Already have a relation

        output = []
        for s1, s2 in sent_pairs.values():
            s1_edus = [e.edu_id for e in docstate.edus.values() if e.sent_id == s1.sent_id]
            s2_edus = [e.edu_id for e in docstate.edus.values() if e.sent_id == s2.sent_id]
            s1_ent_vals = re.findall(r'Entity=([^|\n]+)',s1.plain_conllu)
            s1_ents = []
            for v in s1_ent_vals:
                s1_ents += re.findall(r'([0-9]+)\)',v)
                s1_ents += re.findall(r'\(([0-9]+)[^(]+\)',v)
            s2_ent_vals = re.findall(r'PronType[^\n]*?Entity=([^|\n]+)',s2.plain_conllu)
            s2_pron_ents = []
            for v in s2_ent_vals:
                s2_pron_ents += re.findall(r'\(([0-9]+)[^(]+\)',v)  # Single token entities on pronoun lines
            if len(set(s1_ents).intersection(set(s2_pron_ents))) > 0:
                new_rel = [docstate.docname, "entrel", "_", "EntRel", "_", s1.plain_text, s2.plain_text, s1_edus, s2_edus, "_", "_"]
            else:
                new_rel = [docstate.docname, "norel", "_", "NoRel", "_", s1.plain_text, s2.plain_text, s1_edus, s2_edus, "_", "_"]
            output.append(new_rel)

        return output

