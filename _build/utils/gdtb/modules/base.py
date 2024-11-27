import io
import re
import os
import json
from glob import glob
from abc import ABC
from typing import Dict, Tuple, Any, DefaultDict
from collections import defaultdict
try:
    from process import read_file
except:
    from .process import read_file
import sys


script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
disco_dir = script_dir + ".." + os.sep + "discodisco" + os.sep
conllu_target_dir = script_dir + ".." + os.sep + ".." + os.sep + ".." + os.sep + "target" + os.sep + "dep" + os.sep + "not-to-release" + os.sep
missing_data_docs = set()
missing_entries = set(open(disco_dir + "eng.pdtb.missing_test.rels").read().strip().split("\n")[1:])

# Ensure that discodisco has a file with all conllu parses associated with our data
conllu_files = glob(conllu_target_dir + "*.conllu")
all_conllu_data = [open(f).read().strip() for f in conllu_files]
all_conllu_data = "\n\n".join(all_conllu_data).strip() + "\n\n"
with open(disco_dir + "eng.pdtb.missing_test.conllu", 'w', encoding="utf8", newline="\n") as f:
    f.write(all_conllu_data)


class ConvertBase(ABC):
    def __init__(self, data_dir, direct_mappings_dir=None, probs_mappings_dir=None, conn_pred_dir=None):
        self.data_dir = data_dir
        self.direct_mappings = self.get_direct_mappings(direct_mappings_dir) if direct_mappings_dir else None
        self.probs_mappings, self.probs_mappings_by_sents, self.force_mapping = self.get_probs_preds(probs_mappings_dir) if probs_mappings_dir else (None, None, None)
        self.conn_preds = self.get_conn_preds(conn_pred_dir) if conn_pred_dir else None

    def convert(self, docname, rel):
        pass

    def process_files(self, conllu_dir: str, rs4_dir: str, docname: str):
        return read_file(conllu_dir, rs4_dir, docname)

    def get_probs_preds(self, pred_dir: str) -> Tuple[DefaultDict[Any, DefaultDict[Any, Dict]], DefaultDict[Any, Dict]]:
        """
        map predicted relations with probs
        """
        # Get basic mappings
        # key:  the source to find the corresponding rel
        # value: the probability map
        mappings = defaultdict(lambda: defaultdict(dict))

        # Get fallback mappings by sentence texts in case of minor changes to source annotations.
        # This only works for inter-sentential relations
        mappings_by_sents = defaultdict(lambda: defaultdict(dict))

        for split in ['train', 'dev', 'test', 'add']:
            raw_f = open(os.path.join(pred_dir, f'eng.rst.gum_{split}.rels'), encoding='utf8').read().strip().split('\n')[1:]
            pred_f = open(os.path.join(pred_dir, f'eng.rst.gum_{split}_predictions.json'), encoding='utf8').read().strip().split('\n')

            assert len(raw_f) == len(pred_f)

            for i, (raw_line, pred) in enumerate(zip(raw_f, pred_f)):
                json_pred = json.loads(pred)
                fields = raw_line.split('\t')
                docname = fields[0]
                src, trg, relname = fields[10].split("-",2)
                mappings[docname][(src, trg, relname)] = json_pred['relation_probs']
                if fields[7] != fields[8]:
                    mappings_by_sents[docname][(fields[7],fields[8])] = json_pred['relation_probs']

        # retrieve force mapping in case the script cannot recognize any mappings in the prediction file
        force_mapping_dir = os.path.join(pred_dir, "force_mapping.tab")
        force_mapping = defaultdict(dict)
        with open(force_mapping_dir, encoding="utf8") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:
                fields = line.split("\t")
                docname = fields[0]
                key_source, key_target = fields[1], fields[2]
                pred_source, pred_target = fields[3], fields[4]
                force_mapping[docname][(key_source, key_target)] = (pred_source, pred_target)

        return mappings, mappings_by_sents, force_mapping

    def get_conn_preds(self, dm_pred_dir: str) -> Dict:
        """
        get predictions of top 5 predicted dm connectors (by implicit NLP module).
            Structure is nested dictionary with keys similar to the get_probs_pred -
            [docname][(arg1_text, arg2_text, dir)]. Values are a list of top5 dm connectors, in order.
        """
        preds = defaultdict(lambda: defaultdict(dict))
        # key:  the source to find the corresponding rel
        # value: the list of top 5 connectors

        for split in ['train', 'dev', 'test', 'add']:
            dm_pred_f = open(os.path.join(dm_pred_dir, f'gum_implicit_{split}_preds.jsonl'), encoding='utf8').read().strip().split('\n')
            for line in dm_pred_f:
                if line != "":
                    json_pred = json.loads(line)
                    arg1_text = json_pred["unit1_txt"]
                    arg2_text = json_pred["unit2_txt"]
                    # no_relations = re.search("Relations:")
                    connectors = json_pred["connectors"]
                    docname = json_pred["docname"]
                    key = (arg1_text, arg2_text)
                    if key in preds[docname]:
                        # print("ALREADY THERE", file=sys.stderr)
                        # print("KEY:", key)
                        # print("Connectors:", preds[docname][key])
                        # print("New Connectors:",connectors)
                        preds[docname][key] += connectors
                    else:
                        preds[docname][key] = connectors

        return preds

    def get_direct_mappings(self, mapping_dir: str) -> Dict:
        """
        map explicit mapping relations
        """
        with open(mapping_dir) as f:
            mappings = json.load(f)
        return mappings

    def get_rel_probs(self, docname, rel, sentences):
        """
        REVERT TO THE _get_rel_probs BELOW IF ANYTHING BREAKS
        """

        # disco preds are in linear order so smaller ID = arg1
        if rel.source.tok_ids[0] < rel.target.tok_ids[0]:
            arg1 = rel.source
            arg2 = rel.target
        else:
            arg1 = rel.target
            arg2 = rel.source

        #mapkey = (span1, span2)
        # Change map key to use dep_src-dep_trg-relname (with sameunit participants switched to their parent)
        mapkey = (rel.head_edu, rel.dep_parent, rel.relname.replace("_m","").replace("_r",""))

        if mapkey in self.probs_mappings[docname]:
            return self.probs_mappings[docname][mapkey]
        elif docname in self.force_mapping and mapkey in self.force_mapping[docname]:
            print('forced :(')
            # case by case handling (from data/discodisco_preds/force_mapping.tab)
            return self.probs_mappings[docname][self.force_mapping[docname][mapkey]]
        else:
            for k in self.probs_mappings[docname]:
                if (k[0],k[1]) == (mapkey[0],mapkey[1]) or (k[0],k[1]) == (mapkey[1],mapkey[0]):
                    # Found a relation with the same endpoints which has predicted probabilities
                    sys.stderr.write(f"WARN: used probabilities of different, same endpoints relation for key {mapkey} in {docname}\n")
                    return self.probs_mappings[docname][k]
            # Old text based mapping - this should never happen when using src-trg-rel keys - AZ
            # Try mapping entire first and last sentences of span, since this relation may span multiple sentences
            # if intersentential, clip to sentence
            if arg1.sent_ids != arg2.sent_ids and not any([sid in arg1.sent_ids for sid in arg2.sent_ids]) and not any(
                    [sid in arg2.sent_ids for sid in arg1.sent_ids]):
                span1 = arg1.head_edu_sent.plain_text
                span2 = arg2.head_edu_sent.plain_text
            else:  # if intrasentential, whole thing
                span1 = arg1.raw_text
                span2 = arg2.raw_text

            all_sents = sorted(list(set(arg1.sent_ids).union(set(arg2.sent_ids))))
            first, last = all_sents[0], all_sents[-1]
            s1 = sentences[first].plain_text
            s2 = sentences[last].plain_text
            sys.stderr.write(f"WARN: using fallback relation probabilities mapping indexed by sentence texts because an exact path key is not found for {mapkey} in {docname}\n")
            mapkey = (s1, s2)
            if mapkey in self.probs_mappings_by_sents[docname]:
                return self.probs_mappings_by_sents[docname][mapkey]
            elif len(all_sents) > 2:  # Maybe the head sentence of one of the arguments is not the first or last sentence
                s_mid = sentences[all_sents[1]].plain_text
                if (s1, s_mid) in self.probs_mappings_by_sents[docname]:
                    return self.probs_mappings_by_sents[docname][(s1, s_mid)]
                elif (s_mid, s2) in self.probs_mappings_by_sents[docname]:
                    return self.probs_mappings_by_sents[docname][(s_mid, s2)]

        if docname not in missing_data_docs:
            sys.stderr.write("! missing relation probability predictions for " + docname + ", adding to discodisco/missing.rels and returning dummy preds for now\n")
            missing_data_docs.add(docname)
        # Check that file has header line
        if not os.path.exists(disco_dir + "eng.pdtb.missing_test.rels"):
            with open(disco_dir + "missing.rels", 'w', encoding="utf8", newline="\n") as f:
                f.write("doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\torig_label\tlabel\n")
        else:
            data = open(disco_dir + "eng.pdtb.missing_test.rels", 'r').read()
            if not data.startswith("doc"):
                with open(disco_dir + "eng.pdtb.missing_test.rels", 'w', encoding="utf8", newline="\n") as f:
                    f.write("doc\tunit1_toks\tunit2_toks\tunit1_txt\tunit2_txt\ts1_toks\ts2_toks\tunit1_sent\tunit2_sent\tdir\torig_label\tlabel\n")

        # Serialize entry like:
        # doc	unit1_toks	unit2_toks	unit1_txt	unit2_txt	s1_toks	s2_toks	unit1_sent	unit2_sent	dir	rel_type	orig_label	label
        # GUM_academic_discrimination	22-42	43-56	Personal experiences of discrimination and bias have been the focus of much social science research . [ 1 - 3 ]	Sociologists have explored the adverse consequences of discrimination [ 3 – 5 ] ;	22-42	43-56	Personal experiences of discrimination and bias have been the focus of much social science research . [ 1 - 3 ]	Sociologists have explored the adverse consequences of discrimination [ 3 – 5 ] ;	1<2	implicit	contingency.cause.reason	contingency.cause

        s1_tokens = [int(t) for t in sentences[first].doc_tokens]
        s2_tokens = [int(t) for t in sentences[last].doc_tokens]
        s1_tokens = f"{min(s1_tokens)}-{max(s1_tokens)}" if len(s1_tokens) > 1 else str(s1_tokens[0])
        s2_tokens = f"{min(s2_tokens)}-{max(s2_tokens)}" if len(s2_tokens) > 1 else str(s2_tokens[0])

        entry = "\t".join([docname, f"{arg1.tok_ids[0] + 1}-{arg1.tok_ids[-1] + 1}",
                 f"{arg2.tok_ids[0] + 1}-{arg2.tok_ids[-1] + 1}", span1, span2, s1_tokens, s2_tokens, s1, s2, "1<2",
                 "expansion.conjunction", "expansion.conjunction"])

        if entry not in missing_entries:
            with open(disco_dir + "eng.pdtb.missing_test.rels", 'a', encoding="utf8", newline="\n") as f:
                f.write(entry + '\n')
            key = rel.key[:-2] if rel.key.endswith("_m") or rel.key.endswith("_r") else rel.key
            with open(disco_dir + "eng.pdtb.missing_test_keys.tab", 'a', encoding="utf8", newline="\n") as f:
                f.write(key + '\n')
        #raise ValueError(f"Mismatch occurs in {docname}, nid {rel.nid}. Source: {span1}. Target: {span2}")
        return {"expansion.conjunction":0.0004224016738589853}


    def _get_rel_probs(self, docname, rel):
        # generate key candidates
        key_candidates = []
        text_options = list(set([rel.source.raw_text, rel.target.raw_text] + rel.source.raw_text_sent + rel.target.raw_text_sent))
        for source in text_options:
            for target in text_options:
                if source and target:
                    key_candidates.append((source, target))
                    mapkey = (source, target)
                    if mapkey in self.probs_mappings[docname]:
                        return self.probs_mappings[docname][mapkey]
                    elif docname in self.force_mapping and mapkey in self.force_mapping[docname]:
                        # case by case handling (from data/discodisco_preds/force_mapping.tab)
                        return self.probs_mappings[docname][self.force_mapping[docname][mapkey]]
        raise ValueError(f"Mismatch occurs in {docname}, nid {rel.nid}. Source: {rel.source.raw_text}. Target: {rel.target.raw_text}")
        # return None

    def output(self, docname, node_id, rel, rel_type):
        if rel.pdtb_rels[rel_type]:
            print(f"###Doc: {docname}\tNode: {node_id}")
            print(f"RST relation: {rel.relname}")
            print(f"PDTB relation: {rel.pdtb_rels[rel_type]}")
            print(f"Source: {rel.source.raw_text}")
            print(f"Target: {rel.target.raw_text}")
            print()
