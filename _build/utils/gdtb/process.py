import io
import re
import sys
from typing import Text, List, Set
from collections import defaultdict
from copy import deepcopy
import conllu
from rst2dep import read_rst, make_rsd
from nodes import EDU, Sentence, Doc, Relation, Signal
import xml.etree.ElementTree as ET


def read_raw_files(conllu_file: Text, rsd: Text, docname: str) -> Doc:
    doc = read_conllu(conllu_file, docname)
    doc = read_rsd(rsd, doc)
    return doc


def read_conllu(conllu_file: Text, docname: str) -> Doc:
    parsed = conllu.parse(conllu_file)
    sents = []
    raw_sents = conllu_file.strip().split('\n\n')
    par_id = -1
    tok_id = 0
    for sent_id, raw_sent in enumerate(raw_sents):
        if '# newpar' in raw_sent:
            par_id += 1
        elif par_id == -1:  # no newpar in the first sentence of the file
            par_id = 0
        sent = Sentence(raw_sent, tok_id, sent_id, par_id)
        sent.raw_conllu = raw_sent
        sent.s_type = parsed[sent_id].metadata['s_type']
        tok_id += sent.num_toks
        sents.append(sent)
    doc = Doc(docname, sents, parsed)
    return doc


def extract_head_line(lines: List[List]) -> int:
    child_to_parent = {}

    tok2line = {}
    seen_tok_ids = []
    for line_id, line in enumerate(lines):
        tok_id = int(line[0])
        head = int(line[6])
        tok2line[tok_id] = line_id
        if head == 0:
            return line_id
        seen_tok_ids.append(tok_id)
        child_to_parent[tok_id] = {'head': head, 'func': line[7]}

    heads = set()
    for child, vals in child_to_parent.items():
        parent = vals['head']
        func = vals['func']
        if parent not in seen_tok_ids and func not in ['punct']:
            heads.add(child)

    return tok2line[list(heads)[0]]


def read_rsd(rsd: Text, doc: Doc) -> Doc:
    for line in rsd.split("\n"):
        if "\t" not in line:
            continue
        fields = line.split("\t")
        edu_id = int(fields[0])
        rsd_rel = fields[7]

        # find head line
        head_line_id = extract_head_line(doc.edus[edu_id].conllu_lines)
        head_line = doc.edus[edu_id].conllu_lines[head_line_id]

        doc.edus[edu_id].head_func = head_line[7]
        doc.edus[edu_id].head_pos = head_line[4]

        try:
            edu_tense = re.search(r'Tense=(\w+)\|', head_line[5]).group(1)
            doc.edus[edu_id].edu_tense = re.search(r'(None|Modal|Pres|Past|Fut)', edu_tense).group(1)
        except:
            doc.edus[edu_id].edu_tense = ''

        doc.edus[edu_id].head_lemma = head_line[2]
        doc.edus[edu_id].head_line_id = head_line_id

        doc.edus[edu_id].child_rels = fields[-1]
        doc.edus[edu_id].rel = fields[7]
        doc.edus[edu_id].dep_parent = int(fields[6])
        if rsd_rel.startswith("same-unit"):
            doc.edus[edu_id].same_unit2main = int(fields[6])
            doc.edus[int(fields[6])].same_unit.add(edu_id)
            doc.main2same_unit[int(fields[6])].add(edu_id)
    return doc


def read_file(conllu_dir: str, rs4_dir: str, docname: str) -> Doc:
    """
    Partially borrow from RST++ at https://github.com/t-aoyam/rstpp/blob/main/code/converter/browse_rs4.py#L106
    """
    conllu_file = io.open(conllu_dir).read()
    rs4 = io.open(rs4_dir).read()
    rsd = make_rsd(rs4, '', as_text=True, algorithm="chain", keep_same_unit=True, output_const_nid=True)

    rsd2nid = {fields.split("\t")[0]: fields.split("\t")[3] for fields in rsd.split("\n") if "\t" in fields}
    nid2head_edu = {fields.split("\t")[3]: fields.split("\t")[0] for fields in rsd.split("\n") if "\t" in fields}
    rsd_parents = {}
    rsd_rels = {}
    for row in rsd.split("\n"):
        if "\t" in row:
            fields = row.split("\t")
            rsd_parents[fields[0]] = fields[6]
            rsd_rels[fields[0]+"-"+fields[6]] = fields[7]
            if fields[0] not in nid2head_edu:
                nid2head_edu[fields[0]] = fields[0]
    secedge_mappings = {}
    secedge_rev_mappings = {}
    for row in rsd.split("\n"):  # Second pass for secedges now that primedges are mapped
        if "\t" in row:
            fields = row.split("\t")
            if fields[8] != "_":  # secedges in row
                mappings = fields[4].split("|")
                for mapping in mappings:
                    secedge_mappings[mapping.split(":")[0]] = mapping.split(":")[1]  # dep2const mapping
                    secedge_rev_mappings[mapping.split(":")[1]] = mapping.split(":")[0]  # const2dep mapping
                secedges = fields[8].split("|")
                for secedge in secedges:
                    dep_trg = secedge.split(":")[0]
                    dep_src = fields[0]
                    const_id = secedge_mappings[dep_src + "-" + dep_trg]
                    rsd_parents[dep_src + "-" + dep_trg] = dep_trg
                    nid2head_edu[const_id] = dep_src
                    rsd2nid[dep_src + "-" + dep_trg] = const_id

    # read conllu
    doc = read_raw_files(conllu_file, rsd, docname)

    main2same_unit = doc.main2same_unit
    edu2head_func, edu2dep_parent, edu2text = doc.get_edu_features()

    # read rs4
    nodes = read_rst(rs4, {}, as_text=True)
    to_remove = set()
    for nid in nodes:
        node = nodes[nid]
        if node.kind == "edu" or "-" in nid:
            if "-" in nid:
                if nid in secedge_rev_mappings:
                    node.dep_parent = rsd_parents[secedge_rev_mappings[nid]]
                else:
                    sys.stderr.write("WARN: Skipping secedge with ID " + nid + " in " + docname + " due to duplicate secedge dependency path\n")
                    to_remove.add(nid)
                    continue
            else:
                node.dep_parent = rsd_parents[nid]
                node.dep_rel = rsd_rels[nid2head_edu[nid]+"-"+node.dep_parent]
        elif node.id in nid2head_edu:
            node.dep_parent = rsd_parents[nid2head_edu[node.id]]
            node.dep_rel = rsd_rels[nid2head_edu[nid]+"-"+node.dep_parent]

    for nid in to_remove:
        del nodes[nid]

    node2descendent_edus = defaultdict(set)
    for nid in nodes:
        if "-" in nid:
            continue
        node = nodes[nid]
        for i in range(node.left, node.right+1):
            node2descendent_edus[nid].add(str(i))

    # find secondary edges and give them clone NODEs instead of SECEDGE objects
    keys = [nid for nid in nodes]
    for nid in keys:
        if "-" in nid:
            const_source = nodes[nid].source
            clone = deepcopy(nodes[const_source])
            relname = nodes[nid].relname + "_r"
            parent = nodes[nid].target
            dep_parent = nodes[nid].dep_parent
            del nodes[nid]
            clone.relname = relname
            clone.parent = parent
            clone.dep_parent = dep_parent
            clone.id = nid
            clone.kind = "secedge"
            nodes[nid] = clone
            node2descendent_edus[nid] = node2descendent_edus[const_source]

    for nid in nodes:
        node = nodes[nid]
        if node.dep_parent in nodes:
            if nodes[node.dep_parent].dep_rel.startswith("same"):  # Set true content parent for same-unit
                node.dep_parent = nodes[node.dep_parent].dep_parent

        if '-' in nid:
            if nid not in nid2head_edu:
                nid2head_edu[nid] = nid2head_edu[nid.split("-")[0]]

        if node.parent == "0" or node.dep_parent == "0" or node.relname == "span" or node.dep_rel.startswith("same") or nid not in nid2head_edu:
            continue  # skip root, span, same-unit
        elif node.relname.endswith("_m") and nodes[node.parent].leftmost_child == nid:
            continue  # skip leftmost multinuc children

        head_edu_id = nid2head_edu[nid]
        descendents = node2descendent_edus[nid]
        head_sent_edus = find_same_sent_edus(doc, doc.edus[int(head_edu_id)], descendents)
        target_edu_id = nodes[nid].dep_parent
        if target_edu_id == "":
            target_edu_id = nodes[rsd2nid[nid]].dep_parent
        if nodes[nid].dep_rel.endswith("_m") and "-" not in nid:
            probe = nodes[nid].parent
            while True:
                if nodes[probe].kind == "multinuc" and nodes[probe].left < nodes[nid].left:
                    break
                probe = nodes[probe].parent
            next_child = [c for c in nodes[probe].children if nodes[c].left < nodes[nid].left]
            target_const_node = max(next_child, key=lambda x: nodes[x].left)
        else:
            target_const_node = nodes[nid].parent
        target_descendents = node2descendent_edus[target_const_node]
        target_sent_edus = find_same_sent_edus(doc, doc.edus[int(target_edu_id)], target_descendents)
        head_edus = [head_edu_id]
        target_head_edus = [target_edu_id]
        if int(head_edu_id) in main2same_unit:
            head_edus.extend(main2same_unit[int(head_edu_id)])
        elif node.relname.startswith("same"):  # Same-unit
            head_edus.append(node.dep_parent)
            if int(node.dep_parent) in main2same_unit:
                head_edus.extend(main2same_unit[int(node.dep_parent)])
        for head_edu in head_edus:
            if head_edu in main2same_unit:
                head_edus.extend(main2same_unit[head_edu])
        head_edus = sorted(list(set([str(x) for x in head_edus])), key=lambda x: int(x))

        if int(target_edu_id) in main2same_unit:
            target_head_edus.extend(main2same_unit[int(target_edu_id)])
        else:
            target_node = nodes[target_edu_id]
            if target_node.relname == "span":
                target_node = nodes[rsd2nid[target_node.id]]
            if target_node.dep_rel.startswith("same"):  # Same-unit
                target_head_edus.append(target_node.dep_parent)
                if int(target_node.dep_parent) in main2same_unit:
                    target_head_edus.extend(main2same_unit[int(target_node.dep_parent)])
        for target_head_edu in target_head_edus:
            if target_head_edu in main2same_unit:
                target_head_edus.extend(main2same_unit[target_head_edu])

        target_head_edus = sorted(list(set([str(x) for x in target_head_edus])), key=lambda x: int(x))
        descendents.update(head_edus)
        target_descendents.update(target_head_edus)
        target_descendents = target_descendents.difference(descendents)

        rel = Relation(node,
                       source_head_edus=head_edus,
                       source_edu_ids=sorted(list([int(i) for i in descendents])),
                       source_sent_edus=head_sent_edus,
                       target_head_edus=target_head_edus,
                       target_edu_ids=sorted(list([int(i) for i in target_descendents])),
                       target_sent_edus=target_sent_edus,
                       edu2head_func=edu2head_func,
                       docstate=doc)
        rel.head_edu = head_edu_id
        rel.dep_parent = node.dep_parent
        rel.head_edu_text = edu2text[int(rel.head_edu)]
        rel.key = rel.head_edu + "-" + rel.dep_parent + "-" + rel.node.dep_rel
        doc.rels[node.id] = rel

    tree = ET.parse(rs4_dir)
    root = tree.getroot()
    for signal in root.iter('signal'):
        attrs = signal.attrib
        source, sigsubtype, tok_ids, sigtype = attrs['source'], attrs['subtype'], attrs['tokens'], attrs['type']
        tids = []
        toks = []
        if tok_ids:
            for tid in sorted([int(tid) for tid in tok_ids.split(",")]):
                tids.append(tid - 1)
                toks.append(doc.id2tok[tid - 1])
        else:
            tids = []
            toks = []
        if source in doc.rels:
            doc.rels[source].signals.append(Signal(sigtype, sigsubtype, tids, toks))
        else:
            #raise ValueError("! Found signal with missing source " + source + " in " + docname)
            sys.stderr.write("WARN: Found signal with missing source " + source + " in " + docname + "; could be a secedge overlapping another secedge path\n")

    return doc


def find_same_sent_edus(doc: Doc, curr_edu: EDU, edu_set: Set) -> List:
    sent_edus_list = []
    curr_sent_id = curr_edu.sent_id
    left_span_tok_id = curr_edu.tok_ids[0]
    edu_candidates = doc.sents[curr_sent_id].sent_edus

    def add_edus(restrition):
        added = []
        for edu in edu_candidates:
            edu_id = edu.edu_id
            if restrition == "whole":
                added.append(edu_id)
            elif restrition == "right":
                if edu.tok_ids[0] >= left_span_tok_id:
                    added.append(edu_id)
            elif restrition == "within":
                if edu_id in edu_set:
                    added.append(edu_id)
        return sorted(added)

    for restriction in ["whole", "right", "within"]:
        added = add_edus(restriction)
        if added and added not in sent_edus_list:
            sent_edus_list.append(added)

    return sent_edus_list
