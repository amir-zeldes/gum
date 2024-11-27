import xml.etree.ElementTree as ET
from collections import defaultdict
from glob import glob
from argparse import ArgumentParser
from depedit import DepEdit
import re, os, sys, io
from copy import deepcopy
from nltk.stem.snowball import SnowballStemmer
try:
    from .classes import RULE, RELATION, TSVReader
except:
    from classes import RULE, RELATION, TSVReader

from rst2dep import read_rst, make_rsd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
XML_ROOT = SCRIPT_DIR + os.sep.join(["..", "src"]) + os.sep
CONLLU_TARGET = SCRIPT_DIR + os.sep.join(["..", "target", "dep", "not-to-release"]) + os.sep
RESOURCE_DIR = SCRIPT_DIR + os.sep.join(["dm-dependencies", "resources"]) + os.sep
DEPED_CACHE = SCRIPT_DIR + os.sep.join(["dm-dependencies", "deped_conllu"]) + os.sep

ALLOW_LEFT_SISTERS = False

# Read signal rules from grammar/non_dm_rules.xlsx
reader = TSVReader(RESOURCE_DIR + "non_dm_rules.xlsx")
all_rules = reader.get_rules()

tsv_lexchain = TSVReader(RESOURCE_DIR + "lexical_chains.xlsx")
tsv_tense = TSVReader(RESOURCE_DIR + "tense.xlsx")
tsv_attribution = TSVReader(RESOURCE_DIR + "attribution_source.xlsx")
tsv_additional = TSVReader(RESOURCE_DIR + "additional.xlsx")
forbidden = set(open(RESOURCE_DIR + "forbidden.tab").read().split("\n"))
forbidden = {tuple(x.split("\t")) for x in forbidden}

snowball = SnowballStemmer("english")

stop_words = {"per","on","with","for","at","from","out","bit","by","part","any","each","group","include","person","up","over",
              "because","using","under","same",
             "one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand","million","both","zero",
              "kind","but","those","a","if","of","that","to","1","2","3","4","5","6","7","8","9","10","100","1000","1000000",
              "below","will","apart","except",
             "my","she","her","his","hers","our","we","they","them","their","theirs","us","I","me","him","he","you","yes","try",
              "nothing","anything","something","everything",
             "can","must","all","the","not","n't","let","what","it","and","in","or","have","be","go","so","make","know","do","get","just",
              "time","more","take","other","see","also","first","then","want","many","most","only","very","well","way","even","thing","there",
              "here","really","as","back","need","right","how","why","where","when","while","much","like","become","put"}
lexical_upos = ["NOUN","VERB","ADJ","ADV"]
tense_pos = {"VBP":"Pres","VBZ":"Pres","VBD":"Past","MD":"Fut"}
multinuc_rels = ["adversative-contrast","joint-list","restatement-repetition","joint-other","joint-disjunction","joint-sequence","same-unit"]

mention_allowed_functions = set(["nsubj","nsubj:pass","nsubj:outer","obj","iobj","nmod:poss","root","nmod","obl",
                                 "obl:tmod","obl:npmod","obl:unmarked","appos","conj"])  # TODO: resolve conj to parent deprel and disallow conj
clause_funcs = set(["root","parataxis","xcomp","conj","ccomp","csubj","csubj:pass","advcl","acl","acl:relcl","advcl:relcl"])

# Note some of these are AltLex (e.g. so+is)
#
# negation not working: polygraph 70 - just because... means... doesn't mean; _reddit_ring 29

parent_ngrams = {"this+cause":"causal.*r","that+way":"mode.*_r","this+create":"causal.*r",
                 "can+cause":"causal.*r","can+create":"causal.*r",
                 "this+lead":"causal.*r","is+why":"explanation.*_r",
                 "Since+then":".*background_r",
                 "opposite":".*antithesis_r"}  # TODO: Currently not lowercase or trigrams, consider lowercasing

altlex = {"base+on","base+upon","base+off","it+follow","give+rise","follow+by","this+cause","so+be","let+alone","especially","that+make",
          "as+is","as+are","as+does","as+do","as+such","for+this+reason","for+that+reason","in+this+case",
          "Since+then","since+then","other+than","as+such","following","Following","at+the+time"}  # TODO: support unigrams, e.g. 'either', 'following', 'too'

pdtb_altlex = set(open(RESOURCE_DIR + "altlex.txt").read().split("\n"))

sigtypes = """<sigtypes>
            <sig type="dm" subtypes="dm" />
            <sig type="graphical" subtypes="colon;dash;items_in_sequence;layout;parentheses;question_mark;quotation_marks;semicolon" />
            <sig type="lexical" subtypes="alternate_expression;indicative_phrase;indicative_word" />
            <sig type="morphological" subtypes="mood;tense" />
            <sig type="numerical" subtypes="same_count" />
            <sig type="orphan" subtypes="orphan" />
            <sig type="reference" subtypes="comparative_reference;demonstrative_reference;personal_reference;propositional_reference" />
            <sig type="semantic" subtypes="antonymy;attribution_source;lexical_chain;meronymy;negation;repetition;synonymy" />
            <sig type="syntactic" subtypes="infinitival_clause;interrupted_matrix_clause;modified_head;nominal_modifier;parallel_syntactic_construction;past_participial_clause;present_participial_clause;relative_clause;reported_speech;subject_auxiliary_inversion" />
            <sig type="unsure" subtypes="unsure" />
		</sigtypes>"""

if "indent" not in ET.__dict__:  # Python <3.9 compat
    def _pretty_print(current, sep, parent=None, index=-1, depth=0):
        current = current.getroot() if isinstance(current, ET.ElementTree) else current
        for i, node in enumerate(current):
            _pretty_print(node, sep, current, i, depth + 1)
        if parent is not None:
            if index == 0:
                parent.text = '\n' + (sep * depth)
            else:
                parent[index - 1].tail = '\n' + (sep * depth)
            if index == len(parent) - 1:
                current.tail = '\n' + (sep * (depth - 1))
    ET.indent = _pretty_print

rsd_deped = DepEdit()
rsd_deped.add_transformation("num=/.*/;num=/(.*)/&func=/(.*)/\t#1>#2\t#1:misc+=$1:$2")  # Add rsd child functions for each EDU
deped = DepEdit(config_file=RESOURCE_DIR + "depedit_preprocess.ini")

antonym_data = open(RESOURCE_DIR + "antonyms.tab").read().strip().split("\n")
antonyms = defaultdict(set)
for l in antonym_data:
    items = l.split(",")
    for i in items:
        for x in items:
            if x != i:
                antonyms[i].add(x)


do_word_pairs = True  # Use ConceptNet word pairs
wordpairs = defaultdict(lambda : defaultdict(str))

if do_word_pairs:
    wordpair_data = open(RESOURCE_DIR + "wordpairs.tab").read().strip().split("\n")
    for l in wordpair_data:
        words, rels = l.split("\t")
        items = words.split(",")
        for i in items:
            if i in stop_words:
                continue
            for x in items:
                if x in stop_words:
                    continue
                if x != i:
                    wordpairs[i][x] = rels

same_count = defaultdict(set)
same_count_data = TSVReader(RESOURCE_DIR + "same_count.xlsx").get_tsv()
for line in same_count_data.strip().split("\n"):
    docname, expression = line.split("\t")[:2]
    same_count[docname.strip()].add(expression.strip())


def make_signal(rst_node_id, maintype, subtype, tokens, status=''):
    elem = ET.Element('signal', attrib={'source': rst_node_id,
                                 'type': maintype,
                                 'subtype': subtype,
                                 'tokens': str(tokens),
                                    'status': status})
    return elem


def add_misc(misc, newanno):
    if misc == "_":
        return newanno
    else:
        key = newanno.split("=")[0]
        annos = [a for a in misc.split("|") if not a.startswith(key)] + [newanno]
        return "|".join(sorted(annos))


def clean_xml(xml):
    xml = xml.replace(" />", "/>").replace("    ", "\t").replace("<?xml version='1.0' encoding='utf8'?>\n", "")
    return xml


def get_cached_signals(gold_rs4, signal_list, docname):
    def make_candidate(signal_line):
        sigtype = re.search(r' type="([^"]+)"',signal_line).group(1)
        sigsubtype = re.search(r' subtype="([^"]+)"',signal_line).group(1)
        source = re.search(r' source="([^"]+)"',signal_line).group(1)
        tokens = re.search(r' tokens="([^"]*)"',signal_line).group(1)
        tokens = tuple([int(x)-1 for x in tokens.split(",")]) if len(tokens) > 0 else ""
        candidate = (tokens, '.*','.*','.*', sigtype, sigsubtype, "nid:" + source)
        return candidate

    # Get DMs and secedges from cached gold eRST file if it exists, otherwise return the current signal list

    # These signal types are always cached
    dms = [l for l in gold_rs4.split("\n") if "<signal" in l and ('type="dm"' in l or 'type="orphan"' in l or 'type="unsure"' in l or 'subtype="relative_conjunction"' in l or 'subtype="parallel' in l or 'subtype="negation' in l)]
    non_dm_gold_lines = [l for l in gold_rs4.split("\n") if "<signal" in l and 'status="gold"' in l and not ('type="dm"' in l or 'type="orphan"' in l or 'type="unsure"' in l or 'subtype="relative_conjunction"' in l or 'subtype="parallel' in l or 'subtype="negation' in l)]
    signal_list = [e for e in signal_list if e.attrib["subtype"] not in ["dm","orphan","unsure","relative_conjunction","parallel_syntactic_construction", "negation"]]
    secedges = [l for l in gold_rs4.split("\n") if 'secedge ' in l]
    for dm in dms:
        sigtype = re.search(r' type="([^"]+)"',dm).group(1)
        sigsubtype = re.search(r' subtype="([^"]+)"',dm).group(1)
        source = re.search(r' source="([^"]+)"',dm).group(1)
        tokens = re.search(r' tokens="([^"]*)"',dm).group(1)
        status = "gold" if 'status' not in dm else re.search(r' status="([^"]+)"',dm).group(1)
        signal = ET.Element('signal', attrib={'source': source, 'type': sigtype, 'subtype': sigsubtype, 'tokens': tokens, 'status': status})
        signal_list.append(signal)

    non_dm_gold = []
    for line in non_dm_gold_lines:
        non_dm_gold.append(make_candidate(line))

    # Always keep cached secedges
    edges = []
    for secedge in secedges:
        eid = re.search(r' id="([^"]+)"',secedge).group(1)
        src = re.search(r' source="([^"]+)"',secedge).group(1)
        trg = re.search(r' target="([^"]+)"',secedge).group(1)
        relname = re.search(r' relname="([^"]+)"',secedge).group(1)
        edge = ET.Element('secedge', attrib={'id': eid, 'source':src, "target":trg, "relname":relname})
        edges.append(edge)
    if len(edges) > 0:
        eblock =  ET.Element('secedges')
        for e in edges[::-1]:
            eblock.insert(0,e)
    else:
        eblock = None

    return signal_list, non_dm_gold, eblock


def conllu_stale(target_conllu, docname):
    def misc_part(field):
        annos = field.split("|")
        included = {"_","Disc","Ent"}
        output = "|".join([a for a in annos if any([a.startswith(x) for x in included])])
        if output == "":
            output = "_"
        return output

    # Check if conllu has changed since last run
    try:
        deped_input_path = SCRIPT_DIR + "dm-dependencies" + os.sep + "deped_inputs" + os.sep + docname + ".conllu"
        conllu_data = open(deped_input_path).read()
    except FileNotFoundError:
        return True
    target_fields = [l.split("\t") for l in target_conllu.split("\n") if "\t" in l]
    deped_fields = [l.split("\t") for l in conllu_data.split("\n") if "\t" in l]

    # First 8 columns excl. FEATS must match, discourse fields must match
    target_fields = ["\t".join(f[0:2] + f[3:5] + f[6:8] + [misc_part(f[-1])]) for f in target_fields]
    deped_fields = ["\t".join(f[0:2] + f[3:5] + f[6:8] + [misc_part(f[-1])]) for f in deped_fields]

    if any([t != d for t,d in zip(target_fields,deped_fields)]):
        return True
    return False


def rm_ellipsis(conllu):
    # Remove all ellipsis tokens with decmial IDs like 1.1 and any references to them in the DEPS field
    output = []
    lines = conllu.split("\n")
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            if "." in fields[0]:
                continue
            if "." in fields[8]:
                edeps = fields[8].split("|")
                edeps = [e for e in edeps if "." not in e]
                fields[8] = "|".join(edeps) if len(edeps) > 0 else "_"
                line = "\t".join(fields)
        output.append(line)
    return "\n".join(output)


def get_non_dm_signals(conllu, rs4, rsd, EDU2rel, genre, connective_idx, non_dm_gold, use_depedit_cache=False,
                       secedges=None,signal_cache=True):
    legacy_mode = False

    conllu = rm_ellipsis(conllu)

    docname = re.search("# newdoc id = ([^\s]+)",conllu).group(1)

    # Read rsd
    rsd = rsd_deped.run_depedit(rsd)

    tok2rsd = {}
    edu2rel = {}
    edu2dep_parent = {}
    edu2head_func = defaultdict(lambda :"none")
    edu2tense = defaultdict(lambda :"None")
    edu2head_lemma = {}
    edu2head_toknum = {}
    edu2child_rels = defaultdict(str)
    edu2lemmas = defaultdict(dict)
    edu2stems = defaultdict(dict)
    edu2tokids = defaultdict(list)
    edu2sent = defaultdict(int)
    sent2edu = defaultdict(set)
    sent2tokids = defaultdict(list)
    edupair2coref_signal_toks = defaultdict(set)
    same_unit2main = {}
    main2same_unit = defaultdict(set)
    toknum = 0
    for line in rsd.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            tokens = fields[1].split()
            rsd_rel = fields[7]
            edu2head_func[int(fields[0])] = re.search(r'head_func=([^|\t]+)',fields[5]).group(1)
            edu_tense = re.search(r'edu_tense=([^|\t]+)', fields[5]).group(1)
            edu2tense[int(fields[0])] = re.search(r'(None|Modal|Pres|Past|Fut)',edu_tense).group(1)
            edu2head_lemma[int(fields[0])] = re.search(r'head_tok=([^|\t]+)',fields[5]).group(1)
            edu2head_toknum[int(fields[0])] = int(re.search(r'head_id=([^|\t]+)',fields[5]).group(1))
            edu2child_rels[int(fields[0])] = fields[-1]
            edu2rel[int(fields[0])] = fields[7]
            edu2dep_parent[int(fields[0])] = int(fields[6])
            if rsd_rel.startswith("same-unit"):
                same_unit2main[int(fields[0])] = int(fields[6])
                main2same_unit[int(fields[6])].add(int(fields[0]))
                if edu2tense[int(fields[6])] == "None" and edu_tense != "None":  # Propagate same-unit tense
                    edu2tense[int(fields[6])] = edu_tense
            for t in tokens:
                tok2rsd[toknum] = rsd_rel
                edu2tokids[int(fields[0])].append(toknum)
                toknum += 1

    # Check that conllu has Discourse in MISC else add it
    if conllu.count("Discourse=") != len(edu2rel):
        temp = []
        toknum = 0
        edunum = 1
        edu_starters = [x[0] for x in edu2tokids.values()]
        for line in conllu.split("\n"):
            line = re.sub(r'Discourse=[^|\n]+','',line)
            if "\t" in line:
                fields = line.split("\t")
                if not ("." in fields[0] or "-" in fields[0]):
                    if toknum in edu_starters:
                        # Make like: Discourse=joint-list_m:14->8
                        relname = tok2rsd[toknum]
                        anno = "Discourse=" + relname.replace("_r","") + ":" + str(edunum)
                        if relname != "ROOT":
                            anno += "->" + str(edu2dep_parent[edunum])
                        fields[-1] = add_misc(fields[-1],anno)
                        line = "\t".join(fields)
                        edunum += 1
                    toknum +=1
            temp.append(line)
        conllu = "\n".join(temp)

    # Check conllu header
    if "global.Entity" not in conllu:
        # Add header
        conllu = re.sub(r'(# newdoc[^\n]+\n)',r'\1# global.Entity = GRP-etype\n',conllu)

    # Remove virtual tokens and MWTs, map sent starts
    tmp = []
    lines = conllu.split("\n")
    sid = 1
    offsets = {1:0}
    toknum = 0
    pos_tags = []
    upos_tags = []
    forms = []
    lemmas = []
    funcs = []
    tok_parents = []
    tok_children = defaultdict(set)
    text = []
    stems = []
    current_edu = 0
    for l in lines:
        if len(l.strip()) == 0:
            sid += 1
            offsets[sid] = toknum
        if "\t" in l:
            fields = l.split("\t")
            if "." in fields[0] or "-" in fields[0]:
                continue
            if "Discourse=" in fields[-1]:
                current_edu += 1
                edu2sent[current_edu] = sid
                sent2edu[sid].add(current_edu)
            sent2tokids[sid].append(toknum)
            forms.append(fields[1])
            lemmas.append(fields[2])
            funcs.append(fields[7])
            if fields[6] != "0":
                tok_parents.append(toknum + int(fields[6]) - int(fields[0]))
                tok_children[toknum + int(fields[6]) - int(fields[0])].add(toknum)
            else:
                tok_parents.append(-1)
            upos_tags.append(fields[3])
            pos_tags.append(fields[4])
            stem = snowball.stem(fields[2])
            stems.append(stem)
            if fields[3] in lexical_upos:
                edu2lemmas[current_edu][fields[2]] = toknum
                edu2stems[current_edu][stem] = toknum
            toknum += 1
        elif line.startswith("# text = "):
            text.append(line.split("=",1).strip())
        tmp.append(l)
    conllu = "\n".join(tmp)

    lemma_form_text = "_".join(lemmas) + "_" + "_".join(text).replace(" ","_")
    lemma_text = "_".join(set(lemmas))
    wordpairs_subset = {k: wordpairs[k] for k in wordpairs if k in lemma_form_text}  # For performance

    grams = set()  # attested ngram cache for rule filtering
    for i in range(len(forms)-2):
        grams.add(forms[i]+"_"+forms[i+1])
        grams.add(lemmas[i]+"_"+lemmas[i+1])
        grams.add(forms[i]+"_"+forms[i+1]+"_"+forms[i+2])
        grams.add(lemmas[i]+"_"+lemmas[i+1]+"_"+lemmas[i+2])
    grams.add(forms[i+1]+"_"+forms[i+2])
    grams.add(lemmas[i+1]+"_"+lemmas[i+2])

    recurring_stems = {stem for stem in stems if stems.count(stem) > 1}
    tok2stem = defaultdict(lambda :defaultdict(set))
    for i, upos in enumerate(upos_tags):
        if lemmas[i] not in stop_words and upos in lexical_upos and stems[i] in recurring_stems:
            tok2stem[i][stems[i]].add((i,))

    edu2sent_head_edu = {}
    for sent in sent2edu:
        for edu in sent2edu[sent]:
            if edu2head_func[edu] == "root":
                root_edu = edu
        for edu in sent2edu[sent]:
            if root_edu in same_unit2main:
                root_edu = same_unit2main[root_edu]
            edu2sent_head_edu[edu] = root_edu

    # Read rs4
    nodes = read_rst(rs4,{},as_text=True)
    edus = list(nodes[nid] for nid in nodes if nodes[nid].kind == "edu")
    edus.sort(key=lambda x: int(x.id))
    keys = [nid for nid in nodes]
    if secedges is None:
        secedges = []
        for nid in keys:
            if "-" in nid:
                secedges.append(nodes[nid])
                del nodes[nid]
    # Get head EDU and height per node
    node2head_edu = {}
    node2multinuc_children = defaultdict(set)
    node2descendent_edus = defaultdict(set)
    node2tokspan = defaultdict(lambda : [0,0])
    for edu in edus:
        node = edu
        edu_id = edu.id
        node2head_edu[node.id] = edu_id
        while node.parent != "0":
            node2tokspan[node.id] = (sorted(list(edu2tokids[node.left]))[0], sorted(list(edu2tokids[node.right]))[-1])
            if nodes[node.parent].kind == "multinuc" and node.relname.endswith("_m"):
                node2multinuc_children[node.parent].add(node.id)
            elif node.relname != "span":  # This is a satellite, clear edu_id for head check
                edu_id = ""
            node = nodes[node.parent]
            if node.kind == "edu":
                edu_id = node.id
            if node.id not in node2head_edu:
                if edu_id != "":
                    node2head_edu[node.id] = edu_id
            elif edu_id != "":
                if int(edu_id) < int(node2head_edu[node.id]):  # Prefer left most child as head
                    node2head_edu[node.id] = edu_id
            node2descendent_edus[node.id] = set([str(x) for x in list(range(node.left,node.right+1))])

    relations = []
    leftmost_sisters = set()  # Left most children of multinucs
    # Add secedges to nodes
    for secedge in secedges:
        nodes[secedge["id"]] = deepcopy(nodes[secedge["source"]])
        nodes[secedge["id"]].relname = secedge["relname"] + "_r"
        nodes[secedge["id"]].parent = secedge["target"]

    for nid in nodes:
        node = nodes[nid]
        if node.relname != "span":
            head_edus = [node2head_edu[node.id]]
            descendents = node2descendent_edus[node.id]
            if node.kind == "edu":
                descendents.add(node.id)
            if "same-unit" in nodes[head_edus[0]].relname:
                head_edus = node2multinuc_children[nodes[head_edus[0]].parent]
            if node.relname.endswith("_m"):
                if "same-unit" in node.relname:
                    continue
                else:  # multinuc
                    sisters = node2multinuc_children[node.parent]
                    leftmost_sisters.add(sorted(list(sisters),key=lambda x:nodes[x].left)[0])
                    sis_head_edus = [node2head_edu[s] for s in sisters if node2head_edu[s] not in descendents]
                    for sis in sis_head_edus:
                        if sis in main2same_unit:
                            for sameunit in main2same_unit[sis]:
                                sis_head_edus.append(sameunit)
                    target_head_edus = set(sis_head_edus)
                    target_descendents = set()
                    for e in target_head_edus:
                        target_descendents.add(e)
                        for d in node2descendent_edus[e]:
                            target_descendents.add(d)
            elif node.relname != "":  # satellite
                target = nodes[node.parent]
                target_head_edus = [node2head_edu[target.id]]
                if target_head_edus[0] in node2multinuc_children:
                    target_sisters = node2multinuc_children[node.parent]
                    target_head_edus = [node2head_edu[s] for s in target_sisters]
                    for edu in target_head_edus:
                        if edu in main2same_unit:
                            for sameunit in main2same_unit[edu]:
                                target_head_edus.append(sameunit)
                    target_head_edus = set(target_head_edus)
                target_descendents = set([d for d in node2descendent_edus[target.id] if d not in descendents])
                if target.kind == "edu":
                    target_descendents.add(target.id)
            else:  # Root
                continue

            if "-" in nid:
                node.id = nid

            rel = RELATION(node,edus=descendents,head_edus=head_edus,target_head_edus=target_head_edus,
                           target_descendents=target_descendents,edu2sent=edu2sent,sent2toks=sent2tokids,
                           node2descendents=node2descendent_edus,edu2toks=edu2tokids,edu2head_func=edu2head_func,
                           edu2tense=edu2tense,docname=docname,tokens=forms,node2head_edu=node2head_edu,
                           edu2dep_parent=edu2dep_parent)
            relations.append(rel)

    # Collect entities mentioned in each EDU
    tok2entities = defaultdict(lambda :defaultdict(set))
    tok2bridging = defaultdict(lambda :defaultdict(set))
    eid2mention_spans = defaultdict(list)
    edu_num = 0
    heads_to_mentions = defaultdict(set)
    mentions_to_heads = defaultdict(list)
    mentions_to_mtype = defaultdict(str)
    mention_text_to_span = defaultdict(list)
    bridging = defaultdict(str)

    open_mentions = []
    mention_span = defaultdict(lambda :defaultdict(list))
    mention_text = defaultdict(lambda :defaultdict(list))
    head_span = defaultdict(lambda :defaultdict(list))
    toknum = 0
    sid = 1
    for line in conllu.split("\n"):
        if "\t" not in line:
            if len(line.strip()) == 0:
                sid += 1
            continue
        fields = line.split("\t")
        if "." in fields[0] or "-" in fields[0]:
            continue
        if "Discourse=" in fields[-1]:
            edu_num += 1
        if "Bridge=" in fields[-1]:
            bridging_info = re.search(r'Bridge=([^|]+)',fields[-1]).group(1)  # Bridge=76<82
            for bridge_instance in bridging_info.split(","):  # Could have multiple bridging like Bridge=12<124,12<125
                parts = bridge_instance.split("<")
                bridging[parts[1]] = parts[0]
        if "Entity=" in fields[-1]:
            ent_string = [x for x in fields[-1].split("|") if x.startswith("Entity=")][0].split("=")[1]
            openers = re.findall(r'\(([0-9]+[^()]+)',ent_string)
            for opener in openers:
                open_mentions.append(opener)
            closers = re.findall(r'\(([0-9]+)-[^()]+\)', ent_string)
            closers += re.findall(r'(?<=[=)])([0-9]+)\)', "="+ent_string)
        else:
            closers = []

        for m in open_mentions:
            minspan = m.split("-")[4]
            head_words = [int(x) for x in minspan.split(",")]  # 1-based
            eid = m.split("-")[0]
            depth = len([x for x in mention_span[eid] if toknum in mention_span[eid][x]])
            mention_span[eid][depth].append(toknum)
            mention_text[eid][depth].append(fields[1])
            if len(mention_text[eid][depth]) in head_words:
                head_span[eid][depth].append(toknum)

        for eid in closers:
            depth = [x.split("-")[0] for x in open_mentions].count(eid) - 1
            opener = [x for x in open_mentions if x.startswith(eid+"-")][-1]
            mention_span[eid][depth] = tuple(sorted(mention_span[eid][depth]))
            mention_text_to_span[" ".join(mention_text[eid][depth])].append(mention_span[eid][depth])
            for tok in head_span[eid][depth]:
                heads_to_mentions[tok].add(mention_span[eid][depth])
                mentions_to_heads[mention_span[eid][depth]].append(tok)
            for tok in []: #m.words:
                if "PronType" in tok.feats:  # Also map relative pronouns to their governing entities
                    if tok.feats["PronType"] == "Rel":
                        if tok.parent.deprel == "acl:relcl":
                            abs_rel_pron_id = tok.ord + offsets[sid] - 1
                            matrix_nominal = tok.parent.parent
                            abs_nominal_id = tok.parent.parent.ord + offsets[sid] - 1
                            if abs_nominal_id in heads_to_mentions:
                                for span in heads_to_mentions[abs_nominal_id]:
                                    heads_to_mentions[abs_rel_pron_id].add(span)

            mentions_to_mtype[mention_span[eid][depth]] = opener.split("-")[5]
            tok2entities[toknum][eid].add(mention_span[eid][depth])
            if mention_span[eid][depth] not in eid2mention_spans[eid]:
                eid2mention_spans[eid].append(mention_span[eid][depth])

            mention_span[eid][depth] = []
            mention_text[eid][depth] = []
            head_span[eid][depth] = []
            open_mentions.reverse()  # LIFO
            open_mentions.remove(opener)
            open_mentions.reverse()
        toknum +=1

    # Bridging pairs
    for bridge_source in bridging:
        bridge_target = bridging[bridge_source]
        mention2 = eid2mention_spans[bridge_source][0]
        for span in sorted(eid2mention_spans[bridge_target]):
            if span[0] < mention2[0]:
                mention1 = span
        for tok in mention1:
            tok2bridging[tok][bridge_target].add(mention1)
        for tok in mention2:
            tok2bridging[tok][bridge_target].add(mention2)

    # Filter irrelevant rules
    rules = []
    for rule in all_rules:
        if 1 in rule.criteria:  # rule targets lemma
            if rule.criteria[1].match_type == "exact":
                if rule.criteria[1].value not in lemmas:
                    continue
            else:
                m = re.search(rule.criteria[1].value,lemma_text)
                if m is None:
                    continue
        if 8 in rule.criteria:  # rule targets MISC
            if rule.criteria[8].value.startswith("gram") and "%" not in rule.criteria[8].value:
                vals = re.sub(r'gram=|\(|\)|_$','',rule.criteria[8].value).split("|")
                if not any([v in grams for v in vals]):
                    continue
        rules.append(rule)

    xsubj_map = {}
    token_lines = [l.split("\t") for l in conllu.split("\n") if "\t" in l]
    for toknum, fields in enumerate(token_lines):
        edeps = fields[8]
        if edeps != "_" and "." not in edeps:
            for edep in edeps.split("|"):
                try:
                    head, edeprel = edep.split(":", 1)
                except ValueError:
                    print("! Broken edep in " + docname + " tok "+str(toknum)+":\n" + "\t".join(fields))
                    continue
                if "xsubj" in edeprel:
                    abs_head = toknum + int(head) - int(fields[0])
                    xsubj_map[abs_head] = toknum
                elif "subj" in edeprel:
                    if head != fields[6]:
                        abs_head = toknum + int(head) - int(fields[0])
                        xsubj_map[abs_head] = toknum

    signal_tokens = non_dm_gold

    signals = []

    # Generate same_count signals
    if signal_cache:
        expressions = set(same_count[docname])
        for mention_text in sorted(mention_text_to_span,key=lambda x:x.count(" ")):
            for expression in same_count[docname]:
                if expression in mention_text and expression in expressions:
                    span = mention_text_to_span[mention_text][0]
                    for t in span:
                        if pos_tags[t] == "CD":
                            candidate = (t, "organization-(preparation|heading).*", ".*", ".*", "numerical", "same_count","initial")
                            signal_tokens.append(candidate)
                            candidate = (t, ".*", ".*", "elaboration-additional.*", "numerical", "same_count","initial")
                            signal_tokens.append(candidate)
                            expressions.remove(expression)
                            break
                if len(expressions) == 0:
                    break

    # Get lexical chains and tense signals
    if signal_cache:
        signal_tokens += tsv_lexchain.parse_gold_table(docname, relations, forms, sigtype="semantic:lexical_chain2")
        signal_tokens += tsv_tense.parse_gold_table(docname, relations, forms, sigtype="morphological:tense")
        signal_tokens += tsv_attribution.parse_gold_table(docname, relations, forms, sigtype="semantic:attribution_source", location="adjacent")
        signal_tokens += tsv_additional.parse_gold_table(docname, relations, forms, sigtype="any:any", location="adjacent")

    # Flag attributions covering only one entity
    for rel in relations:
        if rel.relname.startswith("attribution"):
            unique = set([])
            ents = [set(tok2entities[t].keys()) for t in range(rel.source["all"][0],rel.source["all"][-1]+1)]
            for e in ents:
                unique.update(e)
            if len(unique) == 1:
                if any([pos_tags[p].startswith("V") for p in range(rel.source["all"][0],rel.source["all"][-1]+1)]):
                    continue
                spans = [span for span in eid2mention_spans[list(unique)[0]] if span[0] >= rel.source["all"][0] and span[-1] <= rel.source["all"][-1]]
                if len(spans)>0:
                    candidate = (spans[0], "attribution..*", ".*", ".*", "semantic", "attribution_source","initial")
                    signal_tokens.append(candidate)

    tok2edu = {}
    current_edu = 0
    parent_funcs = []
    toknum = 0
    starts = {}
    s_type = ""
    edu2verbs = defaultdict(lambda :defaultdict(set))
    edu2antonyms = defaultdict(lambda :defaultdict(set))
    edu2wordpairs = defaultdict(lambda :defaultdict(set))
    seen_paired = set([])
    tok2tense = defaultdict(lambda :defaultdict(set))
    for line in conllu.split("\n"):
        if line.startswith("# s_type"):
            s_type = line.split("=")[1].strip()
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
            paired_partner = None
            if "Paired=" in fields[-1]:  # Find closing partner
                tnum = re.search(r'Paired=([0-9]+)',fields[-1]).group(1)
                distance = int(tnum) - int(fields[0])
                paired_partner = toknum + distance
            if "Discourse=" in fields[-1]:
                current_edu += 1
                if "->" in fields[-1]:
                    parent_edu = int(re.search(r'->([0-9]+)',fields[-1]).group(1))
                else:
                    parent_edu = 0
                current_rel = re.search(r'Discourse=([^:]+):',fields[-1]).group(1)
                if current_rel.startswith("same-unit"):
                    parent_edu = edu2dep_parent[same_unit2main[current_edu]]
                # Get paired tense signals
                if "sequence" in current_rel and not signal_cache:
                    tense1 = re.search(r'(None|Modal|Pres|Past|Fut)',edu2tense[parent_edu]).group(1)
                    tense2 = re.search(r'(None|Modal|Pres|Past|Fut)',edu2tense[current_edu]).group(1)
                    if (tense1, tense2) in [("Pres","Fut"),("Past","Pres"),("Past","Fut")]:
                        sig_toks = []
                        target_tokens = edu2tokids[current_edu] + edu2tokids[parent_edu]
                        if current_edu in main2same_unit:
                            for edu in main2same_unit[current_edu]:
                                target_tokens += edu2tokids[edu]
                        if parent_edu in main2same_unit:
                            for edu in main2same_unit[parent_edu]:
                                target_tokens += edu2tokids[edu]
                        for t in target_tokens:
                            if pos_tags[t] in ["VBP","VBZ","VBD"]:
                                sig_toks.append(t)
                        if len(sig_toks) >= 2:
                            candidate = (tuple(sorted(sig_toks)), "joint-sequence.*", ".*", ".*", "morphological", "tense","initial")
                            signal_tokens.append(candidate)
                # Coref
                if current_edu > parent_edu:
                    location = "final"
                    early_edu = parent_edu
                    late_edu = current_edu
                else:
                    location = "initial"
                    early_edu = current_edu
                    late_edu = parent_edu

            tok2edu[toknum] = current_edu

            # Collect lexical items for synonym/antonym/repetition checks
            current_grams = re.findall(r'gram=([^|]+)_\|',fields[-1])
            if fields[4].startswith("V") and fields[7] in clause_funcs:  # Collect verbs to check for repetitions on restatement
                edu2verbs[current_edu][fields[2]].add(toknum)
            if fields[2] in antonyms:
                edu2antonyms[current_edu][fields[2]].add(toknum)
            if fields[2] in wordpairs_subset:
                edu2wordpairs[current_edu][fields[2]].add(toknum)
            for gram in current_grams:
                if gram in wordpairs_subset:
                    edu2wordpairs[current_edu][gram].add(toknum)

            # Candidate rule format: toknum, container_dr_regex, container_root_deprel, children_dr_regex, maintype, subtype, location of relation EDU relative to signals: initial or final
            input_fields = fields[1:] + [s_type, genre]
            for r, rule in enumerate(rules):
                m, seen_paired = rule.match(input_fields,toknum,connective_idx,heads_to_mentions,xsubj_map,paired_partner,seen_paired)
                if m is not None:
                    signal_tokens.append(m)

            if "Negated" in fields[-1]: # Contrast via negation
                if parent_edu != 0:
                    if fields[2] == edu2head_lemma[parent_edu]:
                        negator_offset = int(re.search(r'Negated=([0-9]+)',fields[-1]).group(1)) - int(fields[0])
                        negator_toknum = toknum + negator_offset
                        partner_lemma_id = edu2head_toknum[parent_edu] - 1
                        this_sig_tokens = tuple(sorted([toknum,negator_toknum,partner_lemma_id]))
                        candidate = (this_sig_tokens, "adversative-.*", ".*", ".*", "semantic", "negation","final")
                        signal_tokens.append(candidate)
            if fields[4].startswith("V") and fields[7] in clause_funcs:
                if fields[2] in edu2verbs[parent_edu]:  # clausal verb lemma from parent EDU repeated as clausal verb lemma
                    prev_tokens = list(edu2verbs[parent_edu][fields[2]])
                    this_sig_tokens = tuple(sorted([toknum] + prev_tokens))
                    edu2verbs[parent_edu].pop(fields[2])  # Ensure we don't re-use this signal
                    candidate = (this_sig_tokens, "restatement-.*", ".*", ".*", "semantic", "lexical_chain","initial")
                    signal_tokens.append(candidate)
            if fields[4] in tense_pos:  # Tensed verb forms
                if fields[4] != "MD" or fields[2] in ["will","shall"]:
                    tok2tense[toknum][tense_pos[fields[4]]].add((toknum,))
            found = False
            if not signal_cache:
                for item in sorted(current_grams,key=lambda x:-len(x)) + [fields[2]]:
                    if item in wordpairs_subset:
                        found = True
                        break
            if found:
                parent_items = [x for x in edu2wordpairs[parent_edu] if x in wordpairs_subset[item]]
                if len(parent_items) > 0:
                    for prev_tokens_positions, rels, parent_item in [(edu2wordpairs[parent_edu][x], wordpairs_subset[item][x], x) for x in parent_items]:
                        for pos in prev_tokens_positions:
                            this_sig_tokens = set()
                            this_sig_tokens.add(pos)
                            for i in range(parent_item.count("_")):
                                this_sig_tokens.add(pos + i + 1)
                            these_toks = [toknum]
                            for i in range(item.count("_")):
                                these_toks.append(toknum + i + 1)
                            this_sig_tokens = tuple(sorted(list(this_sig_tokens) + these_toks))
                            candidate = (this_sig_tokens, rels, ".*", ".*", "semantic", "lexical_chain2","final")  # formerly indicative_word_pair
                            signal_tokens.append(candidate)
            elif fields[2] in antonyms:
                parent_items = [x for x in edu2antonyms[parent_edu] if x in antonyms[fields[2]]]
                if len(parent_items) > 0:
                    prev_tokens_positions = [edu2antonyms[parent_edu][x] for x in parent_items]
                    this_sig_tokens = set()
                    for i in prev_tokens_positions:
                        for pos in i:
                            this_sig_tokens.add(pos)
                    this_sig_tokens = tuple(sorted(list(this_sig_tokens) + [toknum]))
                    candidate = (this_sig_tokens, "(adversative-.*|.*disjunction).*", ".*", ".*", "semantic", "antonymy","final")
                    signal_tokens.append(candidate)

            if toknum not in connective_idx:  # Make sure indicative words are not already part of a connective
                for gram in parent_ngrams:  # Currently only supports bigrams
                    if 'LemmaBigram=' + gram + "+" in fields[-1]:
                        gram_type = "indicative_phrase" if gram not in altlex else "alternate_expression"
                        main_type = "lexical"
                        candidate = ((toknum, toknum + 1), ".*", ".*", parent_ngrams[gram], main_type, gram_type, "final")
                        signal_tokens.append(candidate)
                        break
            if "<date" in fields[-1] or "<time" in fields[-1]:
                timetype = re.search(r'<(date|time)',fields[-1]).group(1)
                starts[timetype] = toknum
            if "</date" in fields[-1] or "</time" in fields[-1]:  # Dates as indications of temporal relations
                timetype = re.search(r'</(date|time)', fields[-1]).group(1)
                start = starts[timetype]
                if start == toknum:
                    tup = toknum
                    ind_type = "indicative_word"
                else:
                    tup = (start,toknum)
                    tup = tuple([x for x in range(start, toknum+1)])
                    ind_type = "indicative_phrase"
                candidate = (tup,"(context-circumstance_r|joint-sequence_m)",".*",".*","lexical",ind_type,"initial")
                signal_tokens.append(candidate)

            toknum += 1

    all_edus = list(range(1,current_edu+1))

    edu2nonspanid = {}
    for edu in all_edus:
        relname = EDU2rel[edu]["relname"]
        edu2nonspanid[edu] = edu
        parent = EDU2rel[edu]["parent"]
        while relname == "span" and parent is not None:
            relname = EDU2rel[parent]["relname"]
            edu2nonspanid[edu] = int(parent)
            parent = EDU2rel[parent]["parent"]
    edu2nonmulti = {}
    for edu in all_edus:
        relname = EDU2rel[edu]["relname"]
        edu2nonmulti[edu] = edu
        parent = EDU2rel[edu]["parent"]
        while relname in ["span"] + multinuc_rels and parent is not None:
            relname = EDU2rel[parent]["relname"]
            edu2nonmulti[edu] = int(parent)
            parent = EDU2rel[parent]["parent"]

    for rel in relations:
        # Coref matching
        signal_tokens += rel.get_pair_signals(tok2entities, pos_tags, lemmas, mentions_to_heads=mentions_to_heads, mentions_to_mtype=mentions_to_mtype, pair_type="coref", pattern="(elaboration-additional|context-background|restatement|organization-(preparation|heading)).*")
        signal_tokens += rel.get_pair_signals(tok2bridging, pos_tags, lemmas, mentions_to_mtype=mentions_to_mtype, pair_type="bridging", pattern="(elaboration-additional|context-background|adversative.*|.*list*)")
        signal_tokens += rel.get_pair_signals(tok2stem, pos_tags, lemmas, pair_type="lexical_chain", pattern=r'(organization-(prep.*|head.*))|(restatement.*)')
        if not signal_cache:  # Get predicted tense sequence signals
            signal_tokens += rel.get_pair_signals(tok2tense, pos_tags, lemmas, pair_type="tense", pattern=r'(joint-sequence.*)')

    seen = set()
    for t, tup in enumerate(signal_tokens):
        toknum, match_rel, edu_headfunc, child_rel, maintype, subtype, location = tup

        if isinstance(toknum, tuple):
            if len(toknum) == 1:  # Prevent single token tuple
                toknum = toknum[0]
        if isinstance(toknum,tuple):
            tokspan = ",".join([str(t+1) for t in sorted(list(toknum))])
            if location != "initial":
                toknum = toknum[-1]
            else:
                toknum = toknum[0]
        elif subtype in ["layout","items_in_sequence"] or toknum == "":  # Signal with no specific associated tokens
            if not (pos_tags[toknum] == "LS" and subtype == "items_in_sequence"):
                tokspan = ""
            else:
                tokspan = str(toknum + 1)
        else:
            tokspan = str(toknum+1)  # 1-based index in rs4

        if (docname,maintype,subtype,tokspan) in forbidden:
            continue

        if location.startswith("nid:"):  # Exact instruction from resource
            nid = location.split(":")[1]
            signals.append(make_signal(nid, maintype, subtype, tokspan, status="gold"))
            continue

        # Go through all relations, sorted from least source domain covered EDUs to most
        signal_assigned = False
        for relnum, rel in enumerate(sorted(relations,key=lambda x: x.source_width)):

            invert = False
            if child_rel not in [".*","_"]:
                match_rel = child_rel
                invert = True
            discont = True if maintype == "graphical" else False

            same_sent = False
            force = False
            domain = "head_sent"

            if location == "same_sent":
                location = "initial"
                same_sent = True
            elif location == "xsubj":
                domain = "head_fullsent"
                force = True
            max_span = None
            if subtype == "indicative_word_pair":
                domain = "head"
                max_span = 30
                subtype = "lexical_chain"
            if subtype == "lexical_chain2":
                domain = "all"
                if signal_cache:
                    subtype = "lexical_chain"
            paired = False if "lexical_chain" not in subtype else True
            if rel.match(tup[0], toknum, match_rel, edu_headfunc, domain=domain, force_sent=force, invert=invert,
                         discontinuous_single=discont, same_sent=same_sent, max_span=max_span, paired=paired):
                nid = rel.nid
                if location == "xsubj":
                    location = "initial"
                if rel.nid in leftmost_sisters and not ALLOW_LEFT_SISTERS:  # Leftmost multinuc sister signals should move to sibling
                    sisters = sorted(list(node2multinuc_children[nodes[rel.nid].parent]))
                    sisters.remove(nid)
                    nid = sisters[0]  # Default to second leg of multinuc
                    if isinstance(tup[0], tuple):
                        for sister in sisters:  # Check if part of the signal is in a specific non-initial sister
                            sister_tokens = list(range(node2tokspan[sister][0], node2tokspan[sister][1] + 1))
                            if any([t in sister_tokens for t in tup[0]]):
                                nid = sister
                                break
                signal = make_signal(nid, maintype, subtype, tokspan)
                unique_tokspan = tokspan if tokspan != "" else str(toknum)
                if signal not in signals and (maintype, subtype, unique_tokspan) not in seen:
                    skip = False
                    if subtype == "lexical_chain":
                        for sig in seen:
                            oldtype, oldsubtype, old_tokspan = sig
                            if oldtype == "reference" or oldsubtype == "repetition":
                                if unique_tokspan.split(",")[-1] in old_tokspan.split(","):
                                    skip = True  # Skip lexical chain if already part of coref signal for the relation
                    if not skip:
                        signals.append(signal)
                        seen.add((maintype, subtype, unique_tokspan))  # No duplicates for same type and token span
                        signal_assigned = True
                        break  # match only one, minimally spanned relation per signal

        if not signal_assigned and not signal_cache:  # Syntactic orphan signal candidate
            if subtype == "indicative_word" and match_rel.startswith("attribution"):  # Secondary attribution
                deprel = "ccomp"
            elif subtype == "modified_head" and child_rel.endswith("attribute_r"):  # Secondary attribute
                deprel = "acl"
            else:
                continue
            if "," not in tokspan:
                index = int(tokspan) - 1
                for tok in tok_children[index]:
                    if funcs[tok].startswith(deprel):  # Has ccomp/acl child
                        word_edu = tok2edu[index] # EDU containing indicative word/modified head word
                        if word_edu in same_unit2main:
                            word_edu = same_unit2main[word_edu]
                        ccomp_edu = tok2edu[tok] # EDU containing ccomp/acl
                        if ccomp_edu in same_unit2main:
                            ccomp_edu = same_unit2main[ccomp_edu]
                        if 'phatic' in edu2rel[word_edu] and deprel == "ccomp":  # Phatic cannot be attribution source
                            break
                        if 'elab' in edu2rel[ccomp_edu] or 'attribute' in edu2rel[ccomp_edu] and deprel == "acl":  # No double attribute/elaboration
                            break
                        # Check if another relation connects them, which prevented the attribution/attribute edge
                        if edu2dep_parent[word_edu] == ccomp_edu:  # ccomp/acl is primary parent of indicative word/modified head
                            if word_edu in edu2nonspanid:
                                word_edu = edu2nonspanid[word_edu]
                            if str(word_edu) in nodes:
                                ccomp_edu = nodes[str(word_edu)].parent  # Target node should match target of RST equivalent of the dependency
                        elif edu2dep_parent[ccomp_edu] == word_edu: # indicative word/modified head is primary parent of ccomp/acl
                            if ccomp_edu in edu2nonspanid:
                                ccomp_edu = edu2nonspanid[ccomp_edu]
                            if str(ccomp_edu) in nodes:
                                word_edu = nodes[str(ccomp_edu)].parent  # Target node should match target of RST equivalent of the dependency
                        else:
                            break  # No primary relation found, abort
                        if word_edu != ccomp_edu:  # Indicative word and ccomp in different nodes, make secedge
                            if deprel == "ccomp":
                                nid = nodes[str(word_edu)].id + "-" + nodes[str(ccomp_edu)].id
                            else:
                                nid = nodes[str(ccomp_edu)].id + "-" + nodes[str(word_edu)].id
                            signal = make_signal(nid, maintype, subtype, tokspan)
                            signals.append(signal)

    # Merge parts of same signals
    final_signals = defaultdict(lambda : defaultdict(list))
    exceptions = [("semantic","repetition"),("lexical","indicative_word"),("lexical","indicative_phrase"),
                  ("semantic","meronymy"),("semantic","antonymy"),("reference","personal_reference"),
                  ("reference","propositional_reference"),("reference","demonstrative_reference"),
                  ("reference","comparative_reference")]
    for sig in signals:
        src = sig.attrib["source"]
        maintype = sig.attrib["type"]
        if signal_cache:  # In uncached runs we distinguish lexical signal origins
            sig.attrib["subtype"] = sig.attrib["subtype"].replace("2","").replace("indicative_word_pair","lexical_chain")
        subtype = sig.attrib["subtype"]
        tokens = sig.attrib["tokens"]
        sig_string = " ".join([forms[int(x)-1] for x in tokens.split(",")]) if "," in tokens else ""
        if sig_string in pdtb_altlex and subtype.startswith("indicative"):
            subtype = "alternate_expression"
        if (maintype,subtype) not in exceptions:
            if (maintype,subtype) in final_signals[src]:
                old_tokens = final_signals[src][(maintype, subtype)][0].attrib["tokens"]
                if subtype in ["indicative_word_pair","lexical_chain"]:  # These types only merge on overlap
                    if all([t not in tokens.split(",") for t in old_tokens.split(",")]):
                        final_signals[src][(maintype,subtype)].append(sig)
                        continue  # No overlap
                if len(tokens) > 0:
                    old_tokens += "," + tokens
                if old_tokens != "":  # Only add to tokens of existing signal copy if there are actually tokens
                    tokens = ",".join([str(x) for x in sorted([int(t) for t in list(set(old_tokens.split(",")))])])
                    final_signals[src][(maintype, subtype)][0].attrib["tokens"] = tokens
                continue

        if (maintype == "lexical" or subtype in ["indicative_word_pair","lexical_chain"]) and not ("-" in src and subtype not in ["orphan","dm"]):
            token_indices = [int(x)-1 for x in tokens.split(",")]
            if any([t in connective_idx for t in token_indices]):
                continue  # Skip if tokens are already used for a dm or lexical signal unless this is a secedge
            for t in token_indices:  # Ensure these tokens will not be used for another lexical signal
                connective_idx.add(t)

        final_signals[src][(maintype,subtype)].append(sig)

    sorted_output = {}
    for src in final_signals:
        for tup in final_signals[src]:
            maintype, subtype = tup
            sigs = final_signals[src][tup]
            for i, sig in enumerate(sigs):
                if "-" in str(src):
                    key_src = [int(x) for x in src.split("-")]
                else:
                    key_src = [int(src),int(src)]
                sort_key = tuple(key_src + [maintype, subtype, i])
                sorted_output[sort_key] = sig

    output = [sorted_output[k] for k in sorted(list(sorted_output.keys()))]

    return output, relations


def update_signals(gold_rs4, docname, xml_root=None, rerun_depedit=False, no_cache=False):

    if xml_root is None:
        xml_root = XML_ROOT

    rsd = make_rsd(gold_rs4, xml_dep_root=xml_root, as_text=True, docname=docname)

    tree = ET.parse(io.StringIO(gold_rs4))
    root = tree.getroot()
    EDU2rel = {}
    sat2nuc = {}  # dct to map a satellite to nucleus
    get_nuc = {}  # same as get_child except that a directed relation is not allowed
    get_parent = {}  # same as EDU2rel[edu_id]['parent'] but for easier access
    cdu_contains = {}  # dct to map a CDU to all edus contained in it
    for child in root:          # first pass to create a dictionary
        if child.tag == "body":
            for gc in child:
                if gc.tag in ["secedges","signals"]:
                    continue
                id = int(gc.get('id'))
                parent = int(gc.get('parent')) if gc.get('parent') is not None else gc.get('parent')
                EDU2rel[id] = {'type': gc.get('type'), 'parent': parent, 'relname': gc.get('relname')}
                get_parent[id] = parent
                sat2nuc[id] = parent
                if parent is None:
                    continue
                if parent not in cdu_contains:
                    cdu_contains[parent] = []
                cdu_contains[parent].append((id, gc.get('relname')))
        elif child.tag == "header":
            if len([c for c in child]) == 1:  # No signals
                sigtypes_xml = ET.fromstring(sigtypes)
                child.append(sigtypes_xml)

    signal_list = []
    signal_list, non_dm_gold, secedges = get_cached_signals(gold_rs4, signal_list, docname)
    if no_cache:
        non_dm_gold = []

    nofile = False
    target_conllu = open(CONLLU_TARGET + docname + ".conllu").read()
    try:
        conllu_data = open(DEPED_CACHE + docname + ".conllu").read()
    except FileNotFoundError:
        conllu_data = target_conllu
        nofile = True
    if conllu_stale(target_conllu, docname) or rerun_depedit or nofile:
        sys.stderr.write("o Rerunning depedit for stale conllu " + docname + "\n")
        conllu_data = deped.run_depedit(target_conllu)
        deped_input_path = SCRIPT_DIR + "dm-dependencies" + os.sep + "deped_inputs" + os.sep + docname + ".conllu"
        with open(deped_input_path,'w',encoding="utf8",newline="\n") as f:
            f.write(target_conllu)
        with open(DEPED_CACHE + docname + ".conllu", 'w', encoding="utf8", newline="\n") as f:
            f.write(conllu_data)

    connective_idx = [s.attrib["tokens"].split(",") for s in signal_list if s.attrib["type"] in ["dm", "orphan"]]
    connective_idx = set([int(t) - 1 for tokens in connective_idx for t in tokens])
    secedge_list = [x.attrib for x in secedges] if secedges is not None else None
    genre = docname.split("_")[1]  # Get genre, since some signals only apply to spoken/written data types
    non_dm_signals, relations = get_non_dm_signals(conllu_data, gold_rs4, rsd, EDU2rel, genre, connective_idx, non_dm_gold,
                                                     secedges=secedge_list, signal_cache=not no_cache)

    signal_list += list(set(non_dm_signals))
    signal_list.sort(key=lambda x: tuple([int(y) for y in str(x.attrib["source"]).split("-")] + [int(y) if y != "" else 0 for y in str(x.attrib["tokens"]).split(",")]))

    # appending to existing .rs4
    signals = ET.Element('signals')
    for child in root:
        if child.tag == "body":
            if secedges is not None and '</secedges>' not in gold_rs4:
                child.append(secedges)
            # Replace old signals with new ones
            child.remove(child.find('signals'))
            child.append(signals)
            for gc in child:
                if gc.tag == "signals":
                    for signal in signal_list:
                        pointer = str(signal.get('source'))
                        signal_type = str(signal.get('type'))
                        subtype = str(signal.get('subtype'))
                        tokens = str(signal.get('tokens'))
                        status = str(signal.get('status', ''))
                        if tokens != "":
                            # sort as integers
                            sorted_tokens = sorted([int(t) for t in tokens.split(',')])
                            tokens = ','.join([str(t) for t in sorted_tokens])
                        if status == '':
                            signal = ET.Element('signal', attrib={'source': pointer, 'type': signal_type, 'subtype': subtype, 'tokens': tokens})
                        else:
                            signal = ET.Element('signal', attrib={'source': pointer, 'type': signal_type, 'subtype': subtype, 'tokens': tokens, 'status': status})
                        gc.append(signal)

    ET.indent(tree, '    ')
    xml_out = ET.tostring(tree.getroot(), encoding="utf8").decode("utf8").replace("<?xml version='1.0' encoding='utf8'?>\n",                                                                                  "")
    xml_out = clean_xml(xml_out)

    return xml_out


if __name__ == "__main__":

    # To get spreadsheets for manual annotation, run with --no_cache and inspect results with browse_rs4.py

    p = ArgumentParser()
    p.add_argument("-n", "--no_cache", action="store_true", help="Do not use gold signal cache spreadsheets")
    p.add_argument("-r", "--rerun", action="store_true", help="Rerun depedit")
    p.add_argument("-i", "--input", default="*.rs4", help="Glob expression for input rs4 files")

    opts = p.parse_args()

    # Test non-dm signals on specific documents
    if os.sep not in opts.input:  # Assume files are in default folder
        pattern = SCRIPT_DIR + ".." + os.sep + "src" + os.sep + "rst" + os.sep + opts.input
    else:
        pattern = opts.input
    files = glob(pattern)

    for f in files:
        docname = os.path.basename(f).replace(".rs4","")
        sys.stderr.write("o Predicting non-DM signals for " + docname + "\n")
        gold_rs4 = open(f).read()

        xml_out = update_signals(gold_rs4, docname, rerun_depedit=opts.rerun, no_cache=opts.no_cache)
        with open(SCRIPT_DIR + "dm-dependencies" + os.sep + "non_dm_preds" + os.sep + docname + ".rs4", 'w', encoding="utf8", newline="\n") as f:
            f.write(xml_out)
