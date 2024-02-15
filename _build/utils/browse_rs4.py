from nltk.tokenize.treebank import TreebankWordDetokenizer
from pickle import dump, load
from classes import RULE, RELATION, TSVReader
from glob import glob
import sys, os, re
from copy import deepcopy
from rst2dep import read_rst
from collections import defaultdict
from argparse import ArgumentParser
from random import shuffle, seed

seed(42)


def rel_match(relname,target_rel):
    if re.escape(target_rel)==target_rel:
        return relname==target_rel
    else:
        return re.search(target_rel,relname) is not None

def anno_match(rel, conllu_annos, annos, direction="source"):
    # Test if all anno pairs are present in the relation's source head EDU tokens (incl. same-unit)
    tokens, lemmas, upos_tags, xpos_tags, deprels = conllu_annos
    anno_pairs = annos.split(";")

    # Get head EDU token indices
    if direction == "source":
        head_edu_tokens = rel.source["head"]
    else:
        head_edu_tokens = rel.target["head"]

    for anno_pair in anno_pairs:
        k, v = anno_pair.split("=")
        if k == "deprel":
            if not any(deprels[t] == v for t in head_edu_tokens):
                return False
        elif k == "xpos":
            if not any(xpos_tags[t] == v for t in head_edu_tokens):
                return False
        elif k == "upos":
            if not any(upos_tags[t] == v for t in head_edu_tokens):
                return False
        elif k == "lemma":
            if not any(lemmas[t] == v for t in head_edu_tokens):
                return False
    return True


def pretty(example):
    output = []
    cursor = 0
    for i,c in enumerate(example):
        cursor +=1
        output.append(c)
        if cursor > 160 and c == " ":
            cursor = 0
            output.append("\n\t")
    return "".join(output)

def print_sig(rel, tokens, counter, hits, distances, sigtokens, output_style="cli", full_sent=False, context_limit=30, sig_words=None, detokenizer=None):
    sig_tok_text = []
    rel_tokens = list(range(rel.source["all"][0], rel.source["all"][1] + 1)) + list(range(rel.target["all"][0], rel.target["all"][1] + 1))
    if full_sent:
        rel_tokens += list(range(rel.source["head_fullsent"][0], rel.source["head_fullsent"][1] + 1))
    rel_tokens = sorted(list(set(rel_tokens)))
    if rel.source["all"][0] < rel.target["all"][0]:  # LTR relation
        if rel.source["all"][-1] > rel.target["all"][-1]:
            boundary = rel.target["all"]  # target is medial
        else:
            boundary = [rel.source["all"][-1]]
    else:
        if rel.source["all"][-1] < rel.target["all"][-1]:
            boundary = rel.source["all"]  # source is medial
        else:
            boundary = [rel.target["all"][-1]]
    text = []
    prev = rel_tokens[0] - 1
    for t in rel_tokens:
        if t != prev + 1:
            text.append("[..]")
        word = tokens[t]
        if t in sigtokens:
            if sig_words is not None:
                sig_words[word] += 1
            sig_tok_text.append(word)
            if output_style == "html":
                word = '%%' + word + '&&'
            else:
                word = '\033[96m' + ":" + word + '\033[0m'
        text.append(word)
        prev = t
        if len(sigtokens) > 0:
            if t > sigtokens[-1] + context_limit:
                text.append("[..]")
                break
        if t in boundary:
            if output_style == "html":
                text.append("||")
            else:
                text.append("\033[93m||\033[0m")
    if detokenizer is not None:
        text = detokenizer.detokenize(text)
    if output_style == "html":
        text = text.replace('"', "``").replace("'", "`")
        head_edu_text = rel.head_edu_text.replace('"', "&quot;").replace("'", "&apos;")
        sig_tok_text = " ".join(sig_tok_text).replace('"', "&quot;").replace("'", "&apos;")
        text = re.sub(r' ?% ?% ? ?([^&]+)& ?& ?', r' <font color="blue">\1</font> ', text)
        text = re.sub(r' ?\| ?\| ?', ' <font color="red">||</font> ', text)
        if not text.endswith("[..]"):
            text += " [..]"
        fields = [str(counter), "", text, rel.relname, str(sigtokens), rel.docname, rel.head_edu, rel.dep_parent, rel.nid,
                  str(rel.source["all"]), str(rel.target["all"]), head_edu_text, sig_tok_text,
                  str(sig_tok_text.count(" ") + 1)]
        fields.append(str(rel.node.parent))  # EXTRA FOR SECEDGE
        if full_sent:
            fields.append(str(rel.source["head_fullsent"]))
        line = "</td><td>".join(fields)
        line = line.replace("&", "&amp;")
        output = "<tr><td>" + line + "</td></tr>"
    else:
        hit = f"(\033[93m{rel.relname}\033[0m) {text}"
        prev = sigtokens[0] - 1 if len(sigtokens)>0 else 0
        for t in sigtokens:
            if t != prev + 1:
                dist = t - prev
                if dist < 30:
                    distances.append(dist)
        output = pretty(hit)
    hits += 1

    return output, hits, distances


def read_doc(rs4, rsd, conllu, docname, allow_sourceless_signals=False):

    tok2rsd = {}
    edu2rel = {}
    edu2dep_parent = {}
    edu2head_func = defaultdict(lambda :"none")
    edu2tense = defaultdict(str)
    edu2head_lemma = {}
    edu2head_toknum = {}
    edu2child_rels = defaultdict(str)
    edu2lemmas = defaultdict(dict)
    edu2stems = defaultdict(dict)
    edu2tokids = defaultdict(list)
    edu2sent = defaultdict(int)
    edu2text = defaultdict(str)
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
            edu2text[int(fields[0])] = fields[1]
            if 'head_func' in line:
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
                    edu2tense[int(fields[6])] = edu2tense
            for t in tokens:
                tok2rsd[toknum] = rsd_rel
                edu2tokids[int(fields[0])].append(toknum)
                toknum += 1

    tmp = []
    lines = conllu.split("\n")
    sid = 1
    offsets = {1:0}
    toknum = 0
    pos_tags = []
    upos_tags = []
    deprels = []
    forms = []
    lemmas = []
    text = []
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
            upos_tags.append(fields[3])
            pos_tags.append(fields[4])
            deprels.append(fields[7])
            toknum += 1
        elif line.startswith("# text = "):
            text.append(line.split("=",1).strip())
        tmp.append(l)

    conllu_annos = (forms, lemmas, upos_tags, pos_tags, deprels)

    # Read rs3
    rs4_no_signals = re.sub(r'<signals>.*?</signals>\n?', '', rs4, flags=re.DOTALL)  # Remove signals
    nodes = read_rst(rs4_no_signals, {}, as_text=True)
    secedges = []
    try:
        edus = list(nodes[nid] for nid in nodes if nodes[nid].kind == "edu")
    except:
        print(docname)
        raise IOError("Can't get nodes EDU list")
    edus.sort(key=lambda x: int(x.id))
    keys = [nid for nid in nodes]
    for nid in keys:
        if "-" in nid:
            secedges.append(nodes[nid])
            del nodes[nid]
    # Get head EDU and height per node
    node2head_edu = {}
    node2multinuc_children = defaultdict(set)
    node2descendent_edus = defaultdict(set)
    node2tokspan = defaultdict(lambda: [0, 0])
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
            node2descendent_edus[node.id] = set([str(x) for x in list(range(node.left, node.right + 1))])

    relations = {}
    leftmost_sisters = set()  # Left most children of multinucs

    # Add secedges to nodes
    for secedge in secedges:
        nodes[secedge.id] = deepcopy(nodes[secedge.source])
        nodes[secedge.id].relname = secedge.relname + "_r"
        nodes[secedge.id].parent = secedge.target

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
                    pass
                    continue
                else:  # multinuc
                    sisters = node2multinuc_children[node.parent]
                    leftmost_sisters.add(sorted(list(sisters), key=lambda x: nodes[x].left)[0])
                    sis_head_edus = [node2head_edu[s] for s in sisters if s not in descendents]
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

            rel = RELATION(node, edus=descendents, head_edus=head_edus, target_head_edus=target_head_edus,
                           target_descendents=target_descendents, edu2sent=edu2sent, sent2toks=sent2tokids,
                           node2descendents=node2descendent_edus, edu2toks=edu2tokids, edu2head_func=edu2head_func,
                           edu2tense=edu2tense,docname=docname,tokens=forms,node2head_edu=node2head_edu,
                           edu2dep_parent=edu2dep_parent)
            rel.signals = []
            relations[node.id] = rel

    signals = re.findall(r'<signal source="([0-9-]+)" type="([^"]+)" subtype="([^"]+)" tokens="([^"]+)"',rs4)
    for sig in signals:
        source, sigtype, sigsubtype, tokens = sig
        tids = []
        for tid in tokens.split(","):
            tids.append(int(tid)-1)
        try:
            relations[source].signals.append((docname,sigtype,sigsubtype,tids))
        except:
            if allow_sourceless_signals:
                # Trapped signal in EDU with span parent
                while nodes[source].relname == "span" and source != "0":
                    source = nodes[source].parent
                    if nodes[source].relname != "span":
                        if source in relations:
                            relations[source].signals.append((docname, sigtype, sigsubtype, tids))
            else:
                print("! Found signal with missing source " + source + " in " + docname)
                return relations, conllu_annos, nodes

    return relations, conllu_annos, nodes

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-f","--fresh_cache",action="store_true",help="Rebuilds cache")
    p.add_argument("-i", "--ignore_signals", action="store_true", help="Ignore signals, just output matching relations for anno search")
    p.add_argument("-r","--relname",action="store",default="any",help="Relation name to match (regex OK)")
    p.add_argument("-n","--negative",action="store_true",help="Negative signal type match")
    p.add_argument("-s", "--subtype",action="store",default=".*",help="Signal subtype to find, e.g. indicative_word (regex OK)")
    p.add_argument("-o", "--output",action="store",default="cli",choices=["cli","html"],help="Output type (use html to paste to spreadsheet with colors)")
    p.add_argument("-e", "--edgetypes",action="store",default="both",choices=["both","prim","sec"],help="Included edge types")
    p.add_argument("-d", "--docname",action="store",default="any",help="Document name filter (regex OK)")
    p.add_argument("-a", "--annos",action="store",default=None,help="semicolon separated key-value pairs like: source:xpos=NN;deprel=nsubj")
    p.add_argument("--full_sent",action="store_true",help="Add the full sentence of the relation source in the context")
    opts = p.parse_args()

    #e.g. python browse_rs4.py -d GUM_court_equality -s indicative_word -n -f -r attribution.* -o cli --full_sent
    docs = []

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
    XML_ROOT = SCRIPT_DIR + os.sep.join(["..", "src"]) + os.sep
    conllu_dir = SCRIPT_DIR + os.sep.join(["..", "target", "dep", "not-to-release"]) + os.sep
    rs4_dir = SCRIPT_DIR + os.sep.join(["..", "target", "rst", "rstweb"]) + os.sep
    rsd_dir = rs4_dir.replace("rstweb", "dependencies")
    # Or for local testing:
    #rs4_dir = SCRIPT_DIR + "dm-dependencies" + os.sep + "non_dm_preds" + os.sep

    files = glob(rs4_dir + "*.rs4")
    shuffle(files)
    if opts.docname != "any":
        files = [f for f in files if opts.docname in f]

    if os.path.exists("browse_rs4.pkl") and not opts.fresh_cache:
        sys.stdout.write("! Loading data from pickle file\n")
        docs = load(open("browse_rs4.pkl",'rb'))
    else:
        for f, file_ in enumerate(files):
            docname = os.path.basename(file_).split(".")[0]
            rsd = open(rsd_dir + docname +".rsd").read()
            conllu = open(conllu_dir + docname +".conllu").read()
            rs4 = open(file_).read()
            rels, conllu_annos, _ = read_doc(rs4, rsd, conllu, docname)
            docs.append((rels,conllu_annos))
        dump(docs,open("browse_rs4.pkl",'wb'))

    detok = TreebankWordDetokenizer()
    target_rel = opts.relname #  DEFAULT: any
    target_subtype = opts.subtype # DEFAULT: indicative_word
    target_doc = opts.docname # DEFAULT: any
    negative = opts.negative
    output_style = opts.output
    edge_types = opts.edgetypes

    hits = 0
    distances = []
    sig_words = defaultdict(int)

    counter = 0
    context_limit = 30
    for doc, conllu_annos in docs:
        tokens, lemmas, upos_tags, pos_tags, deprels = conllu_annos
        for rel_id in doc:
            rel = doc[rel_id]
            if edge_types == "prim" and "-" in rel.nid or edge_types == "sec" and "-" not in rel.nid:
                continue
            if not rel_match(rel.relname,target_rel) and not target_rel == "any":
                continue
            if opts.annos is not None:
                direction = "source"
                if opts.annos.startswith("source:"):
                    direction = "source"
                elif opts.annos.startswith("target:"):
                    direction = "target"
                if not anno_match(rel, conllu_annos, opts.annos.replace("target:","").replace("source:",""), direction=direction):
                    continue
            if negative or opts.ignore_signals:
                # Print relations that DO NOT have the signal in question
                if all(not rel_match(sig[2],target_subtype) for sig in rel.signals) or opts.ignore_signals:
                    counter += 1
                    sig_text, hits, distances = print_sig(rel, tokens, counter, hits, distances,[], output_style=output_style, full_sent=opts.full_sent, context_limit=context_limit, sig_words=sig_words, detokenizer=detok)
                    print(sig_text)
            else:
                for sig in rel.signals:
                    counter += 1
                    docname,sigtype,sigsubtype,sigtokens = sig
                    if rel_match(sigsubtype, target_subtype):
                        sig_text, hits, distances = print_sig(rel,tokens, counter, hits, distances,sigtokens, output_style=output_style, full_sent=opts.full_sent, context_limit=context_limit, sig_words=sig_words, detokenizer=detok)
                        print(sig_text)

    print(f"TOTAL: {hits} hits")


    do_top_words = False
    if do_top_words:
        for i,w in enumerate(sorted(sig_words,key=lambda x:-sig_words[x])):
            print(w,sig_words[w])
            if i > 50:
                break

    do_plot = False
    if do_plot:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.hist(np.array(distances))
        plt.show()

