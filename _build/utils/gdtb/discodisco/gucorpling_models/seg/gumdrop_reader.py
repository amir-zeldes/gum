"""
Contains reader code from https://github.com/gucorpling/GumDrop2/blob/master/lib/conll_reader.py#L301
"""
import io
import re
from collections import defaultdict


class DepFeatures:
    def __init__(self, model="eng.rst.gum", genre_pat="^(..)"):
        self.name = "DepFeatures"
        self.model = model
        self.ext = "bmes"  # use bio for bio encoding
        self.genre_pat = genre_pat

    # @profile
    def extract_depfeatures(self, train_feats):
        clausal_deprels = {
            "csubj",
            "ccomp",
            "xcomp",
            "advcl",
            "acl",
            "acl:relcl",
            "list",
            "parataxis",
            "appos",
            "conj",
            "nmod:prep",
        }

        # [s_id, wid, line_id ,head, deprel, word]
        feats_tups = [
            tuple([
                x["s_id"],
                 x["wid"],
                 x["line_id"],
                 x["head"],
                 x["deprel"],
                 x["word"]
            ])
            for x in train_feats
        ]
        num_s = sorted(list(set([x[0] for x in feats_tups])))
        feats_parents = defaultdict(list)

        all_feats_s = defaultdict(list)

        sorted_feats_s = sorted([x for x in feats_tups], key=lambda x: x[1])

        # TODO: why is this necessary? not all tokens have this key set
        for x in train_feats:
            x["parentclauses"] = ""

        # Do only one pass through data to break up into sentences
        for x in sorted_feats_s:
            s_num = x[0]
            all_feats_s[s_num].append(x)

        # looping through sentences
        for s in num_s:
            # feats_s = sorted([x for x in  feats_tups if x[0]==s], key=lambda x: x[1])
            feats_s = all_feats_s[s]
            feats_parents.clear()

            # looping through tokens in a sentence
            wid2lineid = {}
            for t in feats_s:
                # finding all (grand)parents (grand-heads)
                wid = t[1]
                head = t[3]
                wid2lineid[wid] = t[2]
                while int(head) != 0 and all(len(x) >= 4 for x in feats_s) and len([x for x in feats_s if str(x[1]) == str(head)]) > 0:
                    head_t = [x for x in feats_s if str(x[1]) == str(head)][0]
                    feats_parents[t].append(head_t)
                    head = head_t[3]

            for t in feats_s:
                wid = t[1]
                parentrels = [x[4] for x in feats_parents[t]]
                parentcls = []
                for r in parentrels:
                    for dr in clausal_deprels:
                        if r.startswith(dr):
                            parentcls.append(r)
                            continue
                train_feats[wid2lineid[wid]]["parentclauses"] = "|".join(parentcls)

            # loop through clausal_deprels (non-conj & conj) and create BIO list for sentence tokens
            dr_d = defaultdict(list)

            # finding all tokens in a sentence who or whose parents has a deprel (dr) -- non-conj
            for id_t, t in enumerate(feats_s):
                t_gen = [t] + feats_parents[t]

                # all including conj
                for dr in clausal_deprels:
                    in_t_gen = [x for x in t_gen if x[4].startswith(dr)]
                    if len(in_t_gen) > 0:
                        dr_d[(in_t_gen[0][1], in_t_gen[0][4])].append(t)

            #  sort dictionary values
            dr_dl = defaultdict(list)
            for k, v in dr_d.items():
                if v:
                    dr_dl[k + (len(v),)] = sorted(list(set([x[1] for x in v])))
                    # sorted(v, key=lambda x: x[1])

            # collect all BIEO features, for conj and non-conj separately
            feats_l = [[] for x in range(len(feats_s))]
            feats_conjl = [[] for x in range(len(feats_s))]
            for i in range(len(feats_s)):
                for k, v in dr_dl.items():
                    # for non-conj
                    if not k[1].startswith("conj"):
                        if not i + 1 in v:
                            feats_l[i].append("_")
                        elif v[0] == i + 1:
                            feats_l[i].append(("B" + k[1], v[0], v[-1]))
                        elif v[-1] == i + 1:
                            feats_l[i].append(("E" + k[1], v[0], v[-1]))
                        else:
                            feats_l[i].append(("I" + k[1], v[0], v[-1]))

                    # for conj
                    else:
                        if not i + 1 in v:
                            feats_conjl[i].append("_")
                        elif v[0] == i + 1:
                            feats_conjl[i].append(("B" + k[1], v[0], v[-1]))
                        elif v[-1] == i + 1:
                            feats_conjl[i].append(("E" + k[1], v[0], v[-1]))
                        else:
                            feats_conjl[i].append(("I" + k[1], v[0], v[-1]))

            # Prioritize Bsmall > Blarge > Elarge > Esmall > Ismall > Ilarge > _
            # non-conj
            for id_l, l in enumerate(feats_l):
                Bsub = sorted([x for x in l if x[0].startswith("B")], key=lambda x: x[2] - x[1])
                Esub = sorted([x for x in l if x[0].startswith("E")], key=lambda x: x[2] - x[1], reverse=True,)
                Isub = sorted([x for x in l if x[0].startswith("I")], key=lambda x: x[2] - x[1])
                if Bsub != []:
                    feats_l[id_l] = Bsub[0][0]
                elif Esub != []:
                    feats_l[id_l] = Esub[0][0]
                elif Isub != []:
                    feats_l[id_l] = Isub[0][0]
                else:
                    feats_l[id_l] = "_"

            # remove sub-deprel after :, e.g. csubj:pass -> csubj (however, acl:relcl stays as acl:relcl)
            feats_l = [re.sub(r":[^r].*$", "", x) if x != "nmod:prep" else x for x in feats_l]

            # add non-conj to train_feats
            for id_l, l in enumerate(feats_l):
                wid = feats_s[id_l][1]
                train_feats[wid2lineid[wid]]["depchunk"] = l

            # conj
            for id_l, l in enumerate(feats_conjl):
                Bsub = sorted([x for x in l if x[0].startswith("B")], key=lambda x: x[2] - x[1])
                Esub = sorted([x for x in l if x[0].startswith("E")], key=lambda x: x[2] - x[1], reverse=True,)
                Isub = sorted([x for x in l if x[0].startswith("I")], key=lambda x: x[2] - x[1])
                if Bsub != []:
                    feats_conjl[id_l] = Bsub[0][0]
                elif Esub != []:
                    feats_conjl[id_l] = Esub[0][0]
                elif Isub != []:
                    feats_conjl[id_l] = Isub[0][0]
                else:
                    feats_conjl[id_l] = "_"

            # add conj to train_feats
            for id_l, l in enumerate(feats_conjl):
                wid = feats_s[id_l][1]
                train_feats[wid2lineid[wid]]["conj"] = l

            # sys.stderr.write('\r Adding deprel BIEO features to train_feats %s ### o Sentence %d' %(corpus, s))

        return train_feats


def get_case(word):
    if word.isdigit():
        return "d"
    elif word.isupper():
        return "u"
    elif word.islower():
        return "l"
    elif word.istitle():
        return "t"
    else:
        return "o"


def get_stype(tokens):
    q = "NoQ"
    root_child_funcs = []
    # Get root
    for t in tokens:
        if t["deprel"] == "root":
            pass
            # root = t["wid"]
            # root_pos = t["cpos"] if t["cpos"] != "_" else t["pos"]
        if t["word"] in ["?", "？"]:
            q = "Q"
    for t in tokens:
        try:
            if t["head"] == "root":
                root_child_funcs.append(t["deprel"])
        except KeyError:
            raise IOError(
                "! Found input sentence without a root label: " + " ".join([t["word"] for t in tokens]) + "\n"
            )
    if any(["subj" in f for f in root_child_funcs]):
        subj = "Subj"
    else:
        subj = "NoSubj"
    if any(["acl" in f or "rcmod" in f for f in root_child_funcs]):
        acl = "Acl"
    else:
        acl = "NoAcl"
    if any(["advcl" in f for f in root_child_funcs]):
        advcl = "Advcl"
    else:
        advcl = "NoAdvcl"
    if "conj" in root_child_funcs:
        conj = "Conj"
    else:
        conj = "NoConj"
    if "cop" in root_child_funcs:
        cop = "Cop"
    else:
        cop = "NoCop"

    s_type = "_".join([q, subj, conj, cop, advcl, acl])

    return s_type


def get_genre(corpus, doc):
    if corpus in ["eng.rst.rstdt", "eng.pdtb.pdtb", "deu.rst.pcc", "por.rst.cstn", "zho.pdtb.cdtb"]:
        return "news"
    if corpus == "rus.rst.rrt":
        if "news" in doc:
            return "news"
        else:
            return doc.split("_")[0]
    if corpus == "eng.rst.gum":
        return doc.split("_")[1]
    if corpus == "eng.rst.stac":
        return "chat"
    if corpus == "eus.rst.ert":
        if doc.startswith("SENT"):
            return doc[4:7]
        else:
            return doc[:3]
    if corpus == "fra.sdrt.annodis":
        if doc.startswith("wik"):
            return "wiki"
        else:
            return doc.split("_")[0]
    if corpus in ["nld.rst.nldt", "spa.rst.rststb"]:
        return doc[:2]
    if corpus in ["spa.rst.sctb", "spa.rst.sctb"]:
        if doc.startswith("TERM"):
            return "TERM"
        else:
            return doc.split("_")[0]

    print("Warning: unknown genres for corpus", corpus)
    return "_"


CORPORA = [
    "deu.rst.pcc",
    "eng.pdtb.pdtb",
    "eng.rst.gum",
    "eng.rst.rstdt",
    "eng.sdrt.stac",
    "eus.rst.ert",
    "fra.sdrt.annodis",
    "fas.rst.prstc",
    "nld.rst.nldt",
    "por.rst.cstn",
    "rus.rst.rrt",
    "spa.rst.rststb",
    "spa.rst.sctb",
    "tur.pdtb.tdb",
    "zho.pdtb.cdtb",
    "zho.rst.sctb",
]


def read_conll_conn(
    input_file_path, mode="seg", as_text=False, cap=None, char_bytes=False,
):
    """
    Read a DISRPT shared task format .conll file
    :param input_file_path: file path for input
    :param mode: 'seg' to read discourse unit segmentation as labels or
                 'sent' to read sentence breaks as labels
    :param genre_pat: Regex pattern with capturing group to extract genre from document names
    :param as_text: Boolean, whether the input is a string, rather than a file name to read
    :param cap: Maximum tokes to read, after which reading stops at next newdoc boundary
    :param char_bytes: if True, first and last letters are replaced with first and last byte of string (for Chinese)
    :return: list of tokens, each a dictionary of features and values including a gold label, and
             vocabulary frequency list
    """
    corpus = None
    for c in CORPORA:
        if c in input_file_path:
            corpus = c
            break
    if corpus is None:
        raise Exception("Unrecognized corpus! file path: " + input_file_path)

    if as_text:
        lines = input_file_path.split("\n")
    else:
        try:
            lines = io.open(input_file_path, encoding="utf8").readlines()
        except:
            lines = io.open(input_file_path + "u", encoding="utf8").readlines()
    docname = input_file_path if len(input_file_path) < 100 else "doc1"
    output = []  # List to hold dicts of each observation's features
    cache = []  # List to hold current sentence tokens before adding complete sentence features for output
    toks = []  # Plain list of token forms
    firsts = set([])  # Attested first characters of words
    lasts = set([])  # Attested last characters of words
    vocab = defaultdict(int)  # Attested token vocabulary counts
    sent_start = True
    tok_id = 0  # Track token ID within document
    line_id = 0
    sent_id = 1
    genre = "_"
    open_quotes = {'"', "«", "``", "”"}
    close_quotes = {'"', "»", "“", "''"}
    open_brackets = {"(", "[", "{", "<"}
    close_brackets = {")", "]", "}", ">"}
    used_feats = ["VerbForm", "PronType", "Person", "Mood"]
    in_quotes = 0
    in_brackets = 0
    last_quote = 0
    last_bracket = 0
    total = 0
    total_sents = 0
    doc_sents = 1
    heading_first = "_"
    heading_last = "_"
    for r, line in enumerate(lines):
        if "\t" in line:
            fields = line.split("\t")
            while len(fields) < 10:
                fields.append("_")
            if "-" in fields[0]:  # conllu super-token
                continue
            total += 1
            word, lemma, pos, cpos, feats, head, deprel = fields[1:-2]
            if "." in fields[0]:
                token_id = float(fields[0])
                pieces = fields[8].split(":")
                head, deprel = pieces[0], ":".join(pieces[1:])
                head = float(head) if "." in head else int(head)
            else:
                token_id = int(fields[0])
            if mode == "seg":
                if "BeginSeg=Yes" in fields[-1]:
                    label = "BeginSeg"
                elif "Seg=B-Conn" in fields[-1]:
                    label = "Seg=B-Conn"
                elif "Seg=I-Conn" in fields[-1]:
                    label = "Seg=I-Conn"
                # elif "Seg=S-Conn" in fields[-1]:
                # 	label = "Seg=S-Conn"
                else:
                    label = "_"
            elif mode == "sent":
                if sent_start:
                    label = "Sent"
                else:
                    label = "_"
            else:
                raise ValueError("read_conll_conn mode must be one of: seg|sent\n")
            # Compose a categorical feature from morphological features of interest
            feats = [f for f in feats.split("|") if "=" in f]
            feat_string = ""
            for feat in feats:
                name, val = feat.split("=")
                if name in used_feats:
                    feat_string += val
            if feat_string == "":
                feat_string = "_"
            vocab[word] += 1
            case = get_case(word)
            head_dist = int(token_id) - int(head)
            if len(word.strip()) == 0:
                raise ValueError("! Zero length word at line " + str(r) + "\n")
            toks.append(word)
            first_char = word[0]
            last_char = word[-1]
            if char_bytes:
                try:
                    first_char = str(first_char.encode("utf8")[0])
                    last_char = str(last_char.encode("utf8")[-1])
                except Exception:
                    pass
            firsts.add(first_char)
            lasts.add(last_char)
            cache.append(
                {
                    "word": word,
                    "lemma": lemma,
                    "pos": pos,
                    "cpos": cpos,
                    "head": head,
                    "head_dist": head_dist,
                    "deprel": deprel,
                    "docname": docname,
                    "case": case,
                    "tok_len": len(word),
                    "label": label,
                    "first": first_char,
                    "last": last_char,
                    "tok_id": tok_id,
                    "genre": genre,
                    "wid": token_id,
                    "quote": in_quotes,
                    "bracket": in_brackets,
                    "morph": feat_string,
                    "heading_first": heading_first,
                    "heading_last": heading_last,
                    "depchunk": "_",
                    "conj": "_",
                    "line_id": line_id,
                }
            )
            if mode == "seg":
                cache[-1]["s_num"] = doc_sents

            tok_id += 1
            line_id += 1
            sent_start = False
            if word in open_quotes:
                in_quotes = 1
                last_quote = tok_id
            elif word in close_quotes:
                in_quotes = 0
            if word in open_brackets:
                in_brackets = 1
                last_bracket = tok_id
            elif word in close_brackets:
                in_brackets = 0
            if tok_id - last_quote > 100:
                in_quotes = 0
            if tok_id - last_bracket > 100:
                in_brackets = 0

        elif "# newdoc id = " in line:
            if cap is not None:
                if total > cap:
                    break
            docname = re.search(r"# newdoc id = (.+)", line).group(1)
            genre = get_genre(corpus, docname)
            doc_sents = 1
            tok_id = 1
        elif len(line.strip()) == 0:
            sent_start = True
            if len(cache) > 0:
                if mode == "seg":  # Don't add s_len in sentencer learning mode
                    sent = " ".join([t["word"] for t in cache])
                    if (
                        sent[0] == sent[0].upper()
                        and len(cache) < 6
                        and sent[-1] not in [".", "?", "!", ";", "！", "？", "。"]
                    ):
                        # Uppercase short sentence not ending in punct - possible heading affecting subsequent data
                        heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
                        heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
                    # Get s_type features
                    s_type = get_stype(cache)
                    for tok in cache:
                        tok["s_len"] = len(cache)
                        tok["s_id"] = sent_id
                        tok["heading_first"] = heading_first
                        tok["heading_last"] = heading_last
                        tok["s_type"] = s_type
                    sent_id += 1
                    doc_sents += 1
                    total_sents += 1
                output += cache
                if mode == "seg":
                    if len(output) > 0:
                        for t in output[-int(fields[0]) :]:
                            # Add sentence percentile of document length in sentences
                            t["sent_doc_percentile"] = t["s_num"] / doc_sents
                cache = []

    # Flush last sentence if no final newline
    if len(cache) > 0:
        if mode == "seg":  # Don't add s_len in sentencer learning mode
            sent = " ".join([t["word"] for t in cache])
            if sent[0] == sent[0].upper() and len(cache) < 6 and sent[-1] not in [".", "?", "!", ";", "！", "？", "。"]:
                # Uppercase short sentence not ending in punctuation - possible heading
                heading_first = str(sent.encode("utf8")[0]) if char_bytes else sent[0]
                heading_last = str(sent.encode("utf8")[-1]) if char_bytes else sent[-1]
            # Get s_type features
            s_type = get_stype(cache)
            for tok in cache:
                tok["s_len"] = len(cache)
                tok["s_id"] = sent_id
                tok["heading_first"] = heading_first
                tok["heading_last"] = heading_last
                tok["s_type"] = s_type

        output += cache
        if mode == "seg":
            for t in output[-int(fields[0]) :]:
                # Add sentence percentile of document length in sentences
                t["sent_doc_percentile"] = 1.0

    if mode == "seg":
        df = DepFeatures()
        output = df.extract_depfeatures(output)

    for t in output:
        if "sent_doc_percentile" not in t:
            t["sent_doc_percentile"] = 0.5

    return output, vocab, toks, firsts, lasts
