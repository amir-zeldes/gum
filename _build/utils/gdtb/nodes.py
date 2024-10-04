from typing import Text, List, Dict, Set, Tuple
from collections import defaultdict
from rst2dep.classes import NODE
from utils import flat_tokens

demonstrative_lemmas = ["this", "that"]


class Doc:
    def __init__(self, docname: str, sents: List, parsed_conllu):
        self.docname = docname
        self.sents = sents
        self.parsed_conllu = parsed_conllu
        self.edus = {}
        self.rels = {}
        self.main2same_unit = defaultdict(set)
        self._extract_edu()
        self.id2tok = self._get_token_features()
        self.id2head = self._get_token_head()          # added by Janet on April 8, 2024
        self.id2deprel = self._get_token_deprel()      # added by Janet on April 17, 2024
        self.mappings = {}
        self.tokens = [tok for sent in self.sents for tok in sent.toks]

    def __repr__(self):
        return f'{self.docname} with {len(self.sents)} sentences'

    def _extract_edu(self) -> None:
        edu_id = 0
        sent_id = 0
        for sent in self.sents:
            sent_id += 1
            for edu in sent.sent_edus:
                edu_id += 1
                edu.edu_id = edu_id
                self.edus[edu_id] = edu

    def _get_token_features(self) -> Dict:
        id2toks = {}
        for sent in self.sents:
            for tok_id, tok in zip(sent.tok_ids, sent.toks):
                id2toks[tok_id] = tok
        return id2toks

    # added by Janet on April 8, 2024; revised from March 25, 2024
    def _get_token_head(self) -> Dict:
        id2tok_head = {}
        for sent in self.sents:
            for tok_id, head in zip(sent.doc_tokens, sent.doc_heads):
                id2tok_head[int(tok_id)] = int(head)
        return id2tok_head

    # added by Janet on April 17, 2024
    def _get_token_deprel(self) -> Dict:
        id2deprel = {}
        for sent in self.sents:
            for tok_id, deprel in zip(sent.doc_tokens, sent.deprels):
                id2deprel[int(tok_id)] = deprel
        return id2deprel

    def get_edu_features(self) -> Tuple:
        edu2head_func, edu2dep_parent, edu2text = {}, {}, {}
        for edu_id, edu in self.edus.items():
            edu2head_func[edu_id] = edu.head_func
            edu2dep_parent[edu_id] = edu.dep_parent
            edu2text[edu_id] = edu.text
        return edu2head_func, edu2dep_parent, edu2text


class EDU:
    def __init__(self):
        self.edu_id = int
        self.tok_ids = []
        self.par_id = int
        self.tokens = []
        self.parent = EDU
        self.text = str
        self.head_func = str
        self.head_pos = str
        self.edu_tense = str
        self.head_lemma = str
        self.head_toknum = int
        self.child_rels = str
        self.rel = str
        self.dep_parent = int
        self.same_unit2main = int
        self.same_unit = set()
        self.conllu_lines = []
        self.head_line_id = []

    def __repr__(self):
        return self.text


class Token:
    def __init__(self, fields: List):
        self.token_id = fields[0]
        self.text = fields[1]
        self.lemma = fields[2]
        self.upos = fields[3]
        self.pos = fields[4]
        self.head = fields[6]
        self.deprel = fields[7]     # added by Janet on April 17, 2024
        self.doc_token_id = str     # doc-wise token ID
        self.doc_head_id = str      # doc-wise syntactic head ID
        self.misc = fields[-1]
        self.cxns = []
        if "Cxn" in self.misc:
            for anno in self.misc.split("|"):
                if anno.startswith("Cxn"):
                    self.cxns = anno.split("=")[1].split(",")

    def __repr__(self):
        return self.text


class Sentence:
    def __init__(self, lines: str, tok_id: int, sent_id: int, par_id: int):
        self.sent_id = sent_id
        self.par_id = par_id
        self.toks = []
        self.pos = []
        self.upos = []
        self.xpos = []
        self.lemmas = []
        self.deprels = []       # added by Janet on April 17, 2024
        self.heads = []         # added by Janet on March 25, 2024
        self.doc_tokens = []    # added by Janet on April 8, 2024
        self.doc_heads = []     # added by Janet on April 8, 2024
        self.plain_conllu = lines
        self.sent_edus = []
        self.tok_ids = []
        self.num_toks = 0
        self.raw_conllu = str
        self.s_type = str
        edu = None
        start_tok_id = tok_id
        plain_text = []
        upos_tags = []
        xpos_tags = []
        deprels = []
        for line in lines.split('\n'):
            if '\t' not in line:
                continue
            fields = line.split("\t")
            if "." in fields[0] or "-" in fields[0]:
                continue
            if "Discourse=" in fields[-1]:
                if edu:
                    edu.text = ' '.join(flat_tokens(edu.tokens))
                    edu.upos = ' '.join(upos_tags)
                    edu.xpos = ' '.join(xpos_tags)
                    edu.deprels = ' '.join(deprels)
                    edu.tok_ids = [i for i in range(start_tok_id, start_tok_id+len(edu.tokens))]
                    start_tok_id += len(edu.tokens)
                    self.sent_edus.append(edu)
                    upos_tags = []
                    xpos_tags = []
                    deprels = []
                edu = EDU()
                edu.sent_id = sent_id
                # is_new_edu = True
            token = Token(fields)
            token.doc_token_id = str(tok_id + int(token.token_id))
            token.doc_head_id = str(tok_id + int(token.head)) if token.head != '0' else token.doc_token_id
            edu.tokens.append(token)
            edu.conllu_lines.append(fields)
            self.toks.append(token)
            self.pos.append(token.pos)
            self.upos.append(token.upos)
            self.xpos.append(token.pos)
            self.lemmas.append(token.lemma)
            # self.heads.append(token.head)                 # added by Janet on March 25, 2024
            self.deprels.append(token.deprel)               # added by Janet on April 17, 2024
            self.doc_tokens.append(token.doc_token_id)      # added by Janet on April 8, 2024
            self.doc_heads.append(token.doc_head_id)        # added by Janet on April 8, 2024
            plain_text.append(token.text)
            upos_tags.append(token.upos)
            xpos_tags.append(token.pos)
            deprels.append(token.deprel)
        edu.text = ' '.join(flat_tokens(edu.tokens))
        edu.upos = ' '.join(upos_tags)
        edu.xpos = ' '.join(xpos_tags)
        edu.deprels = ' '.join(deprels)
        edu.tok_ids = [i for i in range(start_tok_id, start_tok_id + len(edu.tokens))]
        self.sent_edus.append(edu)
        self.tok_ids = [i for i in range(tok_id, tok_id + len(plain_text))]
        self.num_toks = len(plain_text)
        self.plain_text = ' '.join(plain_text)
        for edu in self.sent_edus:
            for fields in edu.conllu_lines:
                if fields[6] != "_":
                    if fields[6] not in [t.token_id for t in edu.tokens]:
                        edu.head_toknum = int(fields[0])
                        if fields[7] != "punct":
                            break  # Stop searching if we have a non-punct local root
            child_funcs = set()  # Collect info on auxiliaries, copula children etc. to verify that this is a finite clause
            for fields in edu.conllu_lines:
                if fields[6] == str(edu.head_toknum):
                    child_funcs.add(fields[7])
            edu.child_funcs = child_funcs
        for tok in self.toks:
            tok.sent_id = sent_id

    def __repr__(self):
        return f'(Sentence {self.sent_id}, Paragraph {self.par_id}): {self.plain_text}'


class Relation:
    def __init__(self,
                 node: NODE,
                 source_head_edus: List,
                 source_edu_ids: List,
                 source_sent_edus: List,
                 target_head_edus: List,
                 target_edu_ids: List,
                 target_sent_edus: List,
                 edu2head_func: Dict,
                 docstate: Doc):
        self.relname = node.relname
        self.nid = node.id
        self.node = node
        self.relkind = "multinuc" if node.relname.endswith("_m") else "rst"
        self.signals = []
        self.relation_probs = {}
        # added by Tatsuya
        self.arg1_head = None
        self.arg2_head = None
        if "-" not in self.nid:
            if int(self.nid) in [docstate.edus[edu].edu_id for edu in docstate.edus]:
                self.arg1_head, self.arg2_head = sorted([int(node.id), docstate.edus[int(node.id)].dep_parent])

        # pdtb relation cadidates for conversion
        self.pdtb_rels = defaultdict(list)

        if "-" in self.nid:  # Secedge, could create coverage cycle
            source_edu_ids = [e for e in source_edu_ids if e not in target_head_edus]
            source_head_edus = [e for e in source_head_edus if e not in target_head_edus]
            temp = [e for e in target_head_edus if e not in source_head_edus]
            if len(temp) > 0:
                target_head_edus = temp
            if len(target_edu_ids) == 0:
                target_edu_ids = [int(i) for i in target_head_edus]

        # initialize source and target spans
        self.source = Span(docstate, source_edu_ids, source_head_edus, source_sent_edus, edu2head_func)
        self.target = Span(docstate, target_edu_ids, target_head_edus, target_sent_edus, edu2head_func)
        if all([t in self.source.tok_ids for t in self.target.tok_ids]):  # Source contains target due to secedge path
            self.source.tok_ids = [t for t in self.source.tok_ids if t not in self.target.tok_ids]
        source_eids = [e.edu_id for e in self.source.edus]
        target_eids = [e.edu_id for e in self.target.edus]
        if all([e in source_eids for e in target_eids]):  # Source contains target due to secedge path
            self.source.edus = [e for e in self.source.edus if e.edu_id not in target_eids]
            if self.source.head_edu[0].edu_id > self.target.head_edu[0].edu_id:  # Right to left
                if not any([e.rel.startswith("same") for e in self.source.edus]) and any([e.edu_id < self.target.head_edu[0].edu_id for e in self.source.edus]):
                    # Discontinuous source around target but not same unit, remove prefix
                    self.source.edus = [e for e in self.source.edus if e.edu_id > self.target.head_edu[0].edu_id]
            else:  # Left to right
                if not any([e.rel.startswith("same") for e in self.source.edus]) and any([e.edu_id > self.target.head_edu[0].edu_id for e in self.source.edus]):
                    # Discontinuous source around target but not same unit, remove suffix
                    self.source.edus = [e for e in self.source.edus if e.edu_id < self.target.head_edu[0].edu_id]

        # direction - if is_forward, source precedes target, otherwise, source follows target
        self.is_forward = True if self.source.edus[0].dep_parent > self.source.edus[0].edu_id else False

    def __repr__(self):
        direction = "->" if self.is_forward else "<-"
        return f"{self.relname}, node_id: {self.nid}, ({self.source.tok_ids[0]},{self.source.tok_ids[-1]}) {direction} ({self.target.tok_ids[0]},{self.target.tok_ids[-1]})"


class Span:
    def __init__(self, docstate: Doc, edu_ids: List, head_edus: List, sent_edus: List, edu2head_func: Dict):
        self.edus = []
        self.head_edu = []
        self.sent_edus = []
        self.head_edu_id = []
        self.tok_ids = []
        self.sent_ids = set()
        self.par_ids = set()
        self.func = str
        self.add(docstate, edu_ids, head_edus, sent_edus, edu2head_func)
        self.raw_text = ' '.join([edu.text for edu in self.head_edu])
        self.raw_text_sent = [' '.join([edu.text for edu in option]) for option in self.sent_edus]
        # added by Tatsuya
        self.head_edu_sent = [sent for sent in docstate.sents if self.head_edu[0] in sent.sent_edus][0]
        self.raw_text_entirespan = ' '.join(docstate.id2tok[tid].text for tid in self.tok_ids)

    def add(self, docstate: Doc, edu_ids: List, head_edus: List, sent_edus: List, edu2head_func: Dict):
        for edu_id in edu_ids:
            curr_edu = docstate.edus[edu_id]
            curr_sent_id = curr_edu.sent_id
            curr_par_id = docstate.sents[curr_sent_id].par_id
            curr_edu.par_id = curr_par_id
            self.edus.append(curr_edu)
            self.tok_ids += curr_edu.tok_ids
            self.sent_ids.add(curr_sent_id)
            self.par_ids.add(docstate.sents[curr_edu.sent_id].par_id)
            if str(edu_id) in head_edus:
                self.head_edu_id.append(edu_id)
                self.head_edu.append(curr_edu)
        for option in sent_edus:
            added = []
            for edu_id in option:
                added.append(docstate.edus[edu_id])
            self.sent_edus.append(added)
        self.func = edu2head_func[sorted([int(x) for x in list(head_edus) if int(x) in docstate.edus])[0]]
        self.sent_ids = sorted(list(self.sent_ids))
        self.par_ids = list(self.par_ids)

    def __repr__(self):
        return self.raw_text


class Signal:
    def __init__(self, sigtype, subtype, tok_ids, toks):
        self.sigtype = sigtype
        self.subtype = subtype
        self.tok_ids = tok_ids
        self.toks = toks
