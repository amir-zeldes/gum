import re, sys, os
import pandas as pd
from collections import defaultdict

demonstrative_lemmas = ["this","that"]
non_marker_bigrams = {"big+brother","Pete+look","ever+seen","look+at","naked+woman","particular+time","few+places","purple+sphere","green+sphere","others+may"}

class DEFINITION:
    __slots__ = ['value','compiled_re','match_func','negative','match_type']

    def __init__(self, value, negative=False, force_in=False):
        self.value = value
        self.compiled_re = None
        self.match_func = None
        self.negative = negative
        self.set_match_type(force_in=force_in)

    def set_match_type(self, force_in=False):
        value = self.value
        if re.escape(value) == value:  # No regex operators within expression
            if force_in:
                self.match_func = self.return_str_not_in if self.negative else self.return_str_in
                self.match_type = "in"
            else:
                self.match_func = self.return_exact_negative if self.negative else self.return_exact
                self.match_type = "exact"
        else:
            if force_in:
                self.compiled_re = re.compile(".*(" + self.value + ").*")
            else:
                if not (self.value.startswith("(") and self.value.endswith(")")):
                    self.value = "(" + self.value + ")"
                if self.value.startswith("(?i)"):
                    self.value = self.value[4:]
                    self.compiled_re = re.compile("^" + self.value + "$", re.IGNORECASE)
                else:
                    self.compiled_re = re.compile("^"+self.value+"$")
            self.match_func = self.return_regex_negative if self.negative else self.return_regex
            self.match_type = "regex"

    def return_exact(self, test_val):
        return test_val == self.value

    def return_exact_negative(self, test_val):
        return test_val != self.value

    def return_regex(self, test_val):
        return self.compiled_re.search(test_val)

    def return_regex_negative(self, test_val):
        return self.compiled_re.search(test_val) is None

    def return_str_in(self, test_val):
        return self.value in test_val

    def return_str_not_in(self, test_val):
        return self.value not in test_val

    def return_in(self, test_val):
        return any([self.value == v for v in test_val])

    def return_not_in(self, test_val):
        return all([self.value != v for v in test_val])

    def return_regex_in(self, test_val):
        matchers = [self.compiled_re.search(v) for v in test_val if v is not None]
        successful = [m for m in matchers if m is not None]
        successful.sort(key=lambda x: x.endpos - x.pos, reverse=True)
        return successful[0] if len(successful) > 0 else None

    def return_regex_not_in(self, test_val):
        return all([self.compiled_re.search(v) is None for v in test_val])

    @staticmethod
    def return_true(test_val):
        return True


class RULE:

    headers = ["form","lemma","xpos","upos","feats","head","deprel","edeps","misc","s_type","genre"]

    def __init__(self, rule_id, fields, rel, childrel, edu_func, sigtype, sigsubtype, position, comments=""):
        self.id = rule_id
        self.criteria = {}
        self.rel = rel
        self.childrel = childrel
        self.edu_func = edu_func
        self.sigtype = sigtype
        self.sigsubtype = sigsubtype
        self.position = position
        self.comments = comments

        for i, field in enumerate(self.headers):
            if fields[i] != "_":
                negative = False
                if fields[i].startswith("!"):
                    negative = True
                    fields[i] = fields[i][1:]

                force_in = False
                if field == "misc":
                    force_in = True
                definition = DEFINITION(fields[i],negative=negative,force_in=force_in)
                self.criteria[i] = definition

    def __repr__(self):
        return f"Rule {self.id}: {self.comments} ({self.sigtype}/{self.sigsubtype})"

    def match(self, fields, toknum, connective_idx, heads_to_mentions, xsubj_map, paired_partner, seen_paired):
        # Skip lexical sigtypes if toknum already in a connective
        if self.sigtype == "lexical" and toknum in connective_idx:
            return None, seen_paired

        for idx in self.criteria:
            if not self.criteria[idx].match_func(fields[idx]):
                return None, seen_paired
        if self.sigtype == "lexical":
            if any(["LemmaBigram=" + x in fields[8] for x in non_marker_bigrams]):  # Stop list
                return None, seen_paired
        position = self.position

        if self.position == "entity":  # Handle expanding signal token to its entity span
            if toknum in heads_to_mentions:
                toknum = sorted(list(heads_to_mentions[toknum]), key=lambda x: len(x))[0]
            position = "initial"
        elif self.position == "xsubj":  # Handle shifting signal token to external subject from xsubj map
            if toknum in xsubj_map:
                xsubj = xsubj_map[toknum]
                if xsubj in heads_to_mentions:
                    toknum = sorted(list(heads_to_mentions[xsubj]), key=lambda x: len(x))[0]
                else:
                    toknum = xsubj
            else:
                return None, seen_paired
        elif self.position == "paired":  # Requires a paired partner not None
            if paired_partner is None:
                return None, seen_paired
            else:
                if toknum in seen_paired:
                    return None, seen_paired
                else:
                    seen_paired.add(toknum)
                    seen_paired.add(paired_partner)
                    toknum = (toknum,paired_partner)
                    position = "initial"
        elif self.position == "ngram":
            # Check how many words were matched
            tok_count = self.criteria[8].match_func(fields[8]).group(1).count("_")
            if tok_count > 2:
                toknum = (toknum, toknum + 1, toknum + 2)
            elif tok_count > 1:
                toknum = (toknum,toknum+1)
            position = "initial"


        return (toknum, self.rel, self.edu_func, self.childrel, self.sigtype, self.sigsubtype, position), seen_paired


class RELATION:

    def __init__(self,node,head_edus,edus,target_head_edus,target_descendents,edu2sent,sent2toks,node2descendents,
                 edu2toks,edu2head_func,edu2tense,docname,tokens,node2head_edu,edu2dep_parent):
        self.node = node
        self.relname = node.relname
        self.nid = node.id
        self.relkind = "multinuc" if node.relname.endswith("_m") else "rst"
        self.docname = docname
        self.source = defaultdict(set)
        self.target = defaultdict(set)
        int_edus = [int(e) for e in edus]
        self.head_edus = head_edus
        self.source_width = max(int_edus) - min(int_edus) + 1

        if "-" in self.nid:
            self.head_edu = node2head_edu[self.nid.split("-")[0]]
            edus = [e for e in edus if e not in target_head_edus]
            head_edus = [e for e in head_edus if e not in target_head_edus]
            temp = [e for e in target_head_edus if e not in head_edus]
            if len(temp) > 0:
                target_head_edus = temp
            if len(target_descendents) == 0:
                target_descendents = target_head_edus
        else:
            self.head_edu = node2head_edu[self.nid]  # Single head EDU based on dependency conversion
        head_edu_toks = edu2toks[int(self.head_edu)]
        self.head_edu_text = " ".join(tokens[t] for t in head_edu_toks)
        if "-" in self.nid:
            self.dep_parent = str(node2head_edu[self.nid.split("-")[1]])
        else:
            self.dep_parent = str(edu2dep_parent[int(self.head_edu)])
        target_edu_toks = edu2toks[int(self.dep_parent)]
        self.target_edu_text = " ".join(tokens[t] for t in target_edu_toks)

        # Assign source and target domains in tokens
        for edu in edus:
            edu_toks = set([t for t in edu2toks[int(edu)]])
            sent_toks = sent2toks[edu2sent[int(edu)]]
            if edu in head_edus:
                self.source["head"].update(edu_toks)
                self.source["tense"] = edu2tense[int(edu)]
            self.source["all"].update(edu_toks)
        source_sent_edus = [str(e) for e in edu2sent if edu2sent[e] == edu2sent[int(min([int(x) for x in list(head_edus)]))] and str(e) not in target_descendents]
        source_sent_edus += list(head_edus)
        source_sent_edus = list(set(source_sent_edus))
        for edu in source_sent_edus:
            edu_toks = set([t for t in edu2toks[int(edu)]])
            sent_toks = sent2toks[edu2sent[int(edu)]]
            self.source["head_sent"].update(sent_toks)

        self.target["tense"] = []
        for edu in target_descendents:
            if edu in edus:
                continue
            edu_toks = set([t for t in edu2toks[int(edu)]])
            sent_toks = sent2toks[edu2sent[int(edu)]]
            if edu in target_head_edus:
                self.target["head"].update(edu_toks)
                self.target["head_sent"].update(sent_toks)
                if all([int(edu)<int(x) for x in head_edus]):
                    self.target["tense"].append(edu2tense[int(edu)])
            self.target["all"].update(edu_toks)

        self.source["head"] = tuple(sorted(list(self.source["head"])))
        self.target["head"] = tuple(sorted(list(self.target["head"])))
        self.source["all"] = (min(self.source["all"]),max(self.source["all"]))
        self.target["all_with_gaps"] = self.target["all"]
        self.target["all"] = (min(self.target["all"]),max(self.target["all"]))
        if self.target["all"][0] < self.source["all"][0]:  # LTR relation, exclude tokens before end of target
            self.source["head_fullsent"] = (max(self.target["all"])+1, max(self.source["head_sent"]))
        else:  # RTL relation, exclude tokens after beginning of target
            self.source["head_fullsent"] = (min(self.source["head_sent"]),min(self.target["all"])-1)
        self.target["head_fullsent"] = (min(self.target["head_sent"]),max(self.target["head_sent"]))
        self.source["head_sent"] = (max(min(self.source["head_sent"]),self.source["all"][0]),min(max(self.source["head_sent"]),self.source["all"][-1]))
        self.target["head_sent"] = (max(min(self.target["head_sent"]),self.target["all"][0]),min(max(self.target["head_sent"]),self.target["all"][-1]))

        self.source["func"] = edu2head_func[sorted([int(x) for x in list(head_edus)])[0]]
        self.target["func"] = edu2head_func[sorted([int(x) for x in list(target_head_edus)])[0]]

        if self.target["head"][0] < self.source["head"][0]:
            self.early = self.target
            self.late = self.source
            self.location = "final"
        else:
            self.early = self.source
            self.late = self.target
            self.location = "initial"

    def match(self, tokspan, toknum, relname, func, force_sent=False, invert=False, domain="head_sent",
              discontinuous_single=False, same_sent=False, max_span=None, paired=False):

        if relname != "_":
            if re.search(relname,self.relname) is None:
                return False
        node = self.target if invert else self.source

        # Check for gaps in signal tokens
        contiguous = True
        if isinstance(tokspan,tuple):
            sorted_span = sorted(list(tokspan))
            if max_span is not None:
                if sorted_span[-1] - sorted_span[0] >  max_span:
                    return False
            contiguous = sorted_span == list(range(sorted_span[0],sorted_span[-1]+1))

        # Prevent matching into source and target being the same sentence
        if force_sent:
            domain = "head_fullsent"
        elif self.target[domain] == self.source[domain]:
                domain = "head"

        # Check if we only want within-sentence relations
        if same_sent:
            if self.target["head_sent"] != self.source["head_sent"]:# and domain!="head_fullsent":
                if not (domain == "head_fullsent" and self.target[domain] == self.source[domain]):
                    return False

        # Check paired signals like lexical chains span both source and target
        if paired:
            if not (any([self.source["all"][0] <= t <= self.source["all"][1] for t in tokspan]) and
                    any([t in self.target["all_with_gaps"] for t in tokspan])):
                return False

        if contiguous:
            if domain == "head":
                if toknum not in node[domain]:
                    return False
            else:
                if not (node[domain][0]<=toknum<=node[domain][1]):
                    return False
        else:
                if domain == "head":
                    if tokspan[0] not in self.early[domain] or tokspan[-1] not in self.late[domain]:
                        if discontinuous_single:
                            if not (all([t in node[domain] for t in tokspan])):
                                return False
                        else:
                            return False
                else:
                    # Check all divided between source and target node
                    if not ((self.early[domain][0] <= tokspan[0] <= self.early[domain][1]) and (self.late[domain][0] <= tokspan[-1] <= self.late[domain][1])):
                        if discontinuous_single:
                            # For a single but discontinuous signal, check all tokens are in relation node
                            if not (all([node[domain][0]<=t<=node[domain][1] for t in tokspan])):
                                return False
                        else:
                            return False
        if func not in [".*","_"]:
            if re.search(func,node["func"]) is None:
                return False
        return True

    def get_pair_signals(self, tok2spans, pos_tags, lemmas, mentions_to_heads=None, mentions_to_mtype=None, pair_type="coref", pattern=".*"):
        signals = []
        if re.search(pattern, self.relname) is None:
            return signals
        late_entities = defaultdict(set)
        early_entities = defaultdict(set)
        domain = "head_sent" if self.early["head_sent"] != self.late["head_sent"] else "head"
        for toknum in range(self.late[domain][0],self.late[domain][-1]+1):
            if toknum in tok2spans:
                for k in tok2spans[toknum]:
                    late_entities[k].update(tok2spans[toknum][k])
        for toknum in range(self.early[domain][0],self.early[domain][-1]+1):
            if toknum in tok2spans:
                for k in tok2spans[toknum]:
                    early_entities[k].update(tok2spans[toknum][k])
        if pair_type != "tense":
            coref_ids = set(late_entities.keys()).intersection(set(early_entities.keys()))
        else:
            coref_ids = []
            for early_tense in self.target["tense"]:
                if (early_tense,self.source["tense"]) in [("Past","Pres"),("Past","Fut"),("Pres","Fut")]:
                    tense_pair = early_tense + self.source["tense"]
                    for tense1 in set(early_entities.keys()):
                        for tense2 in set(late_entities.keys()):
                            if tense1 == early_tense and tense2 == self.source["tense"]:
                                before_and_in_head = set([tup for tup in early_entities[tense1] if tup[0]<self.source["all"][0] and tup[0] in self.target["head"]])
                                early_entities[tense_pair].update(before_and_in_head)
                                late_entities[tense_pair].update(late_entities[tense2])
                                coref_ids.append(tense_pair)
        for eid in coref_ids:
            for late_tup in sorted(late_entities[eid]):
                candidate = None
                for early_tup in sorted(early_entities[eid]):
                    last_token = early_tup[-1]
                    if last_token < self.late[domain][0]:  # Valid non-nested coref signal
                        tokens = tuple(list(early_tup) + list(late_tup))
                        if pair_type == "coref":
                            sigtype = "semantic"
                            subtype = "repetition"
                            early_heads = mentions_to_heads[early_tup]
                            late_heads = mentions_to_heads[late_tup]
                            if len(early_heads) == len(late_heads):
                                for i, early_idx in enumerate(early_heads):
                                    late_idx = late_heads[i]
                                    if lemmas[early_idx].lower() != lemmas[late_idx].lower():
                                        subtype = "synonymy"  # Different aligned head lemma, not same expression
                            else:
                                subtype = "synonymy"  # Different number of head tokens, not same expression
                            if lemmas[late_tup[0]] in demonstrative_lemmas:
                                sigtype = "reference"
                                subtype = "demonstrative_reference"
                            if len(late_tup) == 1 and pos_tags[late_tup[0]] in ["PRP", "PRP$"]:
                                sigtype = "reference"
                                subtype = "personal_reference"
                            if subtype in ["synonymy","demonstrative_reference"] and lemmas[late_tup[-1]] in ["thing","matter","issue","fact"]:
                                subtype = "general_word"
                            if mentions_to_mtype[early_tup] == "disc":
                                sigtype = "reference"
                                subtype = "propositional_reference"
                            candidate = (tokens, pattern, ".*", ".*", sigtype, subtype, self.location)
                            break
                        elif pair_type == "bridging":
                            sigtype = "semantic"
                            subtype = "meronymy"  # TODO: add other subtypes
                            if lemmas[late_tup[-1]] in ["other","another"] or lemmas[late_tup[0]] in ["other","another"]:
                                sigtype = "reference"
                                subtype = "comparative_reference"
                                candidate = (tokens, "(adversative.*|.*list*)", ".*", ".*", sigtype, subtype, self.location)
                            else:
                                candidate = (tokens, pattern, ".*", ".*", sigtype, subtype, self.location)
                        elif pair_type == "lexical_chain":
                            sigtype = "semantic"
                            subtype = "lexical_chain"
                            candidate = (tokens, pattern, ".*", ".*", sigtype, subtype, self.location)
                        elif pair_type == "tense":

                            sigtype = "morphological"
                            subtype = "tense"
                            candidate = (tokens, pattern, ".*", ".*", sigtype, subtype, self.location)
                if candidate is not None:
                    signals.append(candidate)
                    break
        return signals

    def __repr__(self):
        return self.relname + " (" + self.nid + ")"

    def __str__(self):
        return self.relname.replace("_m","").replace("_r","") + str(self.source["all"]) +"-" + str(self.target["all"])  #+ " (" + self.nid + ")"

class TSVReader():

    def __init__(self, path, sheet="Sheet1"):
        # Read excel file into a dataframe
        self.df = pd.read_excel(path, sheet)
        self.filename = os.path.basename(path)
        self.tsv = None

    @staticmethod
    def findnth(haystack, needle, last=False):
        if "!!" in needle:
            needle = needle.replace("!!","")
            n=2
            parts = haystack.split(needle, n + 1)
            if len(parts) <= n + 1:
                return -1
            return len(haystack) - len(parts[-1]) - len(needle)
        else:
            if last:
                return haystack.rfind(needle)
            else:
                return haystack.find(needle)

    def get_tsv(self):
        if self.tsv is None:
            v = [int(x) for x in pd.__version__.split(".")]
            if v[0] > 1 or (v[0] == 1 and v[1] >= 5):
                self.tsv = self.df.to_csv(sep='\t', encoding='utf-8', index=False, lineterminator='\n')
            else:
                self.tsv = self.df.to_csv(sep='\t', encoding='utf-8', index=False, line_terminator='\n')
        return self.tsv

    def get_rules(self):
        # Read a grammar rules Excel file with either tab-delimited definition fields or variable declarations, for example:
        #   $attrib_verb = (add|admit|agree|announce|argue)
        #
        #   ID  form	lemma	xpos	upos	feats	head	deprel	edeps	misc	s_type	rel	childrel	func	type	subtype	position	comments
        #   1   _	_	W(P\$?|DT)	_	.*:relcl	_	_	_	_	_	elaboration-.*_r	_	_	syntactic	relative_clause	initial	_

        tsv = self.get_tsv()
        rules = []
        variables = {}

        for l,line in enumerate(tsv.strip().split("\n")):
            if "\t" in line and not line.startswith("#") and not " = " in line:
                if line.startswith("ID\tform\t"):
                    continue
                if line.startswith("-1"):
                    continue
                for v in variables:
                    line = line.replace(v, variables[v])
                if re.search(r"\$[a-zA-Z]",line) is not None:
                    print("WARN: rule with unresolved '$' variable detected on line " + str(l))
                fields = line.split("\t")
                for i in range(len(fields)):
                    if fields[i] == "":  # Ensure empty values marked by underscore
                        fields[i] = "_"
                rule_id = fields.pop(0)
                rel, childrel, edu_func, sigtype, sigsubtype, position, comments = fields[-7:]
                fields = fields[:-7]
                rule = RULE(rule_id, fields, rel, childrel, edu_func, sigtype, sigsubtype, position, comments)
                rules.append(rule)
            elif "=" in line and line.startswith("0"):  # Variable
                k, v = line.replace("0\t","").split("=", 1)
                variables[k.strip()] = v.strip()

        return rules

    def parse_gold_table(self, docname, relations, tokens, sigtype="semantic:lexical_chain", location="within"):
        """
        * Retrieves signals from an adjudicated spreadsheet, identifying target relations by node IDs, or on failed match,
        by matching covered text (in case document graph has changed, fallback criterion 'text' rather than 'span').
        * The location argument specifies whether text search operates on the span of the relation itself ('within')
        or on the preceding/following span anywhere in the document (for LTR/RTL relations respectively).
        """
        def match_gold_rel(relations, fields, signals, tokens, row_location, criterion="span", location="within",sigtype="sigtype"):
            """
            * Target tokens are identified via their indices in the sigtoks column if the instruction column has 'y', else
            each instruction (semicolon separated) triggers a search for matching text in the relatin span.
            """
            instruction = fields[1]
            relname = fields[3]
            sig_toks = eval(fields[4])
            doc = fields[5]
            head_edu = fields[6]
            source_span = eval(fields[9])
            head_text = fields[11]
            if docname != doc or instruction == "n" or instruction.strip() == "":
                return False
            for rel in relations:
                matched = False
                if criterion =="span":  # Match by spanned tokens in resource file
                    matched = True if head_edu in rel.head_edus and rel.source["all"] == source_span and rel.relname == relname else False
                else:  # Tree may have changed, attempt match by source head text
                    if rel.head_edu_text == head_text.replace("&apos;","'").replace("&quot;",'"'):
                        matched = True
                        sys.stderr.write("! Fallback match for signal type "+sigtype+" for "+str(row_location)+ \
                                         ", consider updating source from "+str(source_span)+ \
                                         " to "+str(rel.source["all"])+"\n")
                if matched:
                    source_tokens = " ".join([tokens[t] for t in range(rel.source["all"][0],rel.source["all"][-1]+1)])
                    target_tokens = " ".join([tokens[t] for t in range(rel.target["all"][0],rel.target["all"][-1]+1)])
                    instances = instruction.split(";") if location == 'within' else [instruction]
                    for inst in instances:
                        inst = inst.strip()
                        if inst == "y":
                            inst = sig_toks
                        else:
                            if location=="within":
                                if "," in inst:
                                    inst = [x.strip() for x in inst.split(",")]
                                else:
                                    inst = [inst, inst]
                                w1 = rel.target["all"][0] + target_tokens.split(inst[0])[0].count(" ")
                                w2 = rel.source["all"][0] + source_tokens.split(inst[1])[0].count(" ")
                                temp = []
                                for i in range(w1,w1 + inst[0].count(" ") + 1):
                                    temp.append(i)
                                for i in range(w2,w2 + inst[1].count(" ") + 1):
                                    temp.append(i)
                                inst = temp
                            else:
                                search_text = " " + inst.replace('``','"').replace("`","'") + " "
                                if rel.source["all"][-1] < rel.target["all"][0]:  # LTR relation
                                    context = " " + " ".join([t for i, t in enumerate(tokens) if i <= rel.source["all"][-1]]) + " "
                                    position = self.findnth(context,search_text,last=True)
                                    if position < 0:  # Try reverse
                                        context = " " + " ".join([t for i, t in enumerate(tokens) if i >= rel.target["all"][0]]) + " "
                                        position = self.findnth(context,search_text)
                                        if position < 0:
                                            raise IOError("! Error in resource at " + row_location + ": text not found - " + search_text + "\n")
                                        tok_idx = rel.target["all"][0] + context[:position].count(" ")
                                    else:
                                        tok_idx = context[:position].count(" ")
                                else:  # RTL relation
                                    context = " " + " ".join([t for i, t in enumerate(tokens) if i >= rel.source["all"][0]]) + " "
                                    position = self.findnth(context,search_text)
                                    if position < 0:
                                        context = " " + " ".join([t for i, t in enumerate(tokens) if i <= rel.target["all"][-1]]) + " "
                                        position = self.findnth(context,search_text,last=True)
                                        if position < 0:
                                            raise IOError("! Error in resource at " + row_location + ": text not found - " + search_text + "\n")
                                        tok_idx = context[:position].count(" ")
                                    else:
                                        tok_idx = rel.source["all"][0] + context[:position].count(" ")
                                inst = list(range(tok_idx, tok_idx + search_text.count(" ")-1)) if search_text.count(" ") > 2 else [tok_idx]
                        inst = tuple(sorted(inst))
                        if len(inst) == 1:
                            inst = inst[0]
                        candidate = (inst, rel.relname, ".*", ".*", maintype, subtype,"nid:" + rel.nid)
                        signals.append(candidate)
                        return signals
            return None

        signals = []
        tsv = self.get_tsv()
        maintype, subtype = sigtype.split(":")
        for r, row in enumerate(tsv.split("\n")):
            if r == 0 or "\t" not in row:
                continue
            fields = row.split("\t")
            if fields[5] != docname:
                continue
            if sigtype == "any:any":
                maintype, subtype = fields[-1].split(":")
            row_location = self.filename + ": " + str(r)
            result = match_gold_rel(relations,fields,signals,tokens, row_location=row_location, location=location,sigtype=maintype)
            if result is None:  # No matching relation, try fallback
                result = match_gold_rel(relations, fields, signals, tokens, row_location=row_location,criterion="text", location=location,sigtype=maintype)
                if result is None:  # No matching fallback
                    raise IOError("! No relation found for " + str(relname) + " with head edu "+head_edu+" in " + docname)
            elif result is False:  # Skip document
                continue
            else:
                signals = result

        return signals

