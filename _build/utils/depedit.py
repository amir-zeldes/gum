#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
DepEdit - A simple configurable tool for manipulating dependency trees

Input: CoNLL10 or CoNLL-U (10 columns, tab-delimited, blank line between sentences, comments with pound sign #)

Author: Amir Zeldes
"""

from __future__ import print_function

import argparse
import os
import re
import sys
from decimal import Decimal
from collections import defaultdict
from copy import copy, deepcopy
from glob import glob
import io
from six import iteritems, iterkeys

__version__ = "3.4.0.1"

ALIASES = {"form":"text","upostag":"pos","xpostag":"cpos","feats":"morph","deprel":"func","deps":"head2","misc":"func2",
           "xpos": "cpos","upos":"pos"}


def escape(string, symbol_to_mask, border_marker):
    inside = False
    output = ""
    # Scan the string looking for border or symbol to mask
    for char in string:
        if char == border_marker:
            inside = not inside
        output += "%%%%%" if char == symbol_to_mask and inside else char
    return output


class ParsedToken:
    __slots__ = ['id', 'text', 'lemma', 'pos', 'cpos', 'morph', 'head', 'func', 'edep', 'head2', 'func2', 'storage', 'storage2', 'storage3', 'num', 'child_funcs', 'position', 'is_super_tok', 'sentence']

    def __init__(self, tok_id, text, lemma, pos, cpos, morph, head, func, head2, func2, num, child_funcs, position, is_super_tok=False, tokoffset=0):
        self.id = tok_id
        self.text = text
        self.pos = pos
        self.cpos = cpos
        self.lemma = lemma
        self.morph = morph
        self.head = head
        self.func = func
        self.edep = []
        self.head2 = head2
        self.func2 = func2
        if ":" in head2:  # edep format
            try:
                edeps = head2.split("|")
                for edep in edeps:
                    h, d = edep.split(":", 1)
                    h = str(float(h) + tokoffset)
                    self.edep.append([h, d])
            except ValueError:
                pass
        self.storage = ""  # Storage field2 for temporary values, never read or written to/from conllu
        self.storage2 = ""
        self.storage3 = ""
        self.num = num
        self.child_funcs = child_funcs
        self.position = position
        self.is_super_tok = is_super_tok

    def __getattr__(self, item):
        if item.startswith("#S:"):
            key = item.split(":",1)[1]
            if key in self.sentence.annotations:
                return self.sentence.annotations[key]
            elif key in self.sentence.input_annotations:
                return self.sentence.input_annotations[key]
            elif key == "docname":
                return self.sentence.docname
            else:
                return ""

    def __repr__(self):
        return str(self.text) + " (" + str(self.pos) + "/" + str(self.lemma) + ") " + "<-" + str(self.func)

    def __deepcopy__(self, memo):
        return self


class Sentence:
    __slots__ = ['sentence_string','length','annotations','input_annotations','sent_num','offset','depedit','docname']

    def __init__(self, sentence_string="", sent_num=0, tokoffset=0, depedit_object=None, docname=""):
        self.sentence_string = sentence_string
        self.length = 0
        self.annotations = {}  # Dictionary to keep sentence annotations added by DepEdit rules
        self.input_annotations = {}  # Dictionary with original sentence annotations (i.e. comment lines) in input conll
        self.sent_num = sent_num
        self.offset = tokoffset
        self.depedit = depedit_object
        self.docname = docname

    def print_annos(self):
        anno_dict = dict((k, v) for k, v in iteritems(self.annotations))
        if "text" in self.input_annotations and "text" in self.annotations:
            self.input_annotations["text"] = self.annotations["text"]
        if self.depedit.kill not in ["comments", "both"]:
            anno_dict.update(dict((k, v) for k, v in iteritems(self.input_annotations)))
        sorted_keys = sorted(list(iterkeys(anno_dict)))
        newdoc_anno = []
        sent_id = []
        if "sent_id" in sorted_keys:
            # Ensure sent_id is first
            sent_id = ["sent_id"]
            sorted_keys.remove("sent_id")
        if "newdoc id" in sorted_keys:
            # Ensure newdoc id is first
            newdoc_anno = ["newdoc id"]
            sorted_keys.remove("newdoc id")
        global_specs = [k for k in sorted_keys if k.startswith("global.")]
        meta = sorted([k for k in sorted_keys if k.startswith("meta::")])
        others = [k for k in sorted_keys if not k.startswith("meta::") and not k.startswith("global.") and k != "text"]
        text = ["text"] if "text" in sorted_keys else []
        sorted_keys = newdoc_anno + global_specs + meta + sent_id + sorted(others) + text
        return ["# " + key.strip() + " = " + anno_dict[key].strip() if len(anno_dict[key].strip())>0 else "# " + key.strip() for key in sorted_keys]


class Transformation:

    def parse_transformation(self, transformation_text, depedit_container, line):
        split_trans = transformation_text.split("\t")
        if len(split_trans) < 3:
            return None
        definition_string, relation_string, action_string = split_trans
        if "~#" in action_string and "edep=" not in action_string and "edep+=" not in action_string:
            sys.stderr.write("WARN: action specifies enhanced edge (~) but no edep label on line " + str(line) + "\n")
        elif "~#" not in action_string and "edep=" in action_string and not action_string.endswith("edep=") and \
                not "edep=;" in action_string and not "ehead=" in action_string:
            sys.stderr.write("WARN: action specifies an edep label but no enhanced edge (~) on line " + str(line) + "\n")
        match_variables = re.findall(r'\{([^}]+)\}',definition_string)
        for m in match_variables:
            if m in depedit_container.variables:
                definition_string = definition_string.replace("{"+m+"}",depedit_container.variables[m])
            else:
                sys.stderr.write("! Definition contains undefined variable: {" + m + "}")
                quit()
        relation_string = self.normalize_shorthand(relation_string)
        for source, target in iteritems(ALIASES):
            relation_string = relation_string.replace(":"+source,":"+target)
        action_string = self.normalize_shorthand(action_string)
        definition_string = escape(definition_string, ";", "/")
        definitions_list = definition_string.split(";")
        escaped_definitions = []
        for _def in definitions_list:
            escaped_definitions.append(_def.replace("%%%%%", ";"))
        definitions = []
        for def_index, esc_string in enumerate(escaped_definitions):
            definitions.append(DefinitionMatcher(esc_string, def_index + 1))
        relations = relation_string.split(";")
        actions = action_string.strip().split(";")
        aliased_actions = []
        for action in actions:
            aliased_actions.append(self.handle_aliases(action))
        return [definitions, relations, aliased_actions]

    @staticmethod
    def handle_aliases(orig_action):
        for source, target in iteritems(ALIASES):
            orig_action = orig_action.replace(":" + source + "=", ":" + target + "=")
            orig_action = orig_action.replace(":" + source + "+=", ":" + target + "+=")
            orig_action = orig_action.replace(":" + source + "-=", ":" + target + "-=")
            orig_action = orig_action.replace(":" + source + ",=", ":" + target + ",=")
        return orig_action

    @staticmethod
    def normalize_shorthand(criterion_string):
        criterion_string = criterion_string.replace('.*', '.1,1000')
        temp = ""
        while temp != criterion_string:
            temp = criterion_string
            criterion_string = re.sub(r'(#[0-9]+)(>|\.(?:[0-9]+(?:,[0-9]+)?)?)(#[0-9]+)([>~]|\.(?:[0-9]+(?:,[0-9]+)?)?)',
                                      r'\1\2\3;\3\4', criterion_string)
        return criterion_string

    def __init__(self, transformation_text, line, depedit_container):
        self.transformation_text = transformation_text
        instructions = self.parse_transformation(transformation_text, depedit_container, line)
        if instructions is None:
            sys.stderr.write("Depedit says: error in configuration file\n"
                  "Malformed instruction on line " + str(line) + " (instruction lines must contain exactly two tabs)\n")
            sys.exit()
        self.definitions, self.relations, self.actions = instructions
        self.line = line

    def __repr__(self):
        return " | ".join([str(self.definitions), str(self.relations), str(self.actions)])

    def validate(self):
        report = ""
        for definition in self.definitions:
            node = escape(definition.def_text, "&", "/")
            criteria = (_crit.replace("%%%%%", "&") for _crit in node.split("&"))
            for criterion in criteria:
                criterion = escape(criterion, "=", "/")
                if re.match(r"(text|pos|cpos|lemma|morph|storage[23]?|edom|func|head|func2|head2|num|form|upos|upostag|xpos|xpostag|feats|deprel|deps|misc|edep|ehead)!?=/[^/=]*/", criterion) is None:
                    if re.match(r"position!?=/(first|last|mid)/", criterion) is None:
                        if re.match(r"#S:[A-Za-z_]+!?=/[^/\t]+/",criterion) is None:
                            report += "Invalid node definition in column 1: " + criterion
        for relation in self.relations:
            if relation == "none" and len(self.relations) == 1:
                if len(self.definitions) > 1:
                    report += "Column 2 setting 'none' invalid with more than one definition in column 1"
            elif relation == "none":
                report += "Setting 'none' invalid in column 2 when multiple relations are defined"
            else:
                criteria = relation.split(";")
                for criterion in criteria:
                    criterion = criterion.strip()
                    if not re.match(r"(#[0-9]+(([>~]|\.([0-9]+(,[0-9]+)?)?)#[0-9]+)+|#[0-9]+:(text|pos|cpos|lemma|morph|storage[23]?|edom|"
                                    r"func|head|func2|head2|num|form|upos|upostag|xpos|xpostag|feats|deprel|deps|misc|edep|ehead)==#[0-9]+)",
                                    criterion):
                        report += "Column 2 relation setting invalid criterion: " + criterion + "."
        for action in self.actions:
            commands = action.split(";")
            for command in commands:  # Node action
                if re.match(r"(#[0-9]+([>~]|><)#[0-9]+|#[0-9]+:(func|lemma|text|pos|cpos|morph|storage[23]?|edom|head|head2|func2|num|form|upos|upostag|xpos|xpostag|feats|deprel|deps|misc|edep|ehead|split)[\+,-]?=[^;]*)$", command) is None:
                    if re.match(r"#S:[A-Za-z_]+=[A-Za-z_]+$|last$|once$", command) is None:  # Sentence annotation action or quit
                        report += "Column 3 invalid action definition: " + command + " and the action was " + action
                        if "#" not in action:
                            report += " (no node declaration with #)"
                if "split=" in action and "|" not in action:
                    report += "Split action specified but no split points marked with | in command: " + command + " in action " + action

        return report


class DefinitionMatcher:
    __slots__ = ['def_text','def_index','groups','defs','sent_def','priority']

    def __init__(self, def_text, def_index):
        self.def_text = escape(def_text, "&", "/")
        self.def_index = def_index
        self.groups = []
        self.defs = []
        self.sent_def = False
        self.priority = 5  # Minimum priority
        if def_text.startswith("#S:"):
            self.sent_def = True

        def_items = self.def_text.split("&")
        for def_item in def_items:
            def_item = def_item.replace("%%%%%", "&")
            criterion = def_item.split("=", 1)[0]
            negative_criterion = (criterion[-1] == "!")
            if negative_criterion:
                criterion = criterion[:-1]

            def_value = def_item.split("=", 1)[1][1:-1]

            # Ensure regex is anchored
            if def_value[0] != "^":
                def_value = "^" + def_value
            if def_value[-1] != "$":
                def_value += "$"
            if "^(?i)" in def_value:
                def_value = def_value.replace("^(?i)","(?i)^")  # ensure case insensitive flag is initial
            definition = Definition(criterion, def_value, negative_criterion)
            self.defs.append(definition)
            if definition.selectivity < self.priority:
                self.priority = definition.selectivity

    def __repr__(self):
        return "#" + str(self.def_index) + ": " + self.def_text

    def match(self, token):
        potential_groups = []
        for def_item in self.defs:
            if def_item.criterion == "edep":
                tok_value = [e[1] for e in token.edep]  # Get list of edeprels
            elif def_item.criterion == "edom":
                tok_value = ["||".join(e) for e in token.edep]  # Get list of edeprels and their heads
            else:
                tok_value = getattr(token, def_item.criterion)
            if def_item.criterion in ["head","ehead"] and len(tok_value) > 0:
                tok_value = str(Decimal(tok_value) - token.sentence.offset)
                if tok_value.endswith('.0'):
                    tok_value = tok_value.replace(".0","")
            match_obj = def_item.match_func(def_item, tok_value)

            if match_obj is None:
                return False
            elif not match_obj:
                return False
            elif match_obj is True:
                pass
            elif match_obj is not None:
                if len(match_obj.groups()) > 0:
                    potential_groups.append(match_obj.groups())
        
        self.groups = potential_groups
        return True


class Definition:
    __slots__ = ['value','match_type','selectivity','compiled_re','match_func','negative','criterion']

    def __init__(self, criterion, value, negative=False):
        # Handle conllu criterion aliases:
        if criterion.startswith("#S:"):  # Sentence annotation
            self.criterion = criterion
        else:
            self.criterion = ALIASES.get(criterion, criterion)
        self.value = value
        self.match_type = ""
        self.compiled_re = None
        self.match_func = None
        self.negative = negative
        self.selectivity = 0
        self.set_match_type(criterion)

    def set_match_type(self, criterion):
        value = self.value[1:-1]
        if self.value == "^.*$" and not self.negative:
            self.match_func = self.return_true
            self.match_type = "return_true"
            self.selectivity = 5  # Minimum selectivity
        elif re.escape(value) == value:  # No regex operators within expression
            if criterion in ["edep","edom"]:
                self.match_func = self.return_not_in if self.negative else self.return_in
                self.match_type = "return_not_in" if self.negative else "return_in"
                self.selectivity = 2
            else:
                self.match_func = self.return_exact_negative if self.negative else self.return_exact
                self.match_type = "return_exact_negative" if self.negative else "return_exact"
                self.selectivity = 1
            self.value = value
        else:  # regex
            self.compiled_re = re.compile(self.value)
            if criterion in ["edep","edom"]:
                self.match_func = self.return_regex_not_in if self.negative else self.return_regex_in
                self.match_type = "return_regex_not_in" if self.negative else "return_regex_in"
                self.selectivity = 4
            else:
                self.match_func = self.return_regex_negative if self.negative else self.return_regex
                self.match_type = "return_regex_negative" if self.negative else "return_regex"
                self.selectivity = 3

    @staticmethod
    def return_exact(definition, test_val):
        return test_val == definition.value

    @staticmethod
    def return_exact_negative(definition, test_val):
        return test_val != definition.value

    @staticmethod
    def return_regex(definition, test_val):
        return definition.compiled_re.search(test_val)

    @staticmethod
    def return_regex_negative(definition, test_val):
        return definition.compiled_re.search(test_val) is None

    @staticmethod
    def return_in(definition, test_val):
        return any([definition.value == v for v in test_val])

    @staticmethod
    def return_not_in(definition, test_val):
        return all([definition.value != v for v in test_val])

    @staticmethod
    def return_regex_in(definition, test_val):
        matchers = [definition.compiled_re.search(v) for v in test_val if v is not None]
        successful = [m for m in matchers if m is not None]
        successful.sort(key=lambda x: x.endpos - x.pos, reverse=True)
        return successful[0] if len(successful) > 0 else None

    @staticmethod
    def return_regex_not_in(definition, test_val):
        return all([definition.compiled_re.search(v) is None for v in test_val])

    @staticmethod
    def return_true(definition, test_val):
        return True


class Match:
    __slots__ = ['def_index','token','groups','sent_def']

    def __init__(self, def_index, token, groups):
        self.def_index = def_index
        self.token = token
        self.groups = groups
        self.sent_def = False  # Whether this is a sentence annotation match

    def __repr__(self):
        return "#" + str(self.def_index) + ": " + str(self.token)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Return a deep copy of the object without copying the object reference
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k in self.__slots__:
            v = self.__getattribute__(k)
            setattr(result, k, deepcopy(v, memo))
        return result


class DepEdit:

    def __init__(self, config_file="", options=None):
        self.variables = {}
        self.transformations = []
        self.user_transformation_counter = 0
        self.quiet = False
        self.kill = None
        if options is not None:
            if "quiet" in options.__dict__ or options.quiet is not None:
                self.quiet = options.quiet
            if "kill" in options.__dict__ or options.kill is not None:
                if options.kill in ["supertoks", "comments", "both"]:
                    self.kill = options.kill
        if config_file != "":
            self.read_config_file(config_file)
        self.docname = self.input_mode = None

    def read_config_file(self, config_file, clear_transformations=False):
        """
        Function to read configuration file. Can be invoked after initialization.

        :param config_file: The file to read instructions from.
        :param clear_transformations: Whether to discard previous instructions.
        :return: void
        """

        if clear_transformations:
            self.transformations = []
            self.user_transformation_counter = 0
        if isinstance(config_file, str):
            if sys.version_info[0] < 3:
                config_file = open(config_file).readlines()
            else:
                config_file = open(config_file, encoding="utf8").readlines()

        line_num = 0
        for instruction in config_file:
            instruction = instruction.strip()
            line_num += 1
            match_variable = re.match(r'\{([^}]+)\}=/([^\n]+)/',instruction)
            if match_variable is not None:
                key = match_variable.group(1)
                val = match_variable.group(2)
                self.add_variable(key,val)
            elif len(instruction) > 0 and not instruction.startswith(";") and not instruction.startswith("#") \
                    or instruction.startswith("#S:"):
                    self.transformations.append(Transformation(instruction, line_num, self))

        trans_report = ""
        for transformation in self.transformations:
            temp_report = transformation.validate()
            if temp_report != "":
                trans_report += "On line " + str(transformation.line) + ": " + temp_report + "\n"
        if len(trans_report) > 0:
            trans_report = "Depedit says: error in configuration file\n\n" + trans_report
            sys.stderr.write(trans_report)
            sys.exit()

    def process_sentence(self, conll_tokens, stepwise=False):
        retval = 0
        for i, transformation in enumerate(self.transformations):
            if stepwise:
                if sys.version_info[0] < 3:
                    print(("# Rule " + str(i+1) + ": " + str(transformation)+"\n").encode("utf-8"),end="")
                else:
                    print("# Rule " + str(i+1) + ": " + str(transformation)+'\n',end="")

            node_matches = defaultdict(list)
            for def_matcher in sorted(transformation.definitions,key=lambda x: x.priority):
                found = False
                for token in conll_tokens:
                    if not token.is_super_tok and def_matcher.match(token):
                        found = True
                        if def_matcher.sent_def:
                            if len(node_matches[def_matcher.def_index])==0:  # Only add a sentence anno definition once
                                node_matches[def_matcher.def_index] = [Match(def_matcher.def_index, token, def_matcher.groups)]
                                node_matches[def_matcher.def_index][0].sent_def = True
                        else:
                            node_matches[def_matcher.def_index].append(Match(def_matcher.def_index, token, def_matcher.groups))
                if not found:
                    break  # Some required node has no matches, stop processing this rule
            if len(node_matches) == 0:
                continue  # No mode matches for this rule, continue to next rule
            result_sets = []
            for relation in transformation.relations:
                if not self.matches_relation(node_matches, relation, result_sets):
                    result_sets = []
            result_sets = self.merge_sets(result_sets, len(transformation.definitions), len(transformation.relations))
            unique_results = []
            for r in result_sets:
                if r not in unique_results:
                    unique_results.append(r)
            result_sets = unique_results
            self.add_groups(result_sets)
            if len(result_sets) > 0:
                for action in transformation.actions:
                    if "><" in action:
                        retval = self.execute_supertoken(result_sets, action, transformation, conll_tokens)
                    elif "split=" in action:
                        retval += self.execute_split(result_sets, action, transformation, conll_tokens)
                    else:
                        retval = self.execute_action(result_sets, action, transformation)
                    if retval == "last":  # Explicit instruction to cease processing
                        return 0
            if stepwise:
                print("\n".join(self.serialize_output_tree(conll_tokens, 0))+"\n")
        if isinstance(retval, int):
            return retval
        else:
            return 0

    def matches_relation(self, node_matches, relation, result_sets):
        if len(relation) == 0:
            return False
        operator = field = None
        if "==" in relation:
            m = re.search(r':(.+)==', relation)
            operator = m.group()
            field = m.group(1)
        elif "." in relation:
            if re.match(r'.*\.[0-9]', relation):
                m = re.match(r'.*\.[0-9]*,?[0-9]*#', relation)
                operator = m.group()
                operator = operator[operator.find("."):operator.rfind("#")]
            else:
                operator = "."
        elif ">" in relation:
            operator = ">"
        elif "~" in relation:
            operator = "~"

        matches = defaultdict(list)

        hits = 0
        if relation == "none":  # Unary operation on one node
            node1 = 1
            for matcher1 in node_matches[node1]:
                tok1 = matcher1.token
                hits += 1
                result = {}
                matches[node1].append(tok1)
                result[node1] = tok1
                result["rel"] = relation
                result["matchers"] = [matcher1]
                result["ID2matcher"] = {node1:matcher1}
                result_sets.append(result)
        elif "==" in relation:
            node1 = relation.split(operator)[0]
            node2 = relation.split(operator)[1]
            node1 = int(node1.replace("#", ""))
            node2 = int(node2.replace("#", ""))

            for matcher1 in node_matches[node1]:
                tok1 = matcher1.token
                for matcher2 in node_matches[node2]:
                    tok2 = matcher2.token
                    if self.test_relation(tok1, tok2, field):
                        result_sets.append({node1: tok1, node2: tok2, "rel": relation, "matchers": [matcher1, matcher2],
                                            "ID2matcher":{node1:matcher1, node2:matcher2}})
                        matches[node1].append(tok1)
                        matches[node2].append(tok2)
                        hits += 1

            for option in [node1, node2]:
                matchers_to_remove = []
                for matcher in node_matches[option]:
                    if matcher.token not in matches[option]:
                        matchers_to_remove.append(matcher)
                for matcher in matchers_to_remove:
                    node_matches[option].remove(matcher)
        else:
            node1 = relation.split(operator)[0]
            node2 = relation.split(operator)[1]

            node1=int(node1.replace("#", ""))
            node2=int(node2.replace("#", ""))
            for matcher1 in node_matches[node1]:
                tok1 = matcher1.token
                for matcher2 in node_matches[node2]:
                    tok2 = matcher2.token
                    if self.test_relation(tok1, tok2, operator) or matcher1.sent_def:  # Sentence dominance always True
                        result_sets.append({node1: tok1, node2: tok2, "rel": relation, "matchers": [matcher1, matcher2],
                                            "ID2matcher":{node1:matcher1, node2:matcher2}})
                        matches[node1].append(tok1)
                        matches[node2].append(tok2)
                        hits += 1

            for option in node1, node2:
                matchers_to_remove = []
                for matcher in node_matches[option]:
                    if matcher.token not in matches[option]:
                        matchers_to_remove.append(matcher)
                for matcher in matchers_to_remove:
                    node_matches[option].remove(matcher)

        if hits == 0:  # No solutions found for this relation
            return False
        else:
            return True

    @staticmethod
    def test_relation(node1, node2, operator):
        if operator == ".":
            if int(float(node2.id)) == int(float(node1.id))+1:
                return True
            else:
                return False
        elif operator == ">":
            if float(node2.head) == float(node1.id):
                return True
            else:
                return False
        elif operator == "~":
            try:
                eheads = [int(float(e[0])) for e in node2.edep if e[0] is not None]  # ignore root edep
                if int(float(node1.id)) in eheads:
                    return True
                return False
            except ValueError:  # Node has no ehead or non-numeric ehead value, so float conversion fails
                return False
        elif "." in operator:
            if int(float(node2.id)) == int(float(node1.id)):  # Same node
                return False
            m = re.match(r'\.([0-9]+)(,[0-9]+)?',operator)
            if len(m.groups()) > 1:
                min_dist = int(m.group(1))
                if not m.group(2) is None:
                    max_dist = int(m.group(2).replace(",",""))
                else:
                    max_dist = min_dist
                if max_dist >= int(float(node2.id)) - int(float(node1.id)) >= min_dist:
                    return True
                else:
                    return False
            else:
                dist = int(m.group(1))
                if int(float(node2.id)) - int(float(node1.id)) == dist:
                    return True
                else:
                    return False
        else:
            val1 = getattr(node1,operator)
            val2 = getattr(node2,operator)
            return val1 == val2

    def merge_sets(self, sets, node_count, rel_count):

        solutions = []
        bins = []
        for set_to_merge in sets:
            new_set = {"rels": [], "matchers": []}
            for key in set_to_merge:
                if key == "rel":
                    new_set["rels"].append(set_to_merge[key])
                elif key == "matchers":
                    new_set["matchers"] += set_to_merge[key]
                else:
                    new_set[key] = set_to_merge[key]

            for my_bin in copy(bins):
                if self.bins_compatible(new_set, my_bin):
                    candidate = self.merge_bins(new_set, my_bin)
                    bins.append(copy(candidate))
            bins.append(copy(new_set))

        for my_bin in bins:
            if len(my_bin) == node_count + 3:
                if len(my_bin["rels"]) == rel_count:  # All required relations have been fulfilled
                    solutions.append(my_bin)
                else:  # Some node pair has multiple relations, check that all are fulfilled
                    for set_to_merge in sets:
                        if set_to_merge["rel"] not in my_bin["rels"]:  # This relation was missing
                            node_ids = list((key) for key in set_to_merge if isinstance(key, int))
                            # Check that the missing rel connects nodes that are already in my_bin
                            ids_are_in_bin = list((nid in my_bin) for nid in node_ids)
                            if all(ids_are_in_bin):
                                nodes_are_identical = list((set_to_merge[nid] == my_bin[nid]) for nid in node_ids)
                                if all(nodes_are_identical):
                                    my_bin["rels"].append(set_to_merge["rel"])
                                    if len(my_bin["rels"]) == rel_count:  # Check if we now have all required relations
                                        solutions.append(my_bin)

        merged_bins = []
        for solution in solutions:
            self.merge_solutions(solution, merged_bins, rel_count)
        self.prune_merged_bins(merged_bins, rel_count)
        return merged_bins

    @staticmethod
    def merge_solutions(solution, merged, rel_count):
        merges_to_add = []
        if solution not in merged:
            merged.append(solution)
        if len(solution["rels"]) != rel_count:  # Otherwise this solution is completed
            # This is an incomplete solution, try to merge it into the merged solutions list
            for candidate in merged:
                if candidate != solution:
                    for key in solution:
                        if key != "rels" and key != "matchers":
                            if key in candidate:  # This is a position, e.g. #1, that is also in the candidate in merged
                                # Check that they are compatible, e.g. #1 is the same token
                                if solution[key] == candidate[key]:
                                    rels_in_candidate = (rel in candidate["rels"] for rel in solution["rels"])
                                    if not all(rels_in_candidate):
                                        rels = solution["rels"] + candidate["rels"]
                                        matchers = []
                                        for matcher in solution["matchers"]:
                                            matchers.append(matcher)
                                        for matcher in candidate["matchers"]:
                                            if matcher not in matchers:
                                                matchers.append(matcher)
                                        merged_solution = copy(solution)
                                        merged_solution.update(candidate)
                                        merged_solution["rels"] = rels
                                        merged_solution["matchers"] = matchers
                                        merges_to_add.append(merged_solution)
        merged.extend(merges_to_add)
        solution["rels"].sort()

    @staticmethod
    def bins_compatible(bin1, bin2):
        overlap = False
        non_overlap = False
        for key in bin1:
            if key in bin2:
                if bin1[key] == bin2[key]:
                    overlap = True
            if key not in bin2:
                non_overlap = True
        if overlap and non_overlap:
            return True
        else:
            return False

    @staticmethod
    def merge_bins(bin1, bin2_orig):
        """
        Merge bins we know are compatible, e.g. bin1 has #1+#2 and bin2 has #2+#3

        :param bin1: a bin dictionary mapping indices to tokens, a list of relations 'rels' and matcher objects 'matchers'
        :param bin2: a bin dictionary mapping indices to tokens, a list of relations 'rels' and matcher objects 'matchers'
        :return: the merged bin with data from both input bins
        """
        bin2 = deepcopy(bin2_orig)
        for matcher in bin1["matchers"]:
            skip = False
            for matcher2 in bin2["matchers"]:
                if matcher2.def_index == matcher.def_index:
                    skip = True
                elif matcher.token == matcher2.token:
                    skip = True  # Same token can't serve two roles in a merged bin
            if not skip:
                if matcher not in bin2["matchers"]:
                    bin2["matchers"].append(matcher)
        for key in bin1:
            if key != "rels":
                if key not in bin2:
                    out_bin = copy(bin2)
                    out_bin[key] = bin1[key]
                    for rel in bin1["rels"]:
                        new_rel = deepcopy(rel)
                        out_bin["rels"] = bin2["rels"] + [new_rel]
                    return out_bin

    @staticmethod
    def prune_merged_bins(merged_bins, rel_count):
        """
        Deletes bins with too few relationships matched after merging is complete

        :param merged_bins: candidates for bins representing complete related chains of nodes
        :param rel_count: how many relations the current transformation has - any bins with less will be discarded now
        :return: void
        """
        bins_to_delete = []
        for merged_bin in merged_bins:
            if len(merged_bin["rels"]) < rel_count:
                bins_to_delete.append(merged_bin)
        for bin_to_delete in bins_to_delete:
            merged_bins.remove(bin_to_delete)

    @staticmethod
    def add_groups(result_sets):
        for result in result_sets:
            groups = []
            sorted_matchers = sorted(result["matchers"], key=lambda x: x.def_index)
            for matcher in sorted_matchers:
                for group in matcher.groups:
                    for g in group:
                        if g is not None:
                            if "\\" in g:
                                g = re.sub(r'\\', r'\\\\', g)
                        groups.append(g)
            result["groups"] = groups[:]

    def execute_supertoken(self, result_sets, action_list, transformation, conll_tokens):
        def remove_misc(token, anno):
            miscs = token.func2.split("|")
            filtered = []
            for a in miscs:
                if not a.startswith(anno+"="):
                    filtered.append(a)
            if len(filtered) == 0:
                token.func2 = "_"
            else:
                token.func2 = "|".join(filtered)

        for res in result_sets:
            begin_idx = conll_tokens.index(res[1])
            if "." in res[1].num or "." in res[2].num:  # Ellipsis tokens cannot form supertokens
                continue
            super_id = res[1].num + "-" + res[2].num
            if any([t.num==super_id for t in conll_tokens]):  # Do not produce already existing supertokens
                continue
            super_text = res[1].text + res[2].text
            misc = "_" if "SpaceAfter=No" not in res[2].func2 else "SpaceAfter=No"
            remove_misc(res[1],"SpaceAfter")
            remove_misc(res[2],"SpaceAfter")
            supertok = ParsedToken(super_id, super_text, "_", "_", "_", "_", "_", "_", "_", misc, super_id, [], "", is_super_tok=True)
            conll_tokens.insert(begin_idx, supertok)

        return 0

    def execute_split(self, result_sets, action_list, transformation, conll_tokens):
        def add_misc(key_val, old_misc):
            annos = old_misc.split("|") if old_misc != "_" else []
            key = key_val.split("=")[0]
            annos = [a for a in annos if not a.startswith(key + "=")]
            annos.append(key_val)
            return "|".join(sorted(list(set(annos))))

        split_offset = 0
        for res in result_sets:
            # Prepare subtoken list for insertion
            subtoks = action_list.split("split=")[1].split("|")
            no_space_after = "SpaceAfter=No" in res[1].func2
            idx = conll_tokens.index(res[1])
            prev_id = float(res[1].id)
            prev_num = int(float(res[1].num))

            # Push up all tokens after insertion point by len(subtoks)-1
            split_offset += len(subtoks)-1
            for tok in conll_tokens:
                float(tok.id)
                if tok.is_super_tok:
                    start, end = tok.id.split("-")
                    start = float(start)
                    end = float(end)
                    if start > float(res[1].num):
                        start += len(subtoks)-1
                    if end > float(res[1].num):
                        end += len(subtoks) - 1
                    tok.id = tok.num = str(int(start)) + "-" + str(int(end))
                else:
                    if float(tok.id) > float(res[1].id):
                        tok.num = str(float(tok.num)+len(subtoks)-1)
                        tok.id = str(float(tok.id)+len(subtoks)-1)
                    if float(tok.head) > float(res[1].id):
                        tok.head = str(float(tok.head)+len(subtoks)-1)

            # Update first token and insert subsequent ones
            for i, subtok in enumerate(subtoks):
                if subtok.startswith("$"):  # subtoken from capturing group in original token being split
                    group_num = int(subtok[1:]) - 1
                    subtok = res["groups"][group_num]
                if i ==0:  # First token, overwrite original token
                    res[1].text = subtok
                    res[1].func2 = add_misc("SpaceAfter=No", res[1].func2)
                else:  # Subsequent new tokens
                    idx += 1
                    prev_id += 1
                    prev_num += 1
                    new_token = ParsedToken(str(prev_id), subtok, "_", "_", "_", "_", res[1].id, "dep", "_", "_", str(prev_num), [], "")
                    conll_tokens.insert(idx, new_token)
                    if i < len(subtoks) -1 or no_space_after:
                        new_token.func2 = add_misc("SpaceAfter=No", new_token.func2)

        return split_offset

    def execute_action(self, result_sets, action_list, transformation):
        actions = action_list.split(";")
        for result in result_sets:
            if len(result) > 0:
                for action in actions:
                    if action == "last":  # Completely ends processing of this sentence
                        return "last"
                    elif action == "once":  # Stop executing action but return to rule cascade
                        return None
                    elif ":" in action:  # Unary instruction
                        if action.startswith("#S:"):  # Sentence annotation instruction
                            key_val = action.split(":")[1]
                            key, val = key_val.split("=", 1)
                            if key == "docname":
                                result[1].sentence.docname = val
                            else:
                                result[1].sentence.annotations[key] = val
                        else:  # node instruction
                            node_position = int(action[1:action.find(":")])
                            if not self.quiet:
                                if node_position in result["ID2matcher"]:
                                    if result["ID2matcher"][node_position].sent_def:
                                        sys.stdout.write("! Warning: Rule is applying a *token* transformation to a *sentence* annotation node:\n")
                                        sys.stdout.write("  " + transformation.transformation_text + "\n")
                                        sys.stdout.write("  Applying the transformation to first token in sentence.\n")
                            prop = action[action.find(":") + 1:action.find("=")]
                            value = action[action.find("=") + 1:].strip()
                            add_val = False
                            subtract_val = False
                            concat_val = False
                            if prop.endswith("+"):  # Add annotation, e.g. feats+=...
                                add_val = True
                                prop = prop[:-1]
                            elif prop.endswith("-"):  # Remove annotation, e.g. feats-=...
                                subtract_val = True
                                prop = prop[:-1]
                            elif prop.endswith(","):  # Add to existing values, separated by , and alphabetized, e.g. Cxn=X,Y,Z
                                concat_val = True
                                prop = prop[:-1]
                            group_num_matches = re.findall(r"(\$[0-9]+[LU]?)", value)
                            if group_num_matches is not None:
                                for g in group_num_matches:
                                    no_dollar = g[1:]
                                    case = ""
                                    if no_dollar[-1] == "U":
                                        case = "upper"
                                        no_dollar = no_dollar[0:-1]
                                    elif no_dollar[-1] == "L":
                                        case = "lower"
                                        no_dollar = no_dollar[0:-1]
                                    group_num = int(no_dollar)
                                    try:
                                        group_value = result["groups"][group_num - 1]
                                        if case == "lower":
                                            group_value = group_value.lower()
                                        elif case == "upper":
                                            group_value = group_value.upper()
                                    except IndexError:
                                        sys.stderr.write("The action '" + action + "' refers to a missing regex bracket group '$" +
                                              str(group_num) + "'\n")
                                        sys.exit()
                                    group_str = str(group_num)
                                    if case == "lower":
                                        group_str += "L"
                                    elif case == "upper":
                                        group_str += "U"
                                    value = re.sub(r"\$" + group_str, group_value, value)
                            if add_val:
                                old_val = getattr(result[node_position],prop)
                                new_vals = sorted(value.split("|"))
                                new_vals_keys = [v.split("=")[0] for v in new_vals]
                                if old_val != "_" and isinstance(old_val,str):  # Some values already exist
                                    kv = []
                                    for ov in sorted(old_val.split("|")):
                                        if not ov.split("=")[0] in new_vals_keys:  # Else this needs to be overwritten
                                            kv.append(ov)
                                    kv += new_vals
                                    value = "|".join(sorted(kv,key=lambda x:x.lower()))
                                else:
                                    value = "|".join(new_vals)
                            elif subtract_val:
                                old_val = getattr(result[node_position],prop)
                                new_vals = sorted(value.split("|"))
                                new_vals_keys = [v.split("=")[0] for v in new_vals]
                                if old_val != "_":  # Some values already exist
                                    kv = []
                                    for ov in sorted(old_val.split("|")):
                                        if not ov.split("=")[0] in new_vals_keys:  # Else this needs to be overwritten
                                            kv.append(ov)
                                    value = "|".join(sorted(kv,key=lambda x:x.lower()))
                                    if value == "":
                                        value = "_"
                                else:
                                    value = "_"
                            elif concat_val:
                                old_val = getattr(result[node_position],prop)
                                new_vals = sorted(value.split("|"))
                                new_vals_keys = defaultdict(set)
                                for pair in new_vals:
                                    this_key, this_val = pair.split("=")
                                    new_vals_keys[this_key].add(this_val)
                                if old_val != "_" and isinstance(old_val,str):  # Some values already exist
                                    kv = []
                                    for ov in sorted(old_val.split("|")+new_vals):
                                        this_key, this_val = ov.split("=")
                                        if this_key not in new_vals_keys:  # Else this needs to be overwritten
                                            kv.append(ov)
                                        else:
                                            kv.append(this_key + "=" + ",".join(sorted(list(new_vals_keys[this_key].union(set(this_val.split(",")))))))
                                    value = "|".join(sorted(kv,key=lambda x:x.lower()))
                                else:
                                    value = "|".join(new_vals)

                            if prop == "edep":
                                if value == "":  # Set empty edep
                                    result[node_position].edep = []
                                elif len(result[node_position].edep) == 0:
                                    result[node_position].edep.append([None,value])
                                    sys.stderr.write("WARN: added an enhanced label before adding its edge in transformation:\n" +
                                                     str(transformation) + "\n")
                                else:  # There is an edge waiting for a label
                                    # Get first edge with None label (FIFO)
                                    try:
                                        index = [i for i, dep in enumerate(result[node_position].edep) if dep[-1] is None][0]
                                    except IndexError:  # All are filled, overwrite top of stack
                                        if add_val:  # += instruction for edep means add without overwriting
                                            continue  # An edge already exists between these nodes and the op is +=, skip this action
                                        else:
                                            index = -1
                                    if add_val:
                                        test_parent = result[node_position].edep[index][0]
                                        if any([x[0]==test_parent and x[1] is not None and not (x[0] == result[node_position].head and x[1] == result[node_position].func) for x in result[node_position].edep]):
                                            # A distinct edep, non-None edge already exists lower than a None labeled edge
                                            # and the op is +=, skip this action and remove the None edge
                                            result[node_position].edep = [x for x in result[node_position].edep if x[1] is not None]
                                            continue
                                    result[node_position].edep[index][1] = value
                                    # Remove all other edeps connecting the same two tokens
                                    this_edep_parent = result[node_position].edep[index][0]
                                    result[node_position].edep = [x for x in result[node_position].edep if x[0] != this_edep_parent or x[1] == value]
                                    result[node_position].head2 = "_"  # Remove any explicit head2 info to accommodate new edeps; it will be generated from edom
                            elif prop == "edom":
                                if "||" in value:
                                    h, rel = value.split("||", 1)
                                    new_rels = []
                                    for dom in result[node_position].edep:
                                        if dom[0] != h:  # Leave out any existing edeps with the same head unless they are substrings
                                            new_rels.append(dom)
                                        elif dom[1].startswith(rel) or rel.startswith(dom[1]):
                                            new_rels.append(dom)
                                    new_rels.append([h, rel])
                                    #result[node_position].edep.append([h, rel])
                                    result[node_position].edep = new_rels
                                else:
                                    sys.stderr.write("WARN: skipped attempt to write edom; value does not follow the format HEAD||EDEP (e.g. 8.0||nsubj:xsubj)\n")
                            elif prop == "ehead":
                                result[node_position].edep.append([value, None])
                            else:
                                setattr(result[node_position], prop, value)
                    else:
                        if ">" in action:  # Binary instruction; head relation
                            operator = ">"
                        elif "~" in action:
                            operator = "~"
                        else:
                            continue
                        node1 = int(action.split(operator)[0].replace("#", ""))
                        node2 = int(action.split(operator)[1].replace("#", ""))
                        tok1 = result[node1]
                        tok2 = result[node2]
                        if tok1 != tok2:
                            if operator == ">":
                                tok2.head = tok1.id
                            else:
                                if tok1.id not in [e[0] for e in tok2.edep] or True:  # Experimental: remove True to disallow overwriting edeps
                                    # This is a new enhanced edge
                                    if len(tok2.edep) == 0:
                                        tok2.edep.append([tok1.id, None])  # Add an enhanced edge, with a pending label None
                                    elif tok2.edep[-1][0] is None:  # There is an existing label with a pending parent
                                        tok2.edep[-1][0] = tok1.id
                                    else:
                                        tok2.edep.append([tok1.id, None])  # Add an enhanced edge, with a pending label None

    def add_variable(self, key, value):
        self.variables[key] = value

    def add_transformation(self, *args, **kwargs):
        """
        Flexible function for adding transformations to an imported DepEdit object, rather than reading them from a configuration file

        :param args: a string specifying a transformation line (three specs separated by two tabs), or a tuples of such strings
        :param kwargs: alternatively, a dictionary supplying three lists for the keys: 'nodes', 'rels', 'actions'
        :return: void
        """

        if len(args)>0:  # Single string transformation(s)

            if not isinstance(args,list): # Make list out of input tuple
                args = list(arg for arg in args)
                if isinstance(args[0], list):  # Flatten list in case user provided a list
                    args = args[0]
            for transformation_string in args:
                try:
                    self.user_transformation_counter += 1
                    user_line_number = "u" + str(self.user_transformation_counter)
                    new_transformation = Transformation(transformation_string,user_line_number, self)
                    self.transformations.append(new_transformation)
                except:
                    raise IOError("Invalid transformation - must be a string")
        elif "nodes" in kwargs and "rels" in kwargs and "actions" in kwargs:
            if not (isinstance(kwargs["nodes"], list) and (isinstance(kwargs["nodes"], list)) and (isinstance(kwargs["nodes"], list))):
                raise IOError("Invalid transformation - must supply three lists in values for keys: nodes, rels, actions")
            else:
                node_clause = ";".join(kwargs["nodes"])
                rel_clause = ";".join(kwargs["rels"])
                actions_clause = ";".join(kwargs["actions"])
                transformation_string = "\t".join([node_clause,rel_clause,actions_clause])

                self.user_transformation_counter += 1
                user_line_number = "u" + str(self.user_transformation_counter)
                new_transformation = Transformation(transformation_string, user_line_number, self)
                self.transformations.append(new_transformation)

    def serialize_output_tree(self, tokens, tokoffset, enhanced=False):
        def order_edep(multi_edep_string):
            # edep with | must be ordered by ehead without duplicates
            parts = multi_edep_string.split("|")
            d = {}
            # sort by len, meaning we prefer longer edeps for duplicates with same ehead
            for p in sorted(parts,key=lambda x: len(x)):
                if ":" in p:
                    eh, ed = p.split(":",1)
                    d[eh] = ed
                else:  # Non-UD edeps field, warn and return naive sort
                    sys.stderr.write("WARN: Non-standard value in column 9: enhanced dependencies should contain ':'\n")
                    sys.stderr.write(multi_edep_string + "\n")
                    return "|".join(sorted(parts))
            try:
                sorted_keys = sorted(list(iterkeys(d)), key=lambda x: float(x))
            except ValueError:
                sys.stderr.write("WARN: Non-numeric enhanced head in column 9:\n")
                sys.stderr.write(multi_edep_string + "\n")
                return "|".join(sorted(parts))
            vals = [":".join([k, d[k]]) for k in sorted_keys]
            return "|".join(vals)

        output_tree_lines = []
        for tok in tokens:
            if tok.is_super_tok:
                if self.kill in ["both", "supertoks"]:
                    continue
                tok_head_string = tok.head
                tok_id = tok.id
            elif tok.head == "0":
                tok_head_string = "0"
                tok_id = str(Decimal(tok.id) - tokoffset)
            else:
                tok_head_string = str(Decimal(tok.head) - tokoffset)
                tok_id = str(Decimal(tok.id) - tokoffset)
            if tok.head2 not in ["","_"]:  # explicitly set head2 field
                tok_ehead_string = tok.head2
            elif len(tok.edep) == 0:
                tok_ehead_string = "_"
                tok.head2 = "_"
            else:
                tok_ehead_string = [":".join([str(Decimal(ehead[0]) - tokoffset).replace(".0", ""),ehead[1]]) for ehead in tok.edep]
                tok_ehead_string = "|".join(tok_ehead_string)
                if "-" in tok_ehead_string:
                    tok_ehead_string = re.sub(r'-[0-9]+(?=:)', '0', tok_ehead_string)
                tok_ehead_string = order_edep(tok_ehead_string)
            tok_id = tok_id.replace(".0", "")
            tok_head_string = tok_head_string.replace(".0", "")
            if "." in tok_id:
                tok_head_string = "_"
            fields = [tok_id, tok.text, tok.lemma, tok.pos, tok.cpos, tok.morph, tok_head_string, tok.func]
            if self.input_mode != "8col":
                if enhanced or tok.edep != []:
                    if tok.edep != []:
                        tok.head2 = tok_ehead_string
                    elif tok.head2 == "_":
                        tok.head2 = ":".join([tok_head_string, tok.func])  # Default edep column contents
                func2 = tok.func2 if len(tok.func2) > 0 else "_"
                fields += [tok.head2, func2]
            output_tree_lines.append("\t".join(fields))
        return output_tree_lines

    def make_sent_id(self, sent_id):
        return self.docname + "-" + str(sent_id)

    def make_sent_text(self, sentence_tokens):
        toks = []
        word = ""
        skip = 0
        super_space_after = True
        for tok in sentence_tokens:
            if tok.is_super_tok:
                word = tok.text
                start, end = tok.num.split("-")
                skip = int(end) - int(start) + 1
                if "SpaceAfter=No" in tok.func2:
                    super_space_after = False
            if tok.id.endswith(".0"):
                if skip > 0:
                    skip -= 1
                    if skip == 0:
                        if "SpaceAfter=No" not in tok.func2 and super_space_after:
                            word += " "
                        toks.append(word)
                        super_space_after = True
                    continue
                word = tok.text
                if "SpaceAfter=No" not in tok.func2:
                    word += " "
                toks.append(word)
        return "".join(toks)

    def run_depedit(self, infile, filename="file", sent_id=False, docname=False, stepwise=False, enhanced=False, sent_text=False):

        children = defaultdict(list)
        child_funcs = defaultdict(list)
        conll_tokens = [0]
        self.input_mode = "10col"
        self.docname = filename
        tokoffset = supertok_offset = sentlength = supertok_length = 0
        output_lines = []
        sentence_lines = []
        current_sentence = Sentence(sent_num=1, depedit_object=self, docname=filename)

        def _process_sentence(stepwise=False, enhanced=False):
            current_sentence.length = sentlength
            conll_tokens[-1].position = "last"
            sentence_tokens = conll_tokens[tokoffset + supertok_offset + 1:]
            insertion_offset = self.process_sentence(sentence_tokens,stepwise=stepwise)
            if sent_id:
                #output_lines.append(self.make_sent_id(current_sentence.sent_num))
                current_sentence.annotations["sent_id"] = self.make_sent_id(current_sentence.sent_num)
            if sent_text:
                current_sentence.annotations["text"] = self.make_sent_text(sentence_tokens)
            transformed = current_sentence.print_annos() + self.serialize_output_tree(sentence_tokens, tokoffset, enhanced=enhanced)
            output_lines.extend(transformed)

        # Check if DepEdit has been fed an unsplit string programmatically
        if isinstance(infile, str):
            infile = infile.splitlines()

        in_data = []
        for myline in infile:
            in_data.append(myline)
            myline = myline.strip()
            if sentlength > 0 and "\t" not in myline:
                _process_sentence(stepwise=stepwise, enhanced=enhanced)
                sentence_lines = []
                tokoffset += sentlength
                supertok_offset += supertok_length
                current_sentence = Sentence(sent_num=current_sentence.sent_num + 1,tokoffset=tokoffset, depedit_object=self, docname=filename)
                sentlength = supertok_length = 0
            if myline.startswith("#"):  # Preserve comment lines unless kill requested
                if self.kill not in ["comments", "both"] and "=" not in myline:
                    current_sentence.input_annotations[myline[1:].strip()] = ""
                if "=" in myline:
                    key, val = myline[1:].split("=",1)
                    current_sentence.input_annotations[key.strip()] = val.strip()
            elif not myline:
                output_lines.append("")
            elif myline.find("\t") > 0:  # Only process lines that contain tabs (i.e. conll tokens)
                sentence_lines.append(myline.strip())
                cols = myline.split("\t")
                if "-" in cols[0]:  # potential conllu super-token, just preserve
                    super_tok = True
                    tok_id = cols[0]
                    head_id = cols[6]
                else:
                    super_tok = False
                    tok_id = str(float(cols[0]) + tokoffset)
                    if cols[6] == "_":
                        if not self.quiet and "." not in cols[0]:
                            sys.stderr.write("DepEdit WARN: head not set for token " + tok_id + " in " + filename + "\n")
                        head_id = str(0 + tokoffset)
                    else:
                        head_id = str(float(cols[6]) + tokoffset)
                args = (tok_id,) + tuple(cols[1:6]) + (head_id, cols[7])
                if len(cols) > 8:
                    # Collect token from line; note that head2 is parsed as a string, often "_" for monoplanar trees
                    args += (cols[8], cols[9])
                else:  # Attempt to read as 8 column Malt input
                    args += (cols[6], cols[7])
                    self.input_mode = "8col"
                args += (cols[0], [], "mid", super_tok, tokoffset)
                this_tok = ParsedToken(*args)
                if cols[0] == "1" and not super_tok:
                    this_tok.position = "first"
                this_tok.sentence = current_sentence
                conll_tokens.append(this_tok)
                if super_tok:
                    supertok_length += 1
                else:
                    sentlength += 1
                    children[str(float(head_id) + tokoffset)].append(tok_id)
                    child_funcs[(float(head_id) + tokoffset)].append(cols[7])

        if sentlength > 0:  # Possible final sentence without trailing new line
            _process_sentence(stepwise=stepwise, enhanced=enhanced)

        if docname:
            newdoc = '# newdoc id = ' + self.docname
            output_lines.insert(0, newdoc)

        # Trailing whitespace
        rev = "".join(in_data)[::-1]
        white = re.match(r'\s*',rev).group()

        return "\n".join(output_lines).strip() + white


def main():

    depedit_version = "DepEdit V" + __version__

    parser = argparse.ArgumentParser(prog=None if globals().get('__spec__') is None else 'python -m {}'.format(__spec__.name.partition('.')[0]))
    parser.add_argument('file', action="store",
                        help="Input single file name or glob pattern to process a batch (e.g. *.conll10)")
    parser.add_argument('-c', '--config', action="store", dest="config", default="config.ini",
                        help="Configuration file defining transformation")
    parser.add_argument('-d', '--docname', action="store_true", dest="docname",
                        help="Begin output with # newdoc id =...")
    parser.add_argument('-t', '--text', action="store_true", help="Add # text =...")
    parser.add_argument('-s', '--sent_id', action="store_true", dest="sent_id", help="Add running sentence ID comments")
    parser.add_argument('--enhanced', action="store_true", help="Add enhanced dependencies column when absent in input")
    parser.add_argument('-k', '--kill', action="store", choices=["supertoks", "comments", "both"],
                        help="Remove supertokens or commments from output")
    parser.add_argument('-q', '--quiet', action="store_true", dest="quiet", help="Do not output warnings and messages")
    parser.add_argument('--stepwise', action="store_true", help="Output sentence repeatedly after each step (useful for debugging)")
    parser.add_argument('--time', action="store_true", help="Print time taken to process file(s)")
    group = parser.add_argument_group('Batch mode options')
    group.add_argument('-o', '--outdir', action="store", dest="outdir", default="",
                       help="Output directory in batch mode")
    group.add_argument('-e', '--extension', action="store", dest="extension", default="",
                       help="Extension for output files in batch mode")
    group.add_argument('-i', '--infix', action="store", dest="infix", default=".depedit",
                       help="Infix to denote edited files in batch mode (default: .depedit)")
    parser.add_argument('--version', action='version', version=depedit_version)
    options = parser.parse_args()

    if options.extension.startswith("."):  # Ensure user specified extension does not include leading '.'
        options.extension = options.extension[1:]
    try:
        config_file = io.open(options.config, encoding="utf8")
    except IOError:
        sys.stderr.write("\nConfiguration file not found (specify with -c or use the default 'config.ini')\n")
        sys.exit()
    depedit = DepEdit(config_file=config_file, options=options)
    if sys.platform == "win32":  # Print \n new lines in Windows
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    files = glob(options.file)
    if len(files) == 0:
        sys.stderr.write("File(s) not found:\n" + options.file)
        sys.exit()
    for filename in files:
        infile = io.open(filename, encoding="utf8")
        basename = os.path.basename(filename)
        docname = basename[:basename.rfind(".")] if options.docname or options.sent_id else filename
        output_trees = depedit.run_depedit(infile, docname, sent_id=options.sent_id, docname=options.docname,
                                           stepwise=options.stepwise, enhanced=options.enhanced, sent_text=options.text)
        if len(files) == 1:
            # Single file being processed, just print to STDOUT
            if sys.version_info[0] < 3:
                print(output_trees.encode("utf-8"))
            else:
                sys.stdout.buffer.write(output_trees.encode("utf8"))
        else:
            # Multiple files, add '.depedit' or other infix from options before extension and write to file
            if options.outdir != "":
                if not options.outdir.endswith(os.sep):
                    options.outdir += os.sep
            outname = options.outdir + basename
            if "." in filename:
                extension = outname[outname.rfind(".") + 1:]
                if options.extension != "":
                    extension = options.extension
                outname = outname[:outname.rfind(".")]
                outname += options.infix + "." + extension
            else:
                outname += options.infix + "." + options.extension if options.extension else options.infix
            if sys.version_info[0] < 3:
                with io.open(outname, 'wb') as f:
                    f.write(output_trees.encode("utf-8"))
            else:
                with io.open(outname, 'w', encoding="utf8", newline="\n") as f:
                    f.write(output_trees)


if __name__ == "__main__":
    if "--time" in sys.argv:
        import time
        start_time = time.time()

    main()

    if "--time" in sys.argv:
        sys.stderr.write("--- %s seconds ---" % (time.time() - start_time))
