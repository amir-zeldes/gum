#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
DepEdit - A simple configurable tool for manipulating dependency trees

Input: CoNLL10 or CoNLLU (10 columns, tab-delimited, blank line between sentences, comments with pound sign #)

Author: Amir Zeldes
"""

from __future__ import print_function
import argparse
import re
from io import open as io_open
from copy import copy, deepcopy
import sys, os
from collections import defaultdict
from glob import glob
from six import iteritems

__version__ = "2.1.1"


def escape(string,symbol_to_mask,border_marker):
	inside = False
	output = ""
	# Scan the string looking for border or symbol to mask
	for char in string:
		if char == border_marker:
			inside = not inside
		if char == symbol_to_mask and inside:
			output += "%%%%%"
		else:
			output += char
	return output


class ParsedToken:
	def __init__(self, tok_id, text, lemma, pos, cpos, morph, head, func, head2, func2, num, child_funcs, position, is_super_tok=False):
		self.id = tok_id
		self.text = text
		self.pos = pos
		self.cpos = cpos
		self.lemma = lemma
		self.morph = morph
		self.head = head
		self.func = func
		self.head2 = head2
		self.func2 = func2
		self.num = num
		self.child_funcs = child_funcs
		self.position = position
		self.is_super_tok = is_super_tok

	def __repr__(self):
		return str(self.text) + " (" + str(self.pos) + "/" + str(self.lemma) + ") " + "<-" + str(self.func)


class Sentence:

	def __init__(self,sentence_string=""):
		self.sentence_string = sentence_string
		self.length = 0
		self.annotations = {}

	def print_annos(self):
		out_string = ""
		for (key,val) in iteritems(self.annotations):
			out_string += "# " + key + "=" + val + "\n"
		return out_string


class Transformation:

	def parse_transformation(self, transformation_text):
		if transformation_text.count("\t") < 2:
			return None
		else:
			split_trans = transformation_text.split("\t")
			definition_string = split_trans[0]
			relation_string = split_trans[1]
			action_string = split_trans[2]
			relation_string = self.normalize_shorthand(relation_string)
			action_string = self.normalize_shorthand(action_string)
			definition_string = escape(definition_string,";","/")
			definitions_list = definition_string.split(";")
			escaped_definitions = []
			for _def in definitions_list:
				escaped_definitions.append(_def.replace("%%%%%",";"))
			definitions = []
			for def_index, esc_string in enumerate(escaped_definitions):
				definitions.append(DefinitionMatcher(esc_string,def_index + 1))
			relations = relation_string.split(";")
			actions = action_string.strip().split(";")
			aliased_actions = []
			for action in actions:
				aliased_actions.append(self.handle_aliases(action))
			return [definitions, relations, aliased_actions]

	@staticmethod
	def handle_aliases(orig_action):
		orig_action = orig_action.replace(":form=",":text=")
		orig_action = orig_action.replace(":upostag=", ":pos=")
		orig_action = orig_action.replace(":xpostag=",":cpos=")
		orig_action = orig_action.replace(":feats=",":morph=")
		orig_action = orig_action.replace(":deprel=",":func=")
		orig_action = orig_action.replace(":deps=",":head2=")
		orig_action = orig_action.replace(":misc=",":func2=")
		return orig_action

	@staticmethod
	def normalize_shorthand(criterion_string):
		criterion_string = criterion_string.replace('.*', '.1,1000')
		temp = ""
		while temp != criterion_string:
			temp = criterion_string
			criterion_string = re.sub(r'(#[0-9]+)(>|\.(?:[0-9]+(?:,[0-9]+)?)?)(#[0-9]+)(>|\.(?:[0-9]+(?:,[0-9]+)?)?)', r'\1\2\3;\3\4', criterion_string)
		return criterion_string

	def __init__(self, transformation_text, line):
		instructions = self.parse_transformation(transformation_text)
		if instructions is None:
			message = "Depedit says: error in configuration file\n"
			message += "Malformed instruction on line " + str(line) + " (instruction lines must contain exactly two tabs)\n"
			sys.stderr.write(message)
			sys.exit()
		self.definitions = instructions[0]
		self.relations = instructions[1]
		self.actions = instructions[2]
		self.line = line

	def validate(self):
		report = ""
		for definition in self.definitions:
			node = definition.def_text
			node = escape(node,"&","/")
			criteria = node.split("&")
			criteria = (_crit.replace("%%%%%","&") for _crit in criteria)
			for criterion in criteria:
				if re.match("(text|pos|cpos|lemma|morph|func|head|func2|head2|num|form|upostag|xpostag|feats|deprel|deps|misc)!?=/[^/=]*/",criterion) is None:
					if re.match(r"position!?=/(first|last|mid)/",criterion) is None:
						report+= "Invalid node definition in column 1: " + criterion
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
					#if not re.match(r"#[0-9]+((>|\.([0-9]+(,[0-9]+)?)?)#[0-9]+)+",criterion):
					if not re.match(r"(#[0-9]+((>|\.([0-9]+(,[0-9]+)?)?)#[0-9]+)+|#[0-9]+:(text|pos|cpos|lemma|morph|func|head|func2|head2|num|form|upostag|xpostag|feats|deprel|deps|misc)==#[0-9]+)", criterion):
						report += "Column 2 relation setting invalid criterion: " + criterion + "."
		for action in self.actions:
			commands = action.split(";")
			for command in commands:
				if not re.match(r"(#[0-9]+>#[0-9]+|#[0-9]+:(func|lemma|text|pos|cpos|morph|head|head2|func2|num|form|upostag|xpostag|feats|deprel|deps|misc)=[^;]*)$",command):  # Node action
					if not re.match(r"#S:[A-Za-z_]+=[A-Za-z_]+$|last$", command):  # Sentence annotation action or quit
						report += "Column 3 invalid action definition: " + command + " and the action was " + action
		return report


class DefinitionMatcher:

	def __init__(self, def_text, def_index):
		self.def_text = escape(def_text,"&","/")
		self.def_index = def_index
		self.groups = []
		self.defs = []

		def_items = self.def_text.split("&")
		for def_item in def_items:
			def_item = def_item.replace("%%%%%","&")
			criterion = def_item.split("=",1)[0]
			if criterion[-1] == "!":
				negative_criterion = True
				criterion = criterion[:-1]
			else:
				negative_criterion = False

			def_value = def_item.split("=",1)[1][1:-1]

			# Ensure regex is anchored
			if def_value[0] != "^":
				def_value = "^" + def_value
			if def_value[-1] != "$":
				def_value += "$"
			self.defs.append(Definition(criterion,def_value,negative_criterion))

	def __repr__(self):
		return "#" + str(self.def_index) + ": " + self.def_text

	def match(self, token):
		potential_groups = []
		for def_item in self.defs:
			tok_value = getattr(token,def_item.criterion)
			match_obj = def_item.match_func(def_item,tok_value)

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

	def __init__(self, criterion, value, negative=False):
		# Handle conllu criterion aliases:
		if criterion == "form":
			criterion = "text"
		elif criterion == "upostag":
			criterion = "pos"
		elif criterion == "xpostag":
			criterion = "cpos"
		elif criterion == "feats":
			criterion = "morph"
		elif criterion == "deprel":
			criterion = "func"
		elif criterion == "deps":
			criterion = "head2"
		elif criterion == "misc":
			criterion = "func2"

		self.criterion = criterion
		self.value = value
		self.match_type = ""
		self.compiled_re = None
		self.negative = negative
		self.set_match_type()

	def set_match_type(self):
		value = self.value[1:-1]

		if self.value == "^.*$" and not self.negative:
			self.match_func = self.return_true
		elif re.escape(value) == value: # No regex operators within  expression
			if self.negative:
				self.match_func = self.return_exact_negative
			else:
				self.match_func = self.return_exact
			self.value = value
		else:  # regex
			self.compiled_re = re.compile(self.value)
			if self.negative:
				self.match_func = self.return_regex_negative
			else:
				self.match_func = self.return_regex

	@staticmethod
	def return_exact(definition,test_val):
		return test_val == definition.value

	@staticmethod
	def return_exact_negative(definition,test_val):
		return test_val != definition.value

	@staticmethod
	def return_regex(definition,test_val):
		return definition.compiled_re.search(test_val)

	@staticmethod
	def return_regex_negative(definition,test_val):
		return definition.compiled_re.search(test_val) is None

	@staticmethod
	def return_true(definition,test_val):
		return True


class Match:

	def __init__(self, def_index, token, groups):
		self.def_index = def_index
		self.token = token
		self.groups = groups

	def __repr__(self):
		return "#" + str(self.def_index) + ": " + self.token.__repr__


class DepEdit():

	def __init__(self, config_file=""):
		self.transformations = []
		self.user_transformation_counter = 0
		if not config_file == "":
			self.read_config_file(config_file)

	def read_config_file(self, config_file, clear_transformations = False):
		"""
		Function to read configuration file. Can be invoked after initialization.

		:param config_file: The file to read instructions from.
		:param clear_transformations: Whether to discard previous instructions.
		:return: void
		"""

		if clear_transformations:
			self.transformations = []
			self.user_transformation_counter = 0
		line_num = 0
		if isinstance(config_file, str):
			if sys.version_info[0] < 3:
				config_file = open(config_file).readlines()
			else:
				config_file = open(config_file,encoding="utf8").readlines()
		for instruction in config_file:
			line_num += 1
			if len(instruction)>0 and not instruction.startswith(";") and not instruction.startswith("#") and not instruction.strip() =="":
				self.transformations.append(Transformation(instruction, line_num))

		trans_report = ""
		for transformation in self.transformations:
			temp_report = transformation.validate()
			if temp_report != "":
				trans_report += "On line " + str(transformation.line) + ": " + temp_report +"\n"
		if len(trans_report) > 0:
			trans_report = "Depedit says: error in configuration file\n\n" + trans_report
			sys.stderr.write(trans_report)
			sys.exit()

	def process_sentence(self, conll_tokens, tokoffset, supertok_offset, transformations):
		for transformation in transformations:
			node_matches = defaultdict(list)
			for def_matcher in transformation.definitions:
				for token in conll_tokens[tokoffset+supertok_offset+1:]:
					if not token.is_super_tok:
						if def_matcher.match(token):
							node_matches[def_matcher.def_index].append(Match(def_matcher.def_index, token, def_matcher.groups))
			result_sets = []
			for relation in transformation.relations:
				found = self.matches_relation(node_matches, relation, result_sets)
				if not found:
					result_sets = []
			result_sets = self.merge_sets(result_sets,len(transformation.definitions),len(transformation.relations))
			self.add_groups(result_sets)
			if len(result_sets) > 0:
				for action in transformation.actions:
					retval = self.execute_action(result_sets, action)
					if retval == "last": # Explicit instruction to cease processing
						return self.serialize_output_tree(conll_tokens[tokoffset + supertok_offset + 1:], tokoffset)
		return self.serialize_output_tree(conll_tokens[tokoffset + supertok_offset + 1:], tokoffset)

	def matches_relation(self, node_matches, relation, result_sets):
		if len(relation) == 0:
			return
		elif "==" in relation:
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

		matches = defaultdict(list)

		hits=0
		if relation == "none": # Unary operation on one node
			node1 = 1
			for matcher1 in node_matches[node1]:
				tok1 = matcher1.token
				hits += 1
				result = {}
				matches[node1].append(tok1)
				result[node1] = tok1
				result["rel"] = relation
				result["matchers"] = [matcher1]
				result_sets.append(result)
		elif "==" in relation:
			node1 = relation.split(operator)[0]
			node2 = relation.split(operator)[1]
			node1=int(node1.replace("#", ""))
			node2=int(node2.replace("#", ""))

			for matcher1 in node_matches[node1]:
				tok1 = matcher1.token
				for matcher2 in node_matches[node2]:
					tok2 = matcher2.token
					if self.test_relation(tok1, tok2, field):
						result_sets.append({node1: tok1, node2: tok2, "rel": relation, "matchers": [matcher1, matcher2] })
						matches[node1].append(tok1)
						matches[node2].append(tok2)
						hits += 1

			for option in [node1,node2]:
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
					if self.test_relation(tok1, tok2, operator):
						result_sets.append({node1: tok1, node2: tok2, "rel": relation, "matchers": [matcher1, matcher2] })
						matches[node1].append(tok1)
						matches[node2].append(tok2)
						hits += 1

			for option in [node1,node2]:
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
	def test_relation(node1,node2,operator):
		if operator == ".":
			if int(float(node2.id)) == int(float(node1.id))+1:
				return True
			else:
				return False
		elif operator == ">":
			if int(float(node2.head)) == int(float(node1.id)):
				return True
			else:
				return False
		elif "." in operator:
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
			new_set = {}
			new_set["rels"] = []
			new_set["matchers"] = []
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
			if len(my_bin) == node_count + 2:
				if len(my_bin["rels"]) == rel_count:  # All required relations have been fulfilled
					solutions.append(my_bin)
				else:  # Some node pair has multiple relations, check that all are fulfilled
					for set_to_merge in sets:
						if set_to_merge["rel"] not in my_bin["rels"]: # This relation was missing
							node_ids = list((key) for key in set_to_merge if isinstance(key,int))
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
			self.merge_solutions(solution,merged_bins,rel_count)
		self.prune_merged_bins(merged_bins,rel_count)
		return merged_bins


	@staticmethod
	def merge_solutions(solution, merged, rel_count):
		merges_to_add = []
		if solution not in merged:
			merged.append(solution)
		if len(solution["rels"]) == rel_count:
			# This solution is completed
			pass
		else:
			# This is an incomplete solution, try to merge it into the merged solutions list
			for candidate in merged:
				if candidate != solution:
					for key in solution:
						if key != "rels" and key !="matchers":
							if key in candidate:  # This is a position, e.g. #1, that is also in the candidate in merged
								if solution[key] == candidate[key]:  # Check that they are compatible, e.g. #1 is the same token
									rels_in_candidate = (rel in candidate["rels"] for rel in solution["rels"])
									if not all(rels_in_candidate):
										rels = solution["rels"]+candidate["rels"]
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
		if len(merges_to_add)>0:
			for merge_to_add in merges_to_add:
				merged.append(merge_to_add)
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
	def merge_bins(bin1, bin2):
		"""
		Merge bins we know are compatible, e.g. bin1 has #1+#2 and bin2 has #2+#3

		:param bin1: a bin dictionary mapping indices to tokens, a list of relations 'rels' and matcher objects 'matchers'
		:param bin2: a bin dictionary mapping indices to tokens, a list of relations 'rels' and matcher objects 'matchers'
		:return: the merged bin with data from both input bins
		"""
		for matcher in bin1["matchers"]:
			skip = False
			for matcher2 in bin2["matchers"]:
				if matcher2.def_index == matcher.def_index:
					skip = True
			if not skip:
				bin2["matchers"].append(matcher)
		for key in bin1:
			if key != "rels":
				if key not in bin2:
					out_bin = copy(bin2)
					out_bin[key]= bin1[key]
					for rel in bin1["rels"]:
						new_rel = deepcopy(rel)
						out_bin["rels"] = bin2["rels"]+[new_rel]
					return out_bin


	@staticmethod
	def prune_merged_bins(merged_bins,rel_count):
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
				if len(matcher.groups) > 0:
					for group in matcher.groups:
						groups.append(group[0])
			result["groups"] = groups[:]

	@staticmethod
	def execute_action(result_sets, action_list):
		actions = action_list.split(";")
		for result in result_sets:
			if len(result) > 0:
				for action in actions:
					if action == "last":
						return "last"
					elif ":" in action:  # Unary instruction
						if action.startswith("#S:"):  # Sentence annotation instruction
							key_val = action.split(":")[1]
							key, val = key_val.split("=",1)
							result[1].sentence.annotations[key] = val
						else: # node instruction
							node_position = int(action[1:action.find(":")])
							property = action[action.find(":")+1:action.find("=")]
							value = action[action.find("=")+1:].strip()
							group_num_match = re.search(r"(\$[0-9]+[LU]?)",value)
							if group_num_match is not None:
								no_dollar = group_num_match.groups(0)[0][1:]
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
									sys.stderr.write("The action '" + action + "' refers to a missing regex bracket group '$" + str(group_num) + "'\n")
									sys.exit()
								value = re.sub(r"\$[0-9]+[LR]?",group_value,value)
							setattr(result[node_position],property,value)
					else:  # Binary instruction
						if ">" in action:  # Head relation
							operator = ">"
							node1 = int(action.split(operator)[0].replace("#", ""))
							node2 = int(action.split(operator)[1].replace("#", ""))
							tok1 = result[node1]
							tok2 = result[node2]
							if tok1 != tok2:
								tok2.head = tok1.id

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
					new_transformation = Transformation(transformation_string,user_line_number)
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
				new_transformation = Transformation(transformation_string, user_line_number)
				self.transformations.append(new_transformation)

	def serialize_output_tree(self,tokens, tokoffset):
		output_tree = ""
		for tok in tokens:
			if tok.is_super_tok:
				tok_head_string = tok.head
				tok_id = tok.id
			elif tok.head == "0":
				tok_head_string = "0"
				tok_id = str(float(tok.id) - tokoffset)
			else:
				tok_head_string = str(float(tok.head)-tokoffset)
				tok_id = str(float(tok.id)-tokoffset)
			# Only keep decimal ID component for non-0 ellipsis IDs, e.g. 10.1 - those tokens have normal head '_'
			tok_id = tok_id.replace(".0","")
			tok_head_string = tok_head_string.replace(".0","")
			if "." in tok_id:
				tok_head_string = "_"
			if self.input_mode == "8col":
				output_tree += tok_id+"\t"+tok.text+"\t"+tok.lemma+"\t"+tok.pos+"\t"+tok.cpos+"\t"+tok.morph+\
								"\t"+tok_head_string+"\t"+tok.func+"\n"
			else:
				output_tree += tok_id+"\t"+tok.text+"\t"+tok.lemma+"\t"+tok.pos+"\t"+tok.cpos+"\t"+tok.morph+\
								"\t"+tok_head_string+"\t"+tok.func+"\t"+tok.head2+"\t"+tok.func2+"\n"
		return output_tree

	def make_sent_id(self, sent_id):
		return "# sent_id = " + self.docname + "-" + str(sent_id) + "\n"

	def run_depedit(self, infile, filename="file", sent_id=False, docname=False):
		children = defaultdict(list)
		child_funcs = defaultdict(list)
		conll_tokens = []
		self.input_mode = "10col"
		self.docname = filename
		tokoffset = 0
		sent_num = 0
		supertok_offset = 0
		sentlength = 0
		supertok_length = 0

		conll_tokens.append(0)
		my_output = ""
		sentence_string = ""
		current_sentence = Sentence()

		# Check if DepEdit has been fed an unsplit string programmatically
		if isinstance(infile,str):
			infile = infile.split("\n")

		for myline in infile:
			if sentlength > 0 and "\t" not in myline:
				current_sentence.length = sentlength
				conll_tokens[-1].position = "last"
				transformed = self.process_sentence(conll_tokens, tokoffset, supertok_offset, self.transformations)
				transformed = current_sentence.print_annos() + transformed
				sent_num += 1
				if sent_id:
					my_output += self.make_sent_id(sent_num)
				my_output += transformed
				sentence_string = ""
				current_sentence = Sentence()
				if sentlength > 0:
					tokoffset += sentlength
					supertok_offset += supertok_length
				sentlength = 0
				supertok_length = 0
			if myline.startswith("#"):  # Preserve comment lines
					my_output += myline.strip() + "\n"
			elif myline.strip() == "":
					my_output += "\n"
			elif myline.find("\t") > 0:  # Only process lines that contain tabs (i.e. conll tokens)
				sentence_string += myline
				cols = myline.split("\t")
				if "-" in cols[0]: # potential conllu super-token, just preserve
					super_tok = True
					tok_id = cols[0]
					head_id = cols[6]
				else:
					super_tok = False
					tok_id = str(float(cols[0]) + tokoffset)
					if cols[6] == "_":
						sys.stderr.write("DepEdit WARN: head not set for token " + tok_id + " in " + filename + "\n")
						head_id = str(0 + tokoffset)
					else:
						head_id = str(float(cols[6]) + tokoffset)
				if len(cols) > 8:
					# Collect token from line; note that head2 is parsed as a string, which is often "_" for monoplanar trees
					this_tok = ParsedToken(tok_id, cols[1], cols[2], cols[3], cols[4], cols[5],head_id, cols[7].strip(), cols[8], cols[9].strip(), cols[0], [], "mid",super_tok)
				else:  # Attempt to read as 8 column Malt input
					this_tok = ParsedToken(tok_id,cols[1],cols[2],cols[3],cols[4],cols[5],head_id,cols[7].strip(),cols[6],cols[7].strip(),cols[0],[], "mid",super_tok)
					self.input_mode = "8col"
				if cols[0] == "1" and not super_tok:
					this_tok.position = "first"
				this_tok.sentence = current_sentence
				conll_tokens.append(this_tok)
				if not super_tok:
					sentlength += 1
					children[str(float(head_id) + tokoffset)].append(tok_id)
					child_funcs[(float(head_id) + tokoffset)].append(cols[7])
				else:
					supertok_length += 1

		if sentlength > 0:  # Possible final sentence without trailing new line
			current_sentence.length = sentlength
			conll_tokens[-1].position = "last"
			transformed = self.process_sentence(conll_tokens, tokoffset, supertok_offset, self.transformations)
			transformed = current_sentence.print_annos() + transformed
			sent_num += 1
			if sent_id:
				my_output += self.make_sent_id(sent_num)
			my_output += transformed

		if docname:
			newdoc = '# newdoc id = ' + self.docname + "\n"
			my_output = newdoc + my_output

		return my_output

if __name__ == "__main__":
	depedit_version = "DepEdit V" + __version__
	parser = argparse.ArgumentParser()
	parser.add_argument('file',action="store",help="Input single file name or glob pattern to process a batch (e.g. *.conll10)")
	parser.add_argument('-c', '--config', action="store", dest="config", default="config.ini", help="Configuration file defining transformation")
	parser.add_argument('-d', '--docname', action="store_true", dest="docname", help="Begin output with # newdoc id =...")
	parser.add_argument('-s', '--sent_id', action="store_true", dest="sent_id", help="Add running sentence ID comments")
	group = parser.add_argument_group('Batch mode options')
	group.add_argument('-o', '--outdir', action="store", dest="outdir", default="", help="Output directory in batch mode")
	group.add_argument('-e', '--extension', action="store", dest="extension", default="", help="Extension for output files in batch mode")
	group.add_argument('-i', '--infix', action="store", dest="infix", default=".depedit", help="Infix to denote edited files in batch mode (default: .depedit)")
	parser.add_argument('--version', action='version', version=depedit_version)
	options = parser.parse_args()

	if options.extension.startswith("."):  # Ensure user specified extension does not include leading '.'
		options.extension = options.extension[1:]

	try:
		config_file = io_open(options.config, encoding="utf8")
	except IOError:
		sys.stderr.write("\nConfiguration file not found (specify with -c or use the default 'config.ini')\n")
		sys.exit()
	depedit = DepEdit(config_file)

	if sys.platform == "win32":  # Print \n new lines in Windows
		import os, msvcrt
		msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

	files = glob(options.file)
	for filename in files:
		infile = io_open(filename, encoding="utf8")
		basename = os.path.basename(filename)
		if options.docname or options.sent_id:
			docname = basename[:basename.rfind(".")]
		else:
			docname = filename
		output_trees = depedit.run_depedit(infile,docname, sent_id=options.sent_id, docname=options.docname)
		if len(files) == 1:
			# Single file being processed, just print to STDOUT
			if sys.version_info[0] < 3:
				print(output_trees.encode("utf-8"))
			else:
				print(output_trees)
		else:
			# Multiple files, add '.depedit' or other infix from options before extension and write to file
			if options.outdir != "":
				if not options.outdir.endswith(os.sep):
					options.outdir += os.sep
			outname = options.outdir + basename
			if "." in filename:
				extension = outname[outname.rfind(".")+1:]
				if options.extension != "":
					extension = options.extension
				outname = outname[:outname.rfind(".")]
				outname += options.infix + "." + extension
			else:
				if options.extension == "":
					outname += options.infix
				else:
					outname += options.infix + "." + options.extension
			if sys.version_info[0] < 3:
				with open(outname,'wb') as f:
					f.write(output_trees.encode("utf-8"))
			else:
				with open(outname,'w',encoding="utf8") as f:
					f.write(output_trees.encode("utf-8"))

