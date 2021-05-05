#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,platform,os,io,re
from collections import OrderedDict, defaultdict
import ntpath
from glob import glob
from six import iteritems

PY2 = sys.version_info[0] < 3
script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
tsv_temp_dir = script_dir + "pepper" + os.sep + "tmp" + os.sep + "tsv" + os.sep + "GUM" + os.sep
if not os.path.exists(tsv_temp_dir):
	os.makedirs(tsv_temp_dir)

def equiv_tok(token):
	replacements = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "’": "'", "—": "-", "&quot;": '"', "&apos;": "'", "(":"-LRB-", ")":"-RRB-", "…":"...",
					"“":'"',"”":'"','–':"-", "é":"e","É":"E","á":"a","ó":"o","í":"i","č":"c"}

	for find, replace in iteritems(replacements):
		token = token.replace(find, replace)
	return token


def unescape_xml(token):
	replacements = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "&quot;": '"', "&apos;": "'"}

	for find, replace in iteritems(replacements):
		token = token.replace(find, replace)
	return token


def fix_tsv(gum_source, gum_target, reddit=False, genitive_s=False):
	file_list = []
	files_ = glob(gum_source + "tsv" + os.sep + "*.tsv")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		file_list.append(file_)

	outdir = gum_target + "coref" + os.sep + "tsv" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for docnum, filename in enumerate(file_list):
		tt_file = filename.replace(".tsv", ".xml").replace("tsv","xml")
		sys.stdout.write("\t+ " + " "*60 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Adjusting borders for " + ntpath.basename(filename) + "\r")
		fix_file(filename,tt_file,outdir,genitive_s=genitive_s)

	print("o Adjusted " + str(len(file_list)) + " WebAnno TSV files" + " " * 40)


### begin functions for genitive s fix
# file i/o layer
def read_tsv_lines(tsv_path):
	"""
	Reads all lines from a WebAnno TSV file
	:param tsv_path: Path to a single WebAnno TSV file
	:return: A list of lines from the file, newline characters included
	"""
	if sys.version_info[0] < 3:
		infile = open(tsv_path, 'rb')
	else:
		infile = open(tsv_path, 'r', encoding="utf8")

	lines = infile.readlines()
	infile.close()
	return lines


def read_xml_lines(xml_path):
	"""
	Reads all lines from a TreeTagger XML file
	:param xml_path: Path to a single XML file
	:return: A list of lines from the file, newline characters included
	"""
	if PY2:
		infile = open(xml_path, 'rb')
	else:
		infile = open(xml_path, 'r', encoding="utf8")

	lines = infile.readlines()
	infile.close()
	return lines


def format_entities(entities):
	"""
	Turn a list of parsed WebAnno entities into a string.
	:param entities: A list of (parsed) entities from a single WebAnno line
	:return: A string representing the entities
	"""
	strs = []
	for entity in entities:
		if entity['id']:
			strs.append(entity['type'] + '[' + str(entity['id']) + ']')
		else:
			strs.append(entity['type'])

	if strs:
		return "|".join(strs)
	else:
		return "_"


def format_relations(relations):
	"""
	Turn a list of parsed WebAnno relations into a string.
	:param relations: A list of (parsed) relations from a single WebAnno line
	:return: A string representing the relations
	"""
	strs = []
	for relation in relations:
		rstr = relation['src_token']

		if not (relation['src'] is None and relation['dest'] is None):
			rstr += '['
			rstr += str(relation['src']) if relation['src'] is not None else '0'
			rstr += '_'
			rstr += str(relation['dest']) if relation['dest'] is not None else '0'
			rstr += ']'

		strs.append(rstr)

	if strs:
		return "|".join(strs)
	else:
		return "_"


def format_attr(entities, attr="infstat"):
	"""
	Turn a list of parsed WebAnno entities into a string representing their annotations.
	:param entities: A list of (parsed) entities from a single WebAnno line
	:param attr: A string giving the name of the annotation to format
	:return: A string representing the annotations of those entities.
	"""
	strs = []
	for entity in entities:
		if entity['id']:
			strs.append(entity[attr] + '[' + str(entity['id']) + ']')
		else:
			strs.append(entity[attr])

	if strs:
		return "|".join(strs)
	else:
		return "_"


def serialize_tsv_lines(lines, parsed_lines, tsv_path, outdir, as_string=False):
	"""
	Writes the in-memory representation of the WebAnno TSV file to disk.
	:param lines: The original lines of the TSV file
	:param parsed_lines: The in-memory representation of the TSV file that was manipulated
	:param tsv_path: The path of the input file
	:param outdir: The directory to write the output file in. The basename of tsv_path is appended to this.
	:param as_string: Return tsv string rather than writing to file
	"""

	tsv_path = outdir + os.sep + os.path.basename(tsv_path)
	output = []
	if not as_string:
		if sys.version_info[0] < 3:
			outfile = open(tsv_path, 'wb')
		else:
			outfile = io.open(tsv_path, 'w', encoding="utf8",newline="\n")

	i = 0
	for line in lines:
		if "\t" not in line:
			if as_string:
				output.append(line)
			else:
				outfile.write(line)
		else:
			cols = line.split("\t")
			cols[3] = format_entities(parsed_lines[i]['entities'])
			cols[4] = format_attr(parsed_lines[i]['entities'],attr="infstat")
			cols[5] = format_attr(parsed_lines[i]['entities'],attr="identity")
			cols[5] = cols[5].replace(" ","_").replace("(","%28").replace(")","%29").replace(",","%2C")
			cols[-2] = format_relations(parsed_lines[i]['relations'])
			if as_string:
				output.append("\t".join(cols))
			else:
				outfile.write("\t".join(cols))
			i += 1

	if as_string:
		return "\n".join(output)
	else:
		outfile.close()


# string layer
def extract_from_bracket(s):
	"""
	Extracts the label and bracketed content from an entity representation
	:param s: something like "xyz[3]"
	:return: 2-tuple like ("xyz", "3"), or ("xyz", None) if input is 'xyz"
	"""
	bracket_index = s.rfind("[")
	if bracket_index < 0:
		return s, None
	else:
		return s[:bracket_index], s[bracket_index+1:-1]


def parse_tsv_line(line):
	"""
	Parse a line from a WebAnno TSV file into a dictionary representing a subset of its contents.
	:param line: A WebAnno TSV file line
	:return: A dictionary with keys 'token_id', 'entities', 'token', and 'relations'.
	"""
	line = [None if x == "_" else x for x in line]

	already_seen_single_tok_entity = False
	entities = []
	if line[3]:
		# parse something like "person[3]" or "person"
		information_statuses = line[4].split("|")
		identities = line[5].split("|") if line[5] is not None else [None for i, _ in enumerate(information_statuses)]
		for i, entity in enumerate(line[3].split("|")):
			entity_type, entity_id = extract_from_bracket(entity)
			entity_id = int(entity_id) if entity_id else None

			entity_infstat, _ = extract_from_bracket(information_statuses[i])
			ident_field = "_"
			for ident_anno in identities:  # See if an identity anno corresponds to the currently processed entity ID
				if ident_anno is not None:
					entity_identity, ident_id = extract_from_bracket(ident_anno)
					entity_identity = entity_identity.replace(" ","_").replace("(","%28").replace(")","%29").replace(",","%2C")
				else:
					continue
				if int(ident_id) == entity_id:
					ident_field = entity_identity
					break
			entities.append({'id': entity_id, 'type': entity_type, 'infstat': entity_infstat, "identity": ident_field})

			# assumption: there should be at most one single-token entity on any given line
			assert not (entity_id is None and already_seen_single_tok_entity)
			if entity_id is None:
				already_seen_single_tok_entity = True

	relations = []
	if line[7]:
		rel_types = line[6].split("|")
		# parse something like "5-1[20_10]",
		# i.e. "this line is where entity 10 begins, and it is related to entity 20 beginning at token 5-1
		for i, relation in enumerate(line[7].split("|")):
			src_token, src_dest = extract_from_bracket(relation)
			if src_dest:
				src, dest = src_dest.split("_")
				src = int(src) if src != "0" else None
				dest = int(dest) if dest != "0" else None
			else:
				src = None
				dest = None
			dst_token = line[0]
			relations.append({'src': src,
							  'dest': dest,
							  'src_token': src_token,
							  'dst_token': dst_token,
							  'rel_type': rel_types[i]})

	return {'token_id': line[0],
			'token': line[2],
			'entities': entities,
			'relations': relations}


def parse_tsv_lines(lines):
	"""
	Parse all WebAnno TSV lines
	:param lines: A list of unprocessed TSV lines
	:return: a list of dictionaries representing a subset of their contents.
	"""
	return [parse_tsv_line(raw_line.rstrip().split("\t"))
			for raw_line in lines
			if "\t" in raw_line]


def enrich_tsv_representation_with_pos(parsed_lines, xml_path):
	"""
	Add POS tags to the in-memory TSV representation
	:param parsed_lines: Parsed list of TSV lines, from parse_tsv_lines
	:param xml_path: The path to the TSV file's corresponding TreeTagger XML file
	"""
	lines = read_xml_lines(xml_path)

	i = 0
	for line in lines:
		if not line.startswith("<") and len(line) > 0 and "\t" in line: # token
			cols = line.split("\t")
			token = cols[0]
			pos = cols[1]

			tsv_line = parsed_lines[i]
			tsv_line['pos'] = pos
			i += 1

	conll_lines = read_xml_lines(xml_path.replace(".xml",".conllu").replace("xml","dep"))

	abs_id = 0
	for line in conll_lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" not in fields[0] and "." not in fields[0]:  # Regular token, not ellipsis or supertoken
				tsv_line = parsed_lines[abs_id]
				abs_id += 1
				tsv_line['abs_id'] = abs_id
				tsv_line['id_in_sent'] = fields[0]
				tsv_line['dep_parent'] = fields[6]
				tsv_line['func'] = fields[7]


# parsed representation layer
def expand_single_length_entities(parsed_lines):
	"""
	A quirk of the WebAnno TSV format: if an entity only spans a single token, it is almost always
	NOT given an ID in the TSV format. This function gives all entities, even single-length entities,
	IDs and adjusts all ID's so that they are consecutive, counting from 1 and increasing by 1 each time.

	Unfortunately, there are some quirks (at least from my perspective--perhaps there are explanations)
	in the TSV files I've observed:
	- Sometimes single-token entities *do* have IDs
	- Sometimes IDs are not strictly consecutive (in one instance, IDs 58 and 60 were present, but not 59)

	The latter is not handled by this tool: if 58 and 60 occur in the original file but 59 doesn't,
	then 58 and 59 will be the corresponding ID's in the output format.

	To handle the former, this function returns a list of all the single-span entities that did not
	have an ID in the original file. Then, their IDs are only deleted later if they appear in this list.
	:param parsed_lines: Parsed TSV lines from parse_tsv_lines
	:return: A list of IDs that were created for single-span entities.
	"""
	new_id_count = 0
	last_seen_entity_id = 0
	# old id -> shifted id
	old_id_index = {}
	# token id -> new id
	new_id_index = {}

	ent_counts = defaultdict(int)
	for line in parsed_lines:
		ents = line["entities"]
		for ent in ents:
			ent_counts[ent['id']] += 1

	# first step--create id's for single-token spans and adjust subsequent id's
	for line in parsed_lines:
		for entity in line['entities']:
			# single-tok span, has no ID--create one
			if not entity['id'] or ent_counts[entity['id']] == 1:
				new_id_count += 1
				last_seen_entity_id += 1
				entity['id'] = last_seen_entity_id
				new_id_index[line['token_id']] = entity['id']
			# non-first line of a multi-tok span
			elif entity['id'] in old_id_index:
				entity['id'] = old_id_index[entity['id']]
			# first line of a multi-tok span--update ID by adding the number of new ID's we've created so far
			else:
				old_id = entity['id']
				entity['id'] += new_id_count
				old_id_index[old_id] = entity['id']
				last_seen_entity_id = entity['id']

	# step 2--adjust relations so they use the new IDs
	for line in parsed_lines:
		for relation in line['relations']:
			# we have a single token span src--need to set it to the id we created
			if not relation['src']:
				assert relation['src_token'] in new_id_index
				relation['src'] = new_id_index[relation['src_token']]
			else:
				try:
					relation['src'] = old_id_index[relation['src']]
				except:
					relation['src'] = new_id_index[relation['src_token']]

			if not relation['dest']:
				assert line['token_id'] in new_id_index
				relation['dest'] = new_id_index[line['token_id']]
			else:
				try:
					relation['dest'] = old_id_index[relation['dest']]
				except:
					relation['dest'] = new_id_index[line['token_id']]

	return new_id_index.values(), old_id_index, new_id_index


def collapse_single_length_entities(parsed_lines, created_ids):
	"""
	Reverse the work of expand_single_length_entities, as much as possible. Cf. that function's documentation
	for quirks.
	:param parsed_lines: Parsed TSV lines from parse_tsv_lines
	:param created_ids: A list of entities whose ID's should be deleted as long as they are still single-span
	"""
	deleted_id_count = 0
	# old id -> shifted id
	old_id_index = {}
	deleted_ids = []

	# first step--create id's for single-token spans and adjust subsequent id's
	for i, line in enumerate(parsed_lines):
		for entity in line['entities']:
			# non-first line of a multi-tok span
			if entity['id'] in old_id_index:
				entity['id'] = old_id_index[entity['id']]
			# single-tok span, remove ID
			# not an 'else' because there were some cases where, for reasons I couldn't determine, a single-tok span
			# entity that looked like it shouldn't have had an ID nevertheless had an ID. Simplest to just do
			# whatever was done in the original doc.
			elif entity['id'] in created_ids and not \
				 (i < len(parsed_lines) - 1 and entity['id'] in [e['id'] for e in parsed_lines[i + 1]['entities']]):
				deleted_id_count += 1
				deleted_ids.append(entity['id'])
				entity['id'] = None
			# first line of a multi-tok span, or a single-tok span that for some reason has an ID.
			# update ID by adding the number of new ID's we've created so far
			else:
				old_id = entity['id']
				entity['id'] -= deleted_id_count
				old_id_index[old_id] = entity['id']

	for line in parsed_lines:
		for relation in line['relations']:
			relation['src'] = None if relation['src'] in deleted_ids else old_id_index[relation['src']]
			relation['dest'] = None if relation['dest'] in deleted_ids else old_id_index[relation['dest']]


def is_genitive_s(line):
	"""
	Test a parsed line to see whether it is an instance of a genitive s.
	:param line:
	:return: True if the line is a genitive s, false otherwise
	"""
	return line['pos'] == "POS"


def merge_genitive_s(parsed_lines, tsv_path, warn_only):
	"""
	Merges trailing genitive 's into immediately preceding entity span ([John] 's -> [John 's])
	:param parsed_lines: list of dictionaries
	:param tsv_path: target path to write to
	:param warn_only: if True, warn but do not modify cases
	"""
	for i, line in enumerate(parsed_lines):
		if is_genitive_s(line):
			entity_difference = [e for e in parsed_lines[i - 1]['entities'] if e not in line['entities'] and \
								 parsed_lines[i - 1]['token'] !="NSW"]  # exception for [University of [NSW] 's]
			if not entity_difference:
				continue

			if not warn_only:
				for e in entity_difference:
					line['entities'].append(e.copy())
					print("token " + line['token_id'] + " in doc '" + tsv_path + "' identified as genitive \"'s\" "
						  + "and merged with immediately preceding markable " + e['type'] + '[' + str(e['id']) + '].')
			else:
				for e in entity_difference:
					print("WARN: token " + line['token_id'] + " in doc '" + tsv_path + "' "
						  + "looks like a genitive s but is not contained in the immediately preceding markable "
						  + e['type'] + '[' + str(e['id']) +"].\n      Per GUM guidelines, it should be included."
						  + " Run _build/utils/repair_tsv.py to correct this issue.")


def fix_genitive_s(tsv_path, xml_path, warn_only=True, outdir=None, string_input=False):
	"""
	Finds occurrences of "genitive s" tokens (e.g. "Joseph 's Coat", "James ' Jacket", but not "John 's gone")
	the genitive s is not included in any markable(s) that include the immediately preceding token. Only
	prints a warning unless warn_only is set to false.
	:param tsv_path: The path to the WebAnno TSV file
	:param xml_path: The path to the TSV file's correspding XML file
	:param warn_only: If False, actually writes the corrected TSV files to outdir. If True, only prints warnings.
	:param outdir: The directory corrected TSV files will be placed in
	:param string_input: Treat tsv_path as a string containing webanno tsv
	"""
	if outdir is None:
		utils_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
		outdir = utils_dir + "pepper" + os.sep + "tmp" + os.sep + "tsv" + os.sep + "GUM" + os.sep

	if string_input:
		lines = [l + "\n" for l in tsv_path.split("\n")]
	else:
		lines = read_tsv_lines(tsv_path)
	orig_lines = lines
	parsed_lines = parse_tsv_lines(lines)
	parsed_orig_lines = parsed_lines
	enrich_tsv_representation_with_pos(parsed_lines, xml_path)

	created_ids, entity_mappings, single_tok_mappings = expand_single_length_entities(parsed_lines)
	merge_genitive_s(parsed_lines, tsv_path, warn_only)

	if not warn_only:
		collapse_single_length_entities(parsed_lines, created_ids)
		serialize_tsv_lines(lines, parsed_lines, tsv_path, outdir)
	else:
		pass
		#collapse_single_length_entities(parsed_orig_lines, created_ids)
		#serialize_tsv_lines(orig_lines, parsed_orig_lines, tsv_path, outdir)

	return parsed_lines, entity_mappings, single_tok_mappings

### end genitive s fix


def adjust_edges(webanno_tsv, parsed_lines, ent_mappings, single_tok_mappings):
	"""
	Fix webanno TSV in several ways:
	  * All edges pointing back from a pronoun receive type 'ana'
	  * All edges pointing from an entity headed by an apposition token to an entity headed by its parent are 'appos'
	  * Chains whose first member is a pronoun receive type 'cata' and edge must point forward
	  * All coref chain initial tokens and cata targets which are not infstat 'acc' receive infstat 'new'
	  * All other coref chain non-initial entities receive infstat 'giv'
	  * All bridging non-initial members receive infstat 'acc'
	  * Named entity identities are propagated to all members of a coref chain (transitive Wikification)

	:param webanno_tsv: input webanno tsv with possibly incorrect edge types
	:param parsed_lines: structured information about each token line
	:return: adjusted webanno tsv output
	"""

	adjusted = []
	entities = {}
	source2rel = defaultdict(list)
	dest2rel = defaultdict(list)

	tokens = {l["abs_id"]:l["token"] for l in parsed_lines}
	lowered_acc_prons = ["i","you","we","us","your","my","mine","yours","ours","me"]  # accessible pronouns, not cataphors

	# Get all entities
	for tid, dct in enumerate(parsed_lines):
		if "entities" in dct:
			for i, e in enumerate(dct["entities"]):
				if e["id"] in entities:
					entities[e["id"]]["end"] = tid
					entities[e["id"]]["toks"].append((int(dct["id_in_sent"]),int(dct["dep_parent"]),dct["pos"],dct["func"],dct["abs_id"]))
					entities[e["id"]]["length"] += 1
				else:
					entities[e["id"]] = {"start":tid, "length":1, "func": dct["func"], "pos": dct["pos"],
									 "infstat": e["infstat"], "type":e["type"], "identity": e["identity"], "relations": [],
									 "head_tok_abs_id": dct["abs_id"], "head_tok_parent_abs_id" : 0, "group":None,
									 "toks":[(int(dct["id_in_sent"]),int(dct["dep_parent"]),dct["pos"],dct["func"],dct["abs_id"])]}
				if "relations" in dct:
					for rel in dct["relations"]:
						source2rel[rel["src"]].append(rel)
						dest2rel[rel["dest"]].append(rel)
						if rel["dest"] == e["id"]:
							e["relations"] = rel

	# Get head token and assign func+pos for each entity
	for e_id in entities:
		for tok in entities[e_id]["toks"]:
			if tok[3] == "punct":
				continue
			if tok[1] == 0 or tok[1] < entities[e_id]["toks"][0][0] or tok[1] > entities[e_id]["toks"][-1][0]:
				# local root
				entities[e_id]["pos"] = tok[2]
				entities[e_id]["func"] = tok[3]
				entities[e_id]["head_tok_abs_id"] = tok[4]
				# abs parent
				if tok[1] == 0:
					entities[e_id]["head_tok_parent_abs_id"] = 0
				else:
					entities[e_id]["head_tok_parent_abs_id"] = tok[4] + tok[1] - tok[0]

	pronouns = set(["PP", "PRP", "PP$", "PRP$", "DT"])
	for e_id, ent in iteritems(entities):
		if e_id not in dest2rel:  # No incoming relations, this is a singleton
			if ent["infstat"] == "giv":
				ent["infstat"] = "new"
		if e_id in source2rel:
			for rel in source2rel[e_id]:
				if rel["rel_type"] == "coref":
					if ent["infstat"] == 'new':  # Source in coref chain can't be new
						ent["infstat"] = 'giv'
					if ent["pos"] in pronouns:
						rel["rel_type"] = "ana"
					elif ent["func"] == "appos":
						if entities[rel["dest"]]["head_tok_abs_id"] == ent["head_tok_parent_abs_id"]:
							rel["rel_type"] = "appos"
				elif rel["rel_type"].startswith("bridge"):
					if ent["infstat"]=='new':  # Source of bridging can't be new
						ent["infstat"] = "acc"
		if e_id in dest2rel:
			if e_id not in source2rel:
				# This is the first member of a chain
				ent = entities[e_id]
				if ent["infstat"] == "giv":
					ent["infstat"] = "new"
				if ent["pos"] in pronouns and ent["length"] == 1 and ent["infstat"] != "acc":  # Cataphora
					rem_rel = None
					for rel in dest2rel[e_id]:
						if rel["rel_type"] in ["coref","ana","appos"]:
							if rel["src_token"].split("-")[0] == rel["dst_token"].split("-")[0]:  # same sentence
								if tokens[ent["head_tok_abs_id"]].lower() not in lowered_acc_prons:  # I, you etc. not cataphors
									entities[rel["src"]]["infstat"] = "new"  # Next mention also considered new
									entities[rel["dest"]]["infstat"] = "new"
									rem_rel = rel
					if rem_rel is not None:
						new_rel = {k:v for k,v in iteritems(rem_rel)}
						new_rel["rel_type"] = "cata"
						new_rel["dest"] = rem_rel["src"]
						new_rel["src"] = e_id
						new_rel["src_token"] = rem_rel["dst_token"]
						new_rel["dst_token"] = rem_rel["src_token"]
						d_rels = [d for d in dest2rel[e_id]]
						for d_rel in d_rels:
							if rem_rel["src"] == d_rel["src"] and rem_rel["dest"] == d_rel["dest"]:
								dest2rel[e_id].remove(d_rel)
						s_rels = [d for d in source2rel[rem_rel["src"]]]
						for s_rel in s_rels:
							if s_rel["src"] == rem_rel["src"] and s_rel["dest"] == rem_rel["dest"]:
								source2rel[rem_rel["src"]].remove(s_rel)
						source2rel[e_id].append(new_rel)
						dest2rel[new_rel["dest"]].append(new_rel)

	max_group = 0
	mapping = {}
	for e_id in source2rel:
		ent = entities[e_id]
		for rel in source2rel[e_id]:
			if rel["rel_type"].startswith("bridg"):  # ignore bridging
				continue
			while ent["group"] in mapping:
				ent["group"] = mapping[ent["group"]]
			if entities[rel["dest"]]["group"] is not None:
				if ent["group"] is not None:
					if ent["group"] != entities[rel["dest"]]["group"]:
						if entities[rel["dest"]]["group"] in mapping:
							if mapping[entities[rel["dest"]]["group"]] == ent["group"]:
								# Source group already mapped to dest group
								entities[rel["dest"]]["group"] = ent["group"]
								continue
						mapping[ent["group"]] = entities[rel["dest"]]["group"]
				ent["group"] = entities[rel["dest"]]["group"]
			else:  # Assign common group ID to this new pair
				if ent["group"] is None:
					ent["group"] = entities[rel["dest"]]["group"] = max_group
					max_group += 1
				else:
					entities[rel["dest"]]["group"] = ent["group"]

	group_identities = {}
	for e_id in entities:
		ent = entities[e_id]
		if ent["group"] is None:  # Not coreferent
			continue
		if ent["identity"] != "_":
			if ent["group"] in group_identities:
				if group_identities[ent["group"]] != ent["identity"]:
					sys.stderr.write("Multiple entity conflict in doc "+ webanno_tsv[webanno_tsv.find("Text"): webanno_tsv.find("Text") + 20]+"\n"+
									 group_identities[ent["group"]] + "<>" + ent["identity"] + "\n")
			group_identities[ent["group"]] = ent["identity"]
		elif ent["group"] in group_identities:
			#continue
			#if ent["pos"][0] == "P":
			#	pass  # pronoun
			ent["identity"] = group_identities[ent["group"]]

	lines = webanno_tsv.split("\n")
	counter = 0

	seen = set()
	for line in lines:
		if "\t" not in line:
			adjusted.append(line)
		else:
			fields = line.split("\t")
			counter +=1
			ents, infs, idents, types, edges = fields[3:8]
			ents = ents.split("|")
			infs = infs.split("|")
			idents = []
			types = []
			edges = []
			for i in range(len(infs)):
				if infs[i] == "_":
					continue
				if "[" in infs[i]:
					e_id = int(infs[i].split("[")[1][:-1])
					if e_id in ent_mappings:
						e_id = ent_mappings[e_id]
					elif fields[0] in single_tok_mappings:
						if entities[single_tok_mappings[fields[0]]]["length"] == 1:  # Single token entity with orig ID
							e_id = single_tok_mappings[fields[0]]
				else:
					e_id = single_tok_mappings[fields[0]]
				ents[i] = entities[e_id]["type"] + "[" + str(e_id) + "]"
				infs[i] = entities[e_id]["infstat"] + "[" + str(e_id) + "]"
				if entities[e_id]["identity"] != "_":
					idents.append(entities[e_id]["identity"] + "[" + str(e_id) + "]")
				if e_id in dest2rel:
					for rel in dest2rel[e_id]:
						if e_id not in seen:
							edge = rel["src_token"] + "[" + str(rel["src"]) + "_" + str(rel["dest"]) + "]"
							if edge not in edges:
								edges.append(edge)
								types.append(rel["rel_type"])
					seen.add(e_id)

			types = "_" if len(types) == 0 else "|".join(types)
			edges = "_" if len(edges) == 0 else "|".join(edges)
			infs = "|".join(infs)
			ents = "|".join(ents)
			idents = "|".join(idents) if len(idents) > 0 else "_"
			fields[3] = ents
			fields[4] = infs
			fields[5] = idents
			fields[-3] = types
			fields[-2] = edges
			line = "\t".join(fields)
			adjusted.append(line)

	return "\n".join(adjusted)


def fix_file(filename, tt_file, outdir, genitive_s=False):

	# Get reference tokens
	tsv_file_name = ntpath.basename(filename)
	tokens = []

	outdir = os.path.abspath(outdir) + os.sep
	last_good_token = ""
	if sys.version_info[0] < 3:
		outfile = open(outdir + tsv_file_name,'wb')
		outtemp = open(tsv_temp_dir + tsv_file_name,'wb')
	else:
		outfile = io.open(outdir + tsv_file_name, 'w', encoding="utf8",newline="\n")
		outtemp = io.open(tsv_temp_dir + tsv_file_name,'w', encoding="utf8", newline="\n")
	tt_file = os.path.abspath(tt_file).replace("tsv" + os.sep,"xml"+os.sep)

	if PY2:
		tt = open(tt_file)
	else:
		tt = open(tt_file,encoding="utf8")
	lines = tt.read().replace("\r","").split("\n")

	current_token = 0
	current_sent = 1
	sent_map = {}
	tok_sent_map = {}
	tok_id_in_sent = 0

	for line in lines:
		if not line.startswith("<") and len(line) > 0 and "\t" in line: # token
			token = line.split("\t")[0]
			current_token += 1
			tok_id_in_sent += 1
			tokens.append(token)
			sent_map[current_token] = current_sent
			tok_sent_map[current_token] = str(current_sent) + "-" + str(tok_id_in_sent)
		if line.startswith("</s>"): # sentence end
			current_sent += 1
			tok_id_in_sent = 0

	total_out_tokens = 0

	if PY2:
		tsv = open(filename)
	else:
		tsv = io.open(filename,encoding="utf8")
	lines = tsv.read().replace("\r","").split("\n")

	out_lines = OrderedDict()
	id_mapping = {}
	current_token = 0
	current_token_text = ""
	first_tsv_line = {}
	line_num = 0
	id_offset = 0
	absolute_id_mapping = {}
	current_char_start = 0

	for line in lines:
		line_num += 1
		if "\t" not in line: # not a token
			out_lines[line_num] = line
		elif len(line) == 0: # sentence break (if text has already started)
			id_offset = 0
			out_lines[line_num] = line
		else:
			fields = line.split("\t")
			this_line_token = fields[2]
			if current_token_text == "": # Start of a new real token, store this line for reference
				first_tsv_line[current_token] = list(fields)
			current_token_text += this_line_token
			if current_token_text == equiv_tok(tokens[current_token]) or current_token_text == tokens[current_token]:  # Correct token is complete
				total_out_tokens += 1
				last_good_token = current_token_text + "\nat line: " + line

				# retrieve first TSV line of this token
				out_fields = first_tsv_line[current_token]

				# calculate correct char range
				if PY2:
					char_range = str(current_char_start) + "-" + str(current_char_start+len(current_token_text.decode("utf-8")))
					current_char_start += len(current_token_text.decode("utf-8")) + 1
				else:
					char_range = str(current_char_start) + "-" + str(current_char_start + len(current_token_text))
				current_char_start += len(current_token_text) + 1
				out_fields[1] = char_range

				# rebuild token
				out_fields[2] = unescape_xml(tokens[current_token])

				# adjust ID and store mapping
				sent_id, tok_id = out_fields[0].split("-")
				tok_id = str(int(tok_id) + id_offset)
				new_id = sent_id + "-" + tok_id
				id_mapping[out_fields[0]] = new_id
				absolute_id_mapping[new_id] = total_out_tokens
				out_fields[0] = new_id

				# store edited line EXCEPT for token IDs in edge pointers, which still needs to be adjusted
				out_lines[line_num] = "\t".join(out_fields)

				# Reset buffers
				current_token += 1
				current_token_text = ""

			else:  # multipart token is growing
				id_offset += 1

	bridging_count = defaultdict(int)
	bridge_words = defaultdict(lambda:"_")  # Track tokens to find bridging type

	edited_lines = []
	for line_num, line in iteritems(out_lines):
		if "\t" in line:
			tok_id = fields[0]
			bridge_words[tok_id] = fields[2]
			fields = line.split("\t")
			links = fields[-2]
			link_annos = fields[-3]
			split_links = links.split("|")
			split_link_annos = link_annos.split("|")
			out_link = ""
			pipe = ""
			for i, link in enumerate(split_links):
				#continue  ##AZ
				if "[" in link:
					tok, spans = link.split("[")
					spans = "[" + spans
				else:
					spans = ""
					tok = link

				if tok in id_mapping:
					tok = id_mapping[tok]

				try:
					link_anno = split_link_annos[i]
				except:
					print("Error on line " + str(line_num) + " of TSV file: " + filename)
					quit()
				if link_anno == "bridge":
					if spans != "" and not spans.startswith("[0_"):# and not spans.endswith("_0]"):
						bridging_count[spans.split("_")[0].replace("[","")] += 1
					else:
						bridging_count[tok] += 1

				if spans != "":
					tok += spans

				out_link += pipe + tok
				if pipe == "":
					pipe = "|"

			fields[-2] = out_link
			line = "\t".join(fields)
		edited_lines.append(line)

	# Now split bridging sub-types
	bridge_fixed = []
	for line in edited_lines:
		if "\t" in line:
			fields = line.split("\t")
			links = fields[-2]
			link_annos = fields[-3]
			split_links = links.split("|")
			split_link_annos = link_annos.split("|")
			edited_annos = []
			#continue ##AZ
			for i, anno in enumerate(split_link_annos):
				link = split_links[i]
				bridge_count_id = link
				if "[" in link:
					if "[0_" not in link: # and not link.endswith("_0]"):
						bridge_count_id = link.split("[")[1].split("_")[0].replace("[","")
					else:
						bridge_count_id = link.split("[")[0]
					link = link.split("[")[0]
				source_word = bridge_words[link]
				if anno == "bridge":
					if bridging_count[bridge_count_id] > 1:
						anno = "bridge:aggr"
					elif re.match(r'(the|this|that|these|those)$',source_word,re.IGNORECASE) is not None:
						anno = "bridge:def"
					else:
						anno = "bridge:other"
				edited_annos.append(anno)
			fields[-3] = "|".join(edited_annos)
			bridge_fixed.append("\t".join(fields))
		else:
			bridge_fixed.append(line)
	edited_lines = bridge_fixed
	if not total_out_tokens == len(tokens):
		raise IOError("Token length conflict: " + str(len(tokens)) + " TT tokens but " + str(total_out_tokens) + " TSV tokens in " + tsv_file_name +". Last good token: " + last_good_token)

	# Handle sentence splits
	current_token = 0
	current_sent = 1
	tok_id_counter = 0
	sent_text = "#Text="
	header_mode = True
	out_lines = []
	output = ""
	first = True

	for line in edited_lines:
		# Print all lines until we reach the text
		if header_mode:
			if line.startswith("#Text"):
				header_mode = False
			else:
				output += line + "\n"
		else:
			# Now we only deal with token lines
			fields = line.split("\t")
			if "\t" in line:
				current_token += 1
				if current_sent == sent_map[current_token]:  # We are still in the current sentence
					sent_text += fields[2] + " "
					tok_id_counter += 1
				else:
					if not first:
						output += "\n\n"
					else:
						first = False
					output += sent_text.strip() + "\n"
					output += "\n".join(out_lines)
					out_lines = []
					sent_text = "#Text=" + fields[2] + " "
					current_sent += 1
					tok_id_counter = 1
				fields[0] = str(current_sent) + "-" + str(tok_id_counter)

				links = fields[-2]
				split_links = links.split("|")
				out_link = ""
				pipe = ""
				for link in split_links:
					if "[" in link:
						tok, spans = link.split("[")
						spans = "[" + spans
					else:
						spans = ""
						tok = link

					if tok in absolute_id_mapping:
						tok_number = absolute_id_mapping[tok]
						tok = tok_sent_map[tok_number]

					if spans != "":
						tok += spans

					out_link += pipe + tok
					if pipe == "":
						pipe = "|"

				fields[-2] = out_link
				line = "\t".join(fields)

				out_lines.append(line)

	output += "\n\n" + sent_text.strip() + "\n"
	output += "\n".join(out_lines) + "\n"
	parsed_lines, entity_mappings, single_tok_mappings = fix_genitive_s(output, tt_file, warn_only=True, string_input=True)

	output = adjust_edges(output, parsed_lines, entity_mappings, single_tok_mappings)

	outfile.write(output)
	outfile.close()
	outtemp.write(output)
	outtemp.close()


if __name__ == "__main__":
	if platform.system() == "Windows":
		import os, msvcrt
		msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

	filename = sys.argv[1]  # e.g. ../src/tsv/*.tsv
	if "*" in filename:
		file_list = glob(sys.argv[1])
	else:
		file_list = [filename]

	outdir = os.path.abspath(".") + os.sep + "out_tsv" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for filename in file_list:
		tt_file = filename.replace("tsv", "xml")
		fix_file(filename, tt_file, outdir)
		fix_genitive_s(filename, tt_file, warn_only=False, outdir=outdir)