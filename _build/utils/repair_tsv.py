#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,platform,os,io
from collections import OrderedDict
import ntpath
from glob import glob
from six import iteritems

PY2 = sys.version_info[0] < 3


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
	if sys.version_info[0] < 3:
		infile = open(tsv_path, 'rb')
	else:
		infile = open(tsv_path, 'r', encoding="utf8")

	lines = infile.readlines()
	infile.close()
	return lines


def read_tt_lines(tt_path):
	if PY2:
		infile = open(tt_path, 'rb')
	else:
		infile = open(tt_path, 'r', encoding="utf8")

	lines = infile.readlines()
	infile.close()
	return lines


def format_entities(entities):
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


def format_infstat(entities):
	strs = []
	for entity in entities:
		if entity['id']:
			strs.append(entity['infstat'] + '[' + str(entity['id']) + ']')
		else:
			strs.append(entity['infstat'])

	if strs:
		return "|".join(strs)
	else:
		return "_"


def serialize_tsv_lines(lines, parsed_lines, tsv_path):
	if sys.version_info[0] < 3:
		outfile = open(tsv_path, 'wb')
	else:
		outfile = open(tsv_path, 'w', encoding="utf8")

	i = 0
	for line in lines:
		if "\t" not in line:
			outfile.write(line)
		else:
			cols = line.split("\t")
			cols[3] = format_entities(parsed_lines[i]['entities'])
			cols[4] = format_infstat(parsed_lines[i]['entities'])
			cols[6] = format_relations(parsed_lines[i]['relations'])
			outfile.write("\t".join(cols))
			i += 1

	outfile.close()


# string layer
def extract_from_bracket(s):
	"""Given something like "xyz[3]", returns 2-tuple ("xyz", "3"), or ("xyz", None) if input is 'xyz"."""
	bracket_index = s.rfind("[")
	if bracket_index < 0:
		return s, None
	else:
		return s[:bracket_index], s[bracket_index+1:-1]


def parse_tsv_line(line):
	line = [None if x == "_" else x for x in line]

	already_seen_single_tok_entity = False
	entities = []
	if line[3]:
		# parse something like "person[3]" or "person"
		information_statuses = line[4].split("|")
		for i, entity in enumerate(line[3].split("|")):
			entity_type, entity_id = extract_from_bracket(entity)
			entity_id = int(entity_id) if entity_id else None

			entity_infstat, _ = extract_from_bracket(information_statuses[i])
			entities.append({'id': entity_id, 'type': entity_type, 'infstat': entity_infstat})

			# assumption: there should be at most one single-token entity on any given line
			assert not (entity_id is None and already_seen_single_tok_entity)
			if entity_id is None:
				already_seen_single_tok_entity = True

	relations = []
	if line[6]:
		# parse something like "5-1[20_10]",
		# i.e. "this line is where entity 10 begins, and it is related to entity 20 beginning at token 5-1
		for relation in line[6].split("|"):
			src_token, src_dest = extract_from_bracket(relation)
			if src_dest:
				src, dest = src_dest.split("_")
				src = int(src) if src != "0" else None
				dest = int(dest) if dest != "0" else None
			else:
				src = None
				dest = None
			relations.append({'src': src,
							  'dest': dest,
							  'src_token': src_token})

	return {'token_id': line[0],
			'token': line[2],
			'entities': entities,
			'relations': relations}


def parse_tsv_lines(lines):
	return [parse_tsv_line(raw_line.rstrip().split("\t"))
			for raw_line in lines
			if "\t" in raw_line]


def enrich_tsv_representation_with_pos(parsed_lines, tt_path):
	lines = read_tt_lines(tt_path)

	i = 0
	for line in lines:
		if not line.startswith("<") and len(line) > 0 and "\t" in line: # token
			cols = line.split("\t")
			token = cols[0]
			pos = cols[1]

			tsv_line = parsed_lines[i]
			tsv_line['pos'] = pos
			i += 1


# parsed representation layer
def expand_single_length_entities(parsed_lines):
	new_id_count = 0
	last_seen_entity_id = 0
	# old id -> shifted id
	old_id_index = {}
	# token id -> new id
	new_id_index = {}

	# first step--create id's for single-token spans and adjust subsequent id's
	for line in parsed_lines:
		for entity in line['entities']:
			# single-tok span, has no ID--create one
			if not entity['id']:
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

	for line in parsed_lines:
		for relation in line['relations']:
			# we have a single token span src--need to set it to the id we created
			if not relation['src']:
				assert relation['src_token'] in new_id_index
				relation['src'] = new_id_index[relation['src_token']]
			else:
				relation['src'] = old_id_index[relation['src']]

			if not relation['dest']:
				assert line['token_id'] in new_id_index
				relation['dest'] = new_id_index[line['token_id']]
			else:
				relation['dest'] = old_id_index[relation['dest']]

	return new_id_index.values()


def collapse_single_length_entities(parsed_lines, created_ids):
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
	return line['pos'] == "POS"


def merge_genitive_s(parsed_lines, tsv_path, dry):
	for i, line in enumerate(parsed_lines):
		if is_genitive_s(line):
			entity_difference = [e for e in parsed_lines[i - 1]['entities'] if e not in line['entities']]
			if not entity_difference:
				continue

			if not dry:
				for e in entity_difference:
					line['entities'].append(e.copy())
					print("token " + line['token_id'] + " in doc '" + tsv_path + "' identified as genitive \"'s\" "
						  + "and merged with immediately preceding markable " + e['type'] + '[' + str(e['id']) + '].')
			else:
				print("WARN: token " + line['token_id'] + " in doc '" + tsv_path + "' "
					  + "is \"'s\" but is not contained in any immediately preceding markable.\n      Per "
					  + "GUM guidelines, the \"'s\" should be included if it is genitive marking.")


def fix_genitive_s(tsv_path, tt_path, dry=True):
	lines = read_tsv_lines(tsv_path)
	parsed_lines = parse_tsv_lines(lines)
	enrich_tsv_representation_with_pos(parsed_lines, tt_path)

	created_ids = expand_single_length_entities(parsed_lines)
	merge_genitive_s(parsed_lines, tsv_path, dry)

	if not dry:
		collapse_single_length_entities(parsed_lines, created_ids)
		serialize_tsv_lines(lines, parsed_lines, tsv_path)
### end genitive s fix


def fix_file(filename,tt_file,outdir,genitive_s=False):

	# Get reference tokens
	tsv_file_name = ntpath.basename(filename)
	tokens = []

	outdir = os.path.abspath(outdir) + os.sep
	last_good_token = ""
	if sys.version_info[0] < 3:
		outfile = open(outdir + tsv_file_name,'wb')
	else:
		outfile = open(outdir + tsv_file_name, 'w', encoding="utf8")
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


	edited_lines = []
	for line_num, line in iteritems(out_lines):
		if "\t" in line:
			fields = line.split("\t")
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

				if tok in id_mapping:
					tok = id_mapping[tok]

				if spans != "":
					tok += spans

				out_link += pipe + tok
				if pipe == "":
					pipe = "|"

			fields[-2] = out_link
			line = "\t".join(fields)
		edited_lines.append(line)

	if not total_out_tokens == len(tokens):
		raise IOError("Token length conflict: " + str(len(tokens)) + " TT tokens but " + str(total_out_tokens) + " TSV tokens in " + tsv_file_name +". Last good token: " + last_good_token)


	# Handle sentence splits
	current_token = 0
	current_sent = 1
	tok_id_counter = 0
	sent_text = "#Text="
	header_mode = True
	out_lines = []
	first = True

	for line in edited_lines:
		# Print all lines until we reach the text
		if header_mode:
			if line.startswith("#Text"):
				header_mode = False
			else:
				outfile.write(line + "\n")
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
						outfile.write("\n\n")
					else:
						first = False
					outfile.write(sent_text.strip() + "\n")
					outfile.write("\n".join(out_lines))
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

	outfile.write("\n\n" + sent_text.strip() + "\n")
	outfile.write("\n".join(out_lines) + "\n")
	outfile.close()

	fix_genitive_s(filename, tt_file, dry=(not genitive_s))

if __name__ == "__main__":
	if platform.system() == "Windows":
		import os, msvcrt
		msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

	filename = sys.argv[1]
	if "*" in filename:
		file_list = glob(sys.argv[1])
	else:
		file_list = [filename]

	outdir = os.path.abspath(".") + os.sep + "out_tsv" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for filename in file_list:
		tt_file = filename.replace(".tsv", ".xml")
		fix_file(filename,tt_file,outdir)
