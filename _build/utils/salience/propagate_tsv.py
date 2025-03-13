import re, os

def parse_tsv_lines(lines):
	"""
	Parse all WebAnno TSV lines
	:param lines: A list of unprocessed TSV lines
	:return: a list of dictionaries representing a subset of their contents.
	"""
	return [parse_tsv_line(raw_line.rstrip().split("\t"))
			for raw_line in lines
			if "\t" in raw_line]


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
		saliences = line[5].split("|")
		identities = line[6].split("|") if line[6] is not None else [None for i, _ in enumerate(information_statuses)]
		for i, entity in enumerate(line[3].split("|")):
			entity_type, entity_id = extract_from_bracket(entity)
			entity_id = int(entity_id) if entity_id else None

			entity_infstat, _ = extract_from_bracket(information_statuses[i])
			entity_salience, _ = extract_from_bracket(saliences[i])
			ident_field = "_"
			for ident_anno in identities:  # See if an identity anno corresponds to the currently processed entity ID
				if ident_anno is not None:
					entity_identity, ident_id = extract_from_bracket(ident_anno)
					entity_identity = entity_identity.replace(" ","_").replace("(","%28").replace(")","%29").replace(",","%2C").replace("-","%2D")
				else:
					continue
				if int(ident_id) == entity_id:
					ident_field = entity_identity
					break
			entities.append({'id': entity_id, 'type': entity_type, 'infstat': entity_infstat, "salience":entity_salience, "identity": ident_field})

			# assumption: there should be at most one single-token entity on any given line
			assert not (entity_id is None and already_seen_single_tok_entity)
			if entity_id is None:
				already_seen_single_tok_entity = True

	relations = []
	if line[8]:
		rel_types = line[7].split("|")
		# parse something like "5-1[20_10]",
		# i.e. "this line is where entity 10 begins, and it is related to entity 20 beginning at token 5-1
		for i, relation in enumerate(line[8].split("|")):
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
			cols[5] = format_attr(parsed_lines[i]['entities'],attr="salience")
			cols[6] = format_attr(parsed_lines[i]['entities'],attr="identity")
			cols[6] = cols[6].replace(" ","_").replace("(","%28").replace(")","%29").replace(",","%2C").replace("-","%2D")
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
	merge_genitive_s(parsed_lines, tsv_path, warn_only, xml_path)

	if not warn_only:
		collapse_single_length_entities(parsed_lines, created_ids)
		serialize_tsv_lines(lines, parsed_lines, tsv_path, outdir)
	else:
		pass
		#collapse_single_length_entities(parsed_orig_lines, created_ids)
		#serialize_tsv_lines(orig_lines, parsed_orig_lines, tsv_path, outdir)

	return parsed_lines, entity_mappings, single_tok_mappings


def adjust_edges(webanno_tsv, parsed_lines, ent_mappings, single_tok_mappings, single_coref_type=False):
	"""
	Fix webanno TSV in several ways:
	  * All edges pointing back from a pronoun receive type 'ana'
	  * All edges pointing from an entity headed by an apposition token to an entity headed by its parent are 'appos'
	  * All edges pointing from an indefinite nominal entity which has a subject dependent are 'pred'
	  * All edges pointing to a verbal entity are 'disc'
	  * Chains whose first member is a pronoun receive type 'cata' and edge must point forward
	  * All coref chain initial tokens and cata targets which are not infstat 'acc' receive infstat 'new'
	  * All other coref chain non-initial entities receive infstat 'giv'
	  * All bridging non-initial members receive infstat 'acc'
	  * Named entity identities are propagated to all members of a coref chain (transitive Wikification)

	:param webanno_tsv: input webanno tsv with possibly incorrect edge types
	:param parsed_lines: structured information about each token line
	:param ent_mappings: maps entity numbers like [320]
	:param single_tok_mappings: maps single token entities identified by their tok ID, like 13-1 (sent. 13, tok 1)
	:param single_coref_type: if True only use coref and bridge in output, do not distinguish ana, cata, pred, disc and appos types
	:return: adjusted webanno tsv output, per-token conllu_data for conllu-a bracket representation in MISC field, dictionary of coref groups to salience values
	"""

	def has_bridging(rels):
		for rel in rels:
			if rel["rel_type"].startswith("bridg"):
				return True
		return False

	def get_min(toks):
		start = toks[0][0]
		end = toks[-1][0]
		head = end
		head_tokens_lowered = {}
		toks_by_id = {0:(0,0,"","","",0)}  # Dummy token for root
		for tok in toks:
			toks_by_id[tok[0]] = tok
			if tok[1] == 0:  # root
				head = tok[0]
				head_tokens_lowered[tok[0]] = tok[-2].lower()
				break
			elif tok[1] > end or tok[1] < start and tok[3] != "punct":
				head = tok[0]
				head_tokens_lowered[tok[0]] = tok[-2].lower()
		min_ids = [head]
		for i, tok in enumerate(toks):
			if tok[3] == "conj" and tok[1] in min_ids:
				min_ids.append(tok[0])
				head_tokens_lowered[tok[0]] = tok[-2].lower()
			elif tok[3] == "flat" and tok[1] in min_ids:
				min_ids.append(tok[0])
				head_tokens_lowered[tok[0]] = tok[-2].lower()
			if tok[2].startswith("NP") and tok[1] in toks_by_id:  # NNP token with parent in span
				if toks_by_id[tok[1]][2].startswith("NP") and tok[3] not in ["nmod:poss"]:
					# NNP child of NNP, not a possessor
					min_ids.append(tok[0])

		min_idx = []
		for i, tok in enumerate(toks):
			if tok[0] in min_ids:
				min_idx.append(str(i+1))
		return ",".join(min_idx)


	adjusted = []
	entities = {}
	source2rel = defaultdict(list)
	dest2rel = defaultdict(list)

	tokens = {l["abs_id"]:l["token"] for l in parsed_lines}
	lowered_acc_prons = ["i","you","we","us","your","my","mine","yours","ours","me"]  # accessible pronouns, not cataphors

	# Get all child dependency functions per token
	child_funcs = defaultdict(set)
	child_lower_toks = defaultdict(set)
	parents = {}
	for tok in parsed_lines:
		if tok["func"] != "root":
			sent_offset = tok["abs_id"] - int(tok["id_in_sent"])
			abs_parent = int(tok["dep_parent"]) + sent_offset
			parents[tok["abs_id"]] = abs_parent if tok["dep_parent"] != "0" else 0
			child_funcs[abs_parent].add(tok["func"])
			child_lower_toks[abs_parent].add(tok["token"].lower())
		else:
			parents[tok["abs_id"]] = 0


	# Compute all token depths in their sentence dependency graphs
	depths = {}
	for tok in parents:
		if parents[tok] == 0:
			depths[tok] = 0
		else:
			par = parents[tok]
			depth = 0
			while par != 0:
				par = parents[par]
				depth += 1
			depths[tok] = depth

	# Get all entities
	for tid, dct in enumerate(parsed_lines):
		if "entities" in dct:
			for i, e in enumerate(dct["entities"]):
				if e["id"] in entities:
					entities[e["id"]]["end"] = tid
					entities[e["id"]]["toks"].append((int(dct["id_in_sent"]),int(dct["dep_parent"]),dct["pos"],dct["func"],dct["token"],dct["abs_id"]))
					entities[e["id"]]["length"] += 1
				else:
					entities[e["id"]] = {"start":tid, "end":tid,"length":1, "func": dct["func"], "pos": dct["pos"],
									 "infstat": e["infstat"], "salience": e["salience"], "type":e["type"], "identity": e["identity"], "relations": [],
									 "head_tok_abs_id": dct["abs_id"], "head_tok_parent_abs_id" : 0, "group":None,
									 "sid": dct["token_id"].split("-")[0],
									 "toks":[(int(dct["id_in_sent"]),int(dct["dep_parent"]),dct["pos"],dct["func"],dct["token"],dct["abs_id"])]}
				if "relations" in dct:
					for rel in dct["relations"]:
						source2rel[rel["src"]].append(rel)
						dest2rel[rel["dest"]].append(rel)
						rel["sent_dist"] = abs(int(rel["src_token"].split("-")[0]) - int(rel["dst_token"].split("-")[0]))
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
				entities[e_id]["head_tok_abs_id"] = tok[-1]
				entities[e_id]["child_funcs"] = child_funcs[tok[-1]]
				entities[e_id]["child_lower_toks"] = child_lower_toks[tok[-1]]
				entities[e_id]["depth"] = depths[entities[e_id]["head_tok_abs_id"]]
				# abs parent
				if tok[1] == 0:
					entities[e_id]["head_tok_parent_abs_id"] = 0
				else:
					entities[e_id]["head_tok_parent_abs_id"] = tok[-1] + tok[1] - tok[0]

	pronouns = set(["PP", "PRP", "PP$", "PRP$", "DT"])
	definites = set(["the","this","that","those","these","both"])
	for e_id, ent in iteritems(entities):
		ent["coref_type"] = "sgl"  # Assume everything is a singleton at first
		if e_id not in dest2rel:  # No incoming relations, this is a singleton
			if not ent["infstat"].startswith("acc") and not ent["infstat"].startswith("split"):
				ent["infstat"] = "new"
		if e_id in source2rel:
			if all([r["rel_type"].startswith("bridg") for r in source2rel[e_id]]):  # Only has bridging
				if ent["infstat"] not in ["split","acc:aggr"]:
					ent["infstat"] = "acc:inf"
			for rel in source2rel[e_id]:
				if rel["rel_type"] in ["coref","disc","ana","appos"]:
					entities[e_id]["sent_dist"] = rel["sent_dist"]
					if ent["infstat"] in ['new','giv']:  # Source in coref chain can't be new
						if rel["sent_dist"] > 1:
							ent["infstat"] = 'giv:inact'
						else:
							ent["infstat"] = 'giv:act'
						if rel["rel_type"] == "ana":
							continue
					if ent["coref_type"] == "sgl":
						ent["coref_type"] = "coref"
					if ent["pos"] in pronouns and not single_coref_type and rel["rel_type"] != "disc":
						rel["rel_type"] = "ana"
						ent["coref_type"] = "ana"
					elif ent["func"] == "appos" and not single_coref_type:
						if entities[rel["dest"]]["head_tok_abs_id"] == ent["head_tok_parent_abs_id"]:
							rel["rel_type"] = "appos"
							ent["coref_type"] = "appos"
							ent["infstat"] = 'giv:act'
					if ent["pos"].startswith("NN") and (any(["subj" in fnc for fnc in ent["child_funcs"]]) or "as" in ent["child_lower_toks"]) and not single_coref_type:
						# nominal with subject
						if not any([t in ent["child_lower_toks"] for t in definites]):  # no definite determiners
							if not any([":poss" in fnc for fnc in ent["child_funcs"]]):  # no possessors
								rel["rel_type"] = "pred"
								ent["coref_type"] = "pred"

				elif rel["rel_type"].startswith("bridg"):
					if ent["infstat"]=='new' or ent["infstat"].startswith("giv"):  # Source of bridging can't be new or given
						ent["infstat"] = "acc:inf"
		if e_id in dest2rel:
			ent = entities[e_id]
			if ent["coref_type"] == "sgl":
				ent["coref_type"] = "coref"
			if e_id not in source2rel:
				# This is the first member of a chain
				if not ent["infstat"].startswith("acc") and not ent["infstat"].startswith("split"):
					ent["infstat"] = "new"
				# Cataphora
				if ent["pos"] in pronouns:
					# First coref type should be ana if it's not cata
					ent["coref_type"] = "ana"
					if ent["length"] == 1 and not ent["infstat"].startswith("acc") and not single_coref_type:
						rem_rel = None
						for rel in dest2rel[e_id]:
							if rel["rel_type"] in ["coref","ana","appos"]:
								if rel["src_token"].split("-")[0] == rel["dst_token"].split("-")[0]:  # same sentence
									if tokens[ent["head_tok_abs_id"]].lower() not in lowered_acc_prons:  # I, you etc. not cataphors
										entities[rel["src"]]["infstat"] = "new"  # Next mention also considered new
										entities[rel["dest"]]["infstat"] = "new"
										rem_rel = rel
						if rem_rel is not None:
							ent["coref_type"] = "cata"
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
			# Discourse deixis
			head_tok_id_in_sent = 0
			verbal = False
			for t in ent["toks"]:
				if t[-1] == ent["head_tok_abs_id"]:
					head_tok_id_in_sent = t[0]
			for t in ent["toks"]:
				if t[1] == head_tok_id_in_sent and t[3] in ["cop","nsubj","csubj"]:  # non-verbal predicate
					verbal = True
			if ent["pos"].startswith("V") or verbal:
				for rel in dest2rel[e_id]:
					if rel["rel_type"] in ["coref", "ana"]:
						rel["rel_type"] = "disc"
						ent["coref_type"]= "disc"
		if ent["infstat"].startswith("acc"):
			if not has_bridging(source2rel[e_id]):
				# Remove accessibles without bridging which are named places and dates/times (times with CD)
				if ent["type"] == "place":
					if ent["pos"].startswith("NP") and ent["identity"] != "":  # Accessible, non-bridge, linked NNP place: Country
						ent["infstat"] = "new"
						continue
				elif ent["type"] == "time":
					if any([t[2] == "CD" for t in ent["toks"]]):  # Accessible, non-bridge, time with number: Date
						ent["infstat"] = "new"
						continue
				if ent["infstat"] != "acc:aggr":
					if ent["pos"] in pronouns or any([x in {"this","that","those","these","here","there"} for x in ent["child_lower_toks"]]):
						ent["infstat"] = "acc:com"  # proxy for acc:sit
					else:
						ent["infstat"] = "acc:com"  # proxy for acc:gen
			else:
				ent["infstat"] = "acc:inf"

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
	group_saliences = defaultdict(lambda :"nonsal")
	for e_id in entities:
		ent = entities[e_id]
		if ent["infstat"] == "split":
			ent["infstat"] = "acc:aggr"
		elif ent["infstat"] == "giv":
			if "sent_dist" in ent:
				if ent["sent_dist"] > 1:
					ent["infstat"] = "giv:inact"
				else:
					ent["infstat"] = "giv:act"
		if ent["group"] is None:  # Not coreferent
			max_group += 1
			ent["group"] = max_group
			continue
		if ent["identity"] != "_":
			if ent["group"] in group_identities:
				if group_identities[ent["group"]] != ent["identity"]:
					sys.stderr.write("Multiple entity conflict in doc "+ webanno_tsv[webanno_tsv.find("Text"): webanno_tsv.find("Text") + 20]+"\n"+
									 group_identities[ent["group"]] + "<>" + ent["identity"] + "\n")
			group_identities[ent["group"]] = ent["identity"]
		if ent["salience"] == "sal":
			group_saliences[ent["group"]] = "sal"
		elif ent["group"] in group_identities:
			#continue
			#if ent["pos"][0] == "P":
			#	pass  # pronoun
			ent["identity"] = group_identities[ent["group"]]

	# Second pass on identities, for early mentions whose group got an identity later
	for e_id in entities:
		ent = entities[e_id]
		if ent["group"] in group_identities:
			ent["identity"] = group_identities[ent["group"]]
		if ent["group"] in group_saliences:
			ent["salience"] = group_saliences[ent["group"]]

	# Add Centering Theory annotations
	if "REDACTED" in webanno_tsv:
		a=4
	entities, centering_transitions = add_centering(entities)

	# Create opener and closer data for later conllu-a serialization
	opener_lists = defaultdict(list)
	closer_lists = defaultdict(list)
	group_mapping = {}
	mapped_saliences = {}
	max_mapped_group = 1
	for e_id in sorted(entities, key=lambda x: (entities[x]["start"],-entities[x]["end"])):
		ent = entities[e_id]
		start = ent["toks"][0][-1]
		end = ent["toks"][-1][-1]
		etype = ent["type"]
		coref_type = ent["coref_type"]
		infstat = ent["infstat"]
		centering = ent["cf_rank"]
		identity = "-" + ent["identity"] if ent["identity"] != "_" else ""
		min_ids = get_min(ent["toks"])
		if ent["group"] not in group_mapping:
			group_mapping[ent["group"]] = max_mapped_group
			max_mapped_group += 1
		group = group_mapping[ent["group"]]
		if ent["salience"] == "sal":
			mapped_saliences[group] = "sal"
		starter = f'({etype}-{group}-{infstat}-{centering}-{min_ids}-{coref_type}{identity}'
		if start == end:
			starter += ")"
		opener_lists[start].append(starter)
		if start != end:
			closer_lists[end].append(str(group) + ")")

	conllua_data = []
	for i in range(len(tokens)):
		if i+1 in opener_lists or i+1 in closer_lists:
			tid = i+1
			open_part = "".join(opener_lists[tid])
			close_part = "".join(closer_lists[tid][::-1])
			conllua_data.append(open_part + close_part)  # Prefer openers before closers for '(1(2)'
			if open_part != "" and close_part != "":
				if open_part[-1].isdigit() and close_part[-1].isdigit():
					# This should never happen, but if we have open (1 and close 2), we must serialize 2)(1 to avoid (12)
					conllua_data[-1] = close_part + open_part
		else:
			conllua_data.append("_")

	lines = webanno_tsv.split("\n")
	counter = 0

	seen = set()
	for line in lines:
		if "\t" not in line:
			adjusted.append(line)
		else:
			fields = line.split("\t")
			counter +=1
			ents, infs, sals, idents, types, edges = fields[3:9]
			ents = ents.split("|")
			infs = infs.split("|")
			sals = sals.split("|")
			idents = []
			types = []
			edges = []
			centers = []
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
				sals[i] = entities[e_id]["salience"] + "[" + str(e_id) + "]"
				centers.append(entities[e_id]["cf_rank"])
				centers[i] += "[" + str(e_id) + "]"
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
			sals = "|".join(sals)
			ents = "|".join(ents)
			idents = "|".join(idents) if len(idents) > 0 else "_"
			centers = "|".join(centers) if len(centers) > 0 else "_"
			fields[3] = ents
			fields[4] = infs
			fields[5] = sals
			fields[6] = idents
			fields.insert(7, centers)
			fields[-3] = types
			fields[-2] = edges
			line = "\t".join(fields)
			adjusted.append(line)

	return "\n".join(adjusted), conllua_data, centering_transitions, mapped_saliences
