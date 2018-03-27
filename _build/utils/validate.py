#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, ntpath, sys
import glob
import re
import xml.etree.ElementTree as ET
import io
from collections import defaultdict

class Markable:
	def __init__(self):
		self.start = 0
		self.end = 0
		self.text = ""
		self.entity = ""
		self.infstat = ""
		self.antecedent = ""
		self.coref_type = ""
		self.anaphor = ""
		self.anaphor_type =""

class rstNode:
	def __init__(self):
		self.id = 0
		self.text = ""
		self.rel = ""
		self.parent = 0
		self.start = 0
		self.end = 0
		self.type = ""


# Function to validate list of XML files against XSD schema
def validate_xsd(file_list, gum_source):
	from lxml import etree
	if sys.version_info[0] < 3:
		with open(gum_source + "gum_schema.xsd") as f:
			schema_root = etree.XML(f.read())
	else:
		with io.open(gum_source + "gum_schema.xsd", encoding="utf8") as f:
			xml = f.read()
			schema_root = etree.XML(bytes(bytearray(xml, encoding='utf-8')))

	valid_files = 0
	errors = ""
	schema = etree.XMLSchema(schema_root)

	for docnum, xml_file in enumerate(file_list):
		sys.stdout.write("\t+ " + " "*40 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Validating " + xml_file + "\r")

		try:
			with io.open(gum_source + "xml" + os.sep + xml_file,encoding="utf8") as f:
				root = etree.parse(f)
				schema.validate(root)
		except Exception as e:
			print(e)
		finally:
			if len(schema.error_log.filter_from_errors()) == 0:
				valid_files += 1
			else:
				errors += "\n" + xml_file + " has errors:\n"
				for err in schema.error_log.filter_from_errors():
					err_match = re.search(r'\.xml:([0-9]+).*?: (.*)',str(err))
					if err_match is not None:
						errors += "  Line " + err_match.group(1) + ": " + err_match.group(2) + "\n"
					else:
						errors += "\n  "+err + "\n"
	print("o " + str(valid_files) + " documents pass XSD validation" + " "*30)
	if len(errors) > 0:
		print(errors)
		print("Aborting due to validation errors")
		sys.exit()


# helper function to recursively count tokens within an element (e.g. sentence) in xml, potentially nested in other elements
def count_tokens(e):
	tok_count = 0
	lines = e.text.split('\n') + e.tail.split('\n')
	for line in lines:
		if line.count('\t') > 0	:
			tok_count += 1
	for child in e:
		tok_count += count_tokens(child)
	return tok_count


def check_reddit(gum_source):

	reddit_docs = glob.glob(gum_source + "xml" + os.sep + "GUM_reddit*.xml")
	if len(reddit_docs) == 0:
		return False
	else:
		first_reddit = io.open(reddit_docs[0],encoding="utf8").read()
		num_underscores = first_reddit.count("_")
		if num_underscores > 2000:
			return False
		else:
			return True


def validate_src(gum_source, reddit=False):

	dirs = [('xml', 'xml'), ('dep', 'conll10'), ('rst', 'rs3'), ('tsv', 'tsv')]

	# check that each dir has same # and names of files (except extensions)
	file_lists = []
	for dir in dirs:
		dir_name = gum_source + dir[0]
		dir_ext = dir[1]
		filenames = []
		for filename in glob.glob(dir_name + os.sep + '*.' + dir_ext):
			if not reddit and "reddit_" in filename:
				continue
			basename = ntpath.basename(filename)
			filename_validate = re.match(r'(\w+)\.' + dir_ext, basename)
			if filename_validate is None:
				print('x Unexpected filename: ' + filename)
			else:
				filenames.append(filename_validate.group(1))

		sys.stdout.write("Found "+ str(len(filenames)) +" in "+ dir_ext + "\r")
		file_lists.append(filenames)
	
	# check that filenames are the same across dirs
	if all(len(x)==len(file_lists[0]) for x in file_lists) is False:
		print('x Different numbers of files in directories:')
		for d in range(len(dirs)):
			print(str(dirs[d][0]) + ": " + str(len(file_lists[d])))
		exit()
	else:
		for i in range(len(file_lists[0])):
			same_names = all(x[i]==file_lists[0][i] for x in file_lists)
			if same_names is False:
				print('Different filenames:')
				for d in range(len(dirs)):
					print(str(dirs[d][0]) + ": " + file_lists[d][i])
	reddit_string = " (excluding reddit)" if not reddit else ""
	print("o Found " + str(len(file_lists[0])) + " documents" + reddit_string)
	print("o File names match")
	
	# check # of tokens
	sys.stdout.write("Checking identical token counts...\r")
	all_tok_counts = []
	for d in range(len(dirs)):
		dir_tok_counts = []
		filenames = file_lists[d]
		for filename in filenames:
			filepath = gum_source + str(dirs[d][0]) + os.sep + filename + "." + dirs[d][1]
			with io.open(filepath,encoding="utf-8") as this_file:
				file_lines = this_file.readlines()
	
				if dirs[d][0] == 'xml':
					tok_count = 0
					for line in file_lines:
						if line.count('\t') > 0:
							tok_count += 1
					dir_tok_counts.append(tok_count)
	
				elif dirs[d][0] == 'dep':
					tok_count = 0
					for line in file_lines:
						if line.count('\t') == 9:
							tok_count += 1
					dir_tok_counts.append(tok_count)
	
				# rst -- use xml reader, add up space-split counts of segment.text
				elif dirs[d][0] == 'rst':
					tok_count = 0
					tree = ET.parse(filepath)
					root = tree.getroot()
					for segment in root.iter('segment'):
						# seg_text = re.sub(r'^\s*', r'', segment.text)
						# seg_text = re.sub(r'\s*$', r'', seg_text)
						seg_tok_count = len(segment.text.strip().split(" "))
						tok_count += seg_tok_count
					dir_tok_counts.append(tok_count)
	
				elif dirs[d][0] == 'tsv':
					tok_count = 0
					for line in file_lines:
						if line.count('\t') > 0:
							tok_count += 1
					dir_tok_counts.append(tok_count)
	
	
		all_tok_counts.append(dir_tok_counts)

	token_counts_match = True
	for i in range(len(all_tok_counts[0])):
		same_count = all(x[i]==all_tok_counts[0][i] for x in all_tok_counts)
		if same_count is False:
			print("x Different token counts in " + file_lists[0][i] + ":")
			for d in range(len(all_tok_counts)):
				print(str(dirs[d][0]) + ": " + str(all_tok_counts[d][i]))
				token_counts_match = False
	if token_counts_match:
		print("o Token counts match across directories")
	
	# check sentences (based on tok count)
	all_sent_lengths = []
	sentence_dirs = [('xml', 'xml'), ('dep', 'conll10')] # just the dirs where we check sentences
	
	for d in range(len(sentence_dirs)):
		dir_sent_lengths = []
		filenames = file_lists[d]
		for filename in filenames:
			filepath = str(sentence_dirs[d][0]) + os.sep + filename + "." + sentence_dirs[d][1]
			file_sent_lengths = []
			with io.open(gum_source + filepath,encoding="utf8") as this_file:
	
				if sentence_dirs[d][0] == 'xml':
					tree = ET.parse(gum_source + filepath)
					root = tree.getroot()
					for s in root.iter('s'):
						sent_length = count_tokens(s)
	
						file_sent_lengths.append(sent_length)
	
				elif sentence_dirs[d][0] == 'dep':
					file_text = this_file.read().strip() + "\n\n"
					sentences = file_text.split('\n\n')
					for sent in sentences[:-1]:
						sent_lines = sent.splitlines()
						sent_length = 0
						for line in sent_lines:
							if line.count('\t') == 9:
								sent_length += 1
						file_sent_lengths.append(sent_length)
	
			dir_sent_lengths.append(file_sent_lengths)
		all_sent_lengths.append(dir_sent_lengths)
	
	for i in range(len(all_sent_lengths[0])):
		same_lengths = all(x[i] == all_sent_lengths[0][i] for x in all_sent_lengths)
		if not same_lengths:
			print("Different sentence lengths in " + file_lists[0][i] + ":")
			for d in range(len(all_sent_lengths)):
				print(str(dirs[d][0]) + ": " + str(all_sent_lengths[d][i]))

	try:
		import lxml
		sys.stdout.write("Perfoming XSD validation of XML files:\r")
		filenames = list(file_ + ".xml" for file_ in file_lists[0])
		validate_xsd(filenames, gum_source)
	except ImportError:
		print("i WARN: module lxml is not installed")
		print("i Skipping XSD validation of XML files")
		print("i (to fix this warning: pip install lxml)")

	validate_annos(gum_source, reddit)
	sys.stdout.write("\r" + " "*40)


def validate_annos(gum_source, reddit=False):
	xml_source = gum_source + "xml" + os.sep

	xmlfiles = []
	files_ = glob.glob(xml_source + "*.xml")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		xmlfiles.append(file_)

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " " * 40 + "\r")
		sys.stdout.write(" " + str(docnum + 1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r")

		# Dictionaries to hold token annotations from conll10 data
		funcs = {}
		tokens = {}
		parent_ids = {}
		lemmas = {}
		sent_types = {}
		parents = {}
		children = defaultdict(list)
		child_funcs = defaultdict(list)
		tok_num = 0

		depfile = xmlfile.replace("xml" + os.sep, "dep" + os.sep).replace("xml", "conll10")
		dep_lines = io.open(depfile,encoding="utf8").read().replace("\r", "").split("\n")
		line_num = 0
		sent_start = 0
		for r, line in enumerate(dep_lines):
			line_num += 1
			if "\t" in line:  # token line
				if line.count("\t") != 9:
					# Shouldn't be possible, since file validation is already complete
					pass
				else:
					tok_num += 1
					fields = line.split("\t")
					funcs[tok_num] = fields[7]
					if fields[6] != "0":  # Root token
						if fields[6] == "_":
							print("Invalid head '_' at line " + str(r) + " in " + depfile)
							sys.exit()
						parent_ids[tok_num] = int(fields[6]) + sent_start
						children[int(fields[6]) + sent_start].append(fields[1])
						child_funcs[int(fields[6]) + sent_start].append(fields[7])
					else:
						parent_ids[tok_num] = 0
					tokens[tok_num] = fields[1]
			elif len(line) == 0:
				sent_start = tok_num

		for i in range(1, len(tokens) + 1, 1):
			if parent_ids[i] == 0:
				parents[i] = "ROOT"
			else:
				parents[i] = tokens[parent_ids[i]]

		xml_lines = io.open(xmlfile, encoding="utf8").read().replace("\r", "").split("\n")
		tok_num = 0

		s_type = ""
		for line in xml_lines:
			if "\t" in line:  # Token
				tok_num += 1
				lemmas[tok_num] = line.split("\t")[2]
				sent_types[tok_num] = s_type
			else:
				m = re.search(r's type="([^"]+)"', line)
				if m is not None:
					s_type = m.group(1)

		tok_num = 0

		for line in xml_lines:
			if "\t" in line:  # Token
				tok_num += 1
				func = funcs[tok_num]
				fields = line.split("\t")
				tok, pos, lemma = fields[0:3]
				parent_string = parents[tok_num]
				parent_id = parent_ids[tok_num]
				parent_lemma = lemmas[parent_ids[tok_num]] if parent_ids[tok_num] != 0 else ""
				flag_dep_warnings(tok_num, tok, pos, lemma, func, parent_string, parent_lemma, parent_id,
								  children[tok_num], child_funcs[tok_num], sent_types[tok_num], docname)


		# Validate WebAnno TSV data
		coref_file = xmlfile.replace("xml" + os.sep, "tsv" + os.sep).replace("xml", "tsv")
		coref_lines = io.open(coref_file,encoding="utf8").read().replace("\r", "").split("\n")

		markables = {}
		antecedents = {}

		for line in coref_lines:
			if "\t" in line:  # Token
				fields = line.strip().split("\t")
				entity_str, infstat_str, coref_str, src_str = fields[-4:]
				if entity_str != "":  # Entity annotation found
					entities = entity_str.split("|")
					infstats = infstat_str.split("|")
					corefs = coref_str.split("|")
					srcs = src_str.split("|")
					tok_id = fields[0]
					text = fields[2]
					for i, entity in enumerate(entities):
						infstat = infstats[i]
						if isinstance(corefs,list):
							coref = corefs[i] if i < len(corefs) else corefs[-1]
							src = srcs[i] if i < len(srcs) else srcs[-1]
						else:
							coref = corefs
							src = srcs
						if not entity.endswith("]"):
							# Single token entity
							id = tok_id
						else:
							id = re.search(r'\[([0-9]+)\]',entity).group(1)
							entity = re.sub(r'\[[^\]]+\]',"",entity)
							infstat = re.sub(r'\[[^\]]+\]',"",infstat)
							coref = re.sub(r'\[[^\]]+\]', "", coref)
						if id not in markables:
							markables[id] = Markable()
							markables[id].start = tok_id
						markables[id].text += " " + text
						markables[id].end = tok_id
						markables[id].entity = entity
						markables[id].anaphor = src
						markables[id].infstat = infstat
						markables[id].anaphor_type = coref

		for mark_id in markables:
			mark = markables[mark_id]
			mark.text = mark.text.strip()
			src = mark.anaphor
			src_tok = re.sub(r'\[.*', '', src)
			if "[" in src:
				target = re.search(r'_([0-9]+)\]', src).group(1)
				if target == mark_id:
					if "[0_" in src:  # source is single token markable
						antecedents[src_tok] = mark_id
					else:  # source is a multi-token markable
						src_id = re.search(r'\[([0-9]+)_', src).group(1)
						antecedents[src_id] = mark_id
			else:  # source and target are single tokens
				antecedents[src_tok] = mark_id

		for anaphor in antecedents:
			if anaphor != "_":
				markables[anaphor].antecedent = markables[antecedents[anaphor]]
				markables[anaphor].coref_type = markables[antecedents[anaphor]].anaphor_type

		for mark_id in markables:
			flag_mark_warnings(markables[mark_id], docname)

		# Validate RST data
		rst_file = xmlfile.replace("xml" + os.sep, "rst" + os.sep).replace("xml", "rs3")
		rst_lines = io.open(rst_file,encoding="utf8").read().replace("\r", "").split("\n")

		nodes = {}
		children = defaultdict(list)

		# TODO: implement with XML parser instead of less robust regex
		for line in rst_lines:
			m = re.search(r'<segment id="([0-9]+)" parent="([0-9]+)" relname="([^"]+)">([^<]+)',line)  # EDU segment
			if m is not None:
				node = rstNode()
				node.id = int(m.group(1))
				node.parent = int(m.group(2))
				node.rel = m.group(3)
				node.type = "edu"
				node.text = m.group(4)
				node.left = node.id
				node.right = node.id
				nodes[node.id] = node
			m = re.search(r'<group id="([0-9]+)" type="([^"]+)" parent="([0-9]+)" relname="([^"]+)"',line)  # group
			if m is not None:
				node = rstNode()
				node.id = int(m.group(1))
				node.parent = int(m.group(3))
				node.rel = m.group(4)
				node.type = m.group(2)
				nodes[node.id] = node
			m = re.search(r'<group id="([0-9]+)" type="([^"]+)"', line)
			if m is not None and "parent" not in line:  # ROOT
				node = rstNode()
				node.id = int(m.group(1))
				node.parent = 0
				node.type = m.group(2)
				nodes[node.id] = node


		for node in nodes:
			children[nodes[node].parent].append(node)

		flag_rst_warnings(nodes,children,docname)


def flag_rst_warnings(nodes,children,docname):
	for node in children:
		if node > 0:
			if nodes[node].type=="span":
				if len(children[node]) == 1:
					if nodes[children[node][0]].type == "span" and len(children[children[node][0]])==1:
						print("WARN: RST span with single span child in " + docname + " (node "+ str(nodes[node].id) +")")

	for node in children:
		if node > 0:
			if nodes[node].type != "multinuc":
				if len(children[node])>1:
					found_children = 0
					for child in children[node]:
						if nodes[child].rel != "span":
							found_children += 1
					if found_children > 1:
						print("WARN: RST non-multinuc with multiple non-span children in " + docname + " (node "+ str(nodes[node].id) +")")


def flag_mark_warnings(mark, docname):
	inname = " in " + docname

	# General checks for all markables
	if isinstance(mark.antecedent,Markable):
		if mark.infstat == "new" and mark.coref_type != "bridge" and mark.coref_type != "cata":
			print("WARN: new markable has an antecedent"+inname + ", " + mark.start + "=" + mark.entity + " -> " + \
				  str(mark.antecedent.start) + "=" + mark.antecedent.entity + \
				  " (" + truncate(mark.text) + "->" + truncate(mark.antecedent.text) +")")

	# Inspect markables that have antecedents
	# if isinstance(mark.antecedent,Markable): // We don't need a second statement for this, do we?
		if mark.antecedent.entity != mark.entity and mark.coref_type != "bridge":
			print("WARN: coref clash" +inname + ", " + mark.start + "=" + mark.entity + " -> " + \
				  str(mark.antecedent.start) + "=" + mark.antecedent.entity + \
				  " (" + truncate(mark.text) + "->" + truncate(mark.antecedent.text) +")")


def truncate(text):
	words = text.split()
	if len(words) > 5:
		words = words[0:5]
		words.append("...")
	return " ".join(words)


def flag_dep_warnings(id, tok, pos, lemma, func, parent, parent_lemma, parent_id, children, child_funcs, s_type,
					  docname):
	# Shorthand for printing errors
	inname = " in " + docname + " @ token " + str(id) + " (" + parent + " -> " + tok + ")"

	if re.search(r"VH.*", pos) is not None and lemma != "have":
		print(str(id) + docname)
		print("WARN: VH.* must be 'have' & not lemma " + lemma + inname)
	if re.search(r"VB.*", pos) is not None and lemma != "be":
		print(str(id) + docname)
		print("WARN: VB.* must be 'be' & not lemma " + lemma + inname)
	if re.search(r"VV.*", pos) is not None and lemma == "be":
		print(str(id) + docname)
		print("WARN: VV.* must not be 'be'" + inname)
	if re.search(r"VV.*", pos) is not None and lemma == "have":
		print(str(id) + docname)
		print("WARN: VV.* must not be 'have'" + inname)

	if func == 'mwe' and id < parent_id:
		print("WARN: back-pointing func mwe" + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func == 'conj' and id < parent_id:
		print("WARN: back-pointing func conj" + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func == "auxpass" and lemma!= "be" and lemma != "get":
		print("WARN: auxpass must be 'be' or 'get'" + inname)

	if func == "possessive" and pos!= "POS":
		print("WARN: possessive function must be tagged POS" + inname)

	if func != "possessive" and pos== "POS":
		print("WARN: tag POS must have function possessive" + inname)

	if re.search(r"never|not|no|n't|n’t|’t|'t|nt", tok, re.IGNORECASE) is None and func == "neg":
		print(str(id) + docname)
		print("WARN: mistagged negative" + inname)

	be_funcs = ["cop", "aux", "root", "csubj", "auxpass", "rcmod", "ccomp", "advcl", "conj","xcomp","parataxis","vmod","pcomp"]
	if lemma == "be" and func not in be_funcs:
		if not parent_lemma == "that" and func=="mwe":  # Exception for 'that is' as mwe
			print("WARN: invalid dependency of lemma 'be' > " + func + inname)

	if func == "aux" and lemma != "be" and lemma != "have" and lemma !="do" and pos!="MD" and pos!="TO":
		print("WARN: aux must be modal, 'be,' 'have,' or 'do'" + inname)

	if re.search(r"“|”|n’t|n`t|[’`](s|ve|d|ll|m|re|t)", lemma, re.IGNORECASE) is not None:
		print(str(id) + docname)
		print("WARN: non-ASCII character in lemma" + inname)

	if pos == "POS" and lemma != "'s":
		print(str(id) + docname)
		print("WARN: tag POS must have lemma " +'"'+ "'s" + '"' + inname)


	mwe_pairs = {("accord", "to"), ("all","but"), ("as","if"), ("as", "well"), ("as", "as"), ("as","in"), ("as","oppose"),("as","to"),
				 ("at","least"),("because","of"),("due","to"),("had","better"),("'d","better"),("in","between"),
				 ("in","case"),("in","of"), ("in","order"),("instead","of"), ("kind","of"),("less","than"),("let","alone"),
				 ("more","than"),("not","to"),("not","mention"),("of","course"),("prior","to"),("rather","than"),("so","as"),
				 ("so", "to"),("sort", "of"),("so", "that"),("such","as"),("that","is"), ("up","to"),("whether","or"),
				 ("whether","not"),("depend","on"),("out","of"),("more","than"),("on","board"),("as","of"),("depend","upon"),
				 ("that","be"),("just","about"),("vice","versa"),("as","such"),("next","to")}

	# Ad hoc listing of triple mwe parts - All in all, in order for
	mwe_pairs.update({("all","in"),("all","all"),("in","for")})

	if func == "mwe":
		if (parent_lemma.lower(), lemma.lower()) not in mwe_pairs:
			print("WARN: unlisted mwe" + inname)

	#if pos != "CD" and "quantmod" in child_funcs:
	#	print("WARN: quantmod must be cardinal number" + inname)

	if tok == "sort" or tok == "kind":
		if "det" in child_funcs and "mwe" in child_funcs:
			print("WARN: mistagged mwe" + inname)

	if tok == "rather" and "mwe" in child_funcs and func != "cc":
		print("WARN: 'rather than' mwe must be cc" + inname)

	if s_type == "imp" or s_type == "frag" or s_type == "ger" or s_type == "inf":
		if func == "root" and "nsubj" in child_funcs:
			print("WARN: " + s_type + " root may not have nsubj" + inname)

	temp_wh = ["when", "how", "where", "why", "whenever", "while", "who", "whom", "which", "whoever", "whatever",
			   "what", "whomever", "however"]

	#if s_type == "wh" and func == "root":
	#	tok_count = 0							#This is meant to keep it from printing an error for every token.
	#	if tok.lower() not in temp_wh:
	#		for wh in children:
	#			if re.search(r"when|how|where|why|whenever|while|who.*|which|what.*", wh, re.IGNORECASE) is None:
	#				tok_count += 1
	#		if tok_count == len(children):
	#			print("WARN: wh root must have wh child" + inname)

	if s_type == "q" and func == "root":
		for wh in children:
			if wh in temp_wh:
				print("WARN: q root may not have wh child " + wh + inname)

