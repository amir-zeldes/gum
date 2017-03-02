# -*- coding: utf-8 -*-

import os, ntpath, sys
import shutil
from glob import glob
import re
from collections import defaultdict
import xml.etree.ElementTree as ET


# Function to validate list of XML files against XSD schema
def validate_xsd(file_list, gum_source):
	from lxml import etree
	with open(gum_source + "gum_schema.xsd", 'r') as f:
		schema_root = etree.XML(f.read())

	valid_files = 0
	errors = ""
	schema = etree.XMLSchema(schema_root)

	for docnum, xml_file in enumerate(file_list):
		print "\t+ " + " " * 40 + "\r",
		print " " + str(docnum + 1) + "/" + str(len(file_list)) + ":\t+ Validating " + xml_file + "\r",

		try:
			with open(gum_source + "xml" + os.sep + xml_file, 'r') as f:
				root = etree.parse(f)
				schema.validate(root)
		finally:
			if len(schema.error_log.filter_from_errors()) == 0:
				valid_files += 1
			else:
				errors += "\n" + xml_file + " has errors:\n"
				for err in schema.error_log.filter_from_errors():
					err_match = re.search(r'\.xml:([0-9]+).*?: (.*)', str(err))
					if err_match is not None:
						errors += "  Line " + err_match.group(1) + ": " + err_match.group(2) + "\n"
					else:
						errors += "\n  " + err + "\n"
	print "o " + str(valid_files) + " documents pass XSD validation" + " " * 30
	if len(errors) > 0:
		print errors
		print "Aborting due to validation errors"
		sys.exit()


# helper function to recursively count tokens within an element (e.g. sentence) in xml, potentially nested in other elements
def count_tokens(e):
	tok_count = 0
	lines = e.text.split('\n') + e.tail.split('\n')
	for line in lines:
		if line.count('\t') > 0:
			tok_count += 1
	for child in e:
		tok_count += count_tokens(child)
	return tok_count


def validate_src(gum_source):
	dirs = [('xml', 'xml'), ('dep', 'conll10'), ('rst', 'rs3'), ('tsv', 'tsv')]

	# check that each dir has same # and names of files (except extensions)
	file_lists = []
	for dir in dirs:
		dir_name = gum_source + dir[0]
		dir_ext = dir[1]
		filenames = []
		for filename in glob(dir_name + os.sep + '*.' + dir_ext):
			basename = ntpath.basename(filename)
			filename_validate = re.match(r'(\w+)\.' + dir_ext, basename)
			if filename_validate is None:
				print 'x Unexpected filename: ' + filename
			else:
				filenames.append(filename_validate.group(1))

		print "Found " + str(len(filenames)) + " in " + dir_ext + "\r",
		file_lists.append(filenames)

	# check that filenames are the same across dirs
	if all(len(x) == len(file_lists[0]) for x in file_lists) is False:
		print 'x Different numbers of files in directories:'
		for d in xrange(len(dirs)):
			print str(dirs[d][0]) + ": " + str(len(file_lists[d]))
		exit()
	else:
		for i in xrange(len(file_lists[0])):
			same_names = all(x[i] == file_lists[0][i] for x in file_lists)
			if same_names is False:
				print 'Different filenames:'
				for d in xrange(len(dirs)):
					print str(dirs[d][0]) + ": " + file_lists[d][i]
	print "o Found " + str(len(file_lists[0])) + " documents"
	print "o File names match"

	# check # of tokens
	print "Checking identical token counts...\r",
	all_tok_counts = []
	for d in xrange(len(dirs)):
		dir_tok_counts = []
		filenames = file_lists[d]
		for filename in filenames:
			filepath = gum_source + str(dirs[d][0]) + os.sep + filename + "." + dirs[d][1]
			# filename = os.path.abspath(filepath)
			with open(filepath) as this_file:
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
	for i in xrange(len(all_tok_counts[0])):
		same_count = all(x[i] == all_tok_counts[0][i] for x in all_tok_counts)
		if same_count is False:
			print "x Different token counts in " + file_lists[0][i] + ":"
			for d in xrange(len(all_tok_counts)):
				print str(dirs[d][0]) + ": " + str(all_tok_counts[d][i])
				token_counts_match = False
	if token_counts_match:
		print "o Token counts match across directories"

	# check sentences (based on tok count)
	all_sent_lengths = []
	sentence_dirs = [('xml', 'xml'), ('dep', 'conll10')]  # just the dirs where we check sentences

	for d in xrange(len(sentence_dirs)):
		dir_sent_lengths = []
		filenames = file_lists[d]
		for filename in filenames:
			filepath = str(sentence_dirs[d][0]) + os.sep + filename + "." + sentence_dirs[d][1]
			file_sent_lengths = []
			with open(gum_source + filepath) as this_file:

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

	for i in xrange(len(all_sent_lengths[0])):
		same_lengths = all(x[i] == all_sent_lengths[0][i] for x in all_sent_lengths)
		if not same_lengths:
			print "Different sentence lengths in " + file_lists[0][i] + ":"
			for d in xrange(len(all_sent_lengths)):
				print str(dirs[d][0]) + ": " + str(all_sent_lengths[d][i])

	try:
		import lxml
		print "Perfoming XSD validation of XML files:\r",
		filenames = list(file_ + ".xml" for file_ in file_lists[0])
		validate_xsd(filenames, gum_source)
	except ImportError:
		print "i WARN: module lxml is not installed"
		print "i Skipping XSD validation of XML files"
		print "i (to fix this warning: pip install lxml)"


def validate_annos(gum_source):
	xml_source = gum_source + "xml" + os.sep

	xmlfiles = glob(xml_source + "*.xml")

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		print "\t+ " + " " * 40 + "\r",
		print " " + str(docnum + 1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r",

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
		dep_lines = open(depfile).read().replace("\r", "").split("\n")
		line_num = 0
		sent_start = 1
		for line in dep_lines:
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
						parent_ids[tok_num] = int(fields[6]) + sent_start
						children[int(fields[6]) + sent_start].append(fields[1])
						child_funcs[int(fields[6]) + sent_start].append(fields[7])
					else:
						parent_ids[tok_num] = 0
					tokens[tok_num] = fields[1]
			elif len(line) == 0:
				sent_start = tok_num

		for i in xrange(1, len(tokens) + 1, 1):
			if parent_ids[i] == 0:
				parents[i] = "ROOT"
			else:
				parents[i] = tokens[parent_ids[i]]

		xml_lines = open(xmlfile).read().replace("\r", "").split("\n")
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


def flag_dep_warnings(id, tok, pos, lemma, func, parent, parent_lemma, parent_id, children, child_funcs, s_type,
					  docname):
	# Shorthand for printing errors
	inname = " in " + docname + " @ token " + str(id) + " (" + parent + " -> " + tok + ")"

	if re.search(r"VH.*", pos) is not None and lemma != "have":
		print "WARN: VH.* must be 'have' & not lemma " + lemma + inname
	if re.search(r"VB.*", pos) is not None and lemma != "be":
		print "WARN: VB.* must be 'be' & not lemma " + lemma + inname
	if re.search(r"VV.*", pos) is not None and lemma == "be":
		print "WARN: VV.* must not be 'be'" + inname
	if re.search(r"VV.*", pos) is not None and lemma == "have":
		print "WARN: VV.* must not be 'have'" + inname

	if func == 'mwe' and id < parent_id:
		print "WARN: back-pointing func mwe" + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")"

	if func == "auxpass" and lemma!= "be" and lemma != "get":
		print "WARN: auxpass must be 'be' or 'get'" + inname

	if re.search(r"never|not|no|n't|n’t|’t|'t", tok, re.IGNORECASE) is None and func == "neg":
		print "WARN: mistagged negative" + inname

	be_funcs = ["cop", "aux", "root", "csubj", "auxpass", "rcmod", "ccomp", "advcl", "conj","xcomp","parataxis","vmod","pcomp"]
	if lemma == "be" and func not in be_funcs:
		print "WARN: invalid dependency of lemma 'be' > " + func + inname

	if func == "aux" and lemma != "be" and lemma != "have" and lemma !="do" and pos!="MD" and pos!="TO":
		print "WARN: aux must be modal, 'be,' 'have,' or 'do'" + inname

	if re.search(r"“|”|…|n’t|n`t|[’`](s|ve|d|ll|m|re|t)", lemma, re.IGNORECASE) is not None:
		print "WARN: non-ASCII character in lemma" + inname

	mwe_pairs = [("accord", "to"), ("all","but"), ("as","if"), ("as", "well"), ("as", "as"), ("as","oppose"),("as","to"),
				 ("at","least"),("because","of"),("due","to"),("had","better"),("'d","better"),("in","between"),
				 ("in","case"),("in","of"), ("in","order"),("instead","of"), ("kind","of"),("less","than"),("let","alone"),
				 ("more","than"),("not","to"),("not","mention"),("of","course"),("prior","to"),("rather","than"),("so","as"),
				 ("so", "to"),("sort", "of"),("so", "that"),("such","as"),("that","is"), ("up","to"),("whether","or"),
				 ("whether","not")]
	if func == "mwe":
		if (parent_lemma, lemma) not in mwe_pairs:
			print "WARN: mistagged mwe" + inname

	#if pos != "CD" and "quantmod" in child_funcs:
	#	print "WARN: quantmod must be cardinal number" + inname

	if tok == "sort" or tok == "kind":
		if "det" in child_funcs and "mwe" in child_funcs:
			print "WARN: mistagged mwe" + inname

	if tok == "rather" and "mwe" in child_funcs and func != "cc":
		print "WARN: 'rather than' mwe must be cc" + inname

	if s_type == "imp" or s_type == "frag" or s_type == "ger" or s_type == "inf":
		if func == "root" and "nsubj" in child_funcs:
			print "WARN: " + s_type + " root may not have nsubj" + inname

	temp_wh = ["when", "how", "where", "why", "whenever", "while", "who", "whom", "which", "whoever", "whatever",
			   "what", "whomever", "however"]

	#if s_type == "wh" and func == "root":
	#	tok_count = 0							#This is meant to keep it from printing an error for every token.
	#	if tok.lower() not in temp_wh:
	#		for wh in children:
	#			if re.search(r"when|how|where|why|whenever|while|who.*|which|what.*", wh, re.IGNORECASE) is None:
	#				tok_count += 1
	#		if tok_count == len(children):
	#			print "WARN: wh root must have wh child" + inname

	if s_type == "q" and func == "root":
		for wh in children:
			if wh in temp_wh:
				print "WARN: q root may not have wh child " + wh + inname
