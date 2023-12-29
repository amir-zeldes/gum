#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, ntpath, sys
import glob
import re
import xml.etree.ElementTree as ET
import io
from collections import defaultdict
from six import iterkeys
from rst2dep import make_rsd, rsd2rs3

class Markable:
	def __init__(self):
		self.start = 0
		self.end = 0
		self.text = ""
		self.entity = ""
		self.infstat = ""
		self.antecedent = ""
		self.coref_type = ""
		self.anaphor = []
		self.anaphor_type = []

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
		sys.stdout.write("\t+ " + " "*70 + "\r")
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
	print("o " + str(valid_files) + " documents pass XSD validation" + " "*70)
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

	lemma_dict = defaultdict(lambda : defaultdict(int))  # collects tok+pos -> lemmas -> count  for consistency checks
	lemma_docs = defaultdict(set)
	rst_extension = "rs4" if os.path.exists(gum_source + "rst" + os.sep + "GUM_academic_art.rs4") else "rs3"
	dirs = [('xml', 'xml'), ('dep', 'conllu'), ('rst', rst_extension), ('tsv', 'tsv')]

	# check that each dir has same # and names of files (except extensions)
	file_lists = []
	for dir in dirs:
		dir_name = gum_source + dir[0]
		dir_ext = dir[1]
		filenames = []
		for filename in sorted(glob.glob(dir_name + os.sep + '*.' + dir_ext)):
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
					doc = os.path.basename(filename).replace(".xml", "")
					tok_count = 0
					for line in file_lines:
						if line.count('\t') > 0:
							tok_count += 1
							tok, pos, lemma = line.strip().split("\t")
							lemma_dict[(tok,pos)][lemma] += 1
							lemma_docs[(tok,pos,lemma)].add(doc)
					dir_tok_counts.append(tok_count)
	
				elif dirs[d][0] == 'dep':
					tok_count = 0
					for line in file_lines:
						if line.count('\t') == 9:
							if "." not in line.split("\t")[0]:  # Ignore ellipsis tokens
								tok_count += 1
					dir_tok_counts.append(tok_count)
	
				# rst -- use xml reader, add up space-split counts of segment.text
				elif dirs[d][0] == 'rst':
					tok_count = 0
					try:
						tree = ET.parse(filepath)
					except Exception as e:
						sys.stderr.write("Can't parse XML file: " + filepath+"\n")
						sys.stderr.write(str(e))
						sys.exit(0)

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
	sentence_dirs = [('xml', 'xml'), ('dep', 'conllu')] # just the dirs where we check sentences
	
	for d in range(len(sentence_dirs)):
		dir_sent_lengths = []
		filenames = file_lists[d]
		for filename in filenames:
			filepath = str(sentence_dirs[d][0]) + os.sep + filename + "." + sentence_dirs[d][1]
			file_sent_lengths = []
			with io.open(gum_source + filepath,encoding="utf8") as this_file:
	
				if sentence_dirs[d][0] == 'xml':
					xml_string = open(gum_source + filepath).read()
					sents = xml_string.split("</s>")[:-1]
					for s in sents:
						file_sent_lengths.append(int(s.count("\t") / 2))
				elif sentence_dirs[d][0] == 'dep':
					file_text = this_file.read().strip() + "\n\n"
					sentences = file_text.split('\n\n')
					for sent in sentences[:-1]:
						sent_lines = sent.splitlines()
						sent_length = 0
						for line in sent_lines:
							if line.count('\t') == 9:
								if "." not in line.split("\t")[0]:  # Ignore ellipsis tokens
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
	validate_lemmas(lemma_dict,lemma_docs)
	sys.stdout.write("\r" + " "*70)

def validate_lemmas(lemma_dict, lemma_docs, use_neaten=False):
	"""
	use_neaten adds validations implemented by @nschneid motivated by EWT-specific tokens
	"""

	exceptions = [("Democratic","JJ","democratic"),("Water","NP","Waters"),("Sun","NP","Sunday"),("a","IN","of"),
		      ("a","IN","as"),("car","NN","card"),("lay","VV","lay"),("that","IN","than"),
		      ("da","NP","Danish"),("all","RB","alright"),("All","RB","alright"),("any","RB","anymore"),
			  ("before","RB","beforehand"),("any","RB","any"),("Black","JJ","black"),("wait","NN","wait")]
	if use_neaten:
		exceptions += [("Jan","NNP","Jan"),("Jan","NNP","January"),
		      ("'s","VBZ","have"),("’s","VBZ","have"),("`s","VBZ","have"),("'d","VBD","do"),("'d","VBD","have")]

	suspicious_types = 0
	for tok, pos in sorted(list(iterkeys(lemma_dict))):
		if len(lemma_dict[(tok,pos)]) > 1:
			for i, lem in enumerate(sorted(lemma_dict[(tok,pos)],key=lambda x:lemma_dict[(tok,pos)][x],reverse=True)):
				if lem == "_":
					continue
				docs = ", ".join(list(lemma_docs[(tok,pos,lem)]))
				if i == 0:
					majority = lem
				else:
					if (tok,pos,lem) not in exceptions:  # known exceptions
						suspicious_types += 1
						sys.stderr.write("! rare lemma " + lem + " for " + tok + "/" + pos + " in " + docs +
									 " (majority: " + majority + ")\n")
	if suspicious_types > 0:
		sys.stderr.write("! "+str(suspicious_types) + " suspicious lemma types detected\n")


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
		sys.stdout.write("\t+ " + " " * 70 + "\r")
		sys.stdout.write(" " + str(docnum + 1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r")

		# Dictionaries to hold token annotations from conllu data
		funcs = {}
		postags = {}
		tokens = {}
		parent_ids = {}
		lemmas = {}
		sent_types = {}
		sent_positions = defaultdict(lambda: "_")
		parents = {}
		children = defaultdict(list)
		child_funcs = defaultdict(list)
		tok_num = 0

		depfile = xmlfile.replace("xml" + os.sep, "dep" + os.sep).replace("xml", "conllu")
		dep_lines = io.open(depfile,encoding="utf8").read().replace("\r", "").split("\n")
		line_num = 0
		sent_start = 0
		for r, line in enumerate(dep_lines):
			line_num += 1
			if "\t" in line:  # token line
				if line.count("\t") != 9:
					# Shouldn't be possible, since file validation is already complete
					pass
				elif "." in line.split("\t")[0]:  # Ignore ellipsis tokens
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
		new_sent = True
		for line in xml_lines:
			if "\t" in line:  # Token
				tok_num += 1
				_, postags[tok_num], lemmas[tok_num] = line.split("\t")
				sent_types[tok_num] = s_type
				if new_sent:
					sent_positions[tok_num] = "first"
					new_sent = False
			else:
				m = re.search(r's type="([^"]+)"', line)
				if m is not None:
					s_type = m.group(1)
					new_sent = True
					if len(sent_positions) > 0:
						sent_positions[tok_num] = "last"
		sent_positions[tok_num] = "last"

		tok_num = 0

		# Extended PTB (TT/AMALGAM) tagset with HYPH
		tagset = ["CC","CD","DT","EX","FW","IN","IN/that","JJ","JJR","JJS","LS","MD","NN","NNS","NP","NPS","PDT","POS",
				  "PP","PP$","RB","RBR","RBS","RP","SENT","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","VH","VHD",
				  "VHG","VHN","VHP","VHZ","VV","VVD","VVG","VVN","VVP","VVZ","WDT","WP","WP$","WRB","``","''","(",")",
				  ",",":","HYPH","$","GW"]
		non_lemmas = ["them","me","him","n't"]
		non_lemma_combos = [("PP","her"),("MD","wo"),("PP","us"),("DT","an")]
		lemma_pos_combos = {"which":"WDT"}
		non_cap_lemmas = ["There","How","Why","Where","When"]

		prev_tok = ""
		prev_pos = ""
		for i, line in enumerate(xml_lines):
			if "\t" in line:  # Token
				tok_num += 1
				func = funcs[tok_num]
				fields = line.split("\t")
				tok, pos, lemma = fields[0:3]
				if pos not in tagset:
					print("WARN: invalid POS tag " + pos + " in " + docname + " @ line " + str(i) + " (token: " + tok + ")")
				if lemma.lower() in non_lemmas:
					print("WARN: invalid lemma " + lemma + " in " + docname + " @ line " + str(i) + " (token: " + tok + ")")
				elif lemma in non_cap_lemmas:
					print("WARN: invalid lemma " + lemma + " in " + docname + " @ line " + str(i) + " (token: " + tok + ")")
				elif (pos,lemma.lower()) in non_lemma_combos:
					print("WARN: invalid lemma " + lemma + " for POS "+pos+" in " + docname + " @ line " + str(i) + " (token: " + tok + ")")
				elif lemma in lemma_pos_combos:
					if pos != lemma_pos_combos[lemma]:
						print("WARN: invalid pos " + pos + " for lemma "+lemma+" in " + docname + " @ line " + str(i) + " (token: " + tok + ")")

				parent_string = parents[tok_num]
				parent_id = parent_ids[tok_num]
				parent_lemma = lemmas[parent_ids[tok_num]] if parent_ids[tok_num] != 0 else ""
				parent_func = funcs[parent_ids[tok_num]] if parent_ids[tok_num] != 0 else ""
				parent_pos = postags[parent_ids[tok_num]] if parent_ids[tok_num] != 0 else ""
				flag_dep_warnings(tok_num, tok, pos, lemma, func, parent_string, parent_lemma, parent_id,
								  children[tok_num], child_funcs[tok_num], sent_types[tok_num], docname,
								  prev_tok, prev_pos, sent_positions[tok_num], parent_func, parent_pos)
				prev_pos = pos
				prev_tok = tok

		# Validate WebAnno TSV data
		coref_file = xmlfile.replace("xml" + os.sep, "tsv" + os.sep).replace("xml", "tsv")
		coref_lines = io.open(coref_file,encoding="utf8").read().replace("\r", "").split("\n")

		markables = {}
		antecedents = defaultdict(list)
		single_tok_ids = False
		tid2abs = {}
		toknum = 0

		for line in coref_lines:
			if "\t" in line:  # Token
				fields = line.strip().split("\t")
				toknum += 1
				entity_str, infstat_str, salience_str, identity_str, coref_str, src_str = fields[-6:]

				if entity_str != "" and entity_str != "_":  # Entity annotation found
					entities = entity_str.split("|")
					infstats = infstat_str.split("|")
					saliences = salience_str.split("|")
					corefs = coref_str.split("|")
					srcs = src_str.split("|")
					tok_id = fields[0]
					text = fields[2]
					tid2abs[tok_id] = toknum
					for i, entity in enumerate(entities):
						try:
							infstat = infstats[i]
							salience = saliences[i]
						except:
							print("ERROR: " + docname)
							print("no infstat for entity: " + str(entity))
							quit()
						if isinstance(corefs,list):
							coref = corefs[i] if i < len(corefs) else corefs[-1]
							src = srcs[i] if i < len(srcs) else srcs[-1]
						else:
							coref = corefs
							src = srcs
						if not entity.endswith("]"):
							# Single token entity
							single_tok_ids = True
							id = tok_id
						else:
							id = re.search(r'\[([0-9]+)\]',entity).group(1)
							entity = re.sub(r'\[[^\]]+\]',"",entity)
							infstat = re.sub(r'\[[^\]]+\]',"",infstat)
							salience = re.sub(r'\[[^\]]+\]',"",salience)
							coref = re.sub(r'\[[^\]]+\]', "", coref)
						if id not in markables:
							markables[id] = Markable()
							markables[id].start = tok_id
							markables[id].anaphor.append(src)
							markables[id].anaphor_type.append(coref)
						markables[id].entity = entity
						markables[id].infstat = infstat
						markables[id].salience = salience
						markables[id].text += " " + text
						markables[id].end = tok_id

					# second pass: add the missing coref relation
					if srcs != ['_']:
						for coref, src in zip(corefs, srcs):
							candidate_id = src.strip(']').split('_')[-1]
							if candidate_id != "0" and src not in markables[candidate_id].anaphor:
								markables[candidate_id].anaphor.append(src)
								markables[candidate_id].anaphor_type.append(coref)

		# Ensure single token markables are given a tok_id-style identifier if the document uses this convention
		mark_ids = list(markables.keys())
		if single_tok_ids:
			for mark_id in mark_ids:
				mark = markables[mark_id]
				if mark.start == mark.end:
					markables[mark.start] = mark
					del markables[mark_id]

		for mark_id in markables:
			mark = markables[mark_id]
			mark.text = mark.text.strip()
			# Add markable head functions
			start_token = tid2abs[mark.start]
			end_token = tid2abs[mark.end]
			for i in range(mark.text.count(" ")+1):
				if parent_ids[start_token+i] < start_token or parent_ids[start_token+i] > end_token or parent_ids[start_token+i] == 0:  # Entity head?
					if funcs[start_token+i] != "punct":
						mark.func = funcs[start_token+i]
			if mark.func.endswith("tmod") and mark.entity != "time":
				if not (mark.entity == "event" and "time" in mark.text):
					print("! WARN: markable " + mark.text + " at " +docname + " token " + str(toknum) + " is " + \
						  mark.entity + " but has head deprel " + mark.func)

			srcs = mark.anaphor
			for src in srcs:
				src_tok = re.sub(r'\[.*', '', src)
				if "[" in src:
					target = re.search(r'_([0-9]+)\]', src).group(1)
					if target == mark_id:
						if "[0_" in src:  # source is single token markable
							for m in markables:
								if markables[m].start == src_tok:
									antecedents[m].append(mark_id)
									break
						else:  # source is a multi-token markable
							src_id = re.search(r'\[([0-9]+)_', src).group(1)
							# if src_id in antecedents:
							# 	raise ValueError(f'The entity {src_id} has multiple antecedents {antecedents[src_id]} and {mark_id}.')
							antecedents[src_id].append(mark_id)
				else:  # source and target are single tokens
					antecedents[src_tok].append(mark_id)

		for anaphor in antecedents:
			if anaphor != "_":
				for mark_id in antecedents[anaphor]:
					if mark_id == '109':
						a = 1
					try:
						markables[anaphor].antecedent = markables[mark_id]
						for i, src in enumerate(markables[mark_id].anaphor):
							if f'[{anaphor}_' in src:
								markables[anaphor].coref_type = markables[mark_id].anaphor_type[i]
					except KeyError as e:
						sys.stderr.write("Exception in " + docname + ": KeyError\n")
						markables[anaphor].antecedent = markables[mark_id]
						for i, src in enumerate(markables[mark_id].anaphor):
							if f'[{anaphor}_' in src:
								markables[anaphor].coref_type = markables[mark_id].anaphor_type[i]
							elif '[0_' in src and anaphor in src:
								markables[anaphor].coref_type = markables[antecedents[anaphor]].anaphor_type[i]

		for mark_id in markables:
			# Flag entity type clashes but skip giv/new since they are set automatically
			flag_mark_warnings(markables[mark_id], docname, flag_giv_new=False)

		# Validate RST data
		rst_file = xmlfile.replace("xml" + os.sep, "rst" + os.sep).replace("xml", "rs3")
		try:
			rst_xml = io.open(rst_file,encoding="utf8").read().replace("\r", "")
		except:
			rst_xml = io.open(rst_file.replace("rs3","rs4"),encoding="utf8").read().replace("\r", "")
		rst_lines = rst_xml.split("\n")

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

		# Run round-trip conversion to dependencies and back - valid, ordered hierarchy should produce same rs3<>rsd
		rsd1 = make_rsd(rst_xml, "", as_text=True)
		rsd1 = re.sub('\t[^\t]+\t[^\t]+\n',r'\t_\t_\n',rsd1)
		generated_rs3 = rsd2rs3(rsd1)
		rsd2 = make_rsd(generated_rs3, "", as_text=True)
		if rsd1 != rsd2:
			sys.stderr.write("! RST file " + docname + " not identical in rsd<>rs3 round-trip conversion; possible broken hierarchy!\n")


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


def flag_mark_warnings(mark, docname, flag_giv_new=False):
	"""
	Check inconsistent entity types and givenness

	:param mark: entity markable
	:param docname: document name to report errors
	:param flag_giv_new: whether to warn on new markable with antecedent/given markable without one
	:return: None
	"""

	inname = " in " + docname

	# General checks for all markables
	if isinstance(mark.antecedent,Markable):
		if mark.infstat == "new" and mark.coref_type != "bridge" and mark.coref_type != "cata" and flag_giv_new:
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
					  docname, prev_tok, prev_pos, sent_position, parent_func, parent_pos):
	# Shorthand for printing errors
	inname = " in " + docname + " @ token " + str(id) + " (" + parent + " -> " + tok + ")"

	if re.search(r"VH.*", pos) is not None and lemma != "have":
		print("WARN: VH.* must be 'have' & not lemma " + lemma + inname)
	if re.search(r"VB.*", pos) is not None and lemma != "be":
		print("WARN: VB.* must be 'be' & not lemma " + lemma + inname)
	if re.search(r"VV.*", pos) is not None and lemma == "be":
		print("WARN: VV.* must not be 'be'" + inname)
	if re.search(r"VV.*", pos) is not None and lemma == "have":
		print("WARN: VV.* must not be 'have'" + inname)

	if func == "amod" and pos in ["VBD","VVD","VHD"]:
		print("WARN: finite past verb labeled amod " + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func in ["amod", "det"] and parent_lemma == "one" and parent_pos == "CD":
		print("WARN: 'one' with " + func + " dependent should be NN/NOUN not CD/NUM in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if lemma != tok:
		if re.search(r'.*[a-z]+[A-Z]+.*',lemma) is not None:
			if lemma + "s" != tok: # plurals like YouTubers
				print("WARN: suspicious capitalization in lemma " + lemma + " for token " + tok + inname)
		elif pos in ["VBP","VB"] and lemma != "be":
			print("WARN: suspicious lemma should be identical to tok for lemma " + lemma + " with pos "+pos+" and token " + tok + inname)

	if func in ['fixed','goeswith','flat', 'conj'] and id < parent_id:
		print("WARN: back-pointing func " + func + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func in ['cc:preconj','cc','nmod:poss','reparandum'] and id > parent_id:
		if tok not in ["mia"]:
			print("WARN: forward-pointing func " + func + " in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func == "aux:pass" and lemma != "be" and lemma != "get":
		print("WARN: aux:pass must be 'be' or 'get'" + inname)

	if lemma == "'s" and pos != "POS":
		print("WARN: possessive 's must be tagged POS" + inname)

	if func not in ["case","reparandum","goeswith"] and pos == "POS":
		print("WARN: tag POS must have function case" + inname)

	if pos in ["VVG","VVN","VVD"] and lemma == tok:
		# check cases where VVN form is same as tok ('know' and 'notice' etc. are recorded typos, l- is a disfluency)
		if tok not in ["shed","put","read","become","come","overcome","cut","pre-cut","hit","split","cast","set","hurt","run","overrun","outrun","broadcast","knit",
			       "undercut","spread","shut","upset","burst","bit","bid","outbid","let","l-","g-","know","notice","reach","raise","beat","forecast"]:
			if not tok =="recommend" and "languages" in inname:
				print("WARN: tag "+pos+" should have lemma distinct from word form" + inname)

	if pos == "NPS" and tok == lemma and tok.endswith("s") and func != "goeswith":
		if tok not in ["Netherlands","Analytics","Olympics","Commons","Paralympics","Vans",
					   "Andes","Forties","Philippines","Maldives"]:
			print("WARN: tag "+pos+" should have lemma distinct from word form" + inname)

	if pos == "NNS" and tok.lower() == lemma.lower() and lemma.endswith("s") and func != "goeswith":
		if lemma not in ["surroundings","energetics","mechanics","politics","jeans","pants","trousers","clothes","electronics","means","feces","remains",
						 "biceps","triceps","news","species","economics","arrears","glasses","thanks","series","ergonomics","sunglasses",
						 "aesthetics","twenties","thirties","fourties","fifties","sixties","seventies","eighties","nineties"]:
			if re.match(r"[0-9]+'?s",lemma) is None:  # 1920s, 80s
				print("WARN: tag "+pos+" should have lemma distinct from word form" + inname)

	if pos == "IN" and func=="compound:prt":
		print("WARN: function " + func + " should have pos RP, not IN" + inname)

	if tok.lower() in ["yours","mine","hers","theirs","ours"] and (pos == "PP$" or pos == "PRP$" or lemma in ["yours","mine","hers","theirs","ours"]):
		print("WARN: substitutive possessive pronoun  "+tok+"/"+pos+"/"+lemma+" should have pos=PP and the plain possessive lemma" + inname)

	if pos in ["VBN","VVN","VHN"] and ("nsubj" in child_funcs or "csubj" in child_funcs) and lemma != "get" and \
			("aux:pass" not in child_funcs and "aux" not in child_funcs and "cop" not in child_funcs and "compound" not in child_funcs):
		print("WARN: passive verb tagged VBN without perfect auxiliary should have :pass subject, not regular subj" + inname)

	if pos == "CC" and func not in ["cc","cc:preconj","conj","reparandum","root","dep"] and not (parent_lemma=="whether" and func=="fixed"):
		if not ("languages" in inname and tok == "and"):  # metalinguistic discussion in whow_languages
			print("WARN: pos " + pos + " should normally have function cc or cc:preconj, not " + func + inname)

	if pos == "RP" and func not in ["compound:prt","conj"] or pos != "RP" and func=="compound:prt":
		print("WARN: pos " + pos + " should not normally have function " + func + inname)

	if pos.startswith("IN") and lemma == "that" and func not in ["mark","fixed","conj","reparandum","ccomp"]:
		print("WARN: lemma " + lemma + " with pos " + pos + " should not normally have function " + func + inname)

	if pos != "CC" and func in ["cc","cc:preconj"]:
		if lemma not in ["/","rather","as","et","+","let","only","-","∪","∩","∖"]:
			print("WARN: function " + func + " should normally have pos CC, not " + pos + inname)

	if pos == "VVG" and "very" in children:
		print("WARN: pos " + pos + " should not normally have child 'very'" + inname)

	if pos == "UH" and func=="advmod":
		print("WARN: pos " + pos + " should not normally have function 'advmod'" + inname)

	if pos =="IN" and func=="discourse":
		print("WARN: pos " + pos + " should not normally have function 'discourse'" + inname)

	if pos == "VVG" and "case" in child_funcs:
		if tok != "following":  # Exception for promoted 'the following'
			print("WARN: pos " + pos + " should not normally have child function 'case'" + inname)

	if pos.startswith("V") and any([f.startswith("nmod") for f in child_funcs]):
		print("WARN: pos " + pos + " should not normally have child function 'nmod.*'" + inname)

	if pos in ["JJR","JJS","RBR","RBS"] and lemma == tok:
		if lemma not in ["least","further","less","more"] and not lemma.endswith("most"):
			print("WARN: comparative or superlative "+tok+" with tag "+pos+" should have positive lemma not " + lemma + inname)

	if re.search(r"never|not|no|n't|n’t|’t|'t|nt|ne|pas|nit", tok, re.IGNORECASE) is None and func == "neg":
		print(str(id) + docname)
		print("WARN: mistagged negative" + inname)

	if pos == "VVG" and func == "compound":
		# Check phrasal compound exceptions where gerund clause is a compound modifier:
		# "'we're *losing* $X - fix it' levels of pressure
		if tok not in ["losing"]:
			print("WARN: gerund compound modifier should be tagged as NN not VVG" + inname)

	if pos in ["VBG","VHG","VVG"] and func in ["obj","nsubj","iobj","nmod","obl"]:
		if not tok == "following" and func=="obj":  # Exception nominalized "the following"
			print("WARN: gerund should not have noun argument structure function " + func + inname)

	if lemma in ["between",'like','of','than','with'] and pos == "RB":
		if parent_lemma != "sink": # Sank zombie-like/RB is legitimate
			print("WARN: lemma " +lemma+ " not have RB tag" + inname)

	if (pos.startswith("NN") or pos == "DT") and func=="amod":
		print("WARN: tag "+ pos + " should not be " + func + inname)

	be_funcs = ["cop", "aux", "root", "csubj", "aux:pass", "acl:relcl", "ccomp", "advcl", "conj","xcomp","parataxis","reparandum"]
	if lemma == "be" and func not in be_funcs:
		if "expl" not in child_funcs:
			if not (parent_lemma == "that" and func == "fixed"):  # Exception for 'that is' as mwe
				pass
				#print("WARN: invalid dependency "+func+" of lemma 'be' > " + func + inname)

	if parent_lemma in ["tell","show","give","pay","teach","owe","text","write"] and \
			tok in ["him","her","me","us","you"] and func=="obj":
		print("WARN: person object of ditransitive expected to be iobj, not obj" + inname)

	if func == "aux" and lemma.lower() != "be" and lemma.lower() != "have" and lemma.lower() !="do" and pos!="MD" and pos!="TO":
		print("WARN: aux must be modal, 'be,' 'have,' or 'do'" + inname)

	if func == "xcomp" and pos in ["VBP","VVP","VHP","VVZ","VBZ","VHZ","VVD","VBD","VHD"]:
		if parent_lemma not in ["=","seem"]:
			print("WARN: xcomp verb should be infinitive, not tag " + pos + inname)

	if func == "xcomp" and pos in ["VV","VB","VH"] and parent_pos.startswith("N"):
		print("WARN: infinitive child of a noun should be acl not xcomp" + inname)

	if func =="xcomp" and parent_lemma == "be":
		print("WARN: verb lemma 'be' should not have xcomp child" + inname)

	IN_not_like_lemma = ["vs", "vs.", "v", "v.", "o'er", "ca", "that", "then", "a", "fro", "too", "til", "wether"]  # incl. known typos
	if pos == "IN" and tok.lower() not in IN_not_like_lemma and lemma != tok.lower() and func != "goeswith" and "goeswith" not in child_funcs:
		print("WARN: pos IN should have lemma identical to lower cased token" + inname)
	if pos == "DT":
		if lemma == "an":
			print("WARN: lemma of 'an' should be 'a'" + inname)
		if lemma not in ["the","a","this","that","all","some","no","any","every","another","each","both","either","neither","yonder","_"]:
			print("WARN: unknown determiner lemma "+lemma+" for POS DT" + inname)

	if re.search(r"“|”|n’t|n`t|[’`](s|ve|d|ll|m|re|t)", lemma, re.IGNORECASE) is not None:
		print(str(id) + docname)
		print("WARN: non-ASCII character in lemma" + inname)

	if pos == "POS" and lemma != "'s" and func != "goeswith":
		print(str(id) + docname)
		print("WARN: tag POS must have lemma " +'"'+ "'s" + '"' + inname)

	if (parent_pos.startswith("RB") or (parent_pos.startswith("JJ") and ("of" not in children and parent_lemma not in ["Democrat","many","local"]))) and func == "nmod":
		print("WARN: nmod child of " + parent_pos + ' should be obl' + inname)

	if func == "goeswith" and lemma != "_":
		print("WARN: deprel goeswith must have lemma '_'" + inname)

	if func == "obj" and "case" in child_funcs and not (pos == "NP" and any([x in children for x in ["'s","’s"]])):
		print("WARN: obj should not have child case" + inname + str(children))

	if func == "ccomp" and "mark" in child_funcs and not any([x in children for x in ["that","That","whether","if","Whether","If","wether","a"]]):
		if not ((lemma == "lie" and "once" in children) or (lemma=="find" and ("see" in children or "associate" in children)) \
				or (lemma=="look" and "directly" in children) or (lemma=="make" and "to" in children)):  # Exceptions
			print("WARN: ccomp should not have child mark" + inname)

	if func == "acl:relcl" and pos in ["VB","VV","VH"] and "to" in children and "cop" not in child_funcs and "aux" not in child_funcs:
		print("WARN: infinitive with tag " + pos + " should be acl not acl:relcl" + inname)

	if pos in ["VBG","VVG","VHG"] and "det" in child_funcs:
		# Exceptions for phrasal compound in GUM_reddit_card and nominalization in GUM_academic_exposure, GENTLE_dictionary_next
		if tok != "prioritizing" and tok not in ["following","coming"]:
			print(str(id) + docname)
			print("WARN: tag "+pos+" should not have a determiner 'det'" + inname)

	if parent_lemma == "let" and func=="ccomp":
		if "maiden" not in inname:  # Known expl exceptions in speech_maiden
			print("WARN: verb 'let' should take xcomp clausal object, not ccomp" + inname)

	if pos == "MD" and lemma not in ["can","must","will","shall","would","could","may","might","ought","should"] and func != "goeswith":
		print("WARN: lemma '"+lemma+"' is not a known modal verb for tag MD" + inname)

	if lemma == "like" and pos == "UH" and func not in ["discourse","conj","reparandum"]:
		print("WARN: lemma '"+lemma+"' with tag UH should have deprel discourse, not "+ func + inname)

	if func in ["iobj","obj"] and parent_lemma in ["become","remain","stay"]:
		print("WARN: verb '"+parent_lemma+"' should take xcomp not "+func+" argument" + inname)

	if func in ["nmod:tmod","nmod:npmod","obl:tmod","obl:npmod"] and "case" in child_funcs:
		print("WARN: function " + func +  " should not have 'case' dependents" + inname)

	if func in ["aux:pass","nsubj:pass"] and parent_pos not in ["VVN","VBN","VHN"]:
		if not (("stardust" in docname and parent_lemma == "would") or parent_lemma == "Rated" or parent_func == "reparandum"):
			print("WARN: function " + func + " should not be the child of pos " + parent_pos + inname)

	if func == "obl:agent" and (parent_pos not in ["VBN","VHN","VVN"] or "by" not in children):
		print("WARN: function " + func +  " must by child of V.N with a 'by' dependent" + parent_pos + inname)

	if child_funcs.count("obl:agent") > 1:
		print("WARN: a token may have at most one obl:agent dependent" + inname)

	if "obl:agent" in child_funcs and ("nsubj" in child_funcs or "csubj" in child_funcs) and not "nsubj:pass" in child_funcs:
		print("WARN: a token cannot have both a *subj relation and obl:agent" + inname)

	if pos in ["VBD","VVD","VHD","VBP","VVP","VHP"] and "aux" in child_funcs:
		print("WARN: tag "+pos+" should not have auxiliaries 'aux'" + inname)

	# 'amod' promotion for EWT "those affluent and those not"
	if lemma == "not" and func not in ["advmod","root","ccomp","amod","parataxis","reparandum","advcl","conj","orphan","fixed"]:
		print("WARN: deprel "+func+" should not be used with lemma '"+lemma+"'" + inname)

	#if func == "xcomp" and parent_lemma in ["see","hear","notice"]:  # find
	#	print("WARN: deprel "+func+" should not be used with perception verb lemma '"+parent_lemma+"' (should this be nsubj+ccomp?)" + inname)

	if "obj" in child_funcs and "ccomp" in child_funcs:
		print("WARN: token has both obj and ccomp children" + inname)

	if func == "acl" and (pos.endswith("G") or pos.endswith("N")) and parent_id == id + 1:  # premodifier V.G/N should be amod not acl
		print("WARN: back-pointing " + func + " for adjacent premodifier (should be amod?) in " + docname + " @ token " + str(id) + " (" + tok + " <- " + parent + ")")

	if func.endswith("tmod") and pos.startswith("RB"):
		print("WARN: adverbs should not be tmod" + inname)

	"""
	Existential construction
	X.xpos=EX <=> X.deprel=expl & X.lemma=there
	"""
	if func!="reparandum":
		_ex_tag = (pos=="EX")
		_expl_there = (func=="expl" and lemma=="there")
		if _ex_tag != _expl_there:
			print("WARN: 'there' with " + pos + inname)
		if lemma=="there" and not _ex_tag and 'nsubj' in func:
			print("WARN: subject 'there' not tagged as EX/expl" + inname)

	"""
	(Pre)determiner 'what'
	X[lemma=what,xpos=WDT] <=> X[lemma=what,deprel=det|det:predet]
	"""
	if lemma=="what" and ((pos=="WDT") != (func in ["det", "det:predet"])):
		print("WARN: what/WDT should correspond with det or det:predet" + inname)

	#if func == "advmod" and lemma in ["where","when"] and parent_func == "acl:relcl":
	#	print("WARN: lemma "+lemma+" should not be func '"+func+"' when it is the child of a '" + parent_func + "'" + inname)

	if (sent_position == "first" and pos == "''") or (sent_position == "last" and pos=="``"):
		print("WARN: incorrect quotation mark tag " + pos + " at "+sent_position+" position in sentence" + inname)

	mwe_pairs = {("accord", "to"), ("all","but"), ("as","if"), ("as", "well"), ("as", "as"), ("as","in"), ("all","of"), ("as","oppose"),("as","to"),
				 ("at","least"),("because","of"),("due","to"),("had","better"),("'d","better"),("in","between"), ("per", "se"),
				 ("in","case"),("in","of"), ("in","order"),("instead","of"), ("kind","of"),("less","than"),("let","alone"),
				 ("more","than"),("not","to"),("not","mention"),("of","course"),("prior","to"),("rather","than"),("so","as"),
				 ("so", "to"),("sort", "of"),("so", "that"),("such","as"),("such","that"),("that","is"), ("up","to"),("depend","on"),
				 ("out","of"),("off","of"),("long","than"),("on","board"),("as","of"),("depend","upon"),
				 ("that","be"),("just","about"),("vice","versa"),("as","such"),("next","to"),("close","to"),("one","another"),
				 ("de","facto"),("each","other"), ("as","many"), ("in","that"), ("few","than"), ("as","for"), ("as","though")}

	# Ad hoc listing of triple mwe parts - All in all, in order for, whether or not
	mwe_pairs.update({("all","in"),("all","all"),("in","for"),("whether","or"),("whether","not")})

	if func == "fixed":
		if (parent_lemma.lower(), lemma.lower()) not in mwe_pairs:
			print("WARN: unlisted fixed expression" + inname)

	#if pos != "CD" and "quantmod" in child_funcs:
	#	print("WARN: quantmod must be cardinal number" + inname)

	if tok == "sort" or tok == "kind":
		if "det" in child_funcs and "fixed" in child_funcs:
			print("WARN: mistagged fixed expression" + inname)

	if tok == "rather" and "fixed" in child_funcs and func not in ["cc","mark"]:
		print("WARN: 'rather than' fixed expression must be cc or mark" + inname)

	if s_type == "imp" or s_type == "frag" or s_type == "ger" or s_type == "inf":
		if func == "root" and "nsubj" in child_funcs:
			# Exception for frag structures like "Whatever it is that ...", which is actually frag
			# and "don't you VERB", which is an imperative with a subject
			if not ("acl:relcl" in child_funcs and "cop" in child_funcs and s_type=="frag") and \
					not (("do" in children or "Do" in children) and ("n't" in children or "not" in children)):
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
				if not any([c.lower()=="do" or c.lower()=="did" for c in children]):
					if not (tok == "Remember" and wh == "when") and not (tok=="know" and wh=="what") and \
						not (tok =="Know" and wh=="when"):  # Listed exceptions in GUM_reddit_bobby, GUM_conversation_christmas, GUM_vlog_covid
						if not "_wine" in inname and lemma == "remember":  # known unmarked pro-drop polar question in GUM_vlog_wine
							print("WARN: q root may not have wh child " + wh + inname)

	suspicious_pos_tok = [("*","DT","only","RB"),
						  ("no", "RB", "matter", "RB")]

	for w1, pos1, w2, pos2 in suspicious_pos_tok:
		if w1 == prev_tok or w1 == "*":
			if pos1 == prev_pos or pos1 == "*":
				if w2 == tok or w2 == "*":
					if pos2 == pos or pos2 == "*":
						print("WARN: suspicious n-gram " + prev_tok + "/" + prev_pos+" " + tok + "/" + pos + inname)

