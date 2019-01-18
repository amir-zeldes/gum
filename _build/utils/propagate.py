# -*- coding: utf-8 -*-

# GUM Build Bot
# propagate module
# v1.0.1

from glob import glob
from .nlp_helper import get_claws, adjudicate_claws, parse
from .depedit import DepEdit
import os, re, sys, io
import ntpath
from collections import defaultdict

PY2 = sys.version_info[0] < 3


def clean_tag(tag):
	if tag == "0":
		raise IOError("unknown tag 0")
	elif tag == '"':
		return "''"
	elif tag == "'":
		return "''"
	else:
		return tag


def tt2vanilla(tag,token):
	tag = tag.replace("VV","VB").replace("VH","VB")
	tag = tag.replace("NP","NNP")
	tag = tag.replace("PP","PRP")
	tag = tag.replace("SENT",".")

	if tag=="IN/that":
		tag = "IN"
	elif tag=="(":
		if token == "[":
			tag = "-LSB-"
		else:
			tag = "-LRB-"
	elif tag == ")":
		if token == "]":
			tag = "-RSB-"
		else:
			tag = "-RRB-"
	return tag


def enrich_dep(gum_source, gum_target, reddit=False):

	no_space_after_strings = {"(","[","{"}
	no_space_before_strings = {".",",",";","?","!","'s","n't","'ve","'d","'m","'ll","]",")","}",":","%"}
	no_space_after_combos = {("'","``"),('"',"``")}
	no_space_before_combos = {("ll","MD"),("d","MD"),("m","VBP"),("ve","VHP"),("s","POS"),("s","VBZ"),("s","VHZ"),("'","POS"),("nt","RB"),("'","''"),('"',"''")}

	dep_source = gum_source + "dep" + os.sep
	dep_target = gum_target + "dep" + os.sep + "stanford" + os.sep
	if not os.path.isdir(dep_target):
		os.makedirs(dep_target)

	depfiles = []
	files_ = glob(dep_source + "*.conll10")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		depfiles.append(file_)

	for docnum, depfile in enumerate(depfiles):
		docname = ntpath.basename(depfile)
		sys.stdout.write("\t+ " + " "*50 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(depfiles)) + ":\t+ " + docname + "\r")
		current_stype = ""
		current_speaker = ""
		current_sic = False
		current_w = False
		output = ""
		stype_by_token = {}
		speaker_by_token = {}
		space_after_by_token = defaultdict(lambda: True)
		sic_by_token = defaultdict(lambda: False)

		# Dictionaries to hold token annotations from XML
		wordforms = {}
		pos = {}
		lemmas = {}

		tok_num = 0

		xmlfile = depfile.replace("dep" + os.sep,"xml" + os.sep).replace("conll10","xml")
		xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		for line in xml_lines:
			if line.startswith("<"):  # XML tag
				if line.startswith("<s type="):
					current_stype = re.match(r'<s type="([^"]+)"',line).group(1)
				elif line.startswith("<sp who="):
					current_speaker = re.search(r' who="([^"]+)"', line).group(1).replace("#","")
				elif line.startswith("</sp>"):
					current_speaker = ""
				elif line.startswith("<w>"):
					space_after_by_token[tok_num+1] = True
				elif line.startswith("</w>"):
					space_after_by_token[tok_num] = False
				elif line.startswith("<sic>"):
					current_sic = True
				elif line.startswith("</sic>"):
					current_sic = False
			elif len(line)>0:  # Token
				fields = line.split("\t")
				word = fields[0].replace("`","'").replace("‘","'").replace("’","'")
				word = word.replace('“','"').replace("”",'"')
				word_pos = fields[1].replace('"',"''")
				tok_num += 1
				if word == "(":
					a=5
				if word in no_space_after_strings:
					space_after_by_token[tok_num] = False
				if word in no_space_before_strings:
					space_after_by_token[tok_num-1] = False
				if (word,word_pos) in no_space_after_combos:
					space_after_by_token[tok_num] = False
				if (word,word_pos) in no_space_before_combos:
					space_after_by_token[tok_num-1] = False
				stype_by_token[tok_num] = current_stype
				speaker_by_token[tok_num] = current_speaker
				sic_by_token[tok_num] = current_sic
				wordforms[tok_num], pos[tok_num], lemmas[tok_num] = fields[:3]

		conll_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0
		for line in conll_lines:
			if "# speaker" in line or "# s_type" in line:
				# Ignore old speaker and s_type annotations in favor of fresh ones
				continue
			if "\t" in line:  # Token
				tok_num += 1
				wordform = wordforms[tok_num]
				lemma = lemmas[tok_num]
				# De-escape XML escapes
				wordform = wordform.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
				lemma = lemma.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
				fields = line.split("\t")
				tt_pos = pos[tok_num]
				tt_pos = clean_tag(tt_pos)
				vanilla_pos = tt2vanilla(tt_pos, fields[1])
				fields[1] = wordform
				fields[2] = lemma
				fields[3] = tt_pos
				fields[4] = vanilla_pos
				misc = []
				if not space_after_by_token[tok_num]:
					misc.append("SpaceAfter=No")
				if sic_by_token[tok_num]:
					misc.append("Typo=Yes")
				fields[-1] = "|".join(misc) if len(misc) > 0 else "_"
				line = "\t".join(fields)
			if line.startswith("1\t"):  # First token in sentence
				# Check for annotations
				if len(stype_by_token[tok_num]) > 0:
					output += "# s_type=" + stype_by_token[tok_num] + "\n"
				if len(speaker_by_token[tok_num]) > 0:
					output += "# speaker=" + speaker_by_token[tok_num] + "\n"
			output += line + "\n"

		output = output.strip() + "\n" + "\n"

		# Attach all punctuation to the root (could also be a vocative root)
		depedit = DepEdit()
		depedit.add_transformation("func=/root/;func=/punct/\t#1.*#2\t#1>#2")
		depedit.add_transformation("func=/root/;func=/punct/\t#2.*#1\t#1>#2")
		depedit.add_transformation("func=/vocative/&head=/0/;func=/punct/\t#1.*#2\t#1>#2")
		depedit.add_transformation("func=/vocative/&head=/0/;func=/punct/\t#2.*#1\t#1>#2")
		output = depedit.run_depedit(output)

		outfile = io.open(dep_target + docname, 'w', encoding="utf8")
		outfile.write(output)
		outfile.close()
	print("o Enriched dependencies in " + str(len(depfiles)) + " documents" + " " *20)


def enrich_xml(gum_source, gum_target, add_claws=False, reddit=False, warn=False):
	xml_source = gum_source + "xml" + os.sep
	xml_target = gum_target + "xml" + os.sep

	xmlfiles = []
	files_ = glob(xml_source + "*.xml")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		xmlfiles.append(file_)

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " "*40 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r")

		# Dictionaries to hold token annotations from conll10 data
		funcs = {}

		tok_num = 0

		depfile = xmlfile.replace("xml" + os.sep,"dep" + os.sep).replace("xml","conll10")
		if PY2:
			dep_lines = open(depfile).read().replace("\r", "").split("\n")
		else:
			try:
				dep_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
			except FileNotFoundError:
				sys.stderr.write("! File not found: " + depfile)
				if warn:
					continue
				else:
					exit()
		line_num = 0
		for line in dep_lines:
			line_num += 1
			if "\t" in line:  # token line
				if line.count("\t") != 9:
					print("WARN: Found line with less than 9 tabs in " + docname + " line: " + str(line_num))
				else:
					tok_num += 1
					fields = line.split("\t")
					funcs[tok_num] = fields[7]

		if PY2:
			xml_lines = open(xmlfile).read().replace("\r", "").split("\n")
		else:
			xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0

		if add_claws:
			tokens = list((line.split("\t")[0]) for line in xml_lines if "\t" in line)
			claws = get_claws("\n".join(tokens))

		for line in xml_lines:
			if "\t" in line:  # Token
				tok_num += 1
				func = funcs[tok_num]
				fields = line.split("\t")
				if add_claws:
					fields = fields[:3]  # Only retain first three columns; the rest can be dynamically generated
					claws_tag = claws[tok_num-1]
					claws_tag = adjudicate_claws(claws_tag,fields[1],fields[0],func)
					fields.append(claws_tag)
				else:
					fields = fields[:-1] # Just delete last column to re-generate func from conll10
				fields.append(func)
				line = "\t".join(fields)
			output += line + "\n"

		output = output.strip() + "\n"

		if PY2:
			outfile = open(xml_target + docname, 'wb')
		else:
			outfile = io.open(xml_target + docname,'w',encoding="utf8")
		outfile.write(output)
		outfile.close()

	if add_claws:
		print("o Retrieved fresh CLAWS5 tags" + " " * 20 + "\r")
	print("o Enriched xml in " + str(len(xmlfiles)) + " documents" + " " *20)


def const_parse(gum_source, gum_target, warn_slash_tokens=False, reddit=False):
	xml_source = gum_source + "xml" + os.sep
	const_target = gum_target + "const" + os.sep

	files_ = glob(xml_source + "*.xml")
	xmlfiles = []
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		xmlfiles.append(file_)

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " "*40 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(xmlfiles)) + ":\t+ Parsing " + docname + "\r")

		# Name for parser output file
		constfile = const_target + docname.replace("xml", "ptb")


		xml_lines = io.open(xmlfile, encoding="utf8").read().replace("\r", "").split("\n")
		line_num = 0
		out_line = ""

		for line in xml_lines:
			if line.startswith("</s>"): # Sentence ended
				output += out_line.strip() + "\n"
				out_line = ""

			elif "\t" in line:  # Token
				line_num += 1
				fields = line.split("\t")
				token, tag = fields[0], fields[1]
				tag = tt2vanilla(tag,token)
				if " " in token:
					print("WARN: space found in token on line " + str(line_num) + ": " + token + "; replaced by '_'")
					token = token.replace(" ","_")
				elif "/" in token and warn_slash_tokens:
					print("WARN: slash found in token on line " + str(line_num) + ": " + token + "; retained as '/'")

				token = token.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<").replace("&apos;","'").replace("&quot;",'"').replace("(","-LRB-").replace(")","-RRB-")
				item = token + "/" + tag + " "
				out_line += item

		# dump sentence wise plain text for parser
		#with open("C:\\temp\\" + os.path.basename(xmlfile).replace("xml","txt"),"w",encoding="utf8") as f:
		#	f.write(output)

		parsed = parse(output)
		#parsed = "x"

		parsed = parsed.strip() + "\n" + "\n"

		outfile = io.open(constfile, 'w', encoding="utf8")
		outfile.write(parsed)
		outfile.close()

	print("o Reparsed " + str(len(xmlfiles)) + " documents" + " " * 20)
