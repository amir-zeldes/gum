#!/usr/bin/python
# -*- coding: utf-8 -*-

import platform
import tempfile
import subprocess
import os, re, sys
from .paths import tt_path, parser_path, core_nlp_path

PY2 = sys.version_info[0] < 3


def exec_via_temp(input_text, command_params, workdir="", cat_out=False):
	if PY2:
		temp = tempfile.NamedTemporaryFile('w',delete=False)
	else:
		temp = tempfile.NamedTemporaryFile('w',delete=False,encoding="utf8")
	exec_out = ""
	try:
		temp.write(input_text)
		temp.close()

		command_params = [x if x != 'tempfilename' else temp.name for x in command_params]
		if workdir == "":
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
			if cat_out:
				proc.communicate()
				stdout = ""
			else:
				(stdout, stderr) = proc.communicate()
		else:
			proc = subprocess.Popen(command_params, stdout=subprocess.PIPE, stdin=subprocess.PIPE,stderr=subprocess.PIPE, cwd=workdir)
			if cat_out:
				proc.communicate()
				stdout = ""
			else:
				(stdout, stderr) = proc.communicate()

		exec_out = stdout
	except Exception as e:
		print(e)
	finally:
		os.remove(temp.name)
		return exec_out


def get_claws(tokens):

	tag = [tt_path + 'bin' + os.sep + 'tree-tagger', tt_path + 'lib' + os.sep + 'claws5.par', '-sgml', 'tempfilename']
	tagged = exec_via_temp(tokens, tag)
	if PY2:
		tagged = tagged.replace("\r","")
	else:
		tagged = tagged.decode("utf8").replace("\r", "")
	tags = tagged.split("\n")
	return tags


def adjudicate_claws(claws,tt,form,func):
	if tt in ["NP","NPS"]:
		return "NP0"
	if claws == "NN0":
		return claws

	# Handle forms of 'do'
	if form.lower() in ["do", "did", "does", "done", "doing"]:
		if tt == "VVP":
			return "VD"
		elif tt == "VVZ":
			return "VDZ"
		elif tt == "VVG":
			return "VDG"
		elif tt == "VVN":
			return "VDN"
		else:
			return claws

	# Simple substitutions
	if tt == "NN":
		return "NN1"
	elif tt == "NNS":
		return "NN2"
	elif tt == "RP":
		return "AVP"
	elif tt in ['"',"''","``"]:
		return "PUQ"
	elif tt == "(":
		return "PUL"
	elif tt == ")":
		return "PUR"
	elif tt == "FW":
		return "UNC"
	elif tt == "VVP":
		return "VVB"
	elif tt == "PP":
		return "PNP"
	elif tt == "PP$":
		return "DPS"
	elif tt == "PRF":
		return "PNX"
	elif tt == "WP":
		return "PNQ"

	# Disambiguate prepositions and subordinating conjunctions
	if not claws.startswith("CJ") and tt == "IN":
		if form.lower() == "of":
			return "PRF"
		else:
			return "PRP"
	if tt == "IN" and func == "mark":
		return "CJS"

	# Cases to use the TT tag itself (note that 'do' filtering is already done for VD tags)
	if tt in ["VVG","VVZ","VVD","VVN","POS"]:
		return tt

	# Handle ambiguous tag TO
	if tt == "TO":
		if func == "prep":
			return "PRP"
		else:
			return "TO0"

	if claws != "DT0" and claws != "ORD":
		if tt == "JJ":
			return "AJ0"
		elif tt == "JJR":
			return "AJC"
		elif tt == "JJS":
			return "AJS"

	# Handle 'not'
	if tt == "RB" and form.lower() in ["n't","not","n`t","n’t"]:
		return "XX0"

	# Handle alphabetic symbols
	if tt == "SYM" and re.match(r'^[A-Za-z]+$',form) is not None:
		return "ZZ0"

	# POS + func deterministic combinations
	if tt == "VV" and func == "xcomp" and form.lower() == "do":
		return "VDI"
	elif tt == "VV" and func == "xcomp":
			return "VVI"
	elif tt == "VB" and func == "xcomp":
		return "VBI"
	elif tt == "VB" and func == "xcomp":
		return "VBI"

	return claws

"""
def parse(sent_per_line):

	parse_command = [parser_path + os.sep + 'lexparser_eng_const_plus.bat', 'tempfilename']
	parsed = exec_via_temp(sent_per_line, parse_command, parser_path)
	if platform.system() == "Windows":
		if PY2:
			parsed = parsed.replace("\r","")
		else:
			parsed = parsed.decode("utf8").replace("\r","")
	else:
		parsed = parsed.replace("\r", "")


	return parsed
"""

def ud_morph(conllu_string, doc_name, const_path):

	ptb_file = const_path + doc_name + ".ptb"

	if platform.system() == "Windows":
		morph_command = ["java", "-cp", '*;', "edu.stanford.nlp.trees.ud.UniversalDependenciesFeatureAnnotator", "tempfilename", ptb_file]
	else:
		morph_command = ["java", "edu.stanford.nlp.trees.ud.UniversalDependenciesFeatureAnnotator", "tempfilename", ptb_file]

	morphed = exec_via_temp(conllu_string, morph_command, core_nlp_path)
	return morphed
