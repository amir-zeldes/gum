#!/usr/bin/python
# -*- coding: utf-8 -*-

import re,sys,platform
import ntpath, os, io
from rst2dep import make_rsd
from .rst2dis import rst2dis
from collections import defaultdict
from .non_dm_signals import update_signals
from glob import glob

PY2 = sys.version_info[0] < 3


def fix_rst(gum_source, gum_target, reddit=False, rsd_algorithm="li"):
	outdir = gum_target + "rst" + os.sep + "rstweb" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	rsddir = gum_target + "rst" + os.sep + "dependencies" + os.sep
	if not os.path.exists(rsddir):
		os.makedirs(rsddir)
	disdir = gum_target + "rst" + os.sep + "lisp_binary" + os.sep
	if not os.path.exists(disdir):
		os.makedirs(disdir)
	disdir = gum_target + "rst" + os.sep + "lisp_nary" + os.sep
	if not os.path.exists(disdir):
		os.makedirs(disdir)

	file_list = []
	files_ = glob(gum_source + "rst" + os.sep + "*.rs3")
	files_ += glob(gum_source + "rst" + os.sep + "*.rs4")  # Accept rs3 or rs4 extension
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		file_list.append(file_)

	conns_by_doc = defaultdict(dict)
	for docnum, filename in enumerate(file_list):
		docname = os.path.basename(filename).replace(".rs3","").replace(".rs4","")
		tt_file = filename.replace(".rs3", ".xml").replace(".rs4", ".xml").replace("rst","xml")
		sys.stdout.write("\t+ " + " "*70 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Adjusting borders for " + ntpath.basename(filename) + "\r")
		fix_file(filename,tt_file,gum_source,gum_target + "rst" + os.sep + "rstweb" + os.sep, rsd_algorithm=rsd_algorithm)
		conn_data = get_conn_data(filename)
		conns_by_doc[docname] = conn_data

	print("o Adjusted " + str(len(file_list)) + " RST files" + " " * 70)
	return conns_by_doc


def get_conn_data(filename):
	lines = open(filename).read().replace("\r","").split("\n")
	conn_data = defaultdict(str)
	for line in lines:
		if '<signal source="' in line and (' type="dm"' in line or ' type="orphan"' in line):
			tokens = re.search(r' tokens="([0-9,]*)"',line).group(1)
			tokens = sorted(tokens.split(","),key=lambda x: int(x))
			for i, tok in enumerate(tokens):
				if i == 0:
					conn_data[int(tok)] = "B"
				else:
					if conn_data[int(tok)] != "B":
						if conn_data[int(tok)-1] != "":
							conn_data[int(tok)] = "I"
						else:
							conn_data[int(tok)] = "B"
	return conn_data


def validate_rsd(rsd_line, linenum, docname):
	inname = " in document " + docname + " on line " + str(linenum) + "\n"
	if re.search(r'\b[Tt]o\b[^\n]+head_pos=V.\|[^\n]+head_func=acl\|[^\n]+elaboration-attr', rsd_line) is not None:
		if "\tthat" not in rsd_line and "\tabout" not in rsd_line and "\tyou " not in rsd_line and \
			'\tto expect "' not in rsd_line and "\tto consider" not in rsd_line and \
			"\tto hold the Office" not in rsd_line and "\tto represent California" not in rsd_line and \
			"\tto shine through" not in rsd_line and "\tto go down in this match" not in rsd_line:  # check for that-clause embedding to-, or about PP
			sys.stderr.write("! adnominal infinitive clause should be purpose-attribute not elaboration-attribute" + inname)
	if re.search(r'(\bn.t\b[^\n]+)attribution-positive_r', rsd_line) is not None:
		if ("surprised" not in rsd_line and "not only" not in rsd_line) and "n't also deny" not in rsd_line and \
			not ("n't think" in rsd_line and "veronique" in docname) and not ("agreeing" in rsd_line and "raven" in docname):
			sys.stderr.write("! suspicious attribution-positive_r with negation" + inname)
	if "\t" in rsd_line:
		fields = rsd_line.split("\t")
		if int(fields[0]) < int(fields[6]) and (fields[7] in ["elaboration-attribute_r","purpose-attribute_r","elaboration-additional_r","restatement_partial_r"]):
			sys.stderr.write("! invalid left to right relation " + fields[7] + inname)
		elif int(fields[0]) > int(fields[6]) and (fields[7] in ["organization-preparation_r","organization-heading_r","topic-question_r"]):
			sys.stderr.write("! invalid right to left relation " + fields[7] + inname)
		if re.search(r'^\( ((19|20)[0-9][0-9] ([–-] )?)+\)',fields[1]) is not None and fields[7] != "context-circumstance_r":
			sys.stderr.write("! suspicious parenthetical year EDU with rsd relation " + fields[7] + inname)


def validate_erst(rs4,docname,sig_stats):
	lines = rs4.split("\n")
	tokens = []
	secedges = set([])
	signal_sources = set([])
	id2rel = defaultdict(str)
	collapse_dm = {"and ... and ... and":"and","and ... and":"and","and and":"and","but ... but":"but",
				   "when ... when":"when","so ... so":"so","then ... then":"then","them":"then", #typo
				   "cause ... cause ... cause":"cause","also ... also":"also", "for for":"for"}
	dm_sig_toks = set([])  # DM signals tokens should not be used by multiple nodes
	for i,line in enumerate(lines):
		if '<segment' in line:
			tokens += line.split(">")[1].split("<")[0].split(" ")
		if 'relname=' in line:
			rel_id = re.search(r' id="([0-9-]+)"',line).group(1)
			rel_name = re.search(r' relname="([^"]+)"',line).group(1)
			id2rel[rel_id] = rel_name
		if re.search(r'<signal source="[0-9]+-[0-9]+".*"dm"',line) is not None:
			sys.stderr.write("! Found dm signal for secondary eRST relation on line "+str(i+1)+" of "+docname+"\n")
		if re.search(r'<signal source="[0-9]+".*"orphan"',line) is not None:
			sys.stderr.write("! Found orphan signal for primary eRST relation on line "+str(i+1)+" of "+docname+"\n")
		if 'lexical_chain2' in line:
			sys.stderr.write("! Found unnormalized signal lexical_chain2 on line " + str(i + 1) + " of " + docname + "\n")
		if '<secedge ' in line:
			secedge_id = re.search(r' id="([0-9]+-[0-9]+)"',line).group(1)
			if secedge_id in secedges:
				sys.stderr.write("! Found duplicate eRST secondary edge on line " + str(i + 1) + " of " + docname + "\n")
			else:
				secedges.add(secedge_id)
		if '<signal source' in line:
			m = re.search(r' source="([0-9-]+)" type="([^"]+)" subtype="([^"]+)" tokens="([^"]*)"',line)
			src = m.group(1)
			sig_type = m.group(2)
			subtype = m.group(3)
			sig_tokens = sorted([int(x)-1 for x in m.group(4).split(",")]) if m.group(4) != "" else []
			if sig_type in ["dm","orphan"]:
				if any([t in dm_sig_toks for t in sig_tokens]):
					signal_text = " ".join([tokens[t] for t in sig_tokens])
					sys.stderr.write("! Found dm/orphan signal tokens used by multiple nodes in "+docname+": "+str([t+1 for t in sig_tokens])+" (" + signal_text + ")\n")
				dm_sig_toks.update(sig_tokens)
			signal_sources.add(src)
			rel_name = id2rel[src]
			if subtype in ["dm","orphan","alternate_expression"]:
				sig_string = ""
				prev = sig_tokens[0]-1
				for t in sig_tokens:
					if t != prev+1:
						sig_string += "... "
					sig_string += tokens[t] + " "
					prev = t
				sig_string = sig_string.lower().strip()
				if sig_string in collapse_dm:
					sig_string = collapse_dm[sig_string]
				if subtype == "alternate_expression":
					sig_stats["altlex2rel"][sig_string][rel_name] += 1
					sig_stats["rel2altlex"][rel_name][sig_string] += 1
				else:
					sig_stats["dm2rel"][sig_string][rel_name] += 1
					sig_stats["rel2dm"][rel_name][sig_string] += 1

	for e in secedges:
		if e not in signal_sources:
			sys.stderr.write("! Found secondary eRST relation with no signal for edge "+str(e)+" in "+docname+"\n")

	return sig_stats

def fix_file(filename, tt_file, gum_source, outdir, rsd_algorithm="li"):

	# Get reference tokens
	rst_file_name = ntpath.basename(filename)
	tokens = []

	if PY2:
		tt = open(tt_file)
	else:
		tt = open(tt_file,'r',encoding="utf8")
	lines = tt.read().replace("\r","").split("\n")

	current_token = 0
	s_splits = []

	for line in lines:
		if not line.startswith("<") and len(line) > 0 and "\t" in line: # token
			token = line.split("\t")[0]
			current_token += 1
			tokens.append(token)
		elif line.startswith("<s type"):
			s_splits.append(current_token)

	if PY2:
		rst = open(filename)
	else:
		rst = open(filename,encoding="utf8")
	lines = rst.read().replace("\r","").split("\n")

	line_num = 0
	out_data = ""
	token_reached = 0
	for line in lines:
		line_num += 1

		if "<segment" not in line:
			out_data += line + "\n"
		else:
			if line.count("<") != 2 or line.count(">") != 2:
				raise IOError("rs3 XML does not follow one segment tag per line on line: " + str(line_num) + " in file: " + rst_file_name)
			m = re.search(r'^(.*<segment[^>]+>)(.*)(</segment>.*)',line)
			t_open = m.group(1)
			seg = m.group(2)
			t_close = m.group(3)
			seg = seg.strip()
			seg_tokens = len(seg.split(" "))
			repaired_seg = " ".join(tokens[token_reached:token_reached+seg_tokens])
			out_data += t_open + repaired_seg + t_close + "\n"
			if any([t in s_splits for t in range(token_reached+1,token_reached+seg_tokens)]):
				sys.stderr.write("! RST segment contains sentence break in " + rst_file_name + ": " + t_open + "\n")
			token_reached += seg_tokens

	with io.open(outdir + rst_file_name,'w',encoding="utf8",newline="\n") as f:
		f.write(out_data)

	docname = os.path.basename(rst_file_name).replace(".rs3","").replace(".rs4","")

	# Make rsd version
	keep_same_unit = True if rsd_algorithm == "chain" else False
	rsd = make_rsd(out_data,gum_source,as_text=True, algorithm=rsd_algorithm, keep_same_unit=keep_same_unit,
				   docname=os.path.basename(rst_file_name.replace(".rs3","").replace(".rs4","")))

	for l, line in enumerate(rsd.split("\n")):
		validate_rsd(line, l+1, docname)

	with io.open(outdir.replace("rstweb","dependencies") + docname + ".rsd",'w',encoding="utf8",newline="\n") as f:
		f.write(rsd)

	# Unescape XML
	out_data = out_data.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")

	# Make binary dis version
	dis = rst2dis(out_data, binarize=True)
	with io.open(outdir.replace("rstweb","lisp_binary") + docname+".dis",'w',encoding="utf8",newline="\n") as f:
		f.write(dis)

	# Make nary dis version
	dis = rst2dis(out_data, binarize=False)
	with io.open(outdir.replace("rstweb","lisp_nary") + docname+".dis",'w',encoding="utf8",newline="\n") as f:
		f.write(dis)


def update_non_dm_signals(gum_source, gum_target, reddit=False, rsd_algorithm="li"):
	gold_rs4_dir = gum_source + "rst" + os.sep
	gold_rs4_files = glob(gold_rs4_dir + "*.rs4")
	gold_target_dir = gum_target + "rst" + os.sep + "rstweb" + os.sep

	if not reddit:
		gold_rs4_files = [f for f in gold_rs4_files if "reddit_" not in f]

	sig_stats = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
	for docnum, file_ in enumerate(gold_rs4_files):
		docname = os.path.basename(file_).replace(".rs4","")
		sys.stdout.write("\t+ " + " " * 70 + "\r")
		sys.stdout.write(" " + str(docnum + 1) + "/" + str(len(gold_rs4_files)) + ":\t+ " + docname + "\r")
		gold_rs4 = open(file_).read()
		gold_rs4 = update_signals(gold_rs4, docname, xml_root=gum_source)
		with open(gold_target_dir + docname + ".rs4",'w',encoding="utf8",newline="\n") as f:
			f.write(gold_rs4)
		updated_rsd = make_rsd(gold_rs4, gum_source, as_text=True, algorithm=rsd_algorithm, docname=docname)
		with open(gold_target_dir.replace("rstweb","dependencies") + docname + ".rsd",'w',encoding="utf8",newline="\n") as f:
			f.write(updated_rsd)
		sig_stats = validate_erst(gold_rs4, docname, sig_stats)

	print("o Updated signals in " + str(len(gold_rs4_files)) + " RST files" + " " * 70)

	dm2rel = ["\t".join(["dm","freq","senses"])]
	for dm in sig_stats["dm2rel"]:
		row = [dm,str(sum([sig_stats["dm2rel"][dm][rel] for rel in sig_stats["dm2rel"][dm]]))]
		per_rel_stats = []
		for relname in sorted(sig_stats["dm2rel"][dm], key=lambda x: sig_stats["dm2rel"][dm][x], reverse=True):
			per_rel_stats.append(relname + " (" + str(sig_stats["dm2rel"][dm][relname]) + ")")
		row.append(", ".join(per_rel_stats))
		dm2rel.append("\t".join(row))
	altlex2rel = ["\t".join(["altlex","freq","senses"])]
	for dm in sig_stats["altlex2rel"]:
		row = [dm,str(sum([sig_stats["altlex2rel"][dm][rel] for rel in sig_stats["altlex2rel"][dm]]))]
		per_rel_stats = []
		for relname in sorted(sig_stats["altlex2rel"][dm], key=lambda x: sig_stats["altlex2rel"][dm][x], reverse=True):
			per_rel_stats.append(relname + " (" + str(sig_stats["altlex2rel"][dm][relname]) + ")")
		row.append(", ".join(per_rel_stats))
		altlex2rel.append("\t".join(row))
	rel2dm = ["\t".join(["rel","freq","dms"])]
	for rel in sig_stats["rel2dm"]:
		row = [rel,str(sum([sig_stats["rel2dm"][rel][dm] for dm in sig_stats["rel2dm"][rel]]))]
		row.append(", ".join([dm + " (" + str(sig_stats["rel2dm"][rel][dm]) + ")" for dm in sorted(sig_stats["rel2dm"][rel], key=lambda x: sig_stats["rel2dm"][rel][x], reverse=True)]))
		rel2dm.append("\t".join(row))
	rel2altlex = ["\t".join(["rel","freq","altlexes"])]
	for rel in sig_stats["rel2altlex"]:
		row = [rel,str(sum([sig_stats["rel2altlex"][rel][dm] for dm in sig_stats["rel2altlex"][rel]]))]
		row.append(", ".join([dm + " (" + str(sig_stats["rel2altlex"][rel][dm]) + ")" for dm in sorted(sig_stats["rel2altlex"][rel], key=lambda x: sig_stats["rel2altlex"][rel][x], reverse=True)]))
		rel2altlex.append("\t".join(row))

	if not os.path.exists(gum_target + "rst" + os.sep + "stats"):
		os.makedirs(gum_target + "rst" + os.sep + "stats")
	with open(gum_target + "rst" + os.sep + "stats" + os.sep + "dm2rel.tab",'w',encoding="utf8",newline="\n") as f:
		f.write("\n".join([dm2rel[0]] + sorted(dm2rel[1:])) + "\n")
	with open(gum_target + "rst" + os.sep + "stats" + os.sep + "altlex2rel.tab",'w',encoding="utf8",newline="\n") as f:
		f.write("\n".join([altlex2rel[0]] + sorted(altlex2rel[1:])) + "\n")
	with open(gum_target + "rst" + os.sep + "stats" + os.sep + "rel2dm.tab",'w',encoding="utf8",newline="\n") as f:
		f.write("\n".join([rel2dm[0]] + sorted(rel2dm[1:])) + "\n")
	with open(gum_target + "rst" + os.sep + "stats" + os.sep + "rel2altlex.tab",'w',encoding="utf8",newline="\n") as f:
		f.write("\n".join([rel2altlex[0]] + sorted(rel2altlex[1:])) + "\n")


if __name__ == "__main__":
	if platform.system() == "Windows":
		import os, msvcrt
		msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

	filename = sys.argv[1]
	if "*" in filename:
		file_list = glob(sys.argv[1])
	else:
		file_list = [filename]

	outdir = os.path.abspath(".") + os.sep + "out_rst" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for filename in file_list:
		tt_file = filename.replace(".rs3", ".xml").replace(".rs4", ".xml")
		fix_file(filename, tt_file, ".." + os.sep + ".." + os.sep + "src" + os.sep, outdir)

