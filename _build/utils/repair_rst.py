#!/usr/bin/python
# -*- coding: utf-8 -*-

import re,sys,platform
import ntpath, os, io
from .rst2dep import make_rsd
from .rst2dis import rst2dis
from glob import glob

PY2 = sys.version_info[0] < 3


def fix_rst(gum_source, gum_target, reddit=False):
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
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		file_list.append(file_)

	for docnum, filename in enumerate(file_list):
		tt_file = filename.replace(".rs3", ".xml").replace("rst","xml")
		sys.stdout.write("\t+ " + " "*70 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Adjusting borders for " + ntpath.basename(filename) + "\r")
		fix_file(filename,tt_file,gum_source,gum_target + "rst" + os.sep + "rstweb" + os.sep)

	print("o Adjusted " + str(len(file_list)) + " RST files" + " " * 70)


def validate_rsd(rsd_line, linenum, docname):
	inname = " in document " + docname + " on line " + str(linenum)
	if re.search(r'\b[Tt]o\b[^\n]+head_pos=V.\|[^\n]+head_func=acl\|[^\n]+elaboration-attr', rsd_line) is not None:
		if "\tthat" not in rsd_line and "\tabout" not in rsd_line and "\tyou " not in rsd_line:  # check for that-clause embedding to-, or about PP
			sys.stderr.write("! adnominal infinitive clause should be purpose-attribute not elaboration-attribute" + inname)
	if re.search(r'(\bn.t\b[^\n]+)attribution-positive_r', rsd_line) is not None:
		if "surprised" not in rsd_line:
			sys.stderr.write("! suspicious attribution-positive_r with negation" + inname)


def fix_file(filename,tt_file,gum_source,outdir):

	# Get reference tokens
	rst_file_name = ntpath.basename(filename)
	tokens = []

	last_good_token = ""

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

		if not "<segment" in line:
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

	docname = os.path.basename(rst_file_name).replace(".rs3","")

	# Make rsd version
	rsd = make_rsd(out_data,gum_source,as_text=True,docname=os.path.basename(rst_file_name.replace(".rs3","")))
	for l, line in enumerate(rsd.split("\n")):
		validate_rsd(line, l, docname)

	with io.open(outdir.replace("rstweb","dependencies") + docname + ".rsd",'w',encoding="utf8",newline="\n") as f:
		f.write(rsd)

	# Make binary dis version
	dis = rst2dis(out_data, binarize=True)
	with io.open(outdir.replace("rstweb","lisp_binary") + docname+".dis",'w',encoding="utf8",newline="\n") as f:
		f.write(dis)

	# Make nary dis version
	dis = rst2dis(out_data, binarize=False)
	with io.open(outdir.replace("rstweb","lisp_nary") + docname+".dis",'w',encoding="utf8",newline="\n") as f:
		f.write(dis)


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
		tt_file = filename.replace(".rs3", ".xml")
		fix_file(filename,tt_file,outdir)

