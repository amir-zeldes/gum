#!/usr/bin/python
# -*- coding: utf-8 -*-

import re,sys,platform
import ntpath, os
from glob import glob

def fix_rst(gum_source, gum_target):
	outdir = gum_target + "rst" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	file_list = glob(gum_source + "rst" + os.sep + "*.rs3")
	for docnum, filename in enumerate(file_list):
		tt_file = filename.replace(".rs3", ".xml").replace("rst","xml")
		print "\t+ " + " "*60 + "\r",
		print " " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Adjusting borders for " + ntpath.basename(filename) + "\r",
		fix_file(filename,tt_file,gum_target + "rst" + os.sep)

	print "o Adjusted " + str(len(file_list)) + " RST files" + " " * 40


def fix_file(filename,tt_file,outdir):

	# Get reference tokens
	rst_file_name = ntpath.basename(filename)
	tokens = []

	last_good_token = ""
	outfile = open(outdir + rst_file_name,'wb')

	with open(tt_file) as tt:
		lines = tt.read().replace("\r","").split("\n")

	current_token = 0

	for line in lines:
		if not line.startswith("<") and len(line) > 0 and "\t" in line: # token
			token = line.split("\t")[0]
			current_token += 1
			tokens.append(token)

	total_out_tokens = 0


	with open(filename) as rst:
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
			token_reached += seg_tokens

	outfile.write(out_data)


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

