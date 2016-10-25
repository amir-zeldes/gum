#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,platform,os
from collections import OrderedDict
import ntpath
from glob import glob

def equiv_tok(token):
	replacements = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "’": "'", "—": "-", "&quot;": '"', "&apos;": "'", "(":"-LRB-", ")":"-RRB-", "…":"...",
					"“":'"',"”":'"','–':"-", "é":"e","á":"a","ó":"o","í":"i"}

	for find, replace in replacements.iteritems():
		token = token.replace(find, replace)
	return token

def unescape_xml(token):
	replacements = {"&amp;": "&", "&gt;": ">", "&lt;": "<", "&quot;": '"', "&apos;": "'"}

	for find, replace in replacements.iteritems():
		token = token.replace(find, replace)
	return token


def fix_tsv(gum_source, gum_target):
	file_list = glob(gum_source + "tsv" + os.sep + "*.tsv")
	outdir = gum_target + "coref" + os.sep + "tsv" + os.sep
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	for docnum, filename in enumerate(file_list):
		tt_file = filename.replace(".tsv", ".xml").replace("tsv","xml")
		print "\t+ " + " "*60 + "\r",
		print " " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Adjusting borders for " + ntpath.basename(filename) + "\r",
		fix_file(filename,tt_file,outdir)

	print "o Adjusted " + str(len(file_list)) + " WebAnno TSV files" + " " * 40


def fix_file(filename,tt_file,outdir):

	# Get reference tokens
	tsv_file_name = ntpath.basename(filename)
	tokens = []

	outdir = os.path.abspath(outdir) + os.sep
	last_good_token = ""
	outfile = open(outdir + tsv_file_name,'wb')
	tt_file = os.path.abspath(tt_file).replace("tsv" + os.sep,"xml"+os.sep)

	with open(tt_file) as tt:
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


	with open(filename) as tsv:
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
				char_range = str(current_char_start) + "-" + str(current_char_start+len(current_token_text))
				out_fields[1] = char_range
				current_char_start += len(current_token_text) + 1

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
	for line_num, line in out_lines.iteritems():
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
