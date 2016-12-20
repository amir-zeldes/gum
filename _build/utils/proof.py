import os, ntpath
from glob import glob

def proof_pos(tok, pos, lemma, func, docname, line, warn=True):
	warnings = ""

	# NOTE we're not recovering parent function in case of conj; maybe we should

	# Check general tag validtity
	tags = ["$","CC","CD","DT","EX","FW","IN","IN/that","JJ","JJR","JJS","LS","MD","NN","NNS","NP","NPS","PDT","POS","PP",'PP$',"RB","RBR","RBS","RP","SENT","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","VH","VHD","VHG","VHN","VHP","VHZ","VV","VVD","VVG","VVN","VVP","VVZ","WDT","WP",'WP$',"WRB","``","''","(",")",",",":"]
	if pos not in tags:
		warnings += "Invalid POS: ' " + pos + " ' in " + docname + " on line " + str(line)

	# Check punctuation
	if func == "punct" and pos == "POS":
		warnings += "punct & POS in " +docname+ " on line " + str(line)
	if tok in ["[","]"] and pos == "SYM":
		warnings += "[ / ] as SYM in " + docname + " on line " + str(line)

	# Check possessives
	if func =="possessive" and pos != "POS":
		warnings += "Suspicious POS in " + docname + " on line " + str(line)
	if func !="possessive" and pos == "POS":
		warnings += "Suspicious POS in " + docname + " on line " + str(line)
	if func not in ["poss","conj"] and pos == "PP$" and tok not in ["mine","yours","hers","ours","theirs"]:
		warnings += "PP$ not poss in " + docname + " on line " + str(line)

	if warn and warnings != "":
		print warnings

	return pos


def proof(gum_source, gum_target, edit=False):
	xml_source = gum_source + "xml" + os.sep
	xml_target = gum_target + "xml" + os.sep

	xmlfiles = glob(xml_source + "*.xml")

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		#print "\t+ " + " "*40 + "\r",
		#print " " + str(docnum+1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r",

		# Dictionaries to hold token annotations from conll10 data
		funcs = {}

		tok_num = 0

		depfile = xmlfile.replace("xml" + os.sep,"dep" + os.sep).replace("xml","conll10")
		dep_lines = open(depfile).read().replace("\r","").split("\n")
		line_num = 0
		for line in dep_lines:
			line_num += 1
			if "\t" in line:  # token line
				if line.count("\t") != 9:
					print "WARN: Found line with less than 9 tabs in " + docname + " line: " + str(line_num)
				else:
					tok_num += 1
					fields = line.split("\t")
					funcs[tok_num] = fields[7]

		xml_lines = open(xmlfile).read().replace("\r","").split("\n")
		tok_num = 0
		line_num = 0

		for line in xml_lines:
			line_num += 1
			if "\t" in line:  # Token
				tok_num += 1
				fields = line.split("\t")
				tok = fields[0]
				pos = fields[1]
				lemma = fields[2]
				func = funcs[tok_num]
				pos = proof_pos(tok,pos,lemma,func,docname,line_num)
				line = "\t".join([tok, pos, lemma])
			output += line + "\n"

		output = output.strip() + "\n"

		if edit:
			outfile = open(xml_target + docname,'wb')
			outfile.write(output)
			outfile.close()

	print "\no Proofed tags in " + str(len(xmlfiles)) + " documents" + " " *20


if __name__=="__main__":

	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("-t", dest="target", action="store", help="GUM build target directory", default="../target")
	parser.add_argument("-s", dest="source", action="store", help="GUM build source directory", default="../src")
	parser.add_argument("-e", dest="edit", action="store_true", help="Whether to edit or just warn")

	options = parser.parse_args()

	gum_source = os.path.abspath(options.source.replace("/",os.sep)) + os.sep
	gum_target = os.path.abspath(options.target.replace("/",os.sep)) + os.sep

	print "="*20
	print "Proofing POS tags"
	print "="*20

	proof(gum_source,gum_source,options.edit)