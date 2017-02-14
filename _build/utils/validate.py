import os, ntpath, sys
import shutil
import glob
import re
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
		print "\t+ " + " "*40 + "\r",
		print " " + str(docnum+1) + "/" + str(len(file_list)) + ":\t+ Validating " + xml_file + "\r",

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
					err_match = re.search(r'\.xml:([0-9]+).*?: (.*)',str(err))
					if err_match is not None:
						errors += "  Line " + err_match.group(1) + ": " + err_match.group(2) + "\n"
					else:
						errors += "\n  "+err + "\n"
	print "o " + str(valid_files) + " documents pass XSD validation" + " "*30
	if len(errors) > 0:
		print errors
		print "Aborting due to validation errors"
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

def validate_src(gum_source):

	dirs = [('xml', 'xml'), ('dep', 'conll10'), ('rst', 'rs3'), ('tsv', 'tsv')]

	# check that each dir has same # and names of files (except extensions)
	file_lists = []
	for dir in dirs:
		dir_name = gum_source + dir[0]
		dir_ext = dir[1]
		filenames = []
		for filename in glob.glob(dir_name + os.sep + '*.' + dir_ext):
			basename = ntpath.basename(filename)
			filename_validate = re.match(r'(\w+)\.' + dir_ext, basename)
			if filename_validate is None:
				print 'x Unexpected filename: ' + filename
			else:
				filenames.append(filename_validate.group(1))

		print "Found "+ str(len(filenames)) +" in "+ dir_ext + "\r",
		file_lists.append(filenames)
	
	# check that filenames are the same across dirs
	if all(len(x)==len(file_lists[0]) for x in file_lists) is False:
		print 'x Different numbers of files in directories:'
		for d in xrange(len(dirs)):
			print str(dirs[d][0]) + ": " + str(len(file_lists[d]))
		exit()
	else:
		for i in xrange(len(file_lists[0])):
			same_names = all(x[i]==file_lists[0][i] for x in file_lists)
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
			#filename = os.path.abspath(filepath)
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
		same_count = all(x[i]==all_tok_counts[0][i] for x in all_tok_counts)
		if same_count is False:
			print "x Different token counts in " + file_lists[0][i] + ":"
			for d in xrange(len(all_tok_counts)):
				print str(dirs[d][0]) + ": " + str(all_tok_counts[d][i])
				token_counts_match = False
	if token_counts_match:
		print "o Token counts match across directories"
	
	# check sentences (based on tok count)
	all_sent_lengths = []
	sentence_dirs = [('xml', 'xml'), ('dep', 'conll10')] # just the dirs where we check sentences
	
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
