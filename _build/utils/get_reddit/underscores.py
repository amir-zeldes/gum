import os, glob, re, io


def deunderscoring(src_folder, textdic):
	make_text(src_folder + "xml" + os.sep, textdic, 0, lemma_col=2)
	make_text(src_folder + "tsv" + os.sep, textdic, 2, unescape_xml=True)
	make_text(src_folder + "dep" + os.sep, textdic, 1, unescape_xml=True)
	make_text_rst(src_folder + "rst" + os.sep, textdic)


def make_text(folder, textdic, tok_col, lemma_col=None, unescape_xml=False):
	files_to_process = glob.glob(folder + "GUM_reddit*")
	print("o Processing " + str(len(files_to_process)) + " files in " + folder + "...")

	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").split("\n")

		tokens = textdic[os.path.basename(f_path)[:os.path.basename(f_path).find(".")]]
		if unescape_xml:
			tokens = tokens.replace("&gt;",">").replace("&lt;","<").replace("&amp;","&")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:
			for i, line in enumerate(in_lines):
				if line.startswith('<'):
					fout.write(line+"\n")
				elif "\t" in line:
					elements = line.split('\t')
					elements[tok_col] = tokens[:len(elements[tok_col])]
					if lemma_col is not None:
						if elements[lemma_col] == '_':
							elements[lemma_col] = elements[tok_col]
						elif elements[lemma_col] == "*LOWER*":
							elements[lemma_col] = elements[tok_col].lower()
					fout.write('\t'.join(elements)+"\n")
					tokens = tokens[len(elements[tok_col]):]
				else:
					fout.write(line)
					if i < len(in_lines) - 1:
						fout.write("\n")


def make_text_rst(folder, textdic):
	files_to_process = glob.glob(folder + "GUM_reddit*.rs3")
	print("o Processing " + str(len(files_to_process)) + " files in "+folder+"...")

	# Delete tokens in .xml files
	for f_path in files_to_process:

		tokens = textdic[os.path.basename(f_path)[:os.path.basename(f_path).find(".")]]

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").split("\n")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:
			cursor = 0
			for i, line in enumerate(in_lines):
				if "<segment" not in line:
					fout.write(line + "\n")
				else:
					m = re.search(r'(.*<segment[^>]*>)(.*)(</segment>)',line)
					pre = m.group(1)
					seg = m.group(2)
					post = m.group(3)
					out_seg = ""
					for c in seg:
						if c == "_":
							out_seg += tokens[cursor]
							cursor += 1
						else:
							out_seg += c

					fout.write(pre + out_seg + post + "\n")


def underscoring(src_folder):
	make_underscores(src_folder + "xml" + os.sep,0,lemma_col=2)
	make_underscores(src_folder + "tsv" + os.sep,2)
	make_underscores(src_folder + "dep" + os.sep,1)
	make_underscores_rst(src_folder + "rst" + os.sep)


def make_underscores_rst(folder):
	files_to_process = glob.glob(folder + "GUM_reddit*.rs3")
	print("o Processing " + str(len(files_to_process)) + " files in "+folder+"...")

	# Delete tokens in .xml files
	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").split("\n")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:
			for i, line in enumerate(in_lines):
				if "<segment" not in line:
					fout.write(line + "\n")
				else:
					m = re.search(r'(.*<segment[^>]*>)(.*)(</segment>)',line)
					pre = m.group(1)
					seg = m.group(2)
					post = m.group(3)
					seg = re.sub(r'[^ ]','_',seg)
					fout.write(pre + seg + post + "\n")


def make_underscores(folder, tok_col, lemma_col=None):

	files_to_process = glob.glob(folder + "GUM_reddit*")
	print("o Processing " + str(len(files_to_process)) + " files in "+folder+"...")

	# Delete tokens in .xml files
	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").split("\n")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:
			for i, line in enumerate(in_lines):
				if line.startswith('<'):
					fout.write(line + "\n")
				elif line.startswith("#Text="):
					fout.write("#Text=_" + "\n")
				elif "\t" in line:
					elements = line.split('\t')
					if lemma_col is not None:
						if elements[lemma_col] == elements[tok_col]:  # Delete lemma if identical to token
							elements[lemma_col] = '_'
						elif elements[tok_col].lower() == elements[lemma_col]:
							elements[lemma_col] = "*LOWER*"
					elements[tok_col] = len(elements[tok_col])*'_'
					fout.write('\t'.join(elements) + "\n")
				else:
					fout.write(line)
					if i < len(in_lines) - 1:
						fout.write("\n")
