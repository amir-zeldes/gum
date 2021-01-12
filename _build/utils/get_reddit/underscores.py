import os, glob, re, io, sys

PY3 = sys.version_info[0] == 3

def deunderscoring(src_folder, textdic):
	make_text(src_folder + "xml" + os.sep, textdic, 0, lemma_col=2)
	make_text(src_folder + "tsv" + os.sep, textdic, 2, unescape_xml=True)
	make_text(src_folder + "dep" + os.sep, textdic, 1, unescape_xml=True)
	make_text_rst(src_folder + "rst" + os.sep, textdic)
	make_text_const(src_folder + "const" + os.sep, textdic)


def make_text(folder, textdic, tok_col, lemma_col=None, unescape_xml=False):
	files_to_process = glob.glob(folder + "GUM_reddit*")
	print("o Processing " + str(len(files_to_process)) + " files in " + folder + "...")

	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").split("\n")

		tokens = textdic[os.path.basename(f_path)[:os.path.basename(f_path).find(".")]]
		if unescape_xml:
			tokens = tokens.replace("&gt;",">").replace("&lt;","<").replace("&amp;","&")
		else:
			if "&" in tokens and not "&amp;" in tokens and not "_ring" in f_path:
				tokens = tokens.replace("&","&amp;")
			tokens = tokens.replace(">","&gt;").replace("<","&lt;")
		if not PY3:
			tokens = tokens.decode("utf8")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:
			for i, line in enumerate(in_lines):
				if line.startswith('<'):
					fout.write(line+"\n")
				elif "\t" in line:
					elements = line.split('\t')
					elements[tok_col] = tokens[:len(elements[tok_col])]
					tokens = tokens[len(elements[tok_col]):]
					#if not unescape_xml:
					#	elements[tok_col] = elements[tok_col].replace("&amp;","&").replace("&","&amp;")
					if lemma_col is not None:
						if elements[lemma_col] == '_':
							elements[lemma_col] = elements[tok_col]
						elif elements[lemma_col] == "*LOWER*":
							elements[lemma_col] = elements[tok_col].lower()
					try:
						fout.write('\t'.join(elements)+"\n")
					except Exception as e:
						a=4
				else:
					fout.write(line)
					if i < len(in_lines) - 1:
						if PY3:
							fout.write("\n")
						else:
							fout.write(unicode("\n"))


def make_text_rst(folder, textdic):
	files_to_process = glob.glob(folder + "GUM_reddit*.rs3")
	print("o Processing " + str(len(files_to_process)) + " files in "+folder+"...")

	# Delete tokens in .xml files
	for f_path in files_to_process:

		tokens = textdic[os.path.basename(f_path)[:os.path.basename(f_path).find(".")]]
		if not PY3:
			tokens = tokens.decode("utf8")
		if "&" in tokens and not "&amp;" in tokens and not "_ring" in f_path:  # Some bigquery entries have no &amp;
			tokens = tokens.replace("&","&amp;")
		tokens = tokens.replace(">","&gt;").replace("<","&lt;")  # Reddit API does not escape lt/gt, but does escape &amp;


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
							try:
								out_seg += tokens[cursor]
							except Exception as e:
								print("WARNING: tried to access tokens at position " + str(cursor) + ", but "
									  + "an exception occurred. Are you sure '" + f_path + "' was downloaded "
									  + "properly? (len(tokens) = " + str(len(tokens)) + ".)")
							cursor += 1
						else:
							out_seg += c

					#out_seg = out_seg.replace("&","&amp;")
					fout.write(pre + out_seg + post + "\n")


def underscoring(src_folder):
	make_underscores(src_folder + "xml" + os.sep,0,lemma_col=2)
	make_underscores(src_folder + "tsv" + os.sep,2)
	make_underscores(src_folder + "dep" + os.sep,1)
	make_underscores_rst(src_folder + "rst" + os.sep)
	make_underscores_const(src_folder + "const" + os.sep)


def make_underscores_rst(folder):
	files_to_process = glob.glob(folder + "GUM_reddit*.rs3")
	print("o Processing " + str(len(files_to_process)) + " files in "+folder+"...")

	# Delete tokens in .xml files
	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").strip().split("\n")

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
					fout.write(pre + seg + post)
					if i < len(in_lines) - 1:  # Trailing newline
						if PY3:
							fout.write("\n")
						else:
							fout.write(unicode("\n"))


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
					if PY3:
						fout.write("#Text=_" + "\n")
					else:
						fout.write(unicode("#Text=_" + "\n"))
				elif "\t" in line:
					#line = line.replace("&amp;","&")
					elements = line.split('\t')
					if lemma_col is not None:
						if elements[lemma_col] == elements[tok_col]:  # Delete lemma if identical to token
							elements[lemma_col] = '_'
						elif elements[tok_col].lower() == elements[lemma_col]:
							elements[lemma_col] = "*LOWER*"
					elements[tok_col] = len(elements[tok_col])*'_'
					if PY3:
						fout.write('\t'.join(elements) + "\n")
					else:
						fout.write(unicode('\t'.join(elements) + "\n"))
				else:
					fout.write(line)
					if i < len(in_lines) - 1:
						if PY3:
							fout.write("\n")
						else:
							fout.write(unicode("\n"))


def make_underscores_const(const_path):

	files_to_process = glob.glob(const_path + "GUM_reddit_*.ptb")
	print("o Processing " + str(len(files_to_process)) + " files in "+const_path+"...")

	for f_path in files_to_process:

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").strip().split("\n")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:

			out_lines = []
			for line in in_lines:
				out_units = []
				units = line.split(" ")
				for i, unit in enumerate(units):
					buffer = ""
					if unit.endswith(")"):
						for c in unit:
							if c != ")":
								buffer += "_"
								# only 1 underscore for these fellas
								if "-LSB-" in unit or "-RSB-" in unit:
									buffer += ')'
									break
								if unit == '—)' and out_units[-1] == '(:': # hack because the new parser replaces '--' with a single '-'
									buffer += '_)'
									break
							else:
								buffer += c
					else:
						buffer = unit
					out_units.append(buffer)
				out_lines.append(" ".join(out_units))

			fout.write("\n".join(out_lines) + "\n")


def make_text_const(const_path, textdic):

	files_to_process = glob.glob(const_path + "GUM_reddit_*.ptb")
	print("o Processing " + str(len(files_to_process)) + " files in "+const_path+"...")

	for f_path in files_to_process:

		tokens = textdic[os.path.basename(f_path)[:os.path.basename(f_path).find(".")]]
		tokens = tokens.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&").replace("(","-LRB-").replace(")","-RRB-")
		if not PY3:
			tokens = tokens.decode("utf8")
		cursor = 0

		with io.open(f_path, 'r', encoding='utf-8') as fin:
			in_lines = fin.read().replace("\r","").strip().split("\n")

		with io.open(f_path, 'w', encoding='utf-8', newline="\n") as fout:

			out_lines = []
			for line in in_lines:
				out_units = []
				units = line.split(" ")
				for i, unit in enumerate(units):
					if unit.endswith(")"):
						for c in unit:
							if c != ")":
								buffer += tokens[cursor]
								cursor += 1
							else:
								try:
									buffer += c
								except Exception as e:
									a=4
					else:
						buffer = unit
					out_units.append(buffer)
					buffer = ""
				out_lines.append(" ".join(out_units))

			fout.write("\n".join(out_lines) + "\n")
