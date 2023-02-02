"""
process_underscores.py

Script to handle licensed data for which underlying text cannot be posted online (e.g. LDC data).
Users need a copy of the LDC distribution of an underlying resource to restore text in some of the corpora.


"""

__author__ = "Amir Zeldes"
__license__ = "Apache 2.0"
__version__ = "2.0.0"

import io, re, os, sys
from glob import glob
from collections import defaultdict
script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

PY3 = sys.version_info[0] == 3


def underscore_files(filenames):
	def underscore_rel_field(text):
		blanked = []
		text = text.replace("<*>","❤")
		for c in text:
			if c!="❤" and c!=" ":
				blanked.append("_")
			else:
				blanked.append(c)
		return "".join(blanked).replace("❤","<*>")

	if isinstance(filenames,str):
		filenames = glob(filenames + "*.*")
	for f_path in filenames:
		skiplen = 0
		with io.open(f_path, 'r', encoding='utf8') as fin:
			lines = fin.readlines()

		with io.open(f_path, 'w', encoding='utf8', newline="\n") as fout:
			output = []
			if f_path.endswith(".rels"):
				for l, line in enumerate(lines):
					line = line.strip()
					if "\t" in line and l > 0:
						doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, direction, orig_label, label = line.split("\t")
						if "GUM" in doc and "reddit" not in doc:
							output.append(line)
							continue
						unit1_txt = underscore_rel_field(unit1_txt)
						unit2_txt = underscore_rel_field(unit2_txt)
						unit1_sent = underscore_rel_field(unit1_sent)
						unit2_sent = underscore_rel_field(unit2_sent)
						fields = doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, direction, orig_label, label
						line = "\t".join(fields)
					output.append(line)
			else:
				doc = ""
				for line in lines:
					line = line.strip()
					if line.startswith("# newdoc id"):
						doc = line.split("=",maxsplit=1)[1].strip()
					if "GUM" in doc and "reddit" not in doc:
						output.append(line)
						continue
					if line.startswith("# text"):
						m = re.match(r'(# text ?= ?)(.+)',line)
						if m is not None:
							line = m.group(1) + re.sub(r'[^\s]','_',m.group(2))
							output.append(line)
					elif "\t" in line:
						fields = line.split("\t")
						tok_col, lemma_col = fields[1:3]
						if lemma_col == tok_col:  # Delete lemma if identical to token
							fields[2] = '_'
						elif tok_col.lower() == lemma_col:
							fields[2] = "*LOWER*"
						if skiplen < 1:
							fields[1] = len(tok_col)*'_'
						else:
							skiplen -=1
						output.append("\t".join(fields))
						if "-" in fields[0]:  # Multitoken
							start, end = fields[0].split("-")
							start = int(start)
							end = int(end)
							skiplen = end - start + 1
					else:
						output.append(line)
			fout.write('\n'.join(output) + "\n")


def restore_docs(path_to_underscores,text_dict):
	def restore_range(range_string, underscored, tid_dict):
		output = []
		tok_ids = []
		range_strings = range_string.split(",")
		for r in range_strings:
			if "-" in r:
				s, e = r.split("-")
				tok_ids += list(range(int(s),int(e)+1))
			else:
				tok_ids.append(int(r))

		for tok in underscored.split():
			if tok == "<*>":
				output.append(tok)
			else:
				tid = tok_ids.pop(0)
				output.append(tid_dict[tid])
		return " ".join(output)

	dep_files = glob(path_to_underscores+os.sep+"*.conllu")
	tok_files = glob(path_to_underscores+os.sep+"*.tok")
	rel_files = glob(path_to_underscores+os.sep+"*.rels")
	skiplen = 0
	token_dict = {}
	tid2string = defaultdict(dict)
	for file_ in dep_files + tok_files + rel_files:
		lines = io.open(file_,encoding="utf8").readlines()
		underscore_len = 0  # Must match doc_len at end of file processing
		doc_len = 0
		if file_.endswith(".rels"):
			output = []
			violation_rows = []
			for l, line in enumerate(lines):
				line = line.strip()
				if l > 0 and "\t" in line:
					fields = line.split("\t")
					docname = fields[0]
					text = text_dict[docname]
					if "GUM_" in docname and "reddit" not in docname:  # Only Reddit documents need reconstruction in GUM
						output.append(line)
						continue
					doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, direction, orig_label, label = line.split("\t")
					underscore_len += unit1_txt.count("_") + unit2_txt.count("_") + unit1_sent.count("_") + unit2_sent.count("_")
					if underscore_len == 0:
						#sys.stderr.write("! Non-underscored file detected - " + os.path.basename(file_) + "\n")
						print("! DISRPT format alreadt restored in " + os.path.basename(file_) + "\n")
						sys.exit(0)
					unit1_txt = restore_range(unit1_toks, unit1_txt, tid2string[docname])
					unit2_txt = restore_range(unit2_toks, unit2_txt, tid2string[docname])
					unit1_sent = restore_range(s1_toks, unit1_sent, tid2string[docname])
					unit2_sent = restore_range(s2_toks, unit2_sent, tid2string[docname])
					plain = unit1_txt + unit2_txt + unit1_sent + unit2_sent
					plain = plain.replace("<*>","").replace(" ","")
					doc_len += len(plain)
					fields = doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, direction, orig_label, label
					line = "\t".join(fields)
					if doc_len != underscore_len and len(violation_rows) == 0:
						violation_rows.append(str(l) + ": " + line)
				output.append(line)

		else:
			tokfile = True if ".tok" in file_ else False
			output = []
			parse_text = ""
			docname = ""
			for line in lines:
				line = line.strip()
				if "# newdoc id " in line:
					tid = 0
					if parse_text !="":
						if not tokfile:
							token_dict[docname] = parse_text
					parse_text = ""
					docname = re.search(r'# newdoc id ?= ?([^\s]+)',line).group(1)
					if "GUM" in docname and "reddit" not in docname:
						output.append(line)
						continue
					if docname not in text_dict:
						raise IOError("! Text for document name " + docname + " not found.\n Please check that your LDC data contains the file for this document.\n")
					if ".tok" in file_:
						if docname not in token_dict:  # Fetch continuous token string from conllu
							parse_conllu = open(os.sep.join([script_dir,"..","..","..","dep",docname + ".conllu"])).read()
							toks = [l.split("\t") for l in parse_conllu.split("\n") if "\t" in l]
							toks = [l[1] for l in toks if "-" not in l[0] and "." not in l[0]]
							toks = "".join(toks)
							token_dict[docname] = toks
						text = token_dict[docname]
					else:
						text = text_dict[docname]
					doc_len = len(text)
					underscore_len = 0

				if "GUM" in docname and "reddit" not in docname:
					output.append(line)
					continue

				if line.startswith("# text"):
					m = re.match(r'(# ?text ?= ?)(.+)',line)
					if m is not None:
						i = 0
						sent_text = ""
						for char in m.group(2).strip():
							if char != " ":
								try:
									sent_text += text[i]
								except:
									raise IOError("Can't fix")
								i+=1
							else:
								sent_text += " "
						line = m.group(1) + sent_text
						output.append(line)
				elif "\t" in line:
					fields = line.split("\t")
					if skiplen < 1:
						underscore_len += len(fields[1])
						fields[1] = text[:len(fields[1])]
					if not "-" in fields[0] and not "." in fields[0]:
						parse_text += fields[1]
						tid += 1
						tid2string[docname][tid] = fields[1]
					if not tokfile:
						if fields[2] == '_' and not "-" in fields[0] and not "." in fields[0]:
							fields[2] = fields[1]
						elif fields[2] == "*LOWER*":
							fields[2] = fields[1].lower()
					if skiplen < 1:
						text = text[len(fields[1]):]
					else:
						skiplen -=1
					output.append("\t".join(fields))
					if "-" in fields[0]:  # Multitoken
						start, end = fields[0].split("-")
						start = int(start)
						end = int(end)
						skiplen = end - start + 1
				else:
					output.append(line)

		if not doc_len == underscore_len:
			if ".rels" in file_:
				sys.stderr.write(
					"\n! Tried to restore file " + os.path.basename(file_) + " but source text has different length than tokens in shared task file:\n" + \
					"  Source text in data/: " + str(doc_len) + " non-whitespace characters\n" + \
					"  Token underscores in " + file_ + ": " + str(underscore_len) + " non-whitespace characters\n" + \
					"  Violation row: " + violation_rows[0])
			else:
				sys.stderr.write("\n! Tried to restore document " + docname + " but source text has different length than tokens in shared task file:\n" + \
						  "  Source text in data/: " + str(doc_len) + " non-whitespace characters\n" + \
						  "  Token underscores in " + file_+": " + str(underscore_len) + " non-whitespace characters\n")
			with io.open("debug.txt",'w',encoding="utf8") as f:
				f.write(text_dict[docname])
				f.write("\n\n\n")
				f.write(parse_text)
			sys.exit(0)

		if not tokfile and parse_text != "":
			token_dict[docname] = parse_text

		with io.open(file_, 'w', encoding='utf8', newline="\n") as fout:
			fout.write("\n".join(output) + "\n")

	print("o Restored text for DISRPT format in " + \
					 #str(len(dep_files)) + " .conllu files, " + \
					 str(len(tok_files)) + " .tok files and "+ str(len(rel_files)) + " .rels files\n")






