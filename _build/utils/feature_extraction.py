"""
Script to extract depdendency and XML markup information from data
in the conllu and CWB XML formats.

"""
import os, re, io


class ParsedToken:
	def __init__(self, tok_id, text, lemma, pos, morph, head, func):
		self.id = tok_id
		self.text = text.strip()
		self.text_lower = text.lower()
		self.pos = pos
		self.lemma = lemma if lemma != "_" else text
		self.morph = morph
		self.head = head
		self.func = func
		self.heading = "_"
		self.caption = "_"
		self.list = "_"
		self.date = "_"
		self.s_type = "_"

	def __repr__(self):
		return str(self.text) + " (" + str(self.pos) + "/" + str(self.lemma) + ") " + "<-" + str(self.func) + "- " + str(self.head_text)


def get_tok_info(docname,corpus_root):

	if corpus_root[-1]!=os.sep:
		corpus_root += os.sep

	xml_file = corpus_root + "xml" + os.sep + docname + ".xml"
	conll_file = corpus_root + "dep" + os.sep + docname + ".conllu"
	tokens = []

	for line in io.open(conll_file,encoding="utf8").read().replace("\r","").split("\n"):
		if "\t" in line:
			cols = line.split("\t")
			if "." not in cols[0] and "-" not in cols[0]:  # Ignore ellipsis and supertokens
				tokens.append(ParsedToken(cols[0],cols[1],cols[2],cols[3],cols[5],cols[6],cols[7]))

	counter = 0
	heading = "_"
	caption = "_"
	date = "_"
	list = "_"
	s_type = "_"
	para = "_"
	item = "_"
	for line in io.open(xml_file,encoding="utf8").read().replace("\r", "").split("\n"):
		if "<s type=" in line:
			m = re.search(r'<s type="([^"]+)"',line)
			s_type = m.group(1)
		if "<head" in line:
			heading = "head"
		elif "<caption" in line:
			caption = "caption"
		elif "</head" in line:
			heading = "_"
		elif "</caption" in line:
			caption = "_"
		elif '<list type="ordered' in line:
			list = "ordered"
		elif '<list type="unordered' in line:
			list = "unordered"
		elif "</list" in line:
			list = "_"
		elif '<date' in line:
			date = "date"
		elif "</date" in line:
			date = "_"
		elif '<p>' in line:
			para = "open_para"
		elif '<item>' in line:
			item = "open_item"
		if "\t" in line:
			fields = line.split("\t")
			tokens[counter].heading = heading
			tokens[counter].caption = caption
			tokens[counter].list = list
			tokens[counter].s_type = s_type
			tokens[counter].date = date
			tokens[counter].para = para
			tokens[counter].item = item
			tokens[counter].pos = fields[1]
			tokens[counter].lemma = fields[2]
			para = "_"
			item = "_"

			counter += 1

	return tokens