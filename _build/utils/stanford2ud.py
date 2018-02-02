import sys, os, ntpath, io
from glob import glob
from collections import defaultdict
from .nlp_helper import ud_morph

PY2 = sys.version_info[0] < 3

try:
	from StringIO import StringIO
except ImportError:
	from io import StringIO

try:
	from udapi.core.document import Document
	from udapi.block.ud.fixpunct import FixPunct
except:
	if not PY2:
		print("      - Unable to import module udapi.core.run -")
		print("      Punctuation behavior in the UD conversion relies on udapi. ")
		print("      Please install it (e.g. pip3 install udapi)")

from .depedit import DepEdit

class Args:

	def __init__(self):
		self.scenario = ['ud.FixPunct', 'write.Conllu']


class Entity:

	def __init__(self, ent_id, type, infstat):
		self.id = ent_id
		self.type = type
		self.infstat = infstat
		self.tokens = []

	def __repr__(self):
		tok_nums = [int(x) for x in self.tokens]
		tok_range = str(min(tok_nums)) + "-" + str(max(tok_nums))
		return "ent " + self.id + ": " + self.type + "|" + self.infstat + " (" + tok_range + ")"

	def assign_tok_nums(self):
		self.tok_nums = [int(x) for x in self.tokens]
		self.start = min(self.tok_nums)
		self.end = max(self.tok_nums)

	def get_length(self):
		return self.end - self.start + 1


def fix_punct(conllu_string):
	doc = Document()
	doc.from_conllu_string(conllu_string)
	fixpunct_block = FixPunct()
	fixpunct_block.process_document(doc)
	output_string = doc.to_conllu_string()
	return output_string


def create_ud(gum_target):
	# Use generated enriched targets in dep/stanford/ as source
	dep_source = gum_target + "dep" + os.sep + "stanford" + os.sep
	dep_target = gum_target + "dep" + os.sep + "ud" + os.sep + "not-to-release" + os.sep
	train_split_target = gum_target + "dep" + os.sep + "ud" + os.sep

	if not os.path.exists(train_split_target):
		os.makedirs(train_split_target)
	if not os.path.exists(dep_target):
		os.makedirs(dep_target)

	ud_dev = ["GUM_interview_peres","GUM_interview_cyclone","GUM_interview_gaming",
			   "GUM_news_iodine","GUM_news_defector","GUM_news_homeopathic",
			   "GUM_voyage_athens","GUM_voyage_isfahan","GUM_voyage_coron",
			   "GUM_whow_joke","GUM_whow_skittles","GUM_whow_overalls"]
	ud_test = ["GUM_interview_mcguire","GUM_interview_libertarian","GUM_interview_hill",
			   "GUM_news_nasa","GUM_news_expo","GUM_news_sensitive",
			   "GUM_voyage_oakland","GUM_voyage_thailand","GUM_voyage_vavau",
			   "GUM_whow_mice","GUM_whow_cupcakes","GUM_whow_cactus"]

	train_string, dev_string, test_string = "", "", ""

	depfiles = glob(dep_source + "*.conll10")

	depedit = DepEdit(config_file="utils" + os.sep + "stan2uni.ini")

	for docnum, depfile in enumerate(depfiles):
		docname = ntpath.basename(depfile).replace('.conll10',"")
		sys.__stdout__.write("\t+ " + " "*40 + "\r")
		sys.__stdout__.write(" " + str(docnum+1) + "/" + str(len(depfiles)) + ":\t+ " + docname + "\r")

		entity_file = depfile.replace("dep" + os.sep + "stanford","coref" + os.sep + "tsv" + os.sep).replace("conll10","tsv")
		tsv_lines = io.open(entity_file,encoding="utf8").read().replace("\r","").split("\n")
		int_max_entity = 10000
		tok_id = 0
		entity_dict = {}
		for line in tsv_lines:
			if "\t" in line:  # Token line
				tok_id += 1
				fields = line.split("\t")
				entity_string, infstat_string = fields[3:5]
				if entity_string != "_":
					entities = entity_string.split("|")
					infstats = infstat_string.split("|")
					for i, entity in enumerate(entities):
						infstat = infstats[i]
						# Make sure all entities are numbered
						if "[" not in entity:  # Single token entity with no ID
							entity += "["+str(int_max_entity)+"]"
							infstat += "[" + str(int_max_entity) + "]"
							int_max_entity += 1
						entity_id = entity[entity.find("[")+1:-1]
						entity = entity[:entity.find("[")]
						infstat = infstat[:infstat.find("[")]
						if entity_id not in entity_dict:
							entity_dict[entity_id] = Entity(entity_id,entity,infstat)
						entity_dict[entity_id].tokens.append(str(tok_id))

		toks_to_ents = defaultdict(list)
		for ent in entity_dict:
			entity_dict[ent].assign_tok_nums()
			for tok in entity_dict[ent].tokens:
				toks_to_ents[tok].append(entity_dict[ent])

		conll_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0
		processed_lines = []
		negative = []
		for line in conll_lines:
			if "\t" in line:  # Token
				tok_num += 1
				fields = line.split("\t")
				if fields[7] == "neg":
					negative.append(tok_num)
				absolute_head_id = tok_num - int(fields[0]) + int(fields[6]) if fields[6] != "0" else 0
				if str(tok_num) in toks_to_ents:
					for ent in sorted(toks_to_ents[str(tok_num)],key=lambda x:x.get_length(),reverse=True):
						# Check if this is the head of that entity
						if absolute_head_id > ent.end or absolute_head_id < ent.start and absolute_head_id > 0:
							# This is the head
							fields[5] = "ent_head=" + ent.type + "|" + "infstat=" + ent.infstat
				line = "\t".join(fields)
			processed_lines.append(line)

		converted = depedit.run_depedit(processed_lines,filename=docname,sent_id=True,docname=True)

		if PY2:
			punct_fixed = converted
		else:
			punct_fixed = fix_punct(converted)

		# Add UD morphology using CoreNLP script - we assume target/const/ already has .ptb tree files
		utils_abs_path = os.path.dirname(os.path.realpath(__file__))
		morphed = ud_morph(punct_fixed, docname, utils_abs_path + os.sep + ".." + os.sep + "target" + os.sep + "const" + os.sep)

		if not PY2:
			# CoreNLP returns bytes in ISO-8859-1
			# ISO-8859-1 mangles ellipsis glyph, so replace manually
			morphed = morphed.decode("ISO-8859-1").replace("\r","").replace("","…").replace("","“").replace("","’").replace("",'—').replace("","–")

		# Add negative polarity
		negatived = []
		tok_num = 0
		for line in morphed.split("\n"):
			if "\t" in line:
				tok_num += 1
				fields = line.split("\t")
				if tok_num in negative:
					if fields[5] == "_":
						fields[5] = "Polarity=Neg"
				negatived.append("\t".join(fields))
			else:
				negatived.append(line)
		negatived = "\n".join(negatived)

		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8", newline="\n") as f:
			f.write(negatived)

		if docname in ud_dev:
			dev_string += negatived
		elif docname in ud_test:
			test_string += negatived
		else:
			train_string += negatived

	with io.open(train_split_target + "en_gum-ud-train.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(train_string)
	with io.open(train_split_target + "en_gum-ud-dev.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(dev_string)
	with io.open(train_split_target + "en_gum-ud-test.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(test_string)

	sys.__stdout__.write("o Converted " + str(len(depfiles)) + " documents to Universal Dependencies" + " " *20 + "\n")

