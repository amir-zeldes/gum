import sys, os, ntpath, io
from glob import glob
from collections import defaultdict

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
	dep_target = gum_target + "dep" + os.sep + "ud" + os.sep

	if not os.path.exists(dep_target):
		os.makedirs(dep_target)

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
		for line in conll_lines:
			if "\t" in line:  # Token
				fields = line.split("\t")
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

		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8") as f:
			f.write(punct_fixed)

	sys.__stdout__.write("o Converted " + str(len(depfiles)) + " documents to Universal Dependencies" + " " *20 + "\n")