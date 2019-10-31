#!/usr/bin/python
# -*- coding: utf-8 -*-

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
		self.line_tokens = []
		self.token_texts = []
		self.coref_type = ""
		self.coref_link = ""

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


def is_neg_lemma(lemma,pos):
	negstems = set(["imposs","improb","immort","inevit","incomp","indirec","inadeq","insuff","ineff","incong","incoh","inacc","invol","infreq","inapp","indist","infin","intol",
					"dislik","dys","dismount","disadvant","disinteg","disresp","disagr","disjoin","disprov","disinterest","discomfort","dishonest","disband","disentangl"])
	neglemmas = set(["nowhere","never","nothing","none","undo","uncover","unclench"])
	if lemma == 'unconscious':
		a=4
	lemma = lemma.lower()
	if lemma in negstems or lemma in neglemmas:
		return True
	elif lemma.startswith("non-"):
		return True
	elif lemma.startswith("not-"):
		return True
	elif lemma.startswith("un") and (pos.startswith("JJ") or pos.startswith("RB")):
		if not lemma.startswith("unique") and not lemma.startswith("under"):
			return True
	for stem in negstems:
		if lemma.startswith(stem):
			return True
	return False


def add_feat(field,feat):
	if field == "_":
		return feat
	else:
		attrs = field.split("|")
		attrs.append(feat)
		return "|".join(sorted(attrs))


def do_hard_replaces(text):
	"""Replace unresolvable conversion problems with hardwired replacements
	"""

	reps = [("""15	ashes	ash	NOUN	NNS	Number=Plur	12	obl	_	SpaceAfter=No
16	"	"	PUNCT	''	_	12	punct	_	_""","""15	ashes	ash	NOUN	NNS	Number=Plur	12	obl	_	SpaceAfter=No
16	"	"	PUNCT	''	_	15	punct	_	_"""),("""32	)	)	PUNCT	-RRB-	_	24	punct	_	_
33	that	that	PRON	WDT	PronType=Rel	34	nsubj	_	_""","""32	)	)	PUNCT	-RRB-	_	27	punct	_	_
33	that	that	PRON	WDT	PronType=Rel	34	nsubj	_	_""")]

	#reps = [] ########
	for f, r in reps:
		text = text.replace(f,r)
	return text

def create_ud(gum_target, reddit=False):
	# Use generated enriched targets in dep/stanford/ as source
	dep_source = gum_target + os.sep + "dep" + os.sep + "stanford" + os.sep
	dep_target = gum_target + os.sep + "dep" + os.sep + "ud" + os.sep + "not-to-release" + os.sep
	pepper_temp = gum_target + os.sep + ".." + os.sep + "utils" + os.sep + "pepper" + os.sep + "tmp" + os.sep + "entidep" + os.sep
	if not os.path.isdir(pepper_temp):
		os.makedirs(pepper_temp)
	train_split_target = gum_target + "dep" + os.sep + "ud" + os.sep

	if not os.path.exists(train_split_target):
		os.makedirs(train_split_target)
	if not os.path.exists(dep_target):
		os.makedirs(dep_target)

	ud_dev = ["GUM_interview_peres","GUM_interview_cyclone","GUM_interview_gaming",
			   "GUM_news_iodine","GUM_news_defector","GUM_news_homeopathic",
			   "GUM_voyage_athens","GUM_voyage_isfahan","GUM_voyage_coron",
			   "GUM_whow_joke","GUM_whow_skittles","GUM_whow_overalls",
			   "GUM_fiction_beast","GUM_bio_emperor","GUM_academic_librarians",
			   "GUM_fiction_lunre","GUM_bio_byron","GUM_academic_exposure"]
	ud_test = ["GUM_interview_mcguire","GUM_interview_libertarian","GUM_interview_hill",
			   "GUM_news_nasa","GUM_news_expo","GUM_news_sensitive",
			   "GUM_voyage_oakland","GUM_voyage_thailand","GUM_voyage_vavau",
			   "GUM_whow_mice","GUM_whow_cupcakes","GUM_whow_cactus",
			   "GUM_fiction_falling","GUM_bio_jespersen","GUM_academic_discrimination",
			   "GUM_academic_eegimaa","GUM_bio_dvorak","GUM_fiction_teeth"]

	train_string, dev_string, test_string = "", "", ""

	depfiles = []
	files_ = glob(dep_source + "*.conll10")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		depfiles.append(file_)

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
		tok_num_to_tsv_id = {}

		for line in tsv_lines:
			if "\t" in line:  # Token line
				tok_id += 1
				fields = line.split("\t")
				line_tok_id = fields[0]
				tok_txt = fields[2]
				tok_num_to_tsv_id[tok_id] = line_tok_id
				entity_string, infstat_string,coref_type_string, coref_link_string  = fields[3:7]
				if entity_string != "_":
					entities = entity_string.split("|")
					infstats = infstat_string.split("|")
					if coref_type_string != "_":
						coref_types = coref_type_string.split("|")
						coref_links = coref_link_string.split("|")
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
						entity_dict[entity_id].token_texts.append(tok_txt)
						entity_dict[entity_id].line_tokens.append(line_tok_id)


						# loop through coref relations
						if coref_type_string != "_":
							for j, coref_link in enumerate(coref_links):
								if "[" not in coref_link:
									entity_dict[entity_id].coref_type = coref_types[j]
									entity_dict[entity_id].coref_link = coref_link
								else:
									with_ids = coref_link[coref_link.find("[")+1:-1].split("_")
									if (entity_id in with_ids) or ("0" in with_ids):
										entity_dict[entity_id].coref_type = coref_types[j]
										entity_dict[entity_id].coref_link = coref_link[:coref_link.find("[")]



		toks_to_ents = defaultdict(list)
		for ent in entity_dict:
			entity_dict[ent].assign_tok_nums()
			for tok in entity_dict[ent].tokens:
				toks_to_ents[tok].append(entity_dict[ent])

		conll_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0
		processed_lines = []
		negative = []
		doc_toks = []
		doc_lemmas = []
		field_cache = {}
		sent_lens = []
		sent_len = 0
		line_id = -1
		coref_line_and_ent = []
		coref_line_and_ent_last_in_sent = {}


		counter = 0
		for line in conll_lines:
			line_id += 1
			if "\t" in line:  # Token
				sent_len += 1
				fields = line.split("\t")
				field_cache[tok_num] = fields
				tok_num += 1
				doc_toks.append(fields[1])
				doc_lemmas.append(fields[2])
				if fields[7] == "neg" or is_neg_lemma(fields[2],fields[3]):
					negative.append(tok_num)
				absolute_head_id = tok_num - int(fields[0]) + int(fields[6]) if fields[6] != "0" else 0
				if str(tok_num) in toks_to_ents:
					for ent in sorted(toks_to_ents[str(tok_num)],key=lambda x:x.get_length(),reverse=True):
						# Check if this is the head of that entity
						if absolute_head_id > ent.end or (absolute_head_id < ent.start and absolute_head_id > 0) or absolute_head_id == 0:
							if fields[7] == 'punct':
								print()
							# This is the head
							fields[5] = "ent_head=" + ent.type + "|" + "infstat=" + ent.infstat

							# store all head lines
							tsv_sent = tok_num_to_tsv_id[tok_num].split("-")[0]
							coref_line_and_ent.append((line_id, ent, tsv_sent))

							# # store all corefed heads
							# if ent.coref_type in ["coref", "ana", "cata"]:
							# 	tsv_sent = tok_num_to_tsv_id[tok_num].split("-")[0]
							# 	link_sent = ent.coref_link.split("-")[0]
							# 	if link_sent == tsv_sent:
							# 		coref_line_and_ent.append((line_id, ent, tsv_sent))
							# 		coref_line_and_ent_last_in_sent[tsv_sent] = counter
							# 		counter += 1


				line = "\t".join(fields)
			else:
				if sent_len > 0:
					sent_lens.append(sent_len)
					sent_len = 0
			processed_lines.append(line)

		# looping through all ent_head lines having coref to convert 'dep' into 'dislocated' (after all ent_heads have been detected)
		for line_ent_triple1 in coref_line_and_ent:
			ent1 = line_ent_triple1[1]
			if ent1.coref_type in ["coref", "ana", "cata"]:
				for line_ent_triple2 in coref_line_and_ent:
					if line_ent_triple1[2] == line_ent_triple2[2]:
						ent2 = line_ent_triple2[1]
						if (ent1.coref_link in ent2.line_tokens) or (ent2.coref_link in ent1.line_tokens):
							fields1 = processed_lines[line_ent_triple1[0]].split("\t")
							fields2 = processed_lines[line_ent_triple2[0]].split("\t")
							if fields1[6] == fields2[6]:
								if fields1[7] == "dep":
									fields1[7] = "dislocated"
									line = "\t".join(fields1)
									processed_lines[line_ent_triple1[0]] = line
								elif fields2[7] == "dep":
									fields2[7] = "dislocated"
									line = "\t".join(fields2)
									processed_lines[line_ent_triple2[0]] = line


		# Serialize entity tagged dependencies for debugging
		with io.open(pepper_temp + docname + ".conll10",'w',encoding="utf8", newline="\n") as f:
			f.write("\n".join(processed_lines) + "\n")

		converted = depedit.run_depedit(processed_lines,filename=docname,sent_id=True,docname=True)

		if PY2:
			punct_fixed = converted
		else:
			punct_fixed = fix_punct(converted)

		# Add UD morphology using CoreNLP script - we assume target/const/ already has .ptb tree files
		utils_abs_path = os.path.dirname(os.path.realpath(__file__))
		# morphed = punct_fixed #Logan
		morphed = ud_morph(punct_fixed, docname, utils_abs_path + os.sep + ".." + os.sep + "target" + os.sep + "const" + os.sep)

		if not PY2 and False:
			# CoreNLP returns bytes in ISO-8859-1
			# ISO-8859-1 mangles ellipsis glyph, so replace manually
			morphed = morphed.decode("ISO-8859-1").replace("","…").replace("","“").replace("","’").replace("",'—').replace("","–").replace("","”").replace("\r","")
			morphed = morphed.decode("ISO-8859-1").replace("\r","")

		# Add negative polarity and imperative mood
		negatived = []
		tok_num = 0
		sent_num = 0
		imp = False
		for line in morphed.split("\n"):
			if "s_type" in line:
				if "s_type=imp" in line:
					imp = True
				else:
					imp = False
			if "\t" in line:
				tok = doc_toks[tok_num]
				lemma = doc_lemmas[tok_num]
				tok_num += 1
				fields = line.split("\t")
				if tok_num in negative:
					fields[5] = add_feat(fields[5],"Polarity=Neg")
				fields[1] = tok  # Restore correct utf8 token and lemma
				fields[2] = lemma
				if imp and fields[5] == "VerbForm=Inf" and fields[7] == "root":  # Inf root in s_type=imp should be Imp
					fields[5] = "Mood=Imp|VerbForm=Fin"
				fields[8] = "_"
				negatived.append("\t".join(fields))
			else:
				if line.startswith("# text = "):  # Regenerate correct utf8 plain text
					sent_tok_count = sent_lens[sent_num]
					sent_text = ""
					for i in range(sent_tok_count):
						sent_text += doc_toks[i+tok_num]
						if "SpaceAfter=No" not in field_cache[i+tok_num][-1]:
							sent_text += " "
					line = "# text = " + sent_text.strip()  # Strip since UD validation does not tolerate trailing whitespace
					sent_num += 1
				negatived.append(line)
		negatived = "\n".join(negatived)

		# Projectivize punctuation
		#depedit = DepEdit(config_file="utils" + os.sep + "projectivize_punct.ini")
		#projectivized = depedit.run_depedit(negatived,filename=docname,sent_id=True,docname=True)
		#negatived = projectivized

		negatived = do_hard_replaces(negatived)

		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8", newline="\n") as f:
			f.write(negatived)

		if docname in ud_dev:
			dev_string += negatived
		elif docname in ud_test:
			test_string += negatived
		elif "reddit_" not in docname:  # Exclude reddit data from UD release
			train_string += negatived

	with io.open(train_split_target + "en_gum-ud-train.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(train_string)
	with io.open(train_split_target + "en_gum-ud-dev.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(dev_string)
	with io.open(train_split_target + "en_gum-ud-test.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(test_string)

	sys.__stdout__.write("o Converted " + str(len(depfiles)) + " documents to Universal Dependencies" + " " *20 + "\n")

