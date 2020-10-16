# -*- coding: utf-8 -*-

# GUM Build Bot
# propagate module
# v1.0.2

from glob import glob
from .nlp_helper import get_claws, adjudicate_claws, parse, ud_morph
from .depedit import DepEdit
import os, re, sys, io
import ntpath
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
	output_string = re.sub(r'# sent_id = [0-9]+\n',r'',output_string)  # remove udapi sent_id
	return output_string


def is_neg_lemma(lemma,pos):
	negstems = set(["imposs","improb","immort","inevit","incomp","indirec","inadeq","insuff","ineff","incong","incoh","inacc","invol","infreq","inapp","indist","infin","intol",
					"dislik","dys","dismount","disadvant","disinteg","disresp","disagr","disjoin","disprov","disinterest","discomfort","dishonest","disband","disentangl"])
	neglemmas = set(["nowhere","never","nothing","none","undo","uncover","unclench","no","not","n't","ne","pas"])

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
		return "|".join(sorted(list(set(attrs))))


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


def clean_tag(tag):
	if tag == "0":
		raise IOError("unknown tag 0")
	elif tag == '"':
		return "''"
	elif tag == "'":
		return "''"
	else:
		return tag


def tt2vanilla(tag,token):
	tag = tag.replace("VV","VB").replace("VH","VB")
	tag = tag.replace("NP","NNP")
	tag = tag.replace("PP","PRP")
	tag = tag.replace("SENT",".")

	if tag=="IN/that":
		tag = "IN"
	elif tag=="(":
		if token == "[":
			tag = "-LSB-"
		else:
			tag = "-LRB-"
	elif tag == ")":
		if token == "]":
			tag = "-RSB-"
		else:
			tag = "-RRB-"
	return tag

def fix_card_lemma(wordform,lemma):
	if lemma == "@card@" and re.match(r'[0-9,]+$',wordform):
		lemma = wordform.replace(",","")
	elif lemma == "@card@" and re.match(r'([0-9]+)/([0-9]+)+$',wordform) and False:  # Fraction (DISABLED due to dates like 9/11)
		parts = wordform.split("/")
		div = float(parts[0])/float(parts[1])
		parts = str(div).split(".")
		if len(parts[1])>3:
			parts[1] = parts[1][:3]
			lemma = ".".join(parts)
		elif parts[1] == "0":
			lemma = parts[0]
		else:
			lemma = ".".join(parts)
	elif lemma == "@card@" and re.match(r'([0-9]+)\.([0-9]+)+$',wordform) and False:  # Decimal, round 3 places
		parts = wordform.split(".")
		if len(parts[1])>3:
			parts[1] = parts[1][:3]
		lemma = ".".join(parts)
	elif lemma == "@card@":
		lemma = wordform.replace(",","")
	return lemma


def enrich_dep(gum_source, tmp, reddit=False):

	no_space_after_strings = {"(","[","{"}
	no_space_before_strings = {".",",",";","?","!","'s","n't","'ve","'d","'m","'ll","]",")","}",":","%"}
	no_space_after_combos = {("'","``"),('"',"``")}
	no_space_before_combos = {("ll","MD"),("d","MD"),("m","VBP"),("ve","VHP"),("s","POS"),("s","VBZ"),("s","VHZ"),("'","POS"),("nt","RB"),("'","''"),('"',"''")}
	dep_source = gum_source + "dep" + os.sep
	dep_target = tmp + "dep" + os.sep + "tmp" + os.sep
	if not os.path.isdir(dep_target):
		os.makedirs(dep_target)

	depfiles = []
	files_ = glob(dep_source + "*.conllu")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		depfiles.append(file_)

	for docnum, depfile in enumerate(depfiles):
		docname = ntpath.basename(depfile)
		sys.stdout.write("\t+ " + " "*50 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(depfiles)) + ":\t+ " + docname + "\r")
		current_stype = ""
		current_speaker = ""
		current_sic = False
		current_w = False
		output = ""
		stype_by_token = {}
		speaker_by_token = {}
		space_after_by_token = defaultdict(lambda: True)
		sic_by_token = defaultdict(lambda: False)

		# Dictionaries to hold token annotations from XML
		wordforms = {}
		pos = {}
		lemmas = {}

		tok_num = 0

		xmlfile = depfile.replace("dep" + os.sep,"xml" + os.sep).replace("conllu","xml")
		xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		for line in xml_lines:
			if line.startswith("<"):  # XML tag
				if line.startswith("<s type="):
					current_stype = re.match(r'<s type="([^"]+)"',line).group(1)
				elif line.startswith("<sp who="):
					current_speaker = re.search(r' who="([^"]+)"', line).group(1).replace("#","")
				elif line.startswith("</sp>"):
					current_speaker = ""
				elif line.startswith("<w>"):
					space_after_by_token[tok_num+1] = True
				elif line.startswith("</w>"):
					space_after_by_token[tok_num] = False
				elif line.startswith("<sic>"):
					current_sic = True
				elif line.startswith("</sic>"):
					current_sic = False
			elif len(line)>0:  # Token
				fields = line.split("\t")
				word = fields[0].replace("`","'").replace("‘","'").replace("’","'")
				word = word.replace('“','"').replace("”",'"')
				word_pos = fields[1].replace('"',"''")
				tok_num += 1
				if word == "(":
					a=5
				if word in no_space_after_strings:
					space_after_by_token[tok_num] = False
				if word in no_space_before_strings:
					space_after_by_token[tok_num-1] = False
				if (word,word_pos) in no_space_after_combos:
					space_after_by_token[tok_num] = False
				if (word,word_pos) in no_space_before_combos:
					space_after_by_token[tok_num-1] = False
				stype_by_token[tok_num] = current_stype
				speaker_by_token[tok_num] = current_speaker
				sic_by_token[tok_num] = current_sic
				wordforms[tok_num], pos[tok_num], lemmas[tok_num] = fields[:3]

		conll_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0
		for line in conll_lines:
			if "# speaker" in line or "# s_type" in line:
				# Ignore old speaker and s_type annotations in favor of fresh ones
				continue
			if "\t" in line:  # Token
				tok_num += 1
				wordform = wordforms[tok_num]
				lemma = lemmas[tok_num]
				# De-escape XML escapes
				wordform = wordform.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
				lemma = lemma.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
				fields = line.split("\t")
				tt_pos = pos[tok_num]
				tt_pos = clean_tag(tt_pos)
				vanilla_pos = tt2vanilla(tt_pos, fields[1])
				# Convert TO to IN for prepositional 'to'
				if tt_pos == "TO" and fields[7] == "case":
					tt_pos = "IN"
				# Pure digits should receive the number as a lemma
				lemma = fix_card_lemma(wordform,lemma)

				fields[1] = wordform
				fields[2] = lemma
				fields[3] = tt_pos
				fields[4] = vanilla_pos
				misc = []
				feats = fields[5].split() if fields[5] != "_" else []
				if not space_after_by_token[tok_num]:
					misc.append("SpaceAfter=No")
				if sic_by_token[tok_num]:
					feats.append("Typo=Yes")
				fields[-1] = "|".join(misc) if len(misc) > 0 else "_"
				fields[5] = "|".join(sorted(feats)) if len(feats) > 0 else "_"
				line = "\t".join(fields)
			if line.startswith("1\t"):  # First token in sentence
				# Check for annotations
				if len(stype_by_token[tok_num]) > 0:
					output += "# s_type=" + stype_by_token[tok_num] + "\n"
				if len(speaker_by_token[tok_num]) > 0:
					output += "# speaker=" + speaker_by_token[tok_num] + "\n"
			output += line + "\n"

		output = output.strip() + "\n" + "\n"

		# Attach all punctuation to the root (could also be a vocative root)
		depedit = DepEdit()
		depedit.add_transformation("func=/root/;func=/punct/\t#1.*#2\t#1>#2")
		depedit.add_transformation("func=/root/;func=/punct/\t#2.*#1\t#1>#2")
		output = depedit.run_depedit(output)

		# output now contains conll string ready for udapi and morph
		with io.open(dep_target + docname,'w',encoding="utf8",newline="\n") as f:
			f.write(output)


def compile_ud(tmp, gum_target, reddit=False):

	if PY2:
		print("WARN: Running on Python 2 - consider upgrading to Python 3. ")
		print("      Punctuation behavior in the UD data relies on udapi ")
		print("      which does not support Python 2. All punctuation will be attached to sentence roots.\n")


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

	dep_source = tmp + "dep" + os.sep + "tmp" + os.sep
	dep_target = gum_target + "dep" + os.sep + "not-to-release" + os.sep
	if not os.path.isdir(dep_target):
		os.makedirs(dep_target)
	dep_merge_dir = tmp + "dep" + os.sep + "ud" + os.sep + "GUM" + os.sep
	if not os.path.isdir(dep_merge_dir):
		os.makedirs(dep_merge_dir)

	depfiles = []
	files_ = glob(dep_source + "*.conllu")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		depfiles.append(file_)

	for docnum, depfile in enumerate(depfiles):

		docname = os.path.basename(depfile).replace(".conllu","")

		sys.stdout.write("\t+ " + " "*50 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(depfiles)) + ":\t+ " + docname + "\r")

		entity_file = tmp + "tsv" + os.sep + "GUM" + os.sep + docname + ".tsv"
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

		# In stanford to UD conversion, we looped through all ent_head lines having coref to convert
		# 'dep' into 'dislocated' (after all ent_heads have been detected) - this is now redundant since switch to UD
		# This code block is only retained in order to create the sometimes useful tmp/entidep/ data
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
									#fields1[7] = "dislocated"  # no need to set dislocated in manual UD parse
									line = "\t".join(fields1)
									processed_lines[line_ent_triple1[0]] = line
								elif fields2[7] == "dep":
									#fields2[7] = "dislocated"
									line = "\t".join(fields2)  # no need to set dislocated in manual UD parse
									processed_lines[line_ent_triple2[0]] = line

		processed_lines = "\n".join(processed_lines) + "\n"
		# Serialize entity tagged dependencies for debugging
		with io.open(tmp + "entidep" + os.sep + docname + ".conllu",'w',encoding="utf8", newline="\n") as f:
			f.write(processed_lines)

		# UPOS
		depedit = DepEdit(config_file="utils" + os.sep + "upos.ini")
		uposed = depedit.run_depedit(processed_lines,filename=docname,sent_id=True,docname=True)
		# Make sure sent_id is first comment except newdox
		uposed = re.sub(r'((?:# [^n][^\t\n]+\n)+)(# sent_id[^\n]+\n)',r'\2\1',uposed)
		uposed = re.sub(r'ent_head=[a-z]+\|infstat=[a-z]+\|?','',uposed)
		if "infstat=" in uposed:
			sys.__stdout__.write("o WARN: invalid entity annotation from tsv for document " + docname)
		processed_lines = uposed

		#depedit = DepEdit(config_file="utils" + os.sep + "fix_flat.ini")
		#processed_lines = depedit.run_depedit(processed_lines,filename=docname)


		if PY2:
			punct_fixed = processed_lines
		else:
			punct_fixed = fix_punct(processed_lines)

		# Add UD morphology using CoreNLP script - we assume target/const/ already has .ptb tree files
		utils_abs_path = os.path.dirname(os.path.realpath(__file__))
		# morphed = punct_fixed
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
				if tok_num in negative and "Polarity" not in fields[5]:
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

		negatived = do_hard_replaces(negatived)

		# Directory with dependency output
		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8", newline="\n") as f:
			f.write(negatived)
		# Directory for SaltNPepper merging, must be nested in a directory 'GUM'
		with io.open(dep_merge_dir + docname + ".conll10",'w',encoding="utf8", newline="\n") as f:
			f.write(negatived)

		if docname in ud_dev:
			dev_string += negatived
		elif docname in ud_test:
			test_string += negatived
		elif "reddit_" not in docname:  # Exclude reddit data from UD release
			train_string += negatived


	train_split_target = dep_target + ".." + os.sep
	with io.open(train_split_target + "en_gum-ud-train.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(train_string.strip() + "\n")
	with io.open(train_split_target + "en_gum-ud-dev.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(dev_string.strip() + "\n")
	with io.open(train_split_target + "en_gum-ud-test.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(test_string.strip() + "\n")

	sys.__stdout__.write("o Enriched dependencies in " + str(len(depfiles)) + " documents" + " " *20)


def enrich_xml(gum_source, gum_target, add_claws=False, reddit=False, warn=False):
	xml_source = gum_source + "xml" + os.sep
	xml_target = gum_target + "xml" + os.sep

	xmlfiles = []
	files_ = glob(xml_source + "*.xml")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		xmlfiles.append(file_)

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " "*40 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(xmlfiles)) + ":\t+ " + docname + "\r")

		# Dictionaries to hold token annotations from conllu data
		funcs = {}

		tok_num = 0

		depfile = xmlfile.replace("xml" + os.sep,"dep" + os.sep).replace("xml","conllu")
		if PY2:
			dep_lines = open(depfile).read().replace("\r", "").split("\n")
		else:
			try:
				dep_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
			except FileNotFoundError:
				sys.stderr.write("! File not found: " + depfile)
				if warn:
					continue
				else:
					exit()
		line_num = 0
		for line in dep_lines:
			line_num += 1
			if "\t" in line:  # token line
				if line.count("\t") != 9:
					print("WARN: Found line with less than 9 tabs in " + docname + " line: " + str(line_num))
				else:
					tok_num += 1
					fields = line.split("\t")
					funcs[tok_num] = fields[7]

		if PY2:
			xml_lines = open(xmlfile).read().replace("\r", "").split("\n")
		else:
			xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0

		if add_claws:
			tokens = list((line.split("\t")[0]) for line in xml_lines if "\t" in line)
			claws = get_claws("\n".join(tokens))

		for line in xml_lines:
			if "\t" in line:  # Token
				tok_num += 1
				func = funcs[tok_num]
				fields = line.split("\t")
				if add_claws:
					fields = fields[:3]  # Only retain first three columns; the rest can be dynamically generated
					claws_tag = claws[tok_num-1]
					claws_tag = adjudicate_claws(claws_tag,fields[1],fields[0],func)
					fields.append(claws_tag)
				else:
					fields = fields[:-1] # Just delete last column to re-generate func from conllu
				fields.append(func)
				# Convert TO to IN for prepositional 'to'
				if fields[1] == "TO" and fields[-1] == "case":
					fields[1] = "IN"
				# Pure digits should receive the number as a lemma
				fields[2] = fix_card_lemma(fields[0],fields[2])
				line = "\t".join(fields)
			output += line + "\n"

		output = output.strip() + "\n"

		if PY2:
			outfile = open(xml_target + docname, 'wb')
		else:
			outfile = io.open(xml_target + docname,'w',encoding="utf8")
		outfile.write(output)
		outfile.close()

	if add_claws:
		print("o Retrieved fresh CLAWS5 tags" + " " * 20 + "\r")
	print("o Enriched xml in " + str(len(xmlfiles)) + " documents" + " " *20)


def const_parse(gum_source, gum_target, warn_slash_tokens=False, reddit=False):
	xml_source = gum_source + "xml" + os.sep
	const_target = gum_target + "const" + os.sep

	files_ = glob(xml_source + "*.xml")
	xmlfiles = []
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		xmlfiles.append(file_)

	for docnum, xmlfile in enumerate(xmlfiles):
		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " "*40 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(xmlfiles)) + ":\t+ Parsing " + docname + "\r")

		# Name for parser output file
		constfile = const_target + docname.replace("xml", "ptb")


		xml_lines = io.open(xmlfile, encoding="utf8").read().replace("\r", "").split("\n")
		line_num = 0
		out_line = ""

		for line in xml_lines:
			if line.startswith("</s>"): # Sentence ended
				output += out_line.strip() + "\n"
				out_line = ""

			elif "\t" in line:  # Token
				line_num += 1
				fields = line.split("\t")
				token, tag = fields[0], fields[1]
				tag = tt2vanilla(tag,token)
				if " " in token:
					print("WARN: space found in token on line " + str(line_num) + ": " + token + "; replaced by '_'")
					token = token.replace(" ","_")
				elif "/" in token and warn_slash_tokens:
					print("WARN: slash found in token on line " + str(line_num) + ": " + token + "; retained as '/'")

				token = token.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<").replace("&apos;","'").replace("&quot;",'"').replace("(","-LRB-").replace(")","-RRB-")
				item = token + "/" + tag + " "
				out_line += item

		parsed = parse(output)

		parsed = parsed.strip() + "\n" + "\n"

		outfile = io.open(constfile, 'w', encoding="utf8")
		outfile.write(parsed)
		outfile.close()

	print("o Reparsed " + str(len(xmlfiles)) + " documents" + " " * 20)


def get_coref_ids(gum_target):

	entity_dict = defaultdict(list)
	conll_coref = glob(gum_target + "coref" + os.sep + "conll" + os.sep + "GUM" + os.sep + "*.conll")
	for file_ in conll_coref:
		doc = os.path.basename(file_).replace(".conll","")
		lines = io.open(file_,encoding="utf8").read().split("\n")
		for line in lines:
			if "\t" in line:
				entity_dict[doc].append(line.split("\t")[-1])

	return entity_dict


def get_rsd_spans(gum_target):

	rsd_spans = defaultdict(dict)
	rsd_files = glob(gum_target + "rst" + os.sep + "dependencies" + os.sep + "*.rsd")
	for file_ in rsd_files:
		doc = os.path.basename(file_).replace(".rsd","")
		lines = io.open(file_,encoding="utf8").read().split("\n")
		tok_num = 0
		for line in lines:
			if "\t" in line:
				fields = line.split("\t")
				edu_id, toks = fields[0:2]
				head, rsd_rel = fields[6:8]
				rsd_rel = rsd_rel.replace("_m","").replace("_r","")
				rsd_spans[doc][tok_num] = (edu_id, rsd_rel, head)
				tok_num += toks.strip().count(" ") + 1

	return rsd_spans


def add_rsd_to_conllu(gum_target,reddit=False):
	if not gum_target.endswith(os.sep):
		gum_target += os.sep
	rsd_spans = get_rsd_spans(gum_target)

	files = glob(gum_target + "dep" + os.sep + "*.conllu")
	files += glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	for file_ in files:
		with io.open(file_,encoding="utf8") as f:
			lines = f.read().split("\n")

		output = []
		toknum = 0
		for line in lines:
			if line.startswith("# newdoc"):
				doc = line.strip().split()[-1]
				toknum = 0

			if "\t" in line:
				fields = line.split("\t")
				if not "-" in fields[0]:  # Regular token
					if toknum in rsd_spans[doc]:
						rsd_data = rsd_spans[doc][toknum]
						if rsd_data[2] == "0":  # ROOT
							misc = add_feat(fields[-1],"Discourse=" + rsd_data[1] + ":" + rsd_data[0])
						else:
							misc = add_feat(fields[-1],"Discourse="+rsd_data[1]+":"+rsd_data[0]+"->"+rsd_data[2])
						fields[-1] = misc
						line = "\t".join(fields)
					toknum += 1
			output.append(line)

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			f.write("\n".join(output) + "\n")


def add_entities_to_conllu(gum_target,reddit=False):
	if not gum_target.endswith(os.sep):
		gum_target += os.sep
	entity_doc = get_coref_ids(gum_target)

	files = glob(gum_target + "dep" + os.sep + "*.conllu")
	files += glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	for file_ in files:
		with io.open(file_,encoding="utf8") as f:
			lines = f.read().split("\n")

		output = []
		toknum = 0
		for line in lines:
			if line.startswith("# newdoc"):
				doc = line.strip().split()[-1]
				toknum = 0

			if "\t" in line:
				fields = line.split("\t")
				if not "-" in fields[0]:  # Regular token
					entity_data = entity_doc[doc][toknum]
					if entity_data != "_":
						misc = add_feat(fields[-1],"Entity="+entity_data)
						fields[-1] = misc
						line = "\t".join(fields)
					toknum += 1
			output.append(line)

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			f.write("\n".join(output) + "\n")

