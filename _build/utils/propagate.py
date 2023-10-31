# -*- coding: utf-8 -*-

# GUM Build Bot
# propagate module
# v1.0.2

from glob import glob
from .nlp_helper import get_claws, adjudicate_claws, ud_morph
from .add_xml_annotations import add_xml
from .depedit import DepEdit
import os, re, sys, io
import ntpath
from collections import defaultdict, OrderedDict


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

utils_abs_path = os.path.dirname(os.path.realpath(__file__))
ud_morph_deped = DepEdit(utils_abs_path + os.sep + "ud_morph.ini")
ud_morph_deped.quiet = True
# depedit script to fix known projective punctuation issues
# note that this script also introduces some post editing to morphology, such as passive Voice
punct_depedit = DepEdit(config_file="utils" + os.sep + "projectivize_punct.ini")
punct_depedit.quiet = True
ud_edep_deped = DepEdit(utils_abs_path + os.sep + "eng_enhance.ini")
ud_edep_deped.quiet = True

efuncs = set(["acl","acl:relcl","advcl","advcl:relcl","advmod","amod","appos","aux","aux:pass","case","cc","cc:preconj","ccomp","compound","compound:prt","conj","cop","csubj","csubj:pass","csubj:xsubj","dep","det","det:predet","discourse","dislocated","expl","fixed","flat","goeswith","iobj","list","mark","nmod","nmod:npmod","nmod:poss","nmod:tmod","nsubj","nsubj:pass","nsubj:xsubj","nummod","obj","obl","obl:npmod","obl:tmod","orphan","parataxis","punct","ref","reparandum","root","vocative","xcomp"])

ud_dev = ["GUM_interview_cyclone", "GUM_interview_gaming",
		  "GUM_news_iodine", "GUM_news_homeopathic",
		  "GUM_voyage_athens", "GUM_voyage_coron",
		  "GUM_whow_joke", "GUM_whow_overalls",
		  "GUM_bio_byron", "GUM_bio_emperor",
		  "GUM_fiction_lunre", "GUM_fiction_beast",
		  "GUM_academic_exposure", "GUM_academic_librarians",
		  "GUM_reddit_macroeconomics", "GUM_reddit_pandas",  # Reddit
		  "GUM_speech_impeachment", "GUM_textbook_labor",
		  "GUM_vlog_radiology", "GUM_conversation_grounded",
		  "GUM_textbook_governments", "GUM_vlog_portland",
		  "GUM_conversation_risk", "GUM_speech_inauguration"]
ud_test = ["GUM_interview_libertarian", "GUM_interview_hill",
		   "GUM_news_nasa", "GUM_news_sensitive",
		   "GUM_voyage_oakland", "GUM_voyage_vavau",
		   "GUM_whow_mice", "GUM_whow_cactus",
		   "GUM_fiction_falling", "GUM_fiction_teeth",
		   "GUM_bio_jespersen", "GUM_bio_dvorak",
		   "GUM_academic_eegimaa", "GUM_academic_discrimination",
		   "GUM_reddit_escape", "GUM_reddit_monsters",  # Reddit
		   "GUM_speech_austria", "GUM_textbook_chemistry",
		   "GUM_vlog_studying", "GUM_conversation_retirement",
		   "GUM_textbook_union", "GUM_vlog_london",
		   "GUM_conversation_lambada", "GUM_speech_newzealand"]


class Args:

	def __init__(self):
		self.scenario = ['ud.FixPunct', 'write.Conllu']


class Entity:

	def __init__(self, ent_id, type, infstat, salience, centering, identity):
		self.id = ent_id
		self.type = type
		self.infstat = infstat
		self.salience = salience
		self.centering = centering
		self.tokens = []
		self.identity = identity
		self.line_tokens = []
		self.coref_type = ""
		self.coref_link = ""

	def __repr__(self):
		tok_nums = [int(x) for x in self.tokens]
		tok_range = str(min(tok_nums)) + "-" + str(max(tok_nums))
		return "ent " + self.id + ": " + self.type + "|" + self.infstat + "|" + self.centering + " (" + tok_range + ")"

	def assign_tok_nums(self):
		self.tok_nums = [int(x) for x in self.tokens]
		self.start = min(self.tok_nums)
		self.end = max(self.tok_nums)

	def get_length(self):
		return self.end - self.start + 1


def fix_punct(conllu_string):
	def preserve_ellipsis_tokens(conllu):
		lines = conllu.split("\n")
		ellipses = defaultdict(list)
		tok_num = 0
		for line in lines:
			if "\t" in line:
				fields=line.split("\t")
				if "-" in fields[0]:
					continue
				elif "." in fields[0]:
					ellipses[tok_num].append(line)
				else:
					tok_num += 1
		return ellipses

	def restore_ellipses(conllu, ellipses):
		lines = conllu.split("\n")
		tok_num = 0
		output = []
		for line in lines:
			if "\t" in line:
				fields = line.split("\t")
				if "-" in fields[0]:
					pass
				elif "." in fields[0]:
					continue
				else:
					if tok_num in ellipses:
						for el_line in ellipses[tok_num]:
							output.append(el_line)
					tok_num += 1
			output.append(line)
		return "\n".join(output).strip() + "\n\n"

	# Protect possessive apostrophe from being treated as punctuation
	conllu_string = conllu_string.strip() + "\n\n"
	ellipses = preserve_ellipsis_tokens(conllu_string)
	conllu_string = re.sub(r"\t'\t([^\t\n]+\tPART\tPOS)", r'\t&udapi_apos;\t\1', conllu_string, flags=re.MULTILINE)  # remove udapi sent_id
	doc = Document()
	doc.from_conllu_string(conllu_string)
	fixpunct_block = FixPunct()
	fixpunct_block.process_document(doc)
	output_string = doc.to_conllu_string()
	output_string = output_string.replace('&udapi_apos;',"'")
	output_string = re.sub(r'# sent_id = [0-9]+\n',r'',output_string)  # remove udapi sent_id
	output_string = restore_ellipses(output_string, ellipses)
	return output_string


def validate_upos(conllu, docname):
	for l, line in enumerate(conllu.split("\n")):
		if "so\tSCONJ\tRB" in line:
			sys.stderr.write("invalid xpos + upos comibation on line " + str(l) + " in doc " + docname + ": " + line + "\n")

def validate_enhanced(conllu, docname):
	def get_descendants(parent,children_dict,seen,snum,docname, rev=0):
		my_descendants = []
		my_descendants += children_dict[parent]
		for child in children_dict[parent]:
			if child in seen and rev < 2:
				rev_descendants = get_descendants(child, children_dict, set(), snum, docname, rev=rev+1)
				if parent in rev_descendants:  # two way cycle, not a DAG
					raise IOError("Cycle detected in "+docname+" in sentence " + str(snum+1) + " -- " + parent)
			elif rev > 1:  # prevent endless loop
				raise IOError("Cycle detected in " + docname + " in sentence " + str(snum + 1) + " -- " + parent)
			else:
				seen.add(child)
		for child in children_dict[parent]:
			if child in children_dict:
				try:
					my_descendants += get_descendants(child, children_dict, seen, snum, docname)
				except RecursionError:
					raise IOError("Cycle detected in " + docname + " in sentence " + str(snum + 1) + " -- " + parent)
		return my_descendants

	for i, line in enumerate(conllu.split("\n")):
		if "\t" in line:
			fields = line.split("\t")
			location = " on line " + str(i) + " in " + docname + "\n"
			if "." in fields[0]:
				if fields[-1].count("=") == 1:
					CopyOf, CopyID = fields[-1].split("=")
					if CopyOf != "CopyOf":
						sys.stderr.write("! edeps missing CopyOf" + location)
					try:
						int(CopyID)
					except ValueError:
						sys.stderr.write("! invalid edep CopyOf value" + location)
					if fields[6] != "_" or fields[7] != "_":
						sys.stderr.write("! ellipsis token with filled normal deps" + location)
				else:
					edeps = fields[8].split("|")
					if len(edeps) == 0:
						sys.stderr.write("! missing edeps line" + location)
					else:
						for edep in edeps:
							head, func = edep.split(":",maxsplit=1)
							try:
								float(head)
							except ValueError:
								sys.stderr.write("! invalid edep head on line " + str(i) + " in " + docname + "\n")
							if func not in efuncs and ":" not in func:
								sys.stderr.write("! invalid edep relation " + func + location)
							elif ":" in func:
								prefix = func.split(":")[0]
								if prefix not in ["acl","obl","nmod","conj","advcl"]:
									sys.stderr.write("! invalid edep relation" + func + location)
				if fields[8] == "_":
					sys.stderr.write("! missing ehead info for ellipsis token" + location)
	# Detect edep cycles
	for i, sent in enumerate(conllu.split("\n\n")):
		toks = [l.split("\t") for l in sent.strip().split("\n") if "\t" in l]
		children = defaultdict(set)
		for tok in toks:
			secdeps = tok[-2]
			for secdep in secdeps.split("|"):
				if ":" in secdep:
					parent, func = secdep.split(":",maxsplit=1)
					if parent == tok[6] and func == tok[7]:  # redundant secdep, ignore
						continue
					children[parent].add(tok[0])
		for tok in toks:
			seen = set()
			if tok[0] in children:
				get_descendants(tok[0], children, seen, snum=i, docname=docname)


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
		if not lemma.startswith("unique") and not lemma.startswith("under") and not lemma.startswith("univers"):
			return True
	for stem in negstems:
		if lemma.startswith(stem):
			return True
	return False


def is_abbr(word, xpos, lemma):
	lemma_eq_tok = word == lemma
	combos = [("wan","VBP"),("gon","VBG"),("na","TO"),("a","TO"),("em","PRP"),("m","NNS"),("h","NN"),
			  ("oz","NNS"),("fl","JJ")]
	abbr = r"(irl|TLDR|BC|BCE|CE|AD|PS|IIRC|BTW|IMO|TL;DR|GRF|US|NASA|NATO|div.|S.F.|U\.S\.|gov't|USI|DH|DAB|UK|IE6|COVID-19|KPA|UNESCO|FTU|LA|VR|MLB|USA|IATA|ROS|CC|IE|OK|ABC|BBC|DSW|NBC|U\.S|KCNA|ACPeds|US-412|WB|CBC|ICI|ISO|JSC|KKK|KSC|PHX|WHO|BART|CNRS|ELI5|FIFA|O\.J\.|NWSC|ROTC|BAFTA|STS-1|US-75|US-169|NEMISIS|STS-133|STS-134|STS-135|NSU|FEDERAL|ANDRILL|AS|AV|CO|CV|CW|DC|FN|GW|JK|KS|LV|MC|NB|NJ|NZ|PC|QC|RA|SC|ST|UC|VM|XP|XV|AFP|AIM|BAK|BBF|BPA|CBS|CEI|CIS|CRA|DBE|DNA|FRS|GIS|GPL|HBO|HIV|IDD|IE9|IFN|IMU|IQA|IRC|JFK|JPL|LIS|LSD|MIT|MSN|MTV|NBA|NFL|NHS|NPP|NSW|NTU|OIR|ROS|RVS|SNY|TUL|UKB|UNC|USD|USS|WTA|XML|ADPL|AIDS|AKMA|B\.A\.|ARES|D\.C\.|DPRK|FFFF|FGCU|HECS|HTML|IOTM|IRIS|K\.C\.|L\.A\.|MASS|MMPI|OSCE|S\.F\.|SETI|TAOM|THEO|U\.N\.|UAAR|WWII|XKCD|DHBs|U\.S\.|BY-SA|CITIC|LIBER|M\.Sc\.|NCLAN|ODIHR|UNMIK|OSU|CC-BY-SA-NC|CBC\.ca|DH+Lib|DH2017|e\.g\.|al\.|etc\.|Mr\.|St\.|i\.e\.|c\.|b\.|Ph\.D\.|Mrs\.|d\.|m\.|p\.|Dr\.|Jr\.|No\.|vs\.?|div\.|approx\.|a\.|Ed\.|Mt\.|Op\.|ca\.|cm\.|Ave\.|Cal\.|E\.g\.|Inc\.|Vol\.|a\.m\.|eds\.|p\.m\.|M\.Sc\.|Mlle\.|Prof\.|evals?|BBQs?|hrs?\.?|kg\.?|ft\.?|km\.?|[Mm]l\.?|g\.?)$"
	if re.match(abbr,word) is not None:
		return True
	# For the following make sure this isn't just the word "Sun" or the name "Jun"
	abbr_diff_lemma = r"(Sun|Mon|Tue|Wed|Thu|Fri|Sat|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[Tt]el|[Vv]ols?)\.?"
	if re.match(abbr_diff_lemma,word) and not lemma_eq_tok:
		return True
	if word in ["it"] and abbr_diff_lemma and xpos == "NP":  # it -> Italian
		return True
	if not lemma_eq_tok:
		if (word, xpos) in combos:
			return True
		elif (word.endswith("in") or word.endswith("in'")) and (xpos=="VBG" or lemma.endswith("ing")):
			return True
	# Substrings
	abbr_subst = r"gov't"
	if re.search(abbr_subst,word) is not None:
		if lemma_eq_tok:
			sys.stderr.write("WARN: abbreviation substring in " + word + " but lemma == word\n")
		return True
	return False


def add_feat(field,feat):
	if field == "_":
		return feat
	else:
		attrs = field.split("|")
		attrs.append(feat)
		return "|".join(sorted(list(set(attrs))))


def remove_entities(misc):
	output = []
	for anno in misc.split("|"):
		if anno.startswith("Entity=") or anno.startswith("Bridge=") or anno.startswith("SplitAnte=") or anno == "_":
			continue
		else:
			output.append(anno)
	if len(output) == 0:
		return "_"
	else:
		return "|".join(sorted(list(set(output))))


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
			tag = "-LRB-"  # "-LSB-"
		else:
			tag = "-LRB-"
	elif tag == ")":
		if token == "]":
			tag = "-RRB-"  # "-RSB-"
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


def enrich_dep(gum_source, gum_target, tmp, reddit=False):

	pre_annotated = defaultdict(lambda: defaultdict(dict))  # Placeholder for explicit annotations in src/dep/
	no_space_after_strings = {"(","[","{"}
	no_space_before_strings = {".",",",";","?","!","'s","n't","'ve","'re","'d","'m","'ll","]",")","}",":","%"}
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
		docname = ntpath.basename(depfile).replace(".conllu","")
		sys.stdout.write("\t+ " + " "*70 + "\r")
		sys.stdout.write(" " + str(docnum+1) + "/" + str(len(depfiles)) + ":\t+ " + docname + ".conllu\r")
		current_stype = ""
		current_transition = ""
		current_speaker = ""
		current_addressee = ""
		current_sic = None
		current_w = False
		output = ""
		stype_by_token = {}
		transition_by_token = {}
		speaker_by_token = {}
		foreign_by_token = {}
		addressee_by_token = {}
		space_after_by_token = defaultdict(lambda: True)
		sic_by_token = defaultdict(lambda : None)
		multi_sic = defaultdict(lambda : False)
		sic_len = 0

		# Dictionaries to hold token annotations from XML
		wordforms = {}
		pos = {}
		lemmas = {}

		tok_num = 0

		xmlfile = gum_target + "xml" + os.sep + docname + ".xml"
		xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		in_w_tag = False
		in_foreign = False
		for line in xml_lines:
			if line.startswith("<"):  # XML tag
				if line.startswith("<s type="):
					current_stype = re.match(r'<s type="([^"]+)"',line).group(1)
					current_transition = re.search(r'transition="([^"]+)"',line).group(1)
				elif line.startswith("<sp who="):
					current_speaker = re.search(r' who="([^"]+)"', line).group(1).replace("#","")
					if "whom=" in line:
						current_addressee = re.search(r' whom="([^"]+)"', line).group(1).replace("#", "")
				elif line.startswith("</sp>"):
					current_speaker = ""
					current_addressee = ""
				elif line.startswith("<w>"):
					in_w_tag = True
				elif line.startswith("</w>"):
					in_w_tag = False
					space_after_by_token[tok_num] = True  # Most recent token is normally followed by space
				elif line.startswith("<foreign"):
					in_foreign = True
					#foreign_by_token[tok_num] = True
				elif line.startswith("</foreign"):
					in_foreign = False
				elif line.startswith("<sic"):
					if 'ana="' in line:
						current_sic = re.search(r' ana="([^"]*)"', line).group(1)
					else:
						current_sic = None
				elif line.startswith("</sic>"):
					current_sic = None
					sic_len = 0
			elif len(line)>0:  # Token
				fields = line.split("\t")
				word = fields[0].replace("`","'").replace("‘","'").replace("’","'")
				word = word.replace('“','"').replace("”",'"')
				word_pos = fields[1].replace('"',"''")
				tok_num += 1
				if current_sic is not None:
					sic_len += 1
				if sic_len > 1:
					multi_sic[tok_num] = True
				if word in no_space_after_strings:
					space_after_by_token[tok_num] = False
				if word in no_space_before_strings:
					space_after_by_token[tok_num-1] = False
				if (word,word_pos) in no_space_after_combos:
					space_after_by_token[tok_num] = False
				if (word,word_pos) in no_space_before_combos:
					space_after_by_token[tok_num-1] = False
				if in_w_tag:
					space_after_by_token[tok_num] = False
				if in_foreign:
					foreign_by_token[tok_num] = True
				stype_by_token[tok_num] = current_stype
				transition_by_token[tok_num] = current_transition
				speaker_by_token[tok_num] = current_speaker
				addressee_by_token[tok_num] = current_addressee
				sic_by_token[tok_num] = current_sic
				wordforms[tok_num], pos[tok_num], lemmas[tok_num] = fields[:3]

		conll_lines = io.open(depfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0
		for line in conll_lines:
			if "# speaker" in line or "# s_type" in line or "# addressee" in line:
				# Ignore old speaker and s_type annotations in favor of fresh ones
				continue
			if "\t" in line:  # Token
				fields = line.split("\t")
				if "." in fields[0]:  # Ignore ellipsis token
					output += line + "\n"
					continue
				for index in [2,3,4,5,8,9]:
					if fields[index] != "_":
						pre_annotated[docname][tok_num][index] = fields[index]
				tok_num += 1
				wordform = wordforms[tok_num]
				lemma = lemmas[tok_num]
				# De-escape XML escapes
				wordform = wordform.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
				lemma = lemma.replace("&amp;","&").replace("&gt;",">").replace("&lt;","<")
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
				feats = fields[5].split("|") if fields[5] != "_" else []
				if not space_after_by_token[tok_num]:
					misc.append("SpaceAfter=No")
				if sic_by_token[tok_num] is not None:
					feats.append("Typo=Yes")
					if multi_sic[tok_num]:
						misc.append("CorrectForm=_")  # Trailing token of a sic whose correction is already added
					else:
						if not (sic_by_token[tok_num].endswith(" ") and sic_by_token[tok_num].strip() == fields[1]):
							misc.append("CorrectForm=" + sic_by_token[tok_num])
						if "SpaceAfter=No" in misc and sic_by_token[tok_num].endswith(" "):
							misc.append("CorrectSpaceAfter=Yes")
				if tok_num in foreign_by_token:
					feats.append("Foreign=Yes")
				fields[-1] = "|".join(misc) if len(misc) > 0 else "_"
				fields[5] = "|".join(sorted(feats)) if len(feats) > 0 else "_"
				line = "\t".join(fields)
			if line.startswith("1\t"):  # First token in sentence
				# Check for annotations
				if len(stype_by_token[tok_num]) > 0:
					output += "# s_type = " + stype_by_token[tok_num] + "\n"
				if len(transition_by_token[tok_num]) > 0:
					output += "# transition = " + transition_by_token[tok_num] + "\n"
				if len(speaker_by_token[tok_num]) > 0:
					output += "# speaker = " + speaker_by_token[tok_num] + "\n"
				if len(addressee_by_token[tok_num]) > 0:
					output += "# addressee = " + addressee_by_token[tok_num] + "\n"
			output += line + "\n"

		output = output.strip() + "\n" + "\n"

		# Attach all punctuation to the root (could also be a vocative root)
		depedit = DepEdit()
		depedit.add_transformation("func=/root/;func=/punct/\t#1.*#2\t#1>#2")
		depedit.add_transformation("func=/root/;func=/punct/\t#2.*#1\t#1>#2")
		output = depedit.run_depedit(output)
		output = output.strip() + "\n\n"  # Ensure exactly two new lines at end

		# output now contains conll string ready for udapi and morph
		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8",newline="\n") as f:
			f.write(output)

	return pre_annotated


def compile_ud(tmp, gum_target, pre_annotated, reddit=False):
	def get_meta(docname,gum_target):
		with io.open(gum_target + "xml" + os.sep + docname + ".xml", encoding="utf8") as f:
			meta_line = f.read().split("\n")[0]
		key_vals = re.findall(r' ([^=]+?)="([^"]+?)"',meta_line)
		meta = {}
		for key, val in key_vals:
			meta[key] = val.replace("&amp;","&").replace("&apos;","'").replace("&quot;",'"')
		return meta

	if PY2:
		print("WARN: Running on Python 2 - consider upgrading to Python 3. ")
		print("      Punctuation behavior in the UD data relies on udapi ")
		print("      which does not support Python 2. All punctuation will be attached to sentence roots.\n")

	train_string, dev_string, test_string = "", "", ""

	dep_source = tmp + "dep" + os.sep + "tmp" + os.sep
	dep_target = gum_target + "dep" + os.sep + "not-to-release" + os.sep
	if not os.path.isdir(dep_target):
		os.makedirs(dep_target)
	dep_merge_dir = tmp + "dep" + os.sep + "ud" + os.sep + "GUM" + os.sep
	if not os.path.isdir(dep_merge_dir):
		os.makedirs(dep_merge_dir)
	entidep_dir = tmp + "entidep" + os.sep	
	if not os.path.isdir(entidep_dir):
		os.makedirs(entidep_dir)
	

	depfiles = []
	files_ = glob(dep_source + "*.conllu")
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		depfiles.append(file_)

	for docnum, depfile in enumerate(depfiles):

		docname = os.path.basename(depfile).replace(".conllu","")

		sys.stdout.write("\t+ " + " "*70 + "\r")
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
				entity_string, infstat_string,salience_string, identity_string, centering_string, coref_type_string, coref_link_string = fields[3:10]
				if entity_string != "_":
					entities = entity_string.split("|")
					infstats = infstat_string.split("|")
					saliences = salience_string.split("|")
					identities = identity_string.split("|")
					centerings = centering_string.split("|")
					if coref_type_string != "_":
						coref_types = coref_type_string.split("|")
						coref_links = coref_link_string.split("|")
					for i, entity in enumerate(entities):
						infstat = infstats[i]
						salience = saliences[i]
						centering = centerings[i]
						# Make sure all entities are numbered
						if "[" not in entity:  # Single token entity with no ID
							entity += "["+str(int_max_entity)+"]"
							infstat += "[" + str(int_max_entity) + "]"
							int_max_entity += 1
						entity_id = entity[entity.find("[")+1:-1]
						entity = entity[:entity.find("[")]
						infstat = infstat[:infstat.find("[")]
						salience = infstat[:salience.find("[")]
						centering = centering[:centering.find("[")]
						match_ident = "_"
						for ident in identities:
							if "[" not in ident:
								ident += "["+entity_id+"]"
								ident_id = entity_id
							else:
								ident_id = ident[ident.find("[")+1:-1]
							if ident_id == entity_id:
								match_ident = ident[:ident.find("[")]
						if entity_id not in entity_dict:
							entity_dict[entity_id] = Entity(entity_id,entity,infstat,salience,centering,match_ident)
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
		morph_store = []  # Retrains original contents of FEATS field

		counter = 0
		for line in conll_lines:
			line_id += 1
			if "\t" in line:  # Token
				fields = line.split("\t")
				if "." in fields[0] or "-" in fields[0]:  # ellipsis tokens or supertokens
					pass
				else:
					sent_len += 1
					field_cache[tok_num] = fields
					tok_num += 1
					morph_store.append(fields[5])
					doc_toks.append(fields[1])
					doc_lemmas.append(fields[2])
					if fields[7] == "neg" or is_neg_lemma(fields[2],fields[3]):
						negative.append(tok_num)
					absolute_head_id = tok_num - int(fields[0]) + int(fields[6]) if fields[6] != "0" else 0
					if str(tok_num) in toks_to_ents:
						for ent in sorted(toks_to_ents[str(tok_num)],key=lambda x: x.get_length(), reverse=True):
							# Check if this is the head of that entity
							if absolute_head_id > ent.end or (absolute_head_id < ent.start and absolute_head_id > 0) or absolute_head_id == 0:
								# This is the head
								fields[5] = "ent_head=" + ent.type + "|" + "infstat=" + ent.infstat

								# store all head lines
								tsv_sent = tok_num_to_tsv_id[tok_num].split("-")[0]
								coref_line_and_ent.append((line_id, ent, tsv_sent))

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

		# Restore contents of FEATS field
		toknum = 0
		tmplines = []
		for line in processed_lines.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if not ("." in fields[0] or "-" in fields[0]):
					fields[5] = morph_store[toknum]
					line = "\t".join(fields)
					toknum += 1
			tmplines.append(line)
		processed_lines = "\n".join(tmplines).strip() + "\n"

		# UPOS
		depedit = DepEdit(config_file="utils" + os.sep + "upos.ini")
		uposed = depedit.run_depedit(processed_lines,filename=docname,sent_id=True,docname=True)
		uposed = re.sub(r'ent_head=[a-z]+\|infstat=[a-z:]+\|?','',uposed)
		if "infstat=" in uposed:
			sys.__stdout__.write("o WARN: invalid entity annotation from tsv for document " + docname)
		validate_upos(uposed, docname)
		processed_lines = uposed

		# Add metadata and global declaration
		lines = processed_lines.split("\n")
		header = ["# global.Entity = GRP-etype-infstat-centering-minspan-link-identity"]
		meta = get_meta(docname,gum_target)
		header.append("# meta::author = "+ meta["author"])
		header.append("# meta::dateCollected = " + meta["dateCollected"])
		header.append("# meta::dateCreated = " + meta["dateCreated"])
		header.append("# meta::dateModified = " + meta["dateModified"])
		#header.append("# meta::shortTitle = " + meta["shortTitle"])
		header.append("# meta::genre = "+ meta["type"])
		header.append("# meta::sourceURL = " + meta["sourceURL"])
		header.append("# meta::speakerCount = " + meta["speakerCount"])
		header.append("# meta::summary = "+ meta["summary"])
		header.append("# meta::title = "+ meta["title"])
		processed_lines = [lines[0]] + header + lines[1:]
		processed_lines = "\n".join(processed_lines)

		if PY2:
			punct_fixed = processed_lines
		else:
			punct_fixed = fix_punct(processed_lines)

		# morphed = punct_fixed

		use_corenlp = False
		if use_corenlp:
			# Add UD morphology using CoreNLP script - we assume target/const/ already has .ptb tree files
			morphed = ud_morph(punct_fixed, docname, utils_abs_path + os.sep + ".." + os.sep + "target" + os.sep + "const" + os.sep)

			if not PY2 and False:
				# CoreNLP returns bytes in ISO-8859-1
				# ISO-8859-1 mangles ellipsis glyph, so replace manually
				morphed = morphed.decode("ISO-8859-1").replace("","…").replace("","“").replace("","’").replace("",'—').replace("","–").replace("","”").replace("\r","")
			morphed = morphed.decode("ISO-8859-1").replace("\r","")
		else:
			morphed = ud_morph_deped.run_depedit(punct_fixed)

		# Add negative polarity and imperative mood
		negatived = []
		upos_list = []
		tok_num = 0
		sent_num = 0
		imp = False
		for line in morphed.split("\n"):
			if "s_type" in line:
				if "s_type" in line and "imp" in line:
					imp = True
				else:
					imp = False
			if "\t" in line:
				fields = line.split("\t")
				if use_corenlp:  # Handle annotations not covered by corenlp
					if tok_num in negative and "Polarity" not in fields[5]:
						fields[5] = add_feat(fields[5],"Polarity=Neg")
					if fields[4] == "CD" and fields[2].isnumeric() and "NumForm" not in fields[5]:
						fields[5] = add_feat(fields[5],"NumForm=Digit")
					elif fields[4] == "CD" and re.match(r'[XIVLMC]+\.?$',fields[2]) is not None and "NumForm" not in fields[5]:
						fields[5] = add_feat(fields[5],"NumForm=Roman")
					elif fields[4] == "CD" and "NumForm" not in fields[5]:
						fields[5] = add_feat(fields[5],"NumForm=Word")
					if is_abbr(fields[1],fields[4],fields[2]) and "Abbr" not in fields[5]:
						fields[5] = add_feat(fields[5],"Abbr=Yes")
					if imp and fields[5] == "VerbForm=Inf" and fields[7] == "root":  # Inf root in s_type=imp should be Imp
						fields[5] = "Mood=Imp|VerbForm=Fin"
				if "." in fields[0] or "-" in fields[0]:  # Ellipsis token or supertoken
					pass
				else:
					tok = doc_toks[tok_num]
					lemma = doc_lemmas[tok_num]
					tok_num += 1
					fields[8] = "_"
					fields[1] = tok  # Restore correct utf8 token and lemma
					fields[2] = lemma
					upos_list.append(fields[3])

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

		# Broken non-projective punctuation
		negatived = punct_depedit.run_depedit(negatived).strip()

		# Add enhanced dependencies
		negatived = ud_edep_deped.run_depedit(negatived).strip()

		# Remove invalid enhanced dependencies
		negatived = re.sub(r'(nmod|obl):(de|en|a)(?=[\|\t])',r'\1',negatived)
		negatived = re.sub(r'((nmod|obl):([a-z]+))_(?=[\|\t])',r'\1',negatived)  # no trailing underscores
		negatived = re.sub(r'(conj):(as_well)(?=[\|\t])',r'\1:as_well_as',negatived)

		# Collect nsubj:outer for xml
		final_funcs = []
		for line in negatived.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if "." in fields[0] or "-" in fields[0]:
					continue
				final_funcs.append(fields[7])

		# Add upos and nsubj:outer to target/xml/
		xml_lines = io.open(gum_target + "xml" + os.sep + docname + ".xml",encoding="utf8").read().split("\n")
		toknum = 0
		output = []
		for line in xml_lines:
			if "\t" in line:
				fields = line.split("\t")
				fields.append(fields[-1])
				fields[-2] = upos_list[toknum]
				line = "\t".join(fields)
				if ":outer" in final_funcs[toknum]:
					line = re.sub(r'\t[nc]subj','\t'+ final_funcs[toknum],line)
				toknum += 1
			output.append(line)
		with io.open(gum_target + "xml" + os.sep + docname + ".xml",'w',encoding="utf8",newline="\n") as f:
			f.write("\n".join(output))

		# Restore explicitly pre-annotated fields from src/dep/
		output = []
		tok_num = 0
		for line in negatived.split("\n"):
			if "\t" in line:
				fields = line.split("\t")
				if "." not in fields[0] and "-" not in fields[0]:
					if tok_num in pre_annotated[docname]:
						for index in pre_annotated[docname][tok_num]:
							fields[index] = pre_annotated[docname][tok_num][index]
					tok_num +=1
				line = "\t".join(fields)
			output.append(line)
		output = "\n".join(output).strip() + "\n\n"

		validate_enhanced(output, docname)

		# Directory with dependency output
		with io.open(dep_target + docname + ".conllu",'w',encoding="utf8", newline="\n") as f:
			f.write(output)
		# Directory for SaltNPepper merging, must be nested in a directory 'GUM'
		with io.open(dep_merge_dir + docname + ".conll10",'w',encoding="utf8", newline="\n") as f:
			f.write(output)

		if docname in ud_dev and "reddit_" not in docname:
			dev_string += output
		elif docname in ud_test and "reddit_" not in docname:
			test_string += output
		elif "reddit_" not in docname:  # Exclude reddit data from UD release
			train_string += output


	train_split_target = dep_target + ".." + os.sep
	with io.open(train_split_target + "en_gum-ud-train.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(train_string.strip())
	with io.open(train_split_target + "en_gum-ud-dev.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(dev_string.strip())
	with io.open(train_split_target + "en_gum-ud-test.conllu",'w',encoding="utf8", newline="\n") as f:
		f.write(test_string.strip())

	sys.__stdout__.write("o Enriched dependencies in " + str(len(depfiles)) + " documents" + " " *20)


def enrich_xml(gum_source, gum_target, centering_data, add_claws=False, reddit=False, warn=False):
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
		sys.stdout.write("\t+ " + " "*70 + "\r")
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
					fields = line.split("\t")
					if "."  in fields[0] or "-" in fields[0]:  # Supertoken or ellipsis token
						continue
					tok_num += 1
					funcs[tok_num] = fields[7]

		if PY2:
			xml_lines = open(xmlfile).read().replace("\r", "").split("\n")
		else:
			xml_lines = io.open(xmlfile,encoding="utf8").read().replace("\r","").split("\n")
		tok_num = 0

		if add_claws:
			tokens = list((line.split("\t")[0]) for line in xml_lines if "\t" in line)
			claws = get_claws("\n".join(tokens))

		sent_num = 0
		centering = centering_data[docname.replace(".xml","")]
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
					if len(fields) > 3:
						fields = fields[:-1] # Just delete last column to re-generate func from conllu
				fields.append(func)
				# Convert TO to IN for prepositional 'to'
				if fields[1] == "TO" and fields[-1] == "case":
					fields[1] = "IN"
				# Pure digits should receive the number as a lemma
				fields[2] = fix_card_lemma(fields[0],fields[2])
				line = "\t".join(fields)
			elif line.startswith("<s "):
				line = line.replace(">",f' transition="{centering[sent_num+1]}">')
				sent_num += 1
			output += line + "\n"

		output = output.strip() + "\n"

		if PY2:
			outfile = open(xml_target + docname, 'wb')
		else:
			outfile = io.open(xml_target + docname,'w',encoding="utf8",newline="\n")
		outfile.write(output)
		outfile.close()

	if add_claws:
		print("o Retrieved fresh CLAWS5 tags" + " " * 70 + "\r")
	print("o Enriched xml in " + str(len(xmlfiles)) + " documents" + " " *20)

"""
def const_parse(gum_source, gum_target, warn_slash_tokens=False, reddit=False):

	xml_source = gum_source + "xml" + os.sep
	const_target = gum_target + "const" + os.sep

	# because this parent function is called just once,
	# init the lal parser here instead of as a global const
	lalparser = LALConstituentParser(const_target)

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
				item = tag + '\t' + token + " "
				out_line += item

		sentences = output.split('\n')
		lalparser.run_parse(sentences,constfile)

	print("o Reparsed " + str(len(xmlfiles)) + " documents" + " " * 20)
"""

def get_coref_ids(gum_target, ontogum=False):
	def clean_closer(instr):
		output = []
		ents = re.findall(r'(\(?[^()]+\)?)',instr)
		for ent in ents:
			if "-" in ent and ent.endswith(")") and not ent.startswith("("):  # Closer
				ent = ent.split("-")[1].replace(")","") + ")"
			output.append(ent)
		return "".join(output)

	entity_dict = defaultdict(list)
	if ontogum:
		conll_coref = glob(gum_target + "coref" + os.sep + "ontogum" + os.sep + "conll" + os.sep + "*.conll")
	else:
		conll_coref = glob(gum_target + "coref" + os.sep + "conll" + os.sep + "GUM" + os.sep + "*.conll")
	for file_ in conll_coref:
		doc = os.path.basename(file_).replace(".conll","")
		lines = io.open(file_,encoding="utf8").read().split("\n")
		for line in lines:
			if "\t" in line:
				if not line.endswith("_"):
					fields = line.split("\t")
					fields[-1] = clean_closer(fields[-1])
					line = "\t".join(fields)
				entity_dict[doc].append(line.split("\t")[-1])

	return entity_dict


def get_rsd_spans(gum_target):

	rsd_spans = defaultdict(dict)
	rsd_files = glob(gum_target + "rst" + os.sep + "dependencies" + os.sep + "*.rsd")
	depths = defaultdict(dict)
	for file_ in rsd_files:
		doc = os.path.basename(file_).replace(".rsd","")
		lines = io.open(file_,encoding="utf8").read().split("\n")
		tok_num = 0
		parents = defaultdict(list)
		rels = {}
		for line in lines:
			if "\t" in line:
				fields = line.split("\t")
				#edu_id, toks, dist, depth, domain = fields[0:5]
				edu_id, toks, dist = fields[0:3]
				head, rsd_rel = fields[6:8]
				parents[edu_id] = head
				#rsd_rel = rsd_rel.replace("_m","").replace("_r","")
				rsd_rel = rsd_rel.replace("_r","")
				rels[edu_id] = rsd_rel
				rsd_spans[doc][tok_num] = (edu_id, rsd_rel, head, dist)#, depth, domain)
				tok_num += toks.strip().count(" ") + 1

		for unit in parents:
			parent = unit
			depth = 0
			while parent != "0":
				if not rels[parent].endswith("_m"):  # Multinucs do not affect depth
					depth += 1
				parent = parents[parent]
			depths[doc][unit] = depth

	return rsd_spans, depths


def add_rsd_to_conllu(gum_target, reddit=False, ontogum=False, relation_set=8):
	def convert_rel(rel, version=8):
		if version < 7:
			if "disjunction" in rel:
				return "contrast_m"
			elif "means" in rel or "manner" in rel:
				return "elaboration"
		elif version < 8:
			if "restatement" in rel and "_r" not in rel and "_m" not in rel:
				return "restatement"
			elif "restatement" in rel and "_r" not in rel:
				return "restatement_m"
			elif rel.startswith("elaboration"):
				return "elaboration"
			elif "phatic" in rel:
				return "phatic"
			elif "preparation" in rel or "organization" in rel:
				return "preparation"
			elif "list" in rel:
				return "joint_m"
		return rel

	if not gum_target.endswith(os.sep):
		gum_target += os.sep
	rsd_spans, depths = get_rsd_spans(gum_target)

	if not ontogum:
		files = glob(gum_target + "dep" + os.sep + "*.conllu")
		files += glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")
	else:
		files = glob(gum_target + "coref" + os.sep + "ontogum" + os.sep + "conllu" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	for file_ in files:
		with io.open(file_,encoding="utf8") as f:
			lines = f.read().split("\n")

		output = []
		toknum = 0
		snum = 0
		sents2depths = defaultdict(set)
		for line in lines:
			if line.startswith("# newdoc"):
				doc = line.strip().split()[-1]
				toknum = 0

			if "\t" in line:
				fields = line.split("\t")
				if not "-" in fields[0] and not "." in fields[0]:  # Regular token, not an ellipsis token or supertok
					if toknum in rsd_spans[doc]:
						rsd_data = rsd_spans[doc][toknum]
						relname = convert_rel(rsd_data[1],version=relation_set)
						if rsd_data[2] == "0":  # ROOT
							misc = add_feat(fields[-1],"Discourse=" + relname + ":" + rsd_data[0] + ":" + rsd_data[3]) #+ ":" + rsd_data[4] + ":" + rsd_data[5])
						else:
							misc = add_feat(fields[-1],"Discourse=" + relname + ":"+rsd_data[0]+"->"+rsd_data[2] + ":" + rsd_data[3]) #+ ":" + rsd_data[4] + ":" + rsd_data[5])
						fields[-1] = misc
						line = "\t".join(fields)
						depth = depths[doc][rsd_data[0]]
						sents2depths[snum].add(depth)
					toknum += 1
			elif len(line.strip()) == 0:
				snum += 1
			output.append(line)

		min_depths = {}
		for snum in sents2depths:
			min_depths[snum] = min(sents2depths[snum])
		unique_depths = sorted(list(set(min_depths.values())))
		depth_mapping = {}
		for i, val in enumerate(unique_depths):
			depth_mapping[val] = str(i+1)

		with_sent_prominence = []
		snum = 0
		for line in output:
			if "# s_type" in line:
				d = depth_mapping[min_depths[snum]]
				with_sent_prominence.append("# s_prominence = " + d)
			elif line == "":
				snum += 1
			with_sent_prominence.append(line)
		output = with_sent_prominence

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			f.write("\n".join(output).strip() + "\n\n")


def add_entities_to_conllu(gum_target,reddit=False,ontogum=False,conllua_data=None,salience_data=None):
	if not gum_target.endswith(os.sep):
		gum_target += os.sep
	if conllua_data is None:
		entity_doc = get_coref_ids(gum_target,ontogum=ontogum)
	else:
		entity_doc = conllua_data
	if salience_data is None:
		salience_data = {}

	files = glob(gum_target + "dep" + os.sep + "*.conllu")
	files += glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	entexp = "(person|place|organization|plant|abstract|object|event|time|animal|substance)-([0-9]+)"
	for file_ in files:
		with io.open(file_,encoding="utf8") as f:
			lines = f.read().split("\n")

		output = []
		toknum = 0
		for i, line in enumerate(lines):
			if line.startswith("# newdoc"):
				doc = line.strip().split()[-1]
				toknum = 0

			if "\t" in line:
				fields = line.split("\t")
				if not "-" in fields[0] and not "." in fields[0]:  # Regular token, not ellipsis or supertok
					try:
						entity_data = entity_doc[doc][toknum]
					except IndexError:
						raise IndexError("Token number " + str(toknum) + " not found in document " + doc)
					misc = remove_entities(fields[-1])
					if entity_data != "_":
						if "-" in entity_data:
							# place GRP first to match CorefUD standard
							entity_data = re.sub(entexp,r'\2-\1',entity_data)

						misc = add_feat(misc,"Entity="+entity_data)
					fields[-1] = misc
					if "-giv-" in misc or "-acc-" in misc:
						sys.stderr.write("! WARN: Entity with unsub-typed infstat at token " + str(toknum) + " in " + doc + "\n")
					line = "\t".join(fields)
					toknum += 1
			if "# global.Entity" in line and ontogum:
				line = '# global.Entity = GRP'  # OntoGUM only has coref IDs

			output.append(line)

		if ontogum:
			onto_conllu = gum_target+"coref" + os.sep + "ontogum" + os.sep + "conllu" + os.sep
			if not os.path.exists(onto_conllu):
				os.makedirs(onto_conllu)
			with io.open(onto_conllu + os.path.basename(file_), 'w', encoding="utf8", newline="\n") as f:
				f.write("\n".join(output).strip() + "\n\n")
		else:
			output = "\n".join(output)
			if salience_data is not None:
				salient_entities = sorted(list(set([int(k) for k in salience_data[doc] if salience_data[doc][k] == "sal"])))
				salient_entities = ", ".join([str(x) for x in salient_entities])
				salient_entities = "# meta::salientEntities = " + salient_entities
				output = output.replace("# meta::sourceURL",salient_entities + "\n" + "# meta::sourceURL")
			with io.open(file_,'w',encoding="utf8",newline="\n") as f:
				f.write(output.strip() + "\n\n")


def get_bridging(webannotsv):
	"""Get entities connected by bridging relations as sources, target, briding type, where
		entities are described by their token spans and edges are mapped from source to target and edge type
	"""

	tid = 0
	edges_by_source = defaultdict(OrderedDict)
	spans_by_id = defaultdict(list)
	for line in webannotsv.split("\n"):
		if "\t" in line:  # Token
			fields = line.split("\t")
			ents = fields[3].split("|")
			edge_types = fields[-3].split("|")
			edges = fields[-2].split("|")
			for i, ent in enumerate(ents):
				if ent != "_":
					if "[" in ent:
						eid = ent.split("[")[1][:-1]
					else:
						eid = fields[0]
					spans_by_id[eid].append(tid)
			for i, edge_type in enumerate(edge_types):
				if edge_type.startswith("bridge"):
					edge = edges[i]
					if "[" not in edge:
						edge += "[0_0]"
					src, target = edge.split("[")[1].split("_")
					target = target[:-1]
					if src == "0":
						src = edge.split("[")[0]
					if target == "0":
						target = fields[0]
					if "aggr" in edge_type:
						edges_by_source[src][(target,"split")] = None
					else:
						edges_by_source[src][(target,"bridge")] = None
			tid += 1

	out_spans = {}
	rev_out_spans = {}

	for eid in spans_by_id:
		start = min(spans_by_id[eid])
		end = max(spans_by_id[eid])
		out_spans[(start,end)] = eid
		rev_out_spans[eid] = (start, end)

	return edges_by_source, out_spans, rev_out_spans


def merge_bridge_conllu(conllu, webannotsv):
	def no_brace(instr):
		if "-" in instr:
			return instr.split("-")[0].replace("(","").replace(")","")
		else:
			return instr.replace("(","").replace(")","")

	edges_by_source, span_to_eid, eid_to_span = get_bridging(webannotsv)

	lines = conllu.split("\n")

	tid = 0
	# Pass 1 - get spans
	opened = defaultdict(list)
	conll_start_to_end = {}
	conll_span_to_ent = {}
	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0] or "." in fields[0]:  # ellipsis or supertok
				continue
			if "Entity=" in fields[-1]:
				ent_field = fields[-1].split("Entity=")[1].split("|")[0]
				ents = re.findall(r'(\(?[^()]+\)?)',ent_field)  # look for opening and closing entities
				for ent in ents:
					plain_ent = no_brace(ent)
					if ent.startswith("("):
						if ent.endswith(")"):  # Single line ent
							conll_start_to_end[(tid,plain_ent)] = tid
							conll_span_to_ent[(tid, tid)] = plain_ent
						else:
							opened[plain_ent].append(tid)
					else:
						start = opened[plain_ent].pop()
						conll_start_to_end[(start,plain_ent)] = tid
						conll_span_to_ent[(start,tid)] = plain_ent
			tid += 1

	# Pass 2 - insert bridge data
	tid = 0
	output = []
	bridging = []
	split_ante = []
	for line in lines:
		if "\t" in line:
			fields = line.split("\t")
			if "-" in fields[0] or "." in fields[0]:  # ellipsis token or supertok
				output.append(line)
				continue
			if "Entity=" in fields[-1]:
				ent_field = fields[-1].split("Entity=")[1].split("|")[0]
				ents = re.findall(r'(\([^()]+\)?)',ent_field)  # only look for opening entities
				for ent in ents:
					plain_ent = no_brace(ent)
					end = conll_start_to_end[(tid,plain_ent)]
					if (tid, end) in span_to_eid:
						eid = span_to_eid[(tid, end)]
						if eid in edges_by_source:
							for target, bridge_type in edges_by_source[eid]:
								target_start, target_end = eid_to_span[target]
								target_ent = conll_span_to_ent[(target_start, target_end)]
								edge = target_ent + "<" + plain_ent
								if bridge_type == "split":
									split_ante.append(edge)
								else:
									bridging.append(edge)
			out_misc = fields[-1].split("|") if fields[-1] != "_" else []
			out_misc = [a for a in out_misc if not a.startswith("Bridg") and not a.startswith("Split")]  # Kill existing values
			if len(bridging) > 0:
				out_misc.append("Bridge=" + ",".join(bridging))
			if len(split_ante) > 0:
				out_misc.append("SplitAnte=" + ",".join(split_ante))
			bridging = []
			split_ante = []
			fields[-1] = "|".join(sorted(out_misc)) if len(out_misc) > 0 else "_"
			line = "\t".join(fields)
			tid += 1
		output.append(line)

	return "\n".join(output).strip() + "\n\n"


def add_bridging_to_conllu(gum_target,reddit=False):
	if not gum_target.endswith(os.sep):
		gum_target += os.sep

	files = glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	all_merged = {}
	for file_ in files:
		tsv_file = gum_target + "coref" + os.sep + "tsv" + os.sep + os.path.basename(file_).replace("conllu","tsv")

		merged = merge_bridge_conllu(io.open(file_,encoding="utf8").read(),io.open(tsv_file,encoding="utf8").read())
		merged = merged.strip() + "\n\n"

		if any(["SplitAnte=" not in l and "acc:aggr" in l for l in merged.split("\n")]):
			sys.stderr.write("WARN: split information status acc:aggr but no split antecedent 'SplitAnte=' detected in " +
							 os.path.basename(file_)+"\n")

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			f.write(merged)

		all_merged[os.path.basename(file_).replace(".conllu","")] = merged

	bigfiles = glob(gum_target + "dep" + os.sep + "*.conllu")

	for file_ in bigfiles:
		docs = re.findall("# newdoc id ?= ?(GUM_[^\n]+)",io.open(file_).read())

		output = []
		for doc in docs:
			output.append(all_merged[doc])

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			f.write("".join(output).strip() + "\n\n")


def add_xml_to_conllu(gum_target, reddit=False, ontogum=False):
	xml_data = {}
	if not gum_target.endswith(os.sep):
		gum_target += os.sep

	for file_ in glob(gum_target + "xml" + os.sep + "*.xml"):
		xml_data[os.path.basename(file_).replace(".xml","")] = io.open(file_,encoding="utf8").read()

	if ontogum:
		files = glob(gum_target+"coref" + os.sep + "ontogum" + os.sep + "conllu" + os.sep + "GUM*.conllu")
		files += glob(gum_target+"coref" + os.sep + "ontogum" + os.sep + "conllu" + os.sep + "en_gum-ud*.conllu")
	else:
		files = glob(gum_target + "dep" + os.sep + "not-to-release" + os.sep + "*.conllu")
		files += glob(gum_target + "dep" + os.sep + "*.conllu")

	if not reddit:
		files = [f for f in files if not "reddit" in f]

	xml_tagged_conllu = {}
	for file_ in files:
		with io.open(file_,encoding="utf8") as f:
			if "en_gum-ud" not in file_:
				docname = os.path.basename(file_).replace(".conllu", "")
				conllu = f.read()
				with_tags = add_xml(conllu,xml_data[docname])
				xml_tagged_conllu[docname] = with_tags
			else:  # Big file
				docnames = []
				docs = f.read().split("# newdoc id")
				for doc in docs[1:]:
					doc = "# newdoc id" + doc
					docname = re.search(r'# newdoc id = (GUM_[^\s]+)',doc).group(1)
					docnames.append(docname)
				with_tags = "\n\n".join([xml_tagged_conllu[d] for d in docnames])

		with io.open(file_,'w',encoding="utf8",newline="\n") as f:
			# Separate newpar and newpar_blocks
			with_tags = with_tags.replace("# newpar ", "# newpar\n# newpar_block ")
			f.write(with_tags.strip() + "\n\n")

