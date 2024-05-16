#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, shutil, sys, io, re
from glob import glob
from argparse import ArgumentParser
from utils.pepper_runner import run_pepper
from utils.make_rst_rel_data import main as make_disrpt
import datetime
import ntpath, platform

if sys.platform == "win32":  # Print \n new lines in Windows
	import os, msvcrt
	msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

PY2 = sys.version_info[0] < 3


def setup_directories(gum_source, gum_target):
	if not os.path.exists(gum_source):
		raise IOError("Source file directory " + gum_source + " not found.")
	if not os.path.exists(gum_target):
		os.makedirs(gum_target)
	for ext in ["dep","const","xml","rst","coref"]:
		if not os.path.exists(gum_target + ext):
			os.makedirs(gum_target + ext)
	pepper_path = os.path.abspath("utils" + os.sep + "pepper") + os.sep
	# Make sure pepper staging area directory exists in utils/pepper/
	if not os.path.exists(pepper_path + "tmp" + os.sep):
		os.makedirs(pepper_path + "tmp" + os.sep)


parser = ArgumentParser()
parser.add_argument("-t",dest="target",action="store",help="GUM build target directory", default=None)
parser.add_argument("-s",dest="source",action="store",help="GUM build source directory", default=None)
parser.add_argument("-p",dest="parse",action="store_true",help="Whether to reparse constituents")
parser.add_argument("-c",dest="claws",action="store_true",help="Whether to reassign claws5 tags")
parser.add_argument("-v",dest="verbose_pepper",action="store_true",help="Whether to print verbose pepper output")
parser.add_argument("-n",dest="no_pepper",action="store_true",help="No pepper conversion, just validation and file fixing")
parser.add_argument("-i",dest="increment_version",action="store",help="A new version number to assign",default="DEVELOP")
parser.add_argument("--pepper_only",action="store_true", help="Just rerun pepper on generated targets")
parser.add_argument("--skip_ptb_labels",action="store_true", help="Skip projecting function labels to PTB trees")
parser.add_argument("--skip_ontogum",action="store_true", help="Skip building OntoGUM version of coref data")
parser.add_argument("--no_secedges",action="store_true", help="No RST++ secedges in conllu")
parser.add_argument("--no_signals",action="store_true", help="No RST++ signals in conllu")
parser.add_argument("--corpus_name",action="store", default="GUM", help="Corpus name / document prefix")

options = parser.parse_args()

corpus_name = options.corpus_name
build_dir = os.path.dirname(os.path.realpath(__file__))
pepper_home = build_dir + os.sep + "utils" + os.sep + "pepper" + os.sep
pepper_tmp = pepper_home + "tmp" + os.sep

if options.source is None:
	gum_source = build_dir + os.sep + "src"
else:
	gum_source = os.path.abspath(options.source)
if options.target is None:
	gum_target = build_dir + os.sep + "target"
else:
	gum_target = os.path.abspath(options.target)

if gum_source[-1] != os.sep:
	gum_source += os.sep
if gum_target[-1] != os.sep:
	gum_target += os.sep
setup_directories(gum_source,gum_target)

######################################
## Step 1:
######################################
# validate input for further steps
from utils.validate import validate_src, check_reddit

print("="*20)
print("Validating files...")
print("="*20 + "\n")

reddit = check_reddit(gum_source)
if not reddit:
	print("Could not find restored tokens in reddit documents.")
	print("Abort conversion or continue without reddit? (You can restore reddit tokens using process_reddit.py)")
	try:
		# for python 2
		response = raw_input("[A]bort/[C]ontinue> ")
	except NameError:
		# for python 3
		response = input("[A]bort/[C]ontinue> ")
	if response.upper() != "C":
		print("Aborting build.")
		sys.exit()
else:
	print("Found reddit source data")
	print("Including reddit data in build")

validate_src(gum_source, reddit=reddit)

######################################
## Step 2: propagate annotations
######################################
from utils.propagate import enrich_dep, enrich_xml, compile_ud, tt2vanilla, fix_gw_tags
from utils.repair_tsv import fix_tsv, make_ontogum
from utils.repair_rst import fix_rst, update_non_dm_signals


# Moved from propagate.py to facilitate lazy loading of the Cython dependencies
def const_parse(gum_source, warn_slash_tokens=False, reddit=False, only_parse_diff=True):
	def check_diff(xml, ptb, docname):
		# Check whether tokens, sentence splits and tags have changed in XML compared to PTB format
		ptb_trees = ptb.strip().split("(ROOT")[1:]
		xml_sents = xml.split("</s>")[:-1]
		if len(xml_sents) != len(ptb_trees):
			sys.stderr.write("! "+docname+": different sentence counts in const/ and xml/ - flagging for reparse\n")
			return True
		else:
			for i, sent in enumerate(ptb_trees):
				ptb_tags = re.findall(r"\(([^()\s]+)\s+[^()\s]+\)",sent)
				xml_tags = [l.split("\t")[:2] for l in xml_sents[i].split("\n") if "\t" in l]
				vanilla_tags = [tt2vanilla(tag,token) for token, tag in xml_tags]
				if len(vanilla_tags) != len(ptb_tags):
					sys.stderr.write("! "+docname+": different token counts in const/ and xml/: '"+ " ".join([t[0] for t in xml_tags]) +"' - flagging for reparse\n")
					return True
				elif any([x != ptb_tags[i] for i, x in enumerate(vanilla_tags)]):
					sys.stderr.write("! "+docname+": different tags in const/ and xml/: '"+ " ".join([t[0] for t in xml_tags]) +"' - flagging for reparse\n")
					return True
		return False

	# added here for lazy loading
	# only load the cython dependencies if need to regenerate the parse trees
	# this avoids need to compile the cython packages unless they are needed (-p option)
	from constituent_parser_lal import LALConstituentParser

	xml_source = gum_source + "xml" + os.sep
	const_target = gum_source + "const" + os.sep

	# because this parent function is called just once,
	# init the lal parser here instead of as a global const
	lalparser = LALConstituentParser(const_target)

	files_ = glob(xml_source + "*.xml")
	xmlfiles = []
	for file_ in files_:
		if not reddit and "reddit_" in file_:
			continue
		try:
			changed = check_diff(io.open(file_,encoding="utf8").read(), io.open(file_.replace(".xml",".ptb").replace("xml","const"),encoding="utf8").read(), os.path.basename(file_))
		except FileNotFoundError:  # No parse exists, create it
			sys.stderr.write("! " + os.path.basename(file_) + ": no parse found - flagging for parse\n")
			changed = True
		if not changed:
			continue
		xmlfiles.append(file_)

	#gold_const = ["GUM_academic_discrimination","GUM_bio_emperor","GUM_fiction_oversite","GUM_interview_peres",
	#			  "GUM_news_nasa","GUM_reddit_polygraph","GUM_voyage_athens","GUM_whow_arrogant"]
	# Gold constituents must now be overwritten since they no longer correspond to HYPH tokenization.
	# Refer to GUM version <=7.2 for gold constituent trees of these 8 documents with old tokenization
	gold_const = []
	xmlfiles = [f for f in xmlfiles if os.path.basename(f).replace(".xml","") not in gold_const]

	for docnum, xmlfile in enumerate(xmlfiles):

		if "_all" in xmlfile:
			continue
		docname = ntpath.basename(xmlfile)
		output = ""
		sys.stdout.write("\t+ " + " "*70 + "\r")
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

# Check and potentially correct POS tags and lemmas based on pooled annotations
#proof(gum_source)

conn_data = {}
if not options.pepper_only:
	# Token and sentence border adjustments
	print("\nAdjusting token and sentence borders:\n" + "="*37)
	# Adjust tsv/ files:
	#   * refresh and re-merge token strings in case they were mangled by WebAnno
	#   * adjust sentence borders to match xml/ <s>-tags
	#   * find instances of "'s" that are not included in any immediately preceding
	#     markables and merge them into those markables if genitive_s is True
	#   * return conllu-a style bracket informatio to add entity data to conllu files later
	conllua_data, centering_data, salience_data = fix_tsv(gum_source, gum_target, reddit=reddit)

	# Adjust rst/ files:
	#   * refresh token strings in case of inconsistency
	#   * note that segment borders are not automatically adjusted around xml/ <s> elements
	conn_data = fix_rst(gum_source, gum_target, reddit=reddit)

	# Add annotations to xml/:
	#   * add CLAWS tags in fourth column
	#   * add fifth column after lemma containing tok_func from dep/
	#   * add Centering Theory transition types to sentences
	print("\n\nEnriching XML files:\n" + "="*23)
	enrich_xml(gum_source, gum_target, centering_data, add_claws=options.claws, reddit=reddit, corpus=corpus_name)

	# Add annotations to dep/:
	#   * fresh token strings, POS tags and lemmas from xml/
	#   * generates vanilla tags in CPOS column from POS
	#   * creates speaker, s_type and centering transition comments from xml/
	# Returns pre_annotated, a dictionary giving pre-annotated fields in src/dep/ which overwrite annotation values
	print("\nEnriching Dependencies:\n" + "="*23)
	pre_annotated = enrich_dep(gum_source, gum_target, pepper_tmp, reddit)

	# Create fresh constituent parses in const/ if desired
	# (either reparse or use dep2const conversion, e.g. https://github.com/ikekonglp/PAD)
	if options.parse:
		print("\nRegenerating constituent trees:\n" + "="*30)
		const_parse(gum_source, reddit=reddit)
	else:
		sys.stdout.write("\ni Skipping fresh parses for const/\n")
		if not os.path.exists(gum_source + "const"):
			sys.stdout.write("x const/ directory missing in target but parsing was set to false! Aborting merge...\n")
			sys.exit()
		elif len(glob(gum_source + "const" + os.sep + "*.ptb")) != len(glob(gum_source + "xml" + os.sep + "*.xml")):
			sys.stdout.write("x parsing was set to false but xml/ and const/ contain different amounts of files! Aborting...\n")
			sys.exit()

	# Compile Universal Dependencies release version
	#   * UD files will be created in <target>/dep/
	#   * UD punctuation guidelines are enforced using udapi, which must be installed to work
	#   * udapi does not support Python 2, meaning punctuation will be attached to the root if using Python 2
	#   * UD morphology generation relies on parses already existing in <target>/const/
	print("\nCompiling Universal Dependencies version:\n" + "=" * 40)
	compile_ud(pepper_tmp, gum_target, pre_annotated, reddit=reddit, corpus=corpus_name)

	fix_gw_tags(gum_target, reddit=reddit)

	if not options.skip_ontogum:
		# Create OntoGUM data (OntoNotes schema version of coref annotations)
		print("\n\nCreating alternate OntoGUM version of coref annotations:\n" + "="*37)
		make_ontogum(gum_target, reddit=reddit)

	# Add labels to PTB trees
	if not options.skip_ptb_labels:
		print("\n\nAdding function labels to PTB constituent trees:\n" + "=" * 40)
		from utils.label_trees import add_ptb_labels
	ptb_files = sorted(glob(gum_source + "const" + os.sep + "*.ptb"))
	entidep_files = sorted(glob(pepper_tmp + "entidep" + os.sep + "*.conllu"))
	if not reddit:
		ptb_files = [f for f in ptb_files if "_reddit" not in f]
		entidep_files = [f for f in entidep_files if "_reddit" not in f]
	for i, ptb_file in enumerate(ptb_files):
		docname = os.path.basename(ptb_file)
		entidep_file = entidep_files[i]
		if not options.skip_ptb_labels:
			labeled = add_ptb_labels(io.open(ptb_file,encoding="utf8").read(),io.open(entidep_file,encoding="utf8").read())
		else:
			labeled = io.open(ptb_file,encoding="utf8").read()
		if not options.skip_ptb_labels:
			sys.stdout.write("\t+ " + " " * 70 + "\r")
			sys.stdout.write(" " + str(i + 1) + "/" + str(len(ptb_files)) + ":\t+ " + docname + "\r")
		with io.open(gum_target + "const" + os.sep + docname,'w',encoding="utf8",newline="\n") as f:
			f.write(labeled)
	sys.stdout.write("\n")
else:
	conllua_data = None
	salience_data = None
	sys.stderr.write("i Pepper only conversion, entities in conllu-a data will be generated from Pepper output (no infsat or min IDs)\n")

## Step 3: merge and convert source formats to target formats
if options.no_pepper:
	sys.__stdout__.write("\ni Skipping Pepper conversion\n")
else:
	sys.__stdout__.write("\nStarting pepper conversion:\n" + "="*30 + "\n")

	# Create Pepper staging erea in utils/pepper/tmp/
	dirs = [('xml','xml','xml','', ''),('dep','ud','conllu','', os.sep + "ud" + os.sep + "not-to-release"),
			('rst'+os.sep+'rstweb','rst','rs[34]','',''),('rst'+os.sep+'dependencies','rsd','rsd','',''),
			('tsv','tsv','tsv','coref' + os.sep,''),('const','const','ptb','','')]
	for dir in dirs:
		files = []
		dir_name, out_dir_name, extension, prefix, suffix = dir
		files_ = glob(gum_target + prefix + dir_name + suffix + os.sep + "*" + extension)
		for file_ in files_:
			if not reddit and "reddit_" in file_:
				continue
			files.append(file_)
		if not os.path.exists(pepper_tmp + out_dir_name + os.sep + "GUM" + os.sep):
			os.makedirs(pepper_tmp + out_dir_name + os.sep + "GUM" + os.sep)
		for file_ in files:
			shutil.copy(file_, pepper_tmp + out_dir_name + os.sep + "GUM" + os.sep)
	if not os.path.exists(gum_target + "coref" + os.sep + "conll" + os.sep):
		os.makedirs(gum_target + "coref" + os.sep + "conll" + os.sep)

	try:
		pepper_params = io.open(pepper_home + "merge_gum.pepperparams", encoding="utf8").read().replace("\r","")
	except:
		sys.__stdout__.write("x Can't find pepper template at: "+"utils" + os.sep + "pepper" + os.sep + "merge_gum.pepperparams"+"\n  Aborting...")
		sys.exit()

	# Inject gum_target in pepper_params and replace os.sep with URI slash
	if platform.system() == "Windows":
		pepper_params = pepper_params.replace("**gum_tmp**",os.path.abspath(pepper_tmp).replace(os.sep,"/"))
		pepper_params = pepper_params.replace("**gum_target**",gum_target.replace(os.sep,"/"))
	else:
		pepper_params = pepper_params.replace("file:/**gum_tmp**", os.path.abspath(pepper_tmp))
		pepper_params = pepper_params.replace("file:/**gum_target**", os.path.abspath(pepper_home) + os.sep + "../../target/")

	# Setup metadata file
	build_date = datetime.datetime.now().date().isoformat()
	meta = io.open(pepper_home + "meta_template.meta", encoding="utf8").read().replace("\r","")
	meta = meta.replace("**gum_version**",options.increment_version)
	meta = meta.replace("**build_date**",build_date)
	meta_out = io.open(pepper_tmp + "xml" + os.sep + "GUM" + os.sep + "GUM.meta",'w')
	meta_out.write(meta)
	meta_out.close()

	# Remove reddit tmp files if not included in build
	if not reddit:
		sys.__stdout__.write("\ni Deleting reddit files under " + pepper_tmp + "**\n")
		reddit_tmp = glob(pepper_tmp + "**\\GUM_reddit*",recursive=True)
		for f in reddit_tmp:
			os.remove(f)

	out = run_pepper(pepper_params,options.verbose_pepper)
	sys.__stdout__.write(out + "\n")

if options.pepper_only:
	quit()

## Step 4: propagate entity types, coref, discourse relations and XML annotations into conllu dep files
from utils.propagate import add_entities_to_conllu, add_rsd_to_conllu, add_bridging_to_conllu, add_xml_to_conllu

add_entities_to_conllu(gum_target, reddit=reddit, ontogum=False, conllua_data=conllua_data, salience_data=salience_data)
if not options.skip_ontogum:
	if options.no_pepper:
		sys.__stdout__.write("\ni Not adding entity information to UD parses in OntoGUM version since Pepper conversion was skipped\n")
		add_entities_to_conllu(gum_target,reddit=reddit,ontogum=True)
	else:
		add_entities_to_conllu(gum_target,reddit=reddit,ontogum=True)
add_bridging_to_conllu(gum_target,reddit=reddit,corpus=corpus_name)

sys.__stdout__.write("\no Added entities, coreference and bridging to UD parses\n")

add_rsd_to_conllu(gum_target,reddit=reddit,output_signals=not options.no_signals,output_secedges=not options.no_secedges)
if not options.skip_ontogum:
	add_rsd_to_conllu(gum_target,reddit=reddit,ontogum=True,output_signals=not options.no_signals,output_secedges=not options.no_secedges)
add_xml_to_conllu(gum_target,reddit=reddit,corpus=corpus_name)
if not options.skip_ontogum:
	add_xml_to_conllu(gum_target,reddit=reddit,ontogum=True,corpus=corpus_name)

sys.__stdout__.write("\no Added discourse relations and XML tags to UD parses\n")

make_disrpt(conn_data,reddit=reddit,corpus="gum")

sys.__stdout__.write("\no Created DISRPT shared task discourse relation formats in target rst/disrpt/\n")

## Step 5: Refresh automatic portion of non-DM signals in RST files

sys.__stdout__.write("\no Adding fresh non-DM signals to RST files:\n" + "=" * 37 + "\n")
update_non_dm_signals(gum_source, gum_target, reddit=reddit)

