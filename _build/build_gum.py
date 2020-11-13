#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, shutil, sys, io
from glob import glob
from argparse import ArgumentParser
from utils.pepper_runner import run_pepper
import datetime

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

options = parser.parse_args()

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
from utils.propagate import enrich_dep, enrich_xml, const_parse, compile_ud
from utils.repair_tsv import fix_tsv
from utils.repair_rst import fix_rst


# Check and potentially correct POS tags and lemmas based on pooled annotations
#proof(gum_source)

# Add annotations to dep/:
#   * fresh token strings, POS tags and lemmas from xml/
#   * generates vanilla tags in CPOS column from POS
#   * creates speaker and s_type comments from xml/
print("\nEnriching Dependencies:\n" + "="*23)
enrich_dep(gum_source, pepper_tmp, reddit)

# Add annotations to xml/:
#   * add CLAWS tags in fourth column
#   * add fifth column after lemma containing tok_func from dep/
print("\n\nEnriching XML files:\n" + "="*23)
enrich_xml(gum_source, gum_target, add_claws=options.claws, reddit=reddit)

# Token and sentence border adjustments
print("\nAdjusting token and sentence borders:\n" + "="*37)
# Adjust tsv/ files:
#   * refresh and re-merge token strings in case they were mangled by WebAnno
#   * adjust sentence borders to match xml/ <s>-tags
#   * find instances of "'s" that are not included in any immediately preceding
#     markables and merge them into those markables if genitive_s is True
fix_tsv(gum_source, gum_target, reddit=reddit)

# Adjust rst/ files:
#   * refresh token strings in case of inconsistency
#   * note that segment borders are not automatically adjusted around xml/ <s> elements
fix_rst(gum_source, gum_target, reddit=reddit)

# Create fresh constituent parses in const/ if desired
# (either reparse or use dep2const conversion, e.g. https://github.com/ikekonglp/PAD)
if options.parse:
	print("\nRegenerating constituent trees:\n" + "="*30)
	const_parse(gum_source, gum_target, reddit=reddit)
else:
	sys.stdout.write("\ni Skipping fresh parses for const/\n")
	if not os.path.exists(gum_target + "const"):
		sys.stdout.write("x const/ directory missing in target but parsing was set to false! Aborting merge...\n")
		sys.exit()
	elif len(glob(gum_target + "const" + os.sep + "*.ptb")) != len(glob(gum_target + "xml" + os.sep + "*.xml")):
		sys.stdout.write("x parsing was set to false but xml/ and const/ contain different amounts of files! Aborting...\n")
		sys.exit()

# Compile Universal Dependencies release version
#   * UD files will be created in <target>/dep/
#   * UD punctuation guidelines are enforced using udapi, which must be installed to work
#   * udapi does not support Python 2, meaning punctuation will be attached to the root if using Python 2
#   * UD morphology generation relies on parses already existing in <target>/const/
print("\nCompiling Universal Dependencies version:\n" + "=" * 40)
compile_ud(pepper_tmp, gum_target, reddit=reddit)

## Step 3: merge and convert source formats to target formats
if options.no_pepper:
	sys.__stdout__.write("\ni Skipping Pepper conversion\n")
else:
	sys.__stdout__.write("\nStarting pepper conversion:\n" + "="*30 + "\n")

	# Create Pepper staging erea in utils/pepper/tmp/
	dirs = [('xml','xml','xml','', ''),('dep','ud','conllu','', os.sep + "ud" + os.sep + "not-to-release"),
			('rst'+os.sep+'rstweb','rst','rs3','',''),('rst'+os.sep+'dependencies','rsd','rsd','',''),
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
		pepper_params = io.open("utils" + os.sep + "pepper" + os.sep + "merge_gum.pepperparams", encoding="utf8").read().replace("\r","")
	except:
		sys.__stdout__.write("x Can't find pepper template at: "+"utils" + os.sep + "pepper" + os.sep + "merge_gum.pepperparams"+"\n  Aborting...")
		sys.exit()

	# Inject gum_target in pepper_params and replace os.sep with URI slash
	pepper_params = pepper_params.replace("**gum_tmp**",os.path.abspath(pepper_tmp).replace(os.sep,"/"))
	pepper_params = pepper_params.replace("**gum_target**",gum_target.replace(os.sep,"/"))


	# Setup metadata file
	build_date = datetime.datetime.now().date().isoformat()
	meta = io.open(pepper_home + "meta_template.meta", encoding="utf8").read().replace("\r","")
	meta = meta.replace("**gum_version**",options.increment_version)
	meta = meta.replace("**build_date**",build_date)
	meta_out = io.open(pepper_tmp + "xml" + os.sep + "GUM" + os.sep + "GUM.meta",'w')
	meta_out.write(meta)
	meta_out.close()

	out = run_pepper(pepper_params,options.verbose_pepper)
	sys.__stdout__.write(out + "\n")


## Step 4: propagate entity types and coref into conllu dep files
if options.no_pepper:
	sys.__stdout__.write("\ni Not adding entity information to UD parses since Pepper conversion was skipped\n")
else:
	from utils.propagate import add_entities_to_conllu, add_rsd_to_conllu
	add_entities_to_conllu(gum_target,reddit=reddit)
	add_rsd_to_conllu(gum_target,reddit=reddit)
	sys.__stdout__.write("\no Added entities, coreference and discourse relations to UD parses\n")
