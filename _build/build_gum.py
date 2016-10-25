#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, shutil, sys
from glob import glob
from argparse import ArgumentParser
from utils.pepper_runner import run_pepper
import datetime

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
parser.add_argument("-t",dest="target",action="store",help="GUM build target directory", default="target")
parser.add_argument("-s",dest="source",action="store",help="GUM build source directory", default="src")
parser.add_argument("-p",dest="parse",action="store_true",help="Whether to reparse constituents")
parser.add_argument("-c",dest="claws",action="store_true",help="Whether to reassign claws5 tags")
parser.add_argument("-v",dest="verbose_pepper",action="store_true",help="Whether to print verbose pepper output")
parser.add_argument("-i",dest="increment_version",action="store",help="A new version number to assign",default="DEVELOP")

options = parser.parse_args()

gum_source = os.path.abspath(options.source)
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
from utils.validate import validate_src

print "="*20
print "Validating files..."
print "="*20 + "\n"

validate_src(gum_source)

######################################
## Step 2: propagate annotations
######################################
from utils.propagate import enrich_dep, enrich_xml, const_parse
from utils.repair_tsv import fix_tsv
from utils.repair_rst import fix_rst


# Check and potentially correct POS tags and lemmas based on pooled annotations
#proof(gum_source)

# Add annotations to dep/:
#   * fresh token strings, POS tags and lemmas from xml/
#   * generates vanilla tags in CPOS column from POS
#   * creates speaker and s_type comments from xml/
print "\nEnriching Dependencies:\n" + "="*23
enrich_dep(gum_source, gum_target)

# Add annotations to xml/:
#   * add CLAWS tags in fourth column
#   * add fifth column after lemma containing tok_func from dep/
print "\nEnriching XML files:\n" + "="*23
enrich_xml(gum_source, gum_target, options.claws)

# Token and sentence border adjustments
print "\nAdjusting token and sentence borders:\n" + "="*40
# Adjust tsv/ files:
#   * refresh and re-merge token strings in case they were mangled by WebAnno
#   * adjust sentence borders to match xml/ <s>-tags
fix_tsv(gum_source,gum_target)

# Adjust rst/ files:
#   * refresh token strings in case of inconsistency
#   * note that segment borders are not automatically adjusted around xml/ <s> elements
fix_rst(gum_source,gum_target)

# Create fresh constituent parses in const/ if desired
# (either reparse or use dep2const conversion, e.g. https://github.com/ikekonglp/PAD)
if options.parse:
	print "\nRegenerating constituent trees:\n" + "="*30
	const_parse(gum_source,gum_target)
else:
	print "\ni Skipping fresh parse for const/"
	if not os.path.exists(gum_target + "const"):
		print "x const/ directory missing in target but parsing was set to false! Aborting..."
		sys.exit()
	elif len(glob(gum_target + "const" + os.sep + "*.ptb")) != len(glob(gum_target + "xml" + os.sep + "*.xml")):
		print "x parsing was set to false but xml/ and const/ contain different amounts of files! Aborting..."
		sys.exit()

## Step 3: merge and convert source formats to target formats

print "\nStarting pepper conversion:\n" + "="*30

# Create Pepper staging erea in utils/pepper/tmp/
pepper_home = "utils" + os.sep + "pepper" + os.sep
dirs = [('xml','xml',''),('dep','conll10',''),('rst','rs3',''),('tsv','tsv','coref' + os.sep),('const','ptb','')]
for dir in dirs:
	dir_name, extension, prefix = dir
	files = glob(gum_target + prefix + dir_name + os.sep + "*" + extension)
	pepper_tmp = pepper_home + "tmp" + os.sep
	if not os.path.exists(pepper_tmp + dir_name + os.sep + "GUM" + os.sep):
		os.makedirs(pepper_tmp + dir_name + os.sep + "GUM" + os.sep)
	for file_ in files:
		shutil.copy(file_, pepper_tmp + dir_name + os.sep + "GUM" + os.sep)
if not os.path.exists(gum_target + "coref" + os.sep + "conll" + os.sep):
	os.makedirs(gum_target + "coref" + os.sep + "conll" + os.sep)

pepper_tmp = pepper_home + "tmp" + os.sep

try:
	pepper_params = open("utils" + os.sep + "pepper" + os.sep + "merge_gum.pepperparams").read().replace("\r","")
except:
	print "x Can't find pepper template at: "+"utils" + os.sep + "pepper" + os.sep + "merge_gum.pepperparams"+"\n  Aborting..."
	sys.exit()

# Inject gum_target in pepper_params and replace os.sep with URI slash
pepper_params = pepper_params.replace("**gum_tmp**",os.path.abspath(pepper_tmp).replace(os.sep,"/"))
pepper_params = pepper_params.replace("**gum_target**",gum_target.replace(os.sep,"/"))


# Setup metadata file
build_date = datetime.datetime.now().date().isoformat()
meta = open(pepper_home + "meta_template.meta").read().replace("\r","")
meta = meta.replace("**gum_version**",options.increment_version)
meta = meta.replace("**build_date**",build_date)
meta_out = open(pepper_tmp + "xml" + os.sep + "GUM" + os.sep + "GUM.meta",'w')
meta_out.write(meta)
meta_out.close()

out = run_pepper(pepper_params,options.verbose_pepper)
print out
