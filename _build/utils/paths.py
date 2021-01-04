import platform, os

if platform.system() != 'Windows':
	tt_path = os.path.abspath('utils/treetagger/') + os.sep # Path to TreeTagger
	parser_path = os.path.abspath('utils/stanford_parser/') # Path to Stanford Parser
	core_nlp_path = "utils/core_nlp/"
else:
	# Alternate paths for running under Windows
	tt_path = 'utils\\treetagger\\'
	parser_path = "utils\\stanford_parser"
	core_nlp_path = "utils\\core_nlp"
