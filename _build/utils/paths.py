import platform, os

if platform.system() == 'Linux':
	tt_path = os.path.abspath('utils/treetagger/') # Path to TreeTagger
	tt_path = os.path.abspath('utils/stanford_parser/') # Path to Stanford Parser
else:
	# Alternate paths for running under Windows or Mac
	tt_path = 'C:\\TreeTagger\\'
	parser_path = "C:\\stanford-parser\\"
