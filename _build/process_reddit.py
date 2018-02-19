import io, os, sys
from utils.get_reddit.fetch_text import run_fetch
from argparse import ArgumentParser
from utils.get_reddit.underscores import underscoring, deunderscoring

PY3 = sys.version_info[0] == 3

if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("-m","--mode",action="store",choices=["add","del"],default="add",help="Whether to add reddit token data or delete it")

	options = parser.parse_args()

	src_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep + "src" + os.sep

	if options.mode == "add":

		textdic = run_fetch()

		deunderscoring(src_dir, textdic)

		print("Completed fetching reddit data.")
		print("You can now use build_gum.py to produce all annotation layers.")

	elif options.mode == "del":

		underscoring(src_dir)
		print("Tokens in reddit data have been replaced with underscores.")