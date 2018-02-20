from json import loads
import ast, re, io, os, sys
import requests
from collections import defaultdict
import datetime
from time import sleep
from argparse import ArgumentParser

PY3 = sys.version_info[0] == 3

docs = {
	"GUM_reddit_macroeconomics": [
		{"year": "2017", "month": "09", "id": "6zm74h", "type": "post"},
		{"year": "2017", "month": "09", "id": "dmwwqlt", "type": "comment"}
	],
	"GUM_reddit_stroke": [
		{"year": "2017", "month": "08", "id": "6ws3eh", "type": "post"},
		{"year": "2017", "month": "08", "id": "dmaei1x", "type": "comment"},
		{"year": "2017", "month": "08", "id": "dmaiwsm", "type": "comment"},
		{"year": "2017", "month": "09", "id": "dmkx8bk", "type": "comment"},
		{"year": "2017", "month": "09", "id": "dmm1327", "type": "comment"},
		{"year": "2017", "month": "08", "id": "dmaoodn", "type": "comment"}
	],
	"GUM_reddit_polygraph": [
		{"year": "2014", "month": "12", "id": "2q6qnv", "type": "post"}
	],
	"GUM_reddit_ring": [
		{"year": "2016", "month": "09", "id": "5570x1", "type": "post"},
		{"year": "2016", "month": "09", "id": "d885ma0", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d8880w7", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d88u7dg", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d88unu3", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d88v0sz", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d88xaqu", "type": "comment"},
		{"year": "2016", "month": "10", "id": "d893mj9", "type": "comment"},
		{"year": "2016", "month": "09", "id": "d88s4bb", "type": "comment"},
		{"year": "2016", "month": "10", "id": "d88zt6x", "type": "comment"}
	],
	"GUM_reddit_space": [
		{"year": "2016", "month": "08", "id": "50hx5c", "type": "post"},
		{"year": "2016", "month": "08", "id": "d7471k5", "type": "comment"},
		{"year": "2016", "month": "08", "id": "d74i5ka", "type": "comment"},
		{"year": "2016", "month": "08", "id": "d74ppi0", "type": "comment"}
	],
	"GUM_reddit_superman": [
		#{"year": "2017", "month": "04", "id": "68e0u3", "type": "post", "title_only": True},  # Post title not included in this document
		{"year": "2017", "month": "05", "id": "dgys1z8", "type": "comment"}
	]
}


def get_proxy_data():
	out_posts = {}
	tab_delim = requests.get("https://corpling.uis.georgetown.edu/gum/fetch_text_proxy.py").text
	for line in tab_delim.split("\n"):
		if "\t" in line:
			post, text = line.split("\t")
			out_posts[post] = text
	return out_posts


def get_via_praw(post_id, post_type,praw_cred):

	if praw_cred is None:
		raise IOError("Missing praw credentials")

	reddit = praw.Reddit(client_id=praw_cred["client_id"], client_secret=praw_cred["client_secret"],
						 password=praw_cred["password"], user_agent=praw_cred["user_agent"],username=praw_cred["user_name"])

	if post_type == "post":
		submission = reddit.submission(post_id)
		created_utc = submission.mod.thing.created_utc
		selftext = submission.mod.thing.selftext
		selftext = re.sub(r'\s+',' ',selftext)
		selftext = selftext.replace("'","\\'")
		title = submission.mod.thing.title
		out_json = "[{'id':'"+post_id+"','selftext':'"+selftext+"','created_utc':"+str(int(created_utc))+",'title':'"+title+"'}]"
	else:
		submission = reddit.comment(post_id)
		created_utc = submission.mod.thing.created_utc
		selftext = submission.mod.thing.body
		selftext = re.sub(r'\s+',' ',selftext)
		selftext = selftext.replace("'","\\'")
		title = ""
		out_json = "[{'id':'"+post_id+"','body':'"+selftext+"','created_utc':"+str(int(created_utc))+"}]"

	return out_json


def get_post(year, month, post_id, post_type):
	from bigquery import get_client

	# JSON key provided by Google
	json_key = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'key.json'

	client = get_client(json_key_file=json_key, readonly=True)

	if post_type == "post":
		post_or_comment = "posts"
	else:
		post_or_comment = "comments"
	table_name = "fh-bigquery:reddit_"+post_or_comment+"."+year+"_"+month

	# Submit an async query.
	query = "SELECT * FROM [" + table_name + "] WHERE id = '"+post_id+"';"
	job_id, _results = client.query(query)

	sleep(3)

	# Check if the query has finished running.
	complete, row_count = client.check_job(job_id)

	# Retrieve the results.
	results = client.get_query_rows(job_id)

	return str(results)


def get_no_space_strings(cache_dict, praw_cred=None, overwrite_cache=False):

	no_space_docs = defaultdict(str)

	for doc in docs:
		for post in docs[doc]:
			if post["id"] in cache_dict:
				json_result = cache_dict[post["id"]]
				if overwrite_cache:
					with io.open(os.path.dirname(os.path.realpath(__file__)) + os.sep + "cache.txt", "a", encoding="utf8") as f:
						f.write(post["id"] + "\t" + json_result.strip() + "\n")
			else:
				if int(post["year"]) >2015 or (int(post["year"]==2015) and post["month"] == "12"):  # Available from bigquery
					json_result = get_post(post["year"],post["month"],post["id"],post["type"])
				else:
					json_result = get_via_praw(post["id"],post["type"],praw_cred)
				with io.open(os.path.dirname(os.path.realpath(__file__)) + os.sep + "cache.txt","a",encoding="utf8") as f:
					f.write(post["id"] + "\t" + json_result.strip() + "\n")
			parsed = ast.literal_eval(json_result)[0]
			if post["type"]=="post":
				plain = parsed["selftext"]
				title = parsed["title"]
				if "title_only" in post:
					if post["title_only"]:
						plain = ""
			else:
				plain = parsed["body"]
				title = ""
			if "_space" in doc:
				plain = plain.replace("&gt;","")  # GUM_reddit_space has formatting &gt; to indicate indented block quotes
			no_space = re.sub(r"\s","",plain).replace("*","")
			no_space = re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',no_space)  # Remove Wiki style links: [text](URL)
			if no_space_docs[doc] == "":
				no_space_docs[doc] += re.sub(r"\s","",title).replace("*","")
			no_space_docs[doc] += no_space

	return no_space_docs


def run_fetch():
	script_dir = os.path.dirname(os.path.realpath(__file__))
	if not os.path.isfile(script_dir + os.sep + "cache.txt"):
		io.open(script_dir + os.sep + "cache.txt", "a").close()  # Make sure cache file exists

	cache = io.open(script_dir + os.sep + "cache.txt", encoding="utf8")
	cache_dict = {}

	for line in cache.read().split("\n"):
		if "\t" in line:
			post_id, text = line.split("\t")
			cache_dict[post_id] = text

	if not os.path.isfile(script_dir + os.sep + "praw.txt"):
		io.open(script_dir + os.sep + "praw.txt", "a").close()  # Make sure praw file existss

	praw_cred = io.open(script_dir + os.sep + "praw.txt",encoding="utf8")
	praw_dict = {}

	for line in praw_cred.read().split("\n"):
		if "\t" in line and not line.startswith("#"):
			key, val = line.split("\t")
			praw_dict[key] = val


	# Check if cache is already complete
	post_ids = []
	for doc in docs:
		for post in docs[doc]:
			post_ids.append(post["id"])
	if any(post not in cache_dict for post in post_ids):
		incomplete = True
	else:
		incomplete = False

	if incomplete:
		# Check that user has valid json and praw credentials
		if not all(key in praw_dict for key in ["client_id","client_secret","password","user_agent","username"]):
			print("Missing praw credentials detected! You cannot download reddit data using praw.")
			has_praw_cred = False
		else:
			has_praw_cred = True
			try:
				import praw
			except ImportError as e:
				print("Library praw not installed (pip install praw). You cannot download reddit data using praw.")
				has_praw_cred = False

		if not os.path.isfile(script_dir + os.sep + "key.json"):
			print("Can't find Google BigQuery json key file. You cannot download reddit data using bigquery")
			has_bigquery = False
		else:
			try:
				has_bigquery = True
				import bigquery
			except ImportError as e:
				print("Library bigquery not installed (pip install bigquery). You cannot download reddit data using bigquery.")
				has_bigquery = False

		if not has_praw_cred or not has_bigquery:
			print("Missing access to bigquery and/or praw.")
			print("Do you want to try downloading reddit data from an available server?")
			print("Confirm: you are solely responsible for downloading reddit data and may only use it for non-commercial purposes:")
			try:
				# for python 2
				response = raw_input("[Y]es/[N]o> ")
			except NameError:
				# for python 3
				response = input("[Y]es/[N]o> ")

			if response == "Y":
				print("Retrieving reddit data by proxy...")
				cache_dict = get_proxy_data()
				out_docs = get_no_space_strings(cache_dict, overwrite_cache=True)
				return out_docs
			else:
				print("Aborting")
				sys.exit()
		else:
			print("Found praw and bigquery credentials.")
			print("Would you like to use them to download reddit data?")
			print("Confirm: you are solely responsible for downloading reddit data and may only use it for non-commercial purposes:")
			try:
				# for python 2
				response = raw_input("[Y]es/[N]o> ")
			except NameError:
				# for python 3
				response = input("[Y]es/[N]o> ")

			if response == "Y":
				print("Retrieving reddit data...")
				out_docs = get_no_space_strings(cache_dict,praw_cred=praw_dict)
			else:
				print("Do you want to try downloading reddit data from an available server?")
				print("Confirm: you are solely responsible for downloading reddit data and may only use it for non-commercial purposes:")
				try:
					# for python 2
					response = raw_input("[Y]es/[N]o> ")
				except NameError:
					# for python 3
					response = input("[Y]es/[N]o> ")

				if response == "Y":
					print("Retrieving reddit data by proxy...")
					cache_dict = get_proxy_data()
					out_docs = get_no_space_strings(cache_dict, overwrite_cache=True)
					return out_docs
				else:
					print("Aborting")
					sys.exit()
	else:
		print("Found complete reddit data in utils/get_reddit/cache.txt ...")
		print("Compiling raw strings")
		out_docs = get_no_space_strings(cache_dict)

	return out_docs



