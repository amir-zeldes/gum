import io, os, sys, argparse

"""
Script to evaluate segmentation f-score and perfect discourse unit segmentation proportion from two files. Two input formats are permitted:

  * One token per line, with ten columns, no sentence breaks (default *.tok format) - segmentation indicated in column 10
  * The same, but with blank lines between sentences (*.conll format)

Token columns follow the CoNLL-U format, with token IDs in the first column and pipe separated key=value pairs in the last column. 

Document boundaries are indicated by a comment: # newdoc id = ...

The evaluation uses micro-averaged F-Scores per corpus (not document macro average).

Example:

```
# newdoc id = GUM_bio_byron
1	Education	_	_	_	_	_	_	_	BeginSeg=Yes
2	and	_	_	_	_	_	_	_	_
3	early	_	_	_	_	_	_	_	_
4	loves	_	_	_	_	_	_	_	_
5	Byron	_	_	_	_	_	_	_	BeginSeg=Yes
6	received	_	_	_	_	_	_	_	_
...
```

Or:

```
# newdoc id = GUM_bio_byron
# sent_id = GUM_bio_byron-1
# text = Education and early loves
1	Education	education	NOUN	NN	Number=Sing	0	root	_	BeginSeg=Yes
2	and	and	CCONJ	CC	_	4	cc	_	_
3	early	early	ADJ	JJ	Degree=Pos	4	amod	_	_
4	loves	love	NOUN	NNS	Number=Plur	1	conj	_	_

# sent_id = GUM_bio_byron-2
# text = Byron received his early formal education at Aberdeen Grammar School, and in August 1799 entered the school of Dr. William Glennie, in Dulwich. [17]
1	Byron	Byron	PROPN	NNP	Number=Sing	2	nsubj	_	BeginSeg=Yes
2	received	receive	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	0	root	_	_
```

For PDTB-style corpora, we calculate exact span-wise f-scores for BIO encoding, without partial credit. In other words, 
predicting an incorrect span with partial overlap is the same as missing a gold span and predicting an incorrect span
somewhere else in the corpus. Note also that spans must begin with B-Conn - predicted spans beginning with I-Conn are ignored.

The file format for PDTB style corpora is similar, but with different labels:

```
1	Fidelity	Fidelity	PROPN	NNP	_	6	nsubj	_	_
2	,	,	PUNCT	,	_	6	punct	_	_
3	for	for	ADP	IN	_	4	case	_	Seg=B-Conn
4	example	example	NOUN	NN	_	6	obl	_	Seg=I-Conn
5	,	,	PUNCT	,	_	6	punct	_	_
6	prepared	prepare	VERB	VBN	_	0	root	_	_
7	ads	ad	NOUN	NNS	_	6	obj	_	_
...
```


Arguments:
 * goldfile: shared task gold test data
 * predfile: same format, with predicted segments positions in column 10 - note **number of tokens must match**  
 * string_input: if specified, files are replaced by strings with file contents instead of file names


"""

__author__ = "Amir Zeldes"
__license__ = "Apache 2.0"
__version__ = "1.0.1"

def parse_data(infile, string_input=False):
	if not string_input:
		data = io.open(infile, encoding="utf8").read().strip().replace("\r", "")
	else:
		data = infile.strip()

	tokens = []
	labels = []
	spans = []
	counter = 0
	span_start = -1
	span_end = -1
	for line in data.split("\n"):
		if "\t" in line:  # Token
			fields = line.split("\t")
			if "-" in fields[0]:
				continue
			label = fields[-1]
			# Ensure correct labeling even if other pipe-delimited annotations found in column 10
			if "BeginSeg=Yes" in label:
				label = "BeginSeg=Yes"
			elif "Seg=B-Conn" in label:
				if span_start > -1:  # Add span
					if span_end == -1:
						span_end = span_start
					spans.append((span_start,span_end))
					span_end = -1
				label ="Seg=B-Conn"
				span_start = counter
			elif "Seg=I-Conn" in label:
				label = "Seg=I-Conn"
				span_end = counter
			else:
				label = "_"
				if span_start > -1:  # Add span
					if span_end == -1:
						span_end = span_start
					spans.append((span_start,span_end))
					span_start = -1
					span_end = -1

			tokens.append(fields[1])
			labels.append(label)
			counter +=1

	if span_start > -1 and span_end > -1:  # Add last span
		spans.append((span_start,span_end))

	return tokens, labels, spans


def get_scores(gold_file, pred_file, string_input=False):
	"""

	:param gold_file: Gold shared task file
	:param pred_file: File with predictions
	:param string_input: If True, files are replaced by strings with file contents (for import inside other scripts)
	:return: dictionary of scores for printing
	"""


	report = ""
	gold_tokens, gold_labels, gold_spans = parse_data(gold_file, string_input)
	pred_tokens, pred_labels, pred_spans = parse_data(pred_file, string_input)

	if os.path.isfile(gold_file):
		doc_name = os.path.basename(gold_file)
	else:
		# Use first few tokens to identify file
		doc_name = " ".join(gold_tokens[0:10]) + "..."

	# Check same number of tokens in both files
	if len(gold_tokens) != len(pred_tokens):
		report += "\nFATAL: different number of tokens detected in gold and pred:\n"
		report += "  o In " + doc_name + ": " + str(len(gold_tokens)) + " gold tokens but " + str(len(pred_tokens)) + " predicted tokens\n\n"
		sys.stderr.write(report)
		sys.exit(0)

	# Check tokens are identical
	for i, tok in enumerate(gold_tokens):
		if tok != pred_tokens[i]:
			report += "\nWARN: token strings do not match in gold and pred:\n"
			report += " o First instance in " + doc_name + " token " + str(i) + "\n"
			report += "Gold: " + tok + " but Pred: " + pred_tokens[i] + "\n\n"
			sys.stderr.write(report)
			break

	# Check if this is EDU or Conn-style data
	if "BeginSeg=Yes" in gold_labels:
		mode = "edu"
		seg_type = "EDUs"
	else:
		mode = "conn"
		seg_type = "conn spans"

	true_positive = 0
	false_positive = 0
	false_negative = 0

	if mode == "edu":
		for i, gold_label in enumerate(gold_labels):
			pred_label = pred_labels[i]
			if gold_label == pred_label:
				if gold_label == "_":
					continue
				else:
					true_positive += 1
			else:
				if pred_label == "_":
					false_negative += 1
				else:
					if gold_label == "_":
						false_positive += 1
					else:  # I-Conn/B-Conn mismatch
						false_positive +=1
	else:
		for span in gold_spans:
			if span in pred_spans:
				true_positive +=1
			else:
				false_negative +=1
		for span in pred_spans:
			if span not in gold_spans:
				false_positive += 1

	try:
		precision = true_positive / (float(true_positive) + false_positive)
	except Exception as e:
		precision = 0

	try:
		recall = true_positive / (float(true_positive) + false_negative)
	except Exception as e:
		recall = 0

	try:
		f_score = 2 * (precision * recall) / (precision + recall)
	except:
		f_score = 0

	score_dict = {}
	score_dict["doc_name"] = doc_name
	score_dict["tok_count"] = len(gold_tokens)
	score_dict["seg_type"] = seg_type
	score_dict["gold_seg_count"] = true_positive+false_negative
	score_dict["pred_seg_count"] = true_positive+false_positive
	score_dict["prec"] = precision
	score_dict["rec"] = recall
	score_dict["f_score"] = f_score

	return score_dict


if __name__ == "__main__":
	p = argparse.ArgumentParser()

	p.add_argument("goldfile",help="Shared task gold file in .tok or .conll format")
	p.add_argument("predfile",help="Corresponding file with system predictions")
	p.add_argument("-s","--string_input",action="store_true",help="Whether inputs are file names or strings")

	opts = p.parse_args()

	score_dict = get_scores(opts.goldfile,opts.predfile,opts.string_input)

	print("File: " + score_dict["doc_name"])
	print("o Total tokens: " + str(score_dict["tok_count"]))
	print("o Gold " +score_dict["seg_type"]+": " + str(score_dict["gold_seg_count"]))
	print("o Predicted "+score_dict["seg_type"]+": " + str(score_dict["pred_seg_count"]))
	print("o Precision: " + str(score_dict["prec"]))
	print("o Recall: " + str(score_dict["rec"]))
	print("o F-Score: " + str(score_dict["f_score"]))
