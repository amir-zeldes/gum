from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus

from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
    WordEmbeddings,
    TransformerWordEmbeddings
)

from flair.models import SequenceTagger
import flair
from torch.optim.adam import Adam
import os, sys, re, io

from lib.reorder_sgml import reorder

import conllu
from collections import OrderedDict, defaultdict

TAGS = [
    "sp",
    "table",
    "row",
    "cell",
    "head",
    "p",
    "figure",
    "caption",
    "list",
    "item",
    "quote",
    "s",
    "q",
    "hi",
    "sic",
    "ref",
    "date",
    "incident",
    "w",
]
BLOCK_TAGS = ["sp", "head", "p", "figure", "caption", "list", "item"]
OPEN_SGML_ELT = re.compile(r"^<([^/ ]+)( .*)?>$")
CLOSE_SGML_ELT = re.compile(r"^</([^/]+)>$")


def maximal_nontoken_span_end(sgml_list, i):
    """Return j such that sgml_list[i:j] does not contain tokens
    and no element that is begun in the MNS is closed in it."""
    opened = []
    j = i
    while j < len(sgml_list):
        line = sgml_list[j]
        open_match = re.match(OPEN_SGML_ELT, line)
        close_match = re.match(CLOSE_SGML_ELT, line)
        if not (open_match or close_match):
            break
        if open_match:
            opened.append(open_match.groups()[0])
        if close_match and close_match.groups()[0] in opened:
            break
        j += 1
    return j


def fix_malformed_sentences(sgml_list):
    """
    Fixing malformed SGML seems to boil down to two cases:

    (1) The sentence is interrupted by the close of a tag that opened before it. In this case,
        update the s boundaries so that we close and begin sentences at the close tag:

                             <a>
                <a>          ...
                ...          <s>
                <s>          ...
                ...    ==>   </s>
                </a>         </a>
                ...          <s>
                </s>         ...
                             </s>

    (2) Some tag opened inside of the sentence and has remained unclosed at the time of sentence closure.
        In this case, we choose not to believe the sentence split, and merge the two sentences:

                <s>
                ...          <s>
                <a>          ...
                ...          <a>
                </s>   ==>   ...
                <s>          ...
                ...          </a>
                </a>         ...
                ...          </s>
                </s>
    """
    tag_opened = defaultdict(list)
    i = 0
    while i < len(sgml_list):
        line = sgml_list[i].strip()
        open_match = re.search(OPEN_SGML_ELT, line)
        close_match = re.search(CLOSE_SGML_ELT, line)
        if open_match:
            tag_opened[open_match.groups()[0]].append(i)
        elif close_match:
            tagname = close_match.groups()[0]
            j = maximal_nontoken_span_end(sgml_list, i + 1)
            mns = sgml_list[i:j]

            # case 1: we've encountered a non-s closing tag. If...
            if (
                    tagname != "s"  # the closing tag is not an s
                    and len(tag_opened["s"]) > 0  # and we're in a sentence
                    and len(tag_opened[tagname]) > 0
                    and len(tag_opened["s"]) > 0  # and the sentence opened after the tag
                    and tag_opened[tagname][-1] < tag_opened["s"][-1]
                    and "</s>" not in mns  # the sentence is not closed in the mns
            ):
                # end sentence here and move i back to the line we were looking at
                sgml_list.insert(i, "</s>")
                i += 1
                # open a new sentence at the end of the mns and note that we are no longer in the sentence
                sgml_list.insert(j + 1, "<s>")
                tag_opened["s"].pop(-1)
                # we have successfully closed this tag
                tag_opened[tagname].pop(-1)
            # case 2: s closing tag and there's some tag that opened inside of it that isn't closed in time
            elif tagname == "s" and any(
                    e != "s" and f"</{e}>" not in mns
                    for e in [
                        e
                        for e in tag_opened.keys()
                        if
                        len(tag_opened[e]) > 0 and len(tag_opened["s"]) > 0 and tag_opened[e][-1] > tag_opened["s"][-1]
                    ]
            ):
                # some non-s element opened within this sentence and has not been closed even in the mns
                assert "<s>" in mns
                sgml_list.pop(i)
                i -= 1
                sgml_list.pop(i + mns.index("<s>"))
            else:
                tag_opened[tagname].pop(-1)
        i += 1
    return sgml_list


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def unescape(token):
    token = token.replace("&quot;", '"')
    token = token.replace("&lt;", "<")
    token = token.replace("&gt;", ">")
    token = token.replace("&amp;", "&")
    token = token.replace("&apos;", "'")
    return token


def tokens2conllu(tokens):
    tokens = [
        OrderedDict(
            (k, v)
            for k, v in zip(
                conllu.parser.DEFAULT_FIELDS,
                [i + 1, unescape(token)] + ["_" for i in range(len(conllu.parser.DEFAULT_FIELDS) - 1)],
            )
        )
        for i, token in enumerate(tokens)
    ]
    tl = conllu.TokenList(tokens)
    return tl


class FlairSentSplitter:
    def __init__(self, model_path=None, span_size=20, stride_size=10):

        self.span_size = span_size  # Each shingle is 20 tokens by default
        self.stride_size = stride_size  # Tag a shingle every stride_size tokens
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.model = None

    def load_model(self, path):

        self.model = SequenceTagger.load(path)

    def train(self, training_dir):
        from flair.trainers import ModelTrainer
        print(training_dir)

        # define columns
        columns = {0: "text", 1: "ner"}

        # init a corpus using column format, data folder and the names of the train, dev and test files
        # note that training data should be unescaped, i.e. tokens like "&", not "&amp;"
        corpus: Corpus = ColumnCorpus(
            training_dir,
            columns,
            train_file="sent_train.txt",
            test_file="sent_test.txt",
            dev_file="sent_dev.txt",
            document_separator_token="-DOCSTART-",
        )

        print(corpus)

        tag_type = "ner"
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)
        embd = "bert-base-multilingual-cased"
        lang = training_dir.split('/')[-2]
        if "deu" in lang:
            embd = "bert-base-german-cased"
        elif "eus" in lang:
            embd = "ixa-ehu/berteus-base-cased"
        elif "zho" in lang:
            embd = "hfl/chinese-roberta-wwm-ext-large"
        elif "tur" in lang:
            embd = "dbmdz/bert-base-turkish-cased"
        elif "eng" in lang:
            embd = "bert-base-cased"
        elif "por" in lang:
            embd = "bert-base-multilingual-cased"
        elif "spa" in lang:
            embd = "dccuchile/bert-base-spanish-wwm-cased"
        elif "fra" in lang:
            embd = "dbmdz/bert-base-french-europeana-cased"
        elif "nld" in lang:
            embd="GroNLP/bert-base-dutch-cased"
        elif "rus" in lang:
            embd="blinoff/roberta-base-russian-v0"
        elif "fas" in lang:
            embd="HooshvareLab/bert-fa-zwnj-base"


        # initialize embeddings
        embedding_types = [
            # WordEmbeddings('glove'),
            # comment in this line to use character embeddings
            CharacterEmbeddings(),
            # comment in these lines to use flair embeddings
            # FlairEmbeddings("news-forward"),
            # FlairEmbeddings("news-backward"),
            TransformerWordEmbeddings(embd),
            # TransformerWordEmbeddings('bert-base-cased'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type=tag_type,
            use_crf=True,
            use_rnn=True,
        )

        trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)

        trainer.train(training_dir, learning_rate=3e-5, mini_batch_size=32, max_epochs=40)
        self.model = tagger

    def predict(self, tt_sgml, modelDir, outmode="binary"):
        def is_tok(sgml_line):
            return len(sgml_line) > 0 and not (sgml_line.startswith("<") and sgml_line.endswith(">"))

        def is_sent(line):
            return line in ["<s>", "</s>"] or line.startswith("<s ")

        if self.model is None:
            self.load_model(modelDir)

        final_mapping = {}  # Map each contextualized token to its (sequence_number, position)
        spans = []  # Holds flair Sentence objects for labeling

        tt_sgml = unescape(tt_sgml)  # Splitter is trained on UTF-8 forms, since LM embeddings know characters like '&'
        lines = tt_sgml.strip().split("\n")
        toks = [l for l in lines if is_tok(l)]
        toks = [re.sub(r"\t.*", "", t) for t in toks]

        # Hack tokens up into overlapping shingles
        wraparound = toks[-self.stride_size:] + toks + toks[: self.span_size]
        idx = 0
        mapping = defaultdict(set)
        snum = 0
        while idx < len(toks):
            if idx + self.span_size < len(wraparound):
                span = wraparound[idx: idx + self.span_size]
            else:
                span = wraparound[idx:]
            sent = Sentence(" ".join(span), use_tokenizer=lambda x: x.split())
            spans.append(sent)
            for i in range(idx - self.stride_size, idx + self.span_size - self.stride_size):
                # start, end, snum
                if i >= 0 and i < len(toks):
                    mapping[i].add((idx - self.stride_size, idx + self.span_size - self.stride_size, snum))
            idx += self.stride_size
            snum += 1

        for idx in mapping:
            best = self.span_size
            for m in mapping[idx]:
                start, end, snum = m
                dist_to_end = end - idx
                dist_to_start = idx - start
                delta = abs(dist_to_end - dist_to_start)
                if delta < best:
                    best = delta
                    final_mapping[idx] = (snum, idx - start)  # Get sentence number and position in sentence

        # Predict
        preds = self.model.predict(spans)

        if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
            preds = spans

        labels = []
        for idx in final_mapping:
            snum, position = final_mapping[idx]
            if str(flair.__version__).startswith("0.4"):
                label = 0 if preds[snum].tokens[position].tags["ner"].value == "O" else 1
            else:
                label = 0 if preds[snum].tokens[position].labels[0].value == "O" else 1

            labels.append(label)

        import pdb; pdb.set_trace();
        if outmode == "binary":
            return labels

        # Generate edited XML if desired
        output = []
        counter = 0
        first = True
        for line in tt_sgml.strip().split("\n"):
            if is_sent(line):  # Remove existing sentence tags
                continue
            if is_tok(line):
                if labels[counter] == 1:
                    if not first:
                        output.append("</s>")
                    output.append("<s>")
                    first = False
                counter += 1
            output.append(line)
        output.append("</s>")  # Final closing </s>

        output = reorder("\n".join(output))

        return output.strip() + "\n"

    def split(self, context):
        xml_data = context["xml"]
        # Sometimes the tokenizer doesn't newline every elt
        xml_data = xml_data.replace("><", ">\n<")
        # Ad hoc fix for a tokenization error
        xml_data = xml_data.replace("°<", "°\n<")
        # Remove empty elements?
        # for elt in TAGS:
        #    xml_data = xml_data.replace(f"<{elt}>\n</{elt}>\n", "")

        # Search for genre in the first 2 lines (in case there's an <?xml version="1.0" ?>
        genre = re.findall(r'type="(.*?)"', "\n".join(xml_data.split("\n")[:2]))
        assert len(genre) == 1
        genre = genre[0]
        # don't feed the sentencer our pos and lemma predictions, if we have them
        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)
        split_indices = self.predict(no_pos_lemma)

        # for xml
        counter = 0
        splitted = []
        opened_sent = False
        para = True

        for line in xml_data.strip().split("\n"):
            if not is_sgml_tag(line):
                # Token
                if split_indices[counter] == 1 or para:
                    if opened_sent:
                        rev_counter = len(splitted) - 1
                        while is_sgml_tag(splitted[rev_counter]):
                            rev_counter -= 1
                        splitted.insert(rev_counter + 1, "</s>")
                    splitted.append("<s>")
                    opened_sent = True
                    para = False
                counter += 1
            elif any(f"<{elt}>" in line for elt in BLOCK_TAGS) or any(
                    f"</{elt}>" in line for elt in BLOCK_TAGS
            ):  # New block, force sentence split
                para = True
            splitted.append(line)

        if opened_sent:
            rev_counter = len(splitted) - 1
            while is_sgml_tag(splitted[rev_counter]):
                rev_counter -= 1
            splitted.insert(rev_counter + 1, "</s>")

        lines = "\n".join(splitted)
        lines = reorder(lines)
        lines = fix_malformed_sentences(lines.split("\n"))
        lines = "\n".join(lines)
        lines = reorder(lines)

        # now, we need to construct the sentences for conllu
        conllu_sentences = []
        tokens = []
        in_sent = False
        for i, line in enumerate(lines.strip().split("\n")):
            if line == "<s>":
                in_sent = True
                if len(tokens) > 0:
                    conllu_sentences.append(tokens2conllu(tokens))
                    tokens = []
            elif line == "</s>":
                in_sent = False
            elif not is_sgml_tag(line):
                if not in_sent:
                    raise Exception(f"Encountered a token '{line}' not in a sentence at line {i}")
                else:
                    tokens.append(line)
        if len(tokens) > 0:
            conllu_sentences.append(tokens2conllu(tokens))

        return {
            "xml": lines,
            "dep": "\n".join(tl.serialize() for tl in conllu_sentences),
        }

    def run(self, input_dir, output_dir):

        self.load_model()
        # Identify a function that takes data and returns output at the document level
        processing_function = self.split

        # use process_files, inherited from NLPModule, to apply this function to all docs
        self.process_files_multiformat(input_dir, output_dir, processing_function, multithreaded=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("file", help="TT SGML file to test sentence splitting on, or training dir")
    p.add_argument("-m", "--mode", choices=["test", "train"], default="test")
    p.add_argument(
        "-o",
        "--out_format",
        choices=["binary", "sgml"],
        help="output list of binary split indices or TT SGML",
        default="sgml",
    )
    p.add_argument("-p", "--partition", default="test", choices=["test", "train", "dev"],
                   help="testing input partition")

    from glob import glob

    opts = p.parse_args()
    sentencer = FlairSentSplitter()
    if opts.mode == "train":
        folders = glob(opts.file + '*/')
        # import pdb; pdb.set_trace();
        for data_dir in folders:
            sentencer.train(data_dir)
    else:
        folders = glob(opts.file + '*/')
        for data_dir in folders:
            sgml = io.open(data_dir + '/sent_' + opts.partition + '.tt', encoding="utf8").read()
            result = sentencer.predict(sgml, data_dir + 'best-model.pt', outmode=opts.out_format)
            print(result)
            with open(data_dir + '/sent_' + opts.partition + '.pred', 'w') as of:
                of.write(result)
