# from seg_eval import get_scores
from json import dumps
from six import iterkeys
from glob import glob
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, BertEmbeddings, XLNetEmbeddings
from flair.models import SequenceTagger
import flair
import torch

import os, sys, io, numpy as np
from random import seed, shuffle

from torch.utils.data import Dataset

seed(42)
np.random.seed(42)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep


from collections import OrderedDict, defaultdict


def is_sgml_tag(line):
    return line.startswith("<") and line.endswith(">")


def unescape(token):
    token = token.replace("&quot;", '"')
    token = token.replace("&lt;", "<")
    token = token.replace("&gt;", ">")
    token = token.replace("&amp;", "&")
    token = token.replace("&apos;", "'")
    return token


class FlairEDUSplitter:
    def __init__(self, corpus="eng.rst.gum", auto="", model_path=None, span_size=6):

        self.name = "FlairEDUSplitter"
        self.auto = auto
        # Number of tokens to include as pre/post context around each sentence
        self.span_size = span_size
        # Numerical stride size only needed for sent mode; EDU mode strides by sentences
        self.corpus = corpus
        if model_path is not None:
            try:
                self.load_model(model_path)
            except FileNotFoundError:
                self.model = None
        else:
            self.model = None

    def load_model(self, path=None):

        path = data_dir + "best-model.pt"
        self.model = SequenceTagger.load(path)

    def make_flair_data(self, multitrain=False):
        def conll2sents(conll):
            conll_toks = []
            segs = []
            sents = conll.split("\n\n")
            sent_tokens = defaultdict(list)
            for i, sent in enumerate(sents):
                toks = []
                for line in sent.split("\n"):
                    if "\t" in line:
                        fields = line.split("\t")
                        if "-" in fields[0]:
                            continue
                        toks.append(fields[1])
                        conll_toks.append(fields[1])
                        if "BeginSeg" in fields[-1]:
                            segs.append("B-SEG")
                        else:
                            segs.append("O")
                sent_tokens[i] = toks

            output = []
            counter = 0
            for i, sent in enumerate(sents):
                prev_s = (
                    sent_tokens[i - 1] if i > 0 else sent_tokens[len(sents) - 1]
                )  # Use last sent as prev if this is sent 1
                pre_context = prev_s[-6:] if len(prev_s) > 5 else prev_s[:]
                pre_context.append("<pre>")
                next_s = (
                    sent_tokens[i + 1] if i < len(sents) - 1 else sent_tokens[0]
                )  # Use first sent as next if this is last sent
                post_context = next_s[:6] if len(next_s) > 5 else next_s[:]
                post_context = ["<post>"] + post_context

                for tok in pre_context:
                    output.append(tok + "\t" + "O")
                for tok in sent_tokens[i]:
                    output.append(tok + "\t" + segs[counter])
                    counter += 1
                for tok in post_context:
                    output.append(tok + "\t" + "O")
                output.append("")
            return "\n".join(output)

        for file_ in glob(shared_task_dir + self.corpus + os.sep + "*.conll"):
            conll = io.open(file_, encoding="utf8").read().strip()
            output = conll2sents(conll)

            outfile = data_dir + os.path.basename(file_).replace(".conll", ".txt")
            os.makedirs(data_dir, exist_ok=True)
            with io.open(outfile, "w", encoding="utf8", newline="\n") as f:
                f.write(output)

        # Make multitrain if needed
        if multitrain:
            conll_sents = (
                io.open(shared_task_dir + self.corpus + os.sep + self.corpus + "_train.conll", encoding="utf8")
                .read()
                .strip()
                .split("\n\n")
            )
            BIO_sents = io.open(data_dir + self.corpus + "_train.txt", encoding="utf8").read().strip().split("\n\n")
            sent_ids = list(range(len(conll_sents)))
            shuffle(sent_ids)

            parts = np.array_split(sent_ids, 5)

            fold_ids = {}
            for fid, part in enumerate(parts):
                for snum, sent in enumerate(part):
                    fold_ids[(fid, snum)] = sent

            for i in range(len(parts)):
                conll_fold = "\n\n".join([conll_sents[j] for j in parts[i]])
                BIO_fold = "\n\n".join([BIO_sents[j] for j in parts[i]])
                with io.open(
                    data_dir + self.corpus + "_train_fold" + str(i + 1) + ".conll", "w", encoding="utf8", newline="\n"
                ) as f:
                    f.write(conll_fold + "\n")
                with io.open(
                    data_dir + self.corpus + "_train_fold" + str(i + 1) + ".txt", "w", encoding="utf8", newline="\n"
                ) as f:
                    f.write(BIO_fold + "\n")

            return fold_ids

    def train(self, training_dir=None, multitrain=False, ensemble_json_dir=None, embeddings_storage_mode="cpu"):

        from flair.trainers import ModelTrainer

        if training_dir is None:
            training_dir = data_dir

        # define columns
        columns = {0: "text", 1: "ner"}

        # make flair_edu_segmenter style training data from shared task format
        fold_sent_to_order = self.make_flair_data(multitrain=multitrain)

        iters = 5 if multitrain else 1

        # place holder for multitrain predictions
        solutions = defaultdict(list)

        for i in range(iters):

            if multitrain:
                sys.stderr.write("o training on fold " + str(i + 1) + "/" + str(iters) + "\n")
                folds = list(range(iters))
                folds.remove(i)
                train_folds = [
                    io.open(data_dir + self.corpus + "_train_fold" + str(j + 1) + ".txt").read().strip() for j in folds
                ]
                train_data = "\n\n".join(train_folds)
                with io.open(data_dir + self.corpus + "_train.txt", "w", encoding="utf8", newline="\n") as f:
                    f.write(train_data)

            # init a corpus using column format, data folder and the names of the train, dev and test files
            corpus: Corpus = ColumnCorpus(
                data_dir,
                columns,
                train_file=self.corpus + "_dev.txt",
                test_file=self.corpus + "_test.txt",
                dev_file=self.corpus + "_train.txt",
            )

            print(corpus)

            tag_type = "ner"
            tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
            print(tag_dictionary)

            # initialize embeddings
            if not self.corpus.startswith("eng."):
                # TODO: choose non English embeddings
                embedding_types = [BertEmbeddings("distilbert-base-multilingual-cased")]
            else:
                embedding_types = [
                    # WordEmbeddings('glove'),
                    # CharacterEmbeddings(),
                    # FlairEmbeddings("news-forward"),
                    # FlairEmbeddings("news-backward"),
                    BertEmbeddings("distilbert-base-cased")
                    # XLNetEmbeddings('xlnet-large-cased')
                ]

            embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

            tagger: SequenceTagger = SequenceTagger(
                hidden_size=100, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=True,
            )

            from torch.optim.adagrad import Adagrad
            from torch.optim.adam import Adam

            trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)  # Adagrad)

            trainer.train(
                training_dir,
                mini_batch_size=4,
                mini_batch_chunk_size=1,
                max_epochs=20,
                embeddings_storage_mode=embeddings_storage_mode,
                # learning_rate=0.1,
                learning_rate=0.0010,
                # weight_decay=1e-4
                patience=3,
            )
            self.model = tagger

            conll_file = shared_task_dir + splitter.corpus + os.sep + splitter.corpus + "_train.conll"
            with open(conll_file, "r") as f:
                self.predict(f.read(), ensemble_json_dir=ensemble_json_dir, split="train")
            conll_file = shared_task_dir + splitter.corpus + os.sep + splitter.corpus + "_dev.conll"
            with open(conll_file, "r") as f:
                self.predict(f.read(), ensemble_json_dir=ensemble_json_dir, split="dev")

            if multitrain:
                conll_test = io.open(
                    data_dir + self.corpus + "_train_fold" + str(i + 1) + ".conll", encoding="utf8"
                ).read()
                preds = self.predict(conll_test)
                preds = list(preds)
                pred_num = 0
                for snum, sent in enumerate(conll_test.strip().split("\n\n")):
                    toks = [l.split("\t") for l in sent.split("\n") if "\t" in l]
                    toks = [t for t in toks if "." not in t[0] and "-" not in t[0]]
                    for tok in toks:
                        pred = preds[pred_num]
                        sid = fold_sent_to_order[(i, snum)]
                        solutions[sid].append(str(pred[0]) + "\t" + str(pred[1]))
                        pred_num += 1

        if multitrain:
            # Serialize multitrain predictions in correct order
            output = []
            for snum in sorted(list(iterkeys(solutions))):
                output += solutions[snum]
            with io.open(
                script_dir + os.sep + "multitrain" + os.sep + self.name + self.auto + "_" + self.corpus,
                "w",
                encoding="utf8",
                newline="\n",
            ) as f:
                f.write("\n".join(output) + "\n")

    def predict_cached(self, train=None):
        pairs = (
            io.open(script_dir + "multitrain" + os.sep + self.name + self.auto + "_" + self.corpus).read().split("\n")
        )
        preds = [(int(pr.split()[0]), float(pr.split()[1])) for pr in pairs if "\t" in pr]
        return preds

    def predict(self, conll_in, eval_gold=False, as_text=True, ensemble_json_dir=None, split=None):

        if self.model is None:
            self.load_model()

        final_mapping = {}  # Map each contextualized token to its (sequence_number, position)
        spans = []  # Holds flair Sentence objects for labeling

        # Do EDU segmentation, input TT SGML has sentence tags like '\n</s>\n'
        sents = conll_in.strip().split("\n\n")
        sent_tokens = defaultdict(list)
        for i, sent in enumerate(sents):
            toks = []
            for line in sent.split("\n"):
                if "\t" in line:
                    fields = line.split("\t")
                    if "-" not in fields[0] and "." not in fields[0]:
                        tok = fields[1]
                        toks.append(tok)
            sent_tokens[i] = toks

        counter = 0
        for i, sent in enumerate(sents):
            span = []
            # Use last sent as prev if this is sent 1
            prev_s = sent_tokens[i - 1] if i > 0 else sent_tokens[len(sents) - 1]
            pre_context = prev_s[-self.span_size :] if len(prev_s) > 5 else prev_s[:]
            pre_context.append("<pre>")
            # Use first sent as next if this is last sent
            next_s = sent_tokens[i + 1] if i < len(sents) - 1 else sent_tokens[0]
            post_context = next_s[: self.span_size] if len(next_s) > 5 else next_s[:]
            post_context = ["<post>"] + post_context

            for tok in pre_context:
                span.append(tok)
            for j, tok in enumerate(sent_tokens[i]):
                span.append(tok)
                # The aligned prediction will be in Sentence i, at the position
                # after span_size + 1 (pre-context + <pre> token) + counter
                final_mapping[counter] = (i, len(pre_context) + j)
                counter += 1
            for tok in post_context:
                span.append(tok)
            spans.append(Sentence(" ".join(span), use_tokenizer=lambda x: x.split()))

        # Predict
        preds = self.model.predict(spans)

        if preds is None:  # Newer versions of flair have void predict method, use modified Sentence list
            preds = spans

        labels = []
        probas = []
        output_json_format = []
        for idx in final_mapping:
            snum, position = final_mapping[idx]
            if str(flair.__version__).startswith("0.4"):
                label = 0 if preds[snum].tokens[position].tags["ner"].value == "O" else 1
                score = preds[snum].tokens[position].tags["ner"].score
            else:
                label = 0 if preds[snum].tokens[position].labels[0].value == "O" else 1
                score = preds[snum].tokens[position].labels[0].score

            score = 1 - score if label == 0 else score
            output_json = {"B": score, "I": 0.0, "O": 1 - score}

            labels.append(label)
            probas.append(score)
            output_json_format.append(output_json)

        # write to json format per token
        if ensemble_json_dir is not None:
            os.makedirs(ensemble_json_dir, exist_ok=True)
            with io.open(
                ensemble_json_dir + os.sep + self.corpus + (f".{split}" if split is not None else "") + ".json",
                "w",
                newline="\n",
            ) as fout:
                for d in output_json_format:
                    fout.write(dumps({k: float(v) for k, v in d.items()}) + "\n")

        return zip(labels, probas)


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("--mode", choices=["test", "train", "multitrain"], default="test")
    p.add_argument("-c", "--corpus", default="eng.rst.gum")
    p.add_argument("-m", "--multitrain", action="store_true", help="perform 5-fold training")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--ensemble_json_dir", default="models/seg_ensemble_jsons/flair")

    opts = p.parse_args()

    model_dir = opts.model_dir + os.sep
    data_dir = model_dir + "flair" + os.sep + opts.corpus + os.sep
    shared_task_dir = "data" + os.sep + "2019" + os.sep

    splitter = FlairEDUSplitter(corpus=opts.corpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_storage_mode = "gpu" if torch.cuda.is_available() else "cpu"
    print("o The device is: ", device, "; embedding storage mode is: ", embeddings_storage_mode)
    flair.device = device

    if opts.mode == "train":
        splitter.train(
            multitrain=opts.multitrain,
            ensemble_json_dir=opts.ensemble_json_dir,
            embeddings_storage_mode=embeddings_storage_mode,
        )
    elif opts.mode == "multitrain":
        splitter = FlairEDUSplitter(corpus=opts.corpus, model_path=data_dir + "best-model.pt")
        conll_file = shared_task_dir + splitter.corpus + os.sep + splitter.corpus + "_train.conll"
        conll = io.open(conll_file, encoding="utf8").read()
        preds = splitter.predict(conll, ensemble_json_dir=opts.ensemble_json_dir)
        output = []
        for pred in preds:
            output.append(str(pred[0]) + "\t" + str(pred[1]))
        with io.open(
            script_dir + os.sep + "multitrain" + os.sep + splitter.name + splitter.auto + "_" + splitter.corpus,
            "w",
            encoding="utf8",
            newline="\n",
        ) as f:
            f.write("\n".join(output) + "\n")

    else:
        conll_file = shared_task_dir + splitter.corpus + os.sep + splitter.corpus + "_test.conll"
        conll = io.open(conll_file, encoding="utf8").read()
        preds = splitter.predict(conll, ensemble_json_dir=opts.ensemble_json_dir, split="test")
        preds = list(preds)
        output = []
        tok_num = 0
        for line in conll.split("\n"):
            if "\t" in line:
                fields = line.split("\t")
                if "-" in fields[0] or "." in fields[0]:
                    continue
                seg = "BeginSeg=Yes" if preds[tok_num][0] == 1 else "_"
                output.append("_" + "\t" + fields[1] + "\t" + seg)
                tok_num += 1

        # scores = get_scores(conll,"\n".join(output),string_input=True)
        # print(scores)
