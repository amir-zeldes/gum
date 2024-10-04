import os
import io
import re
import json
import sys
import statistics

import numpy as np
import pandas as pd
from glob import glob
from random import seed, shuffle
from collections import defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
sys.path.insert(0, script_dir + "../../")
# sys.path.append(os.path.abspath('../../'))

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from seg_scripts.seg_eval_2019_modified import get_scores

seed(42)
np.random.seed(42)


eduseg2bi = {"BeginSeg=Yes": "B", "_": "O"}
bi2eduseg = {v: k for k, v in eduseg2bi.items()}
connseg2bio = {"Seg=B-Conn": "B", "Seg=I-Conn": "I", "_": "O"}
bio2connseg = {v: k for k, v in connseg2bio.items()}


def read_json(jsonfile):
    if not os.path.exists(jsonfile):
        return None
    with io.open(jsonfile, "r", encoding="utf8") as f:
        lines = f.read().strip().split("\n")
        probas = [json.loads(line) for line in lines]
        return probas


def read_gold(goldfile):
    if not os.path.exists(goldfile):
        return None
    with io.open(goldfile, "r", encoding="utf8") as f:
        lines = f.read().strip().split("\n")
        labs = [x.split("\t")[9].strip() for x in lines if "\t" in x and "-" not in x.split("\t")[0].strip()]
        toks = [x.split("\t")[1].strip() for x in lines if "\t" in x and "-" not in x.split("\t")[0].strip()]
        assert ("BeginSeg=Yes" in labs) ^ ("Seg=B-Conn" in labs)
        if "Seg=B-Conn" in labs:
            labs = [connseg2bio[x] for x in labs]
        elif "BeginSeg=Yes" in labs:
            labs = [eduseg2bi[x] for x in labs]
    return labs, toks


def eval_gold(golds, preds, toks, corpus, method="mode"):
    conf_mat = confusion_matrix(golds, preds)
    # print(golds[:10], preds[:10])
    sys.stderr.write("\n\nConfusion matrix for corpus %s using method %s" % (corpus, method) + "\n")
    sys.stderr.write(str(conf_mat) + "\n")
    true_positive = conf_mat[1][1]
    false_positive = conf_mat[0][1]
    false_negative = conf_mat[1][0]
    prec = true_positive / (true_positive + false_positive)
    rec = true_positive / (true_positive + false_negative)
    f1 = 2 * prec * rec / (prec + rec)
    sys.stderr.write("P: " + str(prec) + "\n")
    sys.stderr.write("R: " + str(rec) + "\n")
    sys.stderr.write("F1: " + str(f1) + "\n")
    with io.open("diff_%s_%s.tab" % (corpus, method), "w", encoding="utf8") as f:
        for i in range(len(golds)):
            f.write("\t".join([toks[i], str(golds[i]), str(preds[i])]) + "\n")


def max_key(d):
    return max(d, key=d.get)


def get_polar_lab(proba):
    return "B" if proba >= 0.5 else "O"


def lr_ensemble(gold_dict, pred_dict):
    X_train = pd.DataFrame(
        list(
            zip(
                [x["B"] for x in pred_dict["allennlp_train"]],
                [x["B"] for x in pred_dict["flair_train"]],
                [x["B"] for x in pred_dict["subtree_train"]],
            )
        ),
        columns=["allennlp", "flair", "subtree"],
    )

    X_test = pd.DataFrame(
        list(
            zip(
                [x["B"] for x in pred_dict["allennlp_test"]],
                [x["B"] for x in pred_dict["flair_test"]],
                [x["B"] for x in pred_dict["subtree_test"]],
            )
        ),
        columns=["allennlp", "flair", "subtree"],
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    Y_train = [1 if x == "B" else 0 for x in gold_dict["gold_train"]]

    clf = LogisticRegressionCV(cv=10, random_state=0, solver="liblinear").fit(X_train, Y_train)
    # clf = LinearRegression(normalize=True).fit(X_train, Y_train)
    clf_pred = clf.predict(X_test)
    clf_pred = ["B" if x == 1 else "O" for x in clf_pred]
    # clf_pred_proba = clf.predict_proba(X_test)
    # print(clf_pred_proba.shape)
    # clf_pred_proba = ['B' if x >= 0.5 else 'O' for x in clf_pred_proba]
    # assert clf_pred_proba == clf_pred
    return clf_pred


def choose_ensemble_method(gold_dict, pred_dict, method="mode"):
    if method == "logisticregression":
        return lr_ensemble(gold_dict, pred_dict)

    # easy ensemble (without training on train)
    ensemble_preds = []
    for itemid in range(len(gold_dict["gold_test"])):
        if method == "mode":
            ensemble_preds.append(
                statistics.mode([max_key(pred_dict[x][itemid]) for x in pred_dict.keys() if "test" in x])
            )
        elif method == "average":
            ensemble_preds.append(
                get_polar_lab(statistics.mean([pred_dict[x][itemid]["B"] for x in pred_dict.keys() if "test" in x]))
            )
    return ensemble_preds


def generate_pred_conll(preds, toks):
    output = []
    assert len(preds) == len(toks)
    for lineid in range(len(toks)):
        output.append("_" + "\t" + toks[lineid] + "\t" + bi2eduseg[preds[lineid]])
    return "\n".join(output)


def print_scores(score_dict, corpus, method="mode"):
    # print("\n\n Scores of corpus %s using method %s" % (corpus, method))
    # print("o Total tokens: " + str(score_dict["tok_count"]))
    # print("o Gold " +score_dict["seg_type"]+": " + str(score_dict["gold_seg_count"]))
    # print("o Predicted "+score_dict["seg_type"]+": " + str(score_dict["pred_seg_count"]))
    # print("o Precision: " + str(score_dict["prec"]))
    # print("o Recall: " + str(score_dict["rec"]))
    # print("o F-Score: " + str(score_dict["f_score"]))
    print("%.4f" % score_dict["f_score"], end="\t")


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("--ensemble_json_dir", default="models/seg_ensemble_jsons/")
    p.add_argument("--data_dir", default="data/2019/")
    p.add_argument("--ensemble_method", default="average")
    p.add_argument(
        "-c",
        "--corpus",
        # default="eng.rst.rstdt",
        default="all",
    )
    opts = p.parse_args()

    if opts.corpus == "all":
        corpora = [
            "deu.rst.pcc",
            "eng.pdtb.pdtb",
            "eng.rst.gum",
            "eng.rst.rstdt",
            "eng.sdrt.stac",
            "eus.rst.ert",
            "fra.sdrt.annodis",
            "nld.rst.nldt",
            "por.rst.cstn",
            "rus.rst.rrt",
            "spa.rst.rststb",
            "spa.rst.sctb",
            "tur.pdtb.tdb",
            "zho.pdtb.cdtb",
            "zho.rst.sctb",
        ]
    else:
        corpora = [opts.corpus]

    predictors = ["allennlp", "subtree", "flair"]

    f1_corpora_dict = defaultdict(list)
    for corpus in corpora:
        if "pdtb" in corpus:
            continue
        print(corpus, end="\t")
        if corpus == "por.rst.cstn":  # corpus with sub-tokens
            print()
            continue
        gold_test_file = opts.data_dir + corpus + os.sep + corpus + "_test.tok"
        gold_test_str = io.open(gold_test_file, "r", encoding="utf8").read().strip()
        gold_test_labs, gold_test_toks = read_gold(gold_test_file)
        gold_dict = {
            "gold_train": read_gold(opts.data_dir + corpus + os.sep + corpus + "_train.tok")[0],
            "gold_test": gold_test_labs,
        }
        pred_dict = {}
        for pred in predictors:
            pred_dict[pred + "_train"] = read_json(opts.ensemble_json_dir + pred + os.sep + corpus + ".train.json")
            pred_dict[pred + "_test"] = read_json(opts.ensemble_json_dir + pred + os.sep + corpus + ".test.json")
            if not None in pred_dict.values():
                if len(pred_dict[pred + "_train"]) != len(gold_dict["gold_train"]):
                    print(corpus, pred, "train", len(pred_dict[pred + "_train"]), len(gold_dict["gold_train"]))
                if len(pred_dict[pred + "_test"]) != len(gold_dict["gold_test"]):
                    print(corpus, pred, "test", len(pred_dict[pred + "_test"]), len(gold_dict["gold_test"]))

        if None in pred_dict.values():
            print()
            # print('Skipping corpus %s due to insufficient predictions' % corpus)
            continue

        # Single predictors
        allennlp_preds = [max_key(x) for x in pred_dict["allennlp_test"]]
        subtree_preds = [max_key(x) for x in pred_dict["subtree_test"]]
        flair_preds = [max_key(x) for x in pred_dict["flair_test"]]
        allennlp_pred_conll = generate_pred_conll(allennlp_preds, gold_test_toks)
        subtree_pred_conll = generate_pred_conll(subtree_preds, gold_test_toks)
        flair_pred_conll = generate_pred_conll(flair_preds, gold_test_toks)
        allennlp_score_dict = get_scores(gold_test_str, allennlp_pred_conll, string_input=True)
        subtree_score_dict = get_scores(gold_test_str, subtree_pred_conll, string_input=True)
        flair_score_dict = get_scores(gold_test_str, flair_pred_conll, string_input=True)
        print_scores(allennlp_score_dict, corpus=corpus, method="single_allennlp")
        print_scores(subtree_score_dict, corpus=corpus, method="single_subtree")
        print_scores(flair_score_dict, corpus=corpus, method="single_flair")
        f1_corpora_dict["allennlp"].append(allennlp_score_dict["f_score"])
        f1_corpora_dict["flair"].append(flair_score_dict["f_score"])
        f1_corpora_dict["subtree"].append(subtree_score_dict["f_score"])

        # eval_gold(gold_dict["gold_test"], allennlp_preds, gold_test_toks, corpus=corpus, method="single_allennlp")
        # eval_gold(gold_dict["gold_test"], subtree_preds, gold_test_toks, corpus=corpus, method="single_subtree")
        # eval_gold(gold_dict["gold_test"], flair_preds, gold_test_toks, corpus=corpus, method="single_flair")

        # Start Ensemble
        ensemble_preds = choose_ensemble_method(gold_dict, pred_dict, method=opts.ensemble_method)
        ensemble_pred_conll = generate_pred_conll(ensemble_preds, gold_test_toks)
        ensemble_score_dict = get_scores(gold_test_str, ensemble_pred_conll, string_input=True)
        print_scores(ensemble_score_dict, corpus=corpus, method=opts.ensemble_method)
        f1_corpora_dict["ensemble"].append(ensemble_score_dict["f_score"])
        print()

    for k in f1_corpora_dict.keys():
        print("Avg score for %s is %.4f" % (k, statistics.mean(f1_corpora_dict[k])))

    print("Done!")
