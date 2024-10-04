import io
import os
import json
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-d", "--dir", help="path to directory which ONLY containing data folders")
p.add_argument("-o", "--out", help="path to output directory")
opts = p.parse_args()

DATA_DIR = opts.dir
OUT_DIR = opts.out
names = os.listdir(DATA_DIR + "/")
partition = ["_train", "_dev","_test"]
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)
for n in names:
    if not os.path.isdir(DATA_DIR + n + "/"):
        continue
    file_ = DATA_DIR + n + "/" + n
    for p in partition:
        file = file_ + p + ".conllu"
        doc_names = []
        all_tokens = []
        lines = io.open(file, encoding="utf8").read().strip().split("\n")
        for line in lines:
            if line.startswith("# newdoc id") or line.startswith("#newdoc id"):
                doc_names.append(line.split(' = ')[1].strip())
                all_tokens.append([])
            if "\t" in line:
                fields = line.split("\t")
                if "-" in fields[0]:
                    continue
                word = fields[1]
                all_tokens[-1].append(word)

        path = OUT_DIR + n + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        data = {'docs': doc_names, 'toks': all_tokens}

        with open(path + 'docs_tokens' + p + '.json', 'w') as f:
            json.dump(data, f)