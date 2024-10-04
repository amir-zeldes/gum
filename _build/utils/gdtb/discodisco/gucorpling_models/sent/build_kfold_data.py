import io
import os
import json
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-d", "--dir", help="path to directory which ONLY containing data folders")
p.add_argument("-i", "--inf", help="path to output directory from get_docs_info.py")
p.add_argument("-o", "--out", help="path to output directory for folds")
opts = p.parse_args()

DATA_DIR = opts.dir
D_DIR = opts.out
S_DIR = opts.inf

names = os.listdir(DATA_DIR + "/")
p = "_train"
folds = {}
for n in names:
    if not os.path.isdir(DATA_DIR + n + "/"):
        continue
    file_ = DATA_DIR + n + "/" + n
    file = file_ + p + ".conllu"
    path = S_DIR + n + "/"
    with open(path + 'docs_tokens' + p + '.json') as f:
        data = json.load(f)
    docs_len = len(data['docs'])
    n_folds = 4
    d = int(docs_len / n_folds)
    for i in range(n_folds):
        if not os.path.exists(D_DIR + str(i)):
            os.makedirs(D_DIR + str(i))
        dest = D_DIR + str(i) + '/' + n + '/'
        os.makedirs(dest)
        os.system("cp " + file_ + "_dev.conllu " + dest)
        if i == n_folds-1:
            test_docs = data['docs'][d * i:]
        else:
            test_docs = data['docs'][d * i:d * (i + 1)]

        lines = io.open(file, encoding="utf8").read().strip().split("\n")
        with io.open(dest + n + "_train.conllu", "w", encoding="utf8", newline="\n") as ftrain:
            with io.open(dest + n + "_test.conllu", "w", encoding="utf8", newline="\n") as ftest:
                for line in lines:
                    if line.startswith("# newdoc id") or line.startswith("#newdoc id"):
                        name = line.split(' = ')[1].strip()
                        if name in test_docs:
                            hnd = ftest
                        else:
                            hnd = ftrain
                    hnd.write(line + '\n')
