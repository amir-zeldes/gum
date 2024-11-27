import io
import os
from collections import defaultdict
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-d", "--dir", help="path to directory which ONLY containing data folders")
p.add_argument("-o", "--out", help="path to output directory")
opts = p.parse_args()

DATA_DIR = opts.dir
OUT_DIR = opts.out

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

names = os.listdir(DATA_DIR + "/")
partition = ["_train", "_dev", "_test"]

for n in names:
    if not os.path.isdir(DATA_DIR + n + "/"):
        continue
    file_ = DATA_DIR + n + "/" + n
    data = defaultdict(list)
    for p in partition:
        file = file_ + p + ".conllu"
        lines = io.open(file, encoding="utf8").read().strip().split("\n")

        for line in lines:
            if line.startswith("# newdoc id") or line.startswith("#newdoc id"):
                sent = "B-SENT"
                data[p].append("-DOCSTART- X")
                data[p].append("")
                counter = 0
            if len(line.strip()) == 0:
                sent = "B-SENT"
                data[p].append("")
            if "\t" in line:
                fields = line.split("\t")
                if "-" in fields[0]:
                    continue
                word = fields[1]
                pos = fields[4]
                data[p].append(word + " " + sent)
                sent = "O"
                counter += 1
                if counter == 21:
                    data[p].append("")
                    data[p].append("-DOCSTART- X")
                    data[p].append("")
                    counter = 0

        data[p].append("")

    path = OUT_DIR + n + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    for d in data:
        lines = data[d]
        with io.open(path + "sent" + d + ".txt", "w", encoding="utf8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
