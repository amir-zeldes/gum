import sys
from json import dumps

if len(sys.argv) == 1:
    print("Provide an output filename created with `allennlp predict`")
    sys.exit(1)

if len(sys.argv) == 2:
    print("Provide a path for the ensemble json output to be written to`")
    sys.exit(1)


import json

with open(sys.argv[1], encoding="utf8") as f:
    s = f.read()


def convert_label(label):
    if label == "B":
        return "B"
    elif label == "B-Conn":
        return "B"
    elif label == "I-Conn":
        return "I"
    elif label == "O":
        return "O"
    else:
        raise Exception("Unknown label: " + str(label))


with open(sys.argv[2], "w", encoding="utf8") as f:
    for line in s.strip().split("\n"):
        data = json.loads(line)
        for i, (token, pred_label) in enumerate(zip(data["tokens"], data["pred_labels"])):
            converted_label = convert_label(pred_label)
            d = {"B": 0.0, "I": 0.0, "O": 0.0}
            d[converted_label] = 1.0
            f.write(dumps(d) + "\n")
