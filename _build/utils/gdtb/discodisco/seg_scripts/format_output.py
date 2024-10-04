import sys

if len(sys.argv) == 1:
    print("Provide an output filename created with `allennlp predict`")
    sys.exit(1)

if len(sys.argv) == 2:
    print("Provide a path for the conll output to be written to`")
    sys.exit(1)


import json

with open(sys.argv[1], encoding="utf8") as f:
    s = f.read()


def convert_label(label):
    if label == "B":
        return "BeginSeg=Yes"
    elif label == "B-Conn":
        return "Seg=B-Conn"
    elif label == "I-Conn":
        return "Seg=I-Conn"
    elif label == "O":
        return "_"
    else:
        return label


with open(sys.argv[2], "w", encoding="utf8") as f:
    for line in s.strip().split("\n"):
        data = json.loads(line)
        for i, (token, pred_label) in enumerate(zip(data["tokens"], data["pred_labels"])):
            converted_label = convert_label(pred_label)
            f.write(f"{i+1}\t{token}" + ("\t_" * 7) + "\t" + converted_label + "\n")
        f.write("\n")
