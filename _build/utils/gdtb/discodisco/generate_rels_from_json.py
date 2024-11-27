"""
Take data from:

- out_test.json (predictions from discodisco)
- eng.pdtb.missing_test_keys.tab (eRST source-target-label keys)
- eng.pdtb.missing_test.rels (relations in DISRPT format that discodisco predicted labels for)

and create two files in data/discodisco_preds/:

- eng.rst.gum_add.rels (DISRPT format, including previous predictions and new ones we are adding)
- eng.rst.gum_add_predictions.json (json format, one relation per line matching the .rels file exactly)
"""

import json
import os

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
discodisco_preds = os.sep.join([script_dir, "..","data","discodisco_preds"]) + os.sep

# Check that .rels file exists and has header
if not os.path.exists(discodisco_preds + "eng.rst.gum_add.rels"):
    with open(discodisco_preds + "eng.rst.gum_add.rels", "w") as f:
        f.write("doc	unit1_toks	unit2_toks	unit1_txt	unit2_txt	s1_toks	s2_toks	unit1_sent	unit2_sent	dir	rel_key	label\n")
if not os.path.exists(discodisco_preds + "eng.rst.gum_add.rels"):
    with open(discodisco_preds + "eng.rst.gum_add_predictions.json", "w") as f:
        f.write("")

existing_add_rels = open(discodisco_preds + "eng.rst.gum_add.rels").read().strip().split("\n")
existing_json_rels = open(discodisco_preds + "eng.rst.gum_add_predictions.json").read().strip().split("\n")

if not existing_add_rels[0].startswith("doc\t"):
    raise ValueError("! eng.rst.gum_add.rels is missing the header.")

if not len(existing_add_rels[1:]) == len(existing_json_rels):
    raise ValueError("! eng.rst.gum_add.rels and eng.rst.gum_add_predictions.json have different lengths.")

new_keys = open("eng.pdtb.missing_test_keys.tab").read().strip().split("\n")
new_rels = open("eng.pdtb.missing_test.rels").read().strip().split("\n")[1:]
new_json = open("out_test.json").read().strip().split("\n")

if not len(new_rels) == len(new_json):
    raise ValueError("! eng.pdtb.missing_test.rels and out_test.json have different lengths.")
if not len(new_rels) == len(new_keys):
    raise ValueError("! eng.pdtb.missing_test.rels and eng.pdtb.missing_test_keys.tab have different lengths.")

for i, (key, rel, json_entry) in enumerate(zip(new_keys, new_rels, new_json)):
    pred = json.loads(json_entry)
    doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, dir, orig_label, _ = rel.split("\t")
    label = pred["pred_relation"]
    entry = "\t".join([doc, unit1_toks, unit2_toks, unit1_txt, unit2_txt, s1_toks, s2_toks, unit1_sent, unit2_sent, dir, key, label])
    if entry not in existing_add_rels:
        existing_add_rels.append(entry)
        existing_json_rels.append(json_entry)

with open(discodisco_preds + "eng.rst.gum_add.rels", "w", encoding="utf8", newline="\n") as f:
    f.write("\n".join(existing_add_rels).strip())
with open(discodisco_preds + "eng.rst.gum_add_predictions.json", "w", encoding="utf8", newline="\n") as f:
    f.write("\n".join(existing_json_rels).strip())
