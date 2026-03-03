"""
Script to clear all relation probability and connective predictions. Only run this if you need a fresh prediction
for all data, for example because underlying RST data has changed substantially (EDU boundaries/numbers) and cached
predictions refer to stale identifiers.
"""
import os
from glob import glob
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument("-m","--missing",action="store_true", help="only clear missing entry indices, not preds")

args = p.parse_args()

disco_files = ["eng.pdtb.missing_test_keys.tab", "eng.pdtb.missing_test.rels", "eng.pdtb.missing_test.conllu"]
disco_dir = "discodisco" + os.sep

# Replace each file with an empty file
for file_name in disco_files:
    file_path = disco_dir + file_name
    with open(file_path, 'w') as f:
        pass  # Just open and close to create an empty file


if not args.missing:
    # Empty *.json and *.rels in preds directory
    disco_pred_dir = "data" + os.sep + "discodisco_preds"

    for file_path in glob(disco_pred_dir + os.sep + "*.json") + glob(disco_pred_dir + os.sep + "*.rels"):
        with open(file_path, 'w') as f:
            pass  # Just open and close to create an empty file

    conn_preds_dir = "data" + os.sep + "connector_preds"

    for file_path in glob(conn_preds_dir + os.sep + "*.jsonl"):
        with open(file_path, 'w') as f:
            pass  # Just open and close to create an empty file