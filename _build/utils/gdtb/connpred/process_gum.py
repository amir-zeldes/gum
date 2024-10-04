import random
import json
import sys
from collections import defaultdict
import argparse


def prepare_gum_data(gum_dir, output_dir, gum_file, mapper_file):
    """generates .jsonl file for the gum testing data. Takes one of the splits files as input"""

    with open(mapper_file) as in_map:
        mapper = json.loads(in_map.read())

    print(mapper["rst2pdtb"], file=sys.stderr)
    input_file = gum_dir + gum_file
    split = "test" if "test" in gum_file else "dev" if "dev" in gum_file else "train"
    output_file = output_dir + "gum_dm_" + split + ".jsonl"
    with open(output_file, "w", encoding="utf-8") as fout:
        with open(input_file) as fin:
            next(fin) #skip heading line
            non_dm_pdtbs = []
            non_pdtbs = []
            for line in fin:
                out_dict = {}
                doc,unit1_toks,unit2_toks,unit1_txt,unit2_txt,s1_toks,s2_toks,unit1_sent,unit2_sent,dir,orig_label,label = line.split("\t")

                #map gold rst label to set of pdtb rels
                pdtb_rels = mapper["rst2pdtb"][orig_label]
                pdtb_rels = list(set(pdtb_rels))
                str_rels = ",".join(pdtb_rels)
                try:
                    assert dir == "1<2"
                except:
                    print(dir, file=sys.stderr)

                #adding in extra fields which will be useful later
                out_dict["docname"] = doc
                out_dict["direction"] = dir
                out_dict["unit1_txt"] = unit1_sent
                out_dict["unit2_txt"] = unit2_sent

                #if direction is 1<2, put unit 1 as sentence 1 in dm prediction. Otherwise reverse order for dm prediction, so that first one is always Sentence 1 for predictor
                if dir == "1<2":
                    out_dict["input"] = "Sentence 1: " + unit1_sent + " Sentence 2: " + unit2_sent + " Relations: " + str_rels
                else:
                    out_dict["input"] = "Sentence 1: " + unit2_sent + " Sentence 2: " + unit1_sent + " Relations: " + str_rels

                #this is test data so no output field
                #out_dict["output"] = "NONE"

                json_line = json.dumps(out_dict, ensure_ascii=False) + '\n'
                fout.write(json_line)
    return



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", help="gum file")
    parser.add_argument("-i", help="gum data dir", default="../data/discodisco_preds/")
    parser.add_argument("-o", help="output dir", default="./gum_test_data/")
    parser.add_argument("-m", help="mapping file", default="../data/mappings.json")
    args = parser.parse_args()

    prepare_gum_data(args.i, args.o, args.g, args.m)

if __name__ == "__main__":
    main()