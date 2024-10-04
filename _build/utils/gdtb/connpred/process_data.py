import random
import json
import sys
from collections import defaultdict
import argparse
import random
import numpy as np
import re

class Mapper():
    def __init__(self, map_file):
        with open(map_file, 'r') as fin:
            data = json.load(fin)

        self.dm_rst_to_pdtb = data["dm@rst2pdtb"]
        self.dm_to_pdtb = data["dm2pdtb"]
        self.rst_to_pdtb = data["rst2pdtb"]
        self.dm_pdtb_to_rst = data["dm@pdtb2rst"]

        #list of all rst relations
        self.rst_list = list(set([rst for rst_vals in self.dm_pdtb_to_rst.values() for rst in rst_vals]))

        #list of all pdtb relations
        self.pdtb_list = list(set([pdtb for pdtb_vals in self.dm_to_pdtb.values() for pdtb in pdtb_vals]))

        #make pdtb_to_rst_dict
        pdtb_rst_dict = defaultdict(list)
        for k in self.dm_pdtb_to_rst:

            rst_senses = self.dm_pdtb_to_rst[k]
            pdtb = k.split("@")[1]
            pdtb_rst_dict[pdtb] += rst_senses

        self.pdtb_to_rst = {}
        for k in pdtb_rst_dict:
            self.pdtb_to_rst[k] = list(set(pdtb_rst_dict[k]))

    def _dm_pdtb_to_rst(self, dm, pdtb):
        """maps onto (ideally 1) rst relation given dm + pdtb sense. If ambiguous, need ranker
        returns: list of rst relations"""
        rst_key = dm.lower() + "@" + pdtb.lower()
        rst_sense = self.dm_pdtb_to_rst[rst_key]
        return rst_sense

    def _dm_to_pdtb(self, dm):
        """maps a dm onto set of pdtb relations"""
        pdtb_sense = self.dm_to_pdtb[dm.lower()]
        return pdtb_sense

    def _rst_to_pdtb(self, rst):
        """maps rst relation onto set of pdtb relations"""
        pdtb_sense = self.rst_to_pdtb[rst.lower()]
        return pdtb_sense

    def _dm_rst_to_pdtb(self, dm, rst):
        """given a dm and rst relation, returns possible pdtb relations"""
        pdtb_key = dm.lower + "@" + rst.lower()
        pdtb_sense = self.dm_rst_to_pdtb[pdtb_key]
        return pdtb_sense

    def _pdtb_to_rst(self, pdtb):
        """given only a pdtb relation, return list of all posible rst relations
        returns: list of rst relations"""

        rst_sense = self.pdtb_to_rst[pdtb.lower()]
        return rst_sense


def get_pdtb_pred_dict(pdtb_rst_pred_file, pdtb_rel_file):
    """
    this function creates a dictionary for accessing the rst predictions for the pdtb data.
    returns a dictionary where the key is (unit1_txt, unit2_txt) and the value is the relation probs
    """
    with open(pdtb_rel_file, 'r') as inf:
        rel_lines = inf.readlines()

    rel_lines = rel_lines[1:]
    
    prob_lines = []
    with open(pdtb_rst_pred_file) as inf2:
        for line in inf2:
            prob_d = json.loads(line.rstrip("\n"))
            probs = prob_d["relation_probs"] #this is a dictionary of the relation probabilities
            prob_lines.append(probs)

    print(len(prob_lines), len(rel_lines))
    assert len(prob_lines) == len(rel_lines)

    probs_dictionary = {} #this is the dict that will house the relation probs, and be returned

    for i in range(len(prob_lines)):
        rel_line = rel_lines[i]
        prob_d = prob_lines[i]
        doc,unit1_toks,unit2_toks,unit1_txt,unit2_txt,s1_toks,s2_toks,unit1_sent,unit2_sent,dir,orig_label,label = rel_line.rstrip("\n").split("\t")
        kee = (unit1_txt, unit2_txt)
        #print(kee, probs_dictionary.keys())
        #assert kee not in probs_dictionary
        if kee in probs_dictionary:
            old_dict = probs_dictionary[kee]
            new_dict = prob_d
            avg_dict = {k: np.mean([old_dict[k], new_dict[k]]) for k in old_dict.keys()}
            probs_dictionary[kee] = avg_dict
        if kee not in probs_dictionary:
            probs_dictionary[kee] = prob_d


    return probs_dictionary

def prepare_training_data(training_dir, output_dir, train_file, pdtb_rst_pred_file, pdtb_rel_file, Mapper):
    """generates .jsonl file for the training data. Takes one of the splits files as input"""
    count1 = 0
    count2 = 0
    input_file = training_dir + train_file
    split = "train" if "train" in train_file else "dev" if "dev" in train_file else "test"
    output_file = output_dir + "pdtb_implicit_" + split + "_disco.jsonl"
    unmapped_count = 0 #how many dm + pdtb don't map
    more_1_count = 0 #how many dm + pdtb map onto more than 1 rst
    ex_1_count = 0
    total_count = 0 #keep running total

    probs_dictionary = get_pdtb_pred_dict(pdtb_rst_pred_file, pdtb_rel_file)

    bad_counts = 0

    with open(output_file, "w") as fout:
        with open(input_file) as fin:
            next(fin) #skip heading line
            non_dm_pdtbs = []
            non_pdtbs = []
            for line in fin:
                out_dict = {}
                pdtb_rels = []
                doc, reltype, arg1, arg2, dm, label, partition = line.split("\t")

                #must take out 'arg2-as' type of things in label, they aren't in mapping
                split_label = label.lower().split(".")
                # print(split_label[-1], file=sys.stderr)
                if "arg1" in split_label[-1] or "arg2" in split_label[-1]:
                    label = ".".join(split_label[:-1])

                #these are being discarded
                if "belief" in label.lower() or "speechact" in label.lower():
                    continue



                #quick fix, need Comparison.Similarity added into mapping file!
                #fixed now
                # if label == "Comparison.Similarity":
                #     label = "Expansion.Manner"

                #lists to keep track of keys that do not have mappings

                #if dm + pdtb doesn't map onto an rst relation, then just map pdtb to rst without dm
                if label != "NONE":
                    total_count += 1
                    try:
                        #find out how many cases actually have an expansion
                        #of those, how many of them map to more than 1
                        rst_rels = Mapper._dm_pdtb_to_rst(dm, label)
                        if len(rst_rels) > 1:
                            more_1_count += 1
                        else:
                            ex_1_count += 1
                    except KeyError: #if there is no dm + pdtb match, map to rst from pdtb only
                        non_dm_pdtbs.append((dm, label))
                        try:
                            #of these how many does it fall back on
                            rst_rels = Mapper._pdtb_to_rst(label)
                            count1 += 1
                            unmapped_count += 1
                        except KeyError: #if there is still no match, resort to picking random rst
                            count2 += 1
                            print(label, dm, file=sys.stderr)
                            non_pdtbs.append(label)
                            rst_rels = [random.choice(Mapper.rst_list)]


                    try:
                        prob_dict_rel = probs_dictionary[(arg1, arg2)]
                    except:
                        try:
                            prob_dict_rel = probs_dictionary[(arg2, arg1)]
                        except:
                            arg1 = re.sub("-- ", "", arg1)
                            arg2 = re.sub("-- ", "", arg2)
                            try:
                                prob_dict_rel = probs_dictionary[(arg1, arg2)]
                            except:
                                try:
                                    prob_dict_rel = probs_dictionary[(arg2, arg1)]
                                except:
                                    arg1 = re.sub(" --", "", arg1)
                                    arg2 = re.sub(" --", "", arg2)
                                    try:
                                        prob_dict_rel = probs_dictionary[(arg1, arg2)]
                                    except:
                                        try:
                                            prob_dict_rel = probs_dictionary[(arg2, arg1)]
                                        except:
                                            print((arg2, arg1))
                                            bad_counts += 1
                                            continue

                    sorted_rels = sorted(prob_dict_rel, key=prob_dict_rel.get, reverse=True)

                    # print(prob_dict_rel[sorted_rels[0]], prob_dict_rel[sorted_rels[1]], prob_dict_rel[sorted_rels[2]])

                    chosen_rst_rel = None

                    for r in sorted_rels:
                        if r in rst_rels:
                            chosen_rst_rel = r
                            break

                    if not chosen_rst_rel:
                        chosen_rst_rel = sorted_rels[0]
                    # for rel in rst_rels: #TAKING THIS OUT BECAUSE IT OVERGENERATES RST SETS. REPLACING WITH ONE RANDOM RST CHOICE
                    #     pdtb_map = Mapper._rst_to_pdtb(rel)]
                    #     pdtb_rels += pdtb_map
                    pdtb_rels = Mapper._rst_to_pdtb(chosen_rst_rel)

                #if norel, then choose a random rst to map back to pdtb
                if label == "NONE":
                    rst_rel = random.choice(Mapper.rst_list)
                    pdtb_rels = Mapper._rst_to_pdtb(rst_rel)
                pdtb_rels = list(set(pdtb_rels))
                str_rels = ",".join(pdtb_rels)
                out_dict["input"] = "Sentence 1: " + arg1 + " Sentence 2: " + arg2 + " Relations: " + str_rels
                out_dict["output"] = dm.lower() if dm != "NONE" else "NONE"

                json_line = json.dumps(out_dict) + '\n'
                fout.write(json_line)


            #just making some quick files to view what are the dm+pdtb and pdtb tags that don't map to rst
            # non_dm_pdtbs = list(set(non_dm_pdtbs))
            # non_pdtbs = list(set(non_pdtbs))
            # with open("./non_dm_pdtb.tsv", "w") as fout2:
            #     for item in non_dm_pdtbs:
            #         print(item, file=fout2)
            #
            # with open("./non_pdtb.tsv", "w") as fout3:
            #     for item in non_pdtbs:
            #         print(item, file=fout3)


    # print("DM + PDTB mismatch", count1, file=sys.stderr)
    # print("PDTB missing", count2, file=sys.stderr)
    print("Bad keys", bad_counts, file=sys.stderr)
    print("TOTAL", total_count, file=sys.stderr)
    print("% exactly 1 pdtb+dm map", ex_1_count / total_count, file=sys.stderr)
    print("% more than 1 pdtb+dm map", more_1_count / total_count, file=sys.stderr)
    print("% no pdtb+dm map, resort to pdtb->rst map", unmapped_count / total_count, file=sys.stderr)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="training file")
    parser.add_argument("-i", help="training data dir", default="./pdtb_impl_norel_data/")
    parser.add_argument("-o", help="output dir", default="./")
    parser.add_argument("-m", help="mapping file", default="../data/mappings.json")
    parser.add_argument("-p", help="pdtb rst predictions", default="./pdtb_rst_predictions/eng.pdtb.pdtb_all_predictions.json")
    parser.add_argument("-pr", help="pdtb all rels", default="./pdtb_rst_predictions/eng.pdtb.pdtb_all_new.rels")
    args = parser.parse_args()

    Mapper1 = Mapper(args.m)
    prepare_training_data(args.i, args.o, args.t, args.p, args.pr, Mapper1)



if __name__=="__main__":
    main()