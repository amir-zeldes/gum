from glob import glob
from argparse import ArgumentParser
import json

if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-f", "--file", help="dir containing subdirectories for each languages, including sent_test.pred "
                                        "file")
    p.add_argument("-m", "--mode", help="train/test/dev",
                   default="dev")
    p.add_argument("-i", "--inf", help="dir containing doc info, output of get_docs_inof.py")
    opts = p.parse_args()

    folders = glob(opts.file + '/' + '*/')
    for data_dir in folders:
        print(data_dir)
        sents = []
        begin = True
        with open(data_dir + '/sent_' + opts.mode + '.pred', 'r') as inp:
            for line in inp:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                if begin and line != "<s>":
                    sents.append("<s>")
                begin = False
                sents.append(line)
        new_sents = []
        count = 0
        name = data_dir.split('/')[-2]
        print('\n'+name+'\n')
        tok_index = 0
        doc_index = 0
        with open(opts.inf + '/' + name + '/docs_tokens_' + opts.mode + '.json') as f:
            inf = json.load(f)
            doc_count=0
        for j in range(len(sents)):
            if sents[j]=="<s>":
                count=0
            #if sents[j]=="Oftmals":
             #   import pdb; pdb.set_trace();
            if j!=len(sents)-1:
                if doc_count==len(inf['toks'][doc_index])-1 and sents[j+1] != "</s>" and sents[j]!="</s>" and sents[j]!="<s>":
                    doc_count=0
                    doc_index+=1
                    new_sents.append(sents[j])
                    new_sents.append("</s>")
                    new_sents.append("<s>")
                    count=0
                    continue
            if count == 256 and sents[j] != "</s>":
                print(">256")
                new_sents.append("</s>")
                new_sents.append("<s>")
                count = 0

            new_sents.append(sents[j])
            count += 1
            if sents[j]!= "<s>" and sents[j]!= "</s>":
                doc_count+=1
            if doc_count==len(inf['toks'][doc_index]):
                doc_count=0
                doc_index+=1
                

        with open(data_dir + '/sent_' + opts.mode + '.predV2', 'w') as out:
            for s in new_sents:
                out.write(s + '\n')
