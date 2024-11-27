from glob import glob
from sklearn.metrics import classification_report

if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument("file", help="path to dir containing flair training outputs (test.tsv)")
    p.add_argument("-p", "--partition", default="test", choices=["test", "train", "dev"],
                               help="testing input partition")
    opts = p.parse_args()

    folders = glob(opts.file + '/*/')
    for data_dir in folders:
        gold_tags = []
        pred_tags = []
        toks = []
        sgml = ''
        print(data_dir)
        with open(data_dir + '/'+opts.partition + '.tsv', "r") as sg:
            for line in sg:
                line = line.rstrip()
                if line:
                    if not line.startswith("-DOCSTART"):
                        pt = line.split(' ')
                        toks.append(pt[0])
                        gold_tags.append(pt[1])
                        pred_tags.append(pt[2])
            for i in range(len(toks)):
                if pred_tags[i] == "B-SENT":
                    if i != 0:
                        sgml += "</s>\n"
                    sgml += "<s>\n"
                sgml += toks[i] + '\n'

        with open(data_dir + '/sent_'+opts.partition + '.pred', 'w') as of:
            of.write(sgml)
        print(set(pred_tags))
        print("Scores:")
        print(classification_report(gold_tags, pred_tags))
