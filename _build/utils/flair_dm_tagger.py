"""
flair_dm_tagger.py

This module trains flair sequence labelers to predict DMs in BIO encoding.
If using to predict RST DMs, remember to run PDTB generation with all_dms=True,
otherwise items like "than" will be excluded.
"""


from argparse import ArgumentParser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, BertEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
import os, sys, io
from glob import glob

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

ud_dev = ["GUM_interview_cyclone", "GUM_interview_gaming",
		  "GUM_news_iodine", "GUM_news_homeopathic",
		  "GUM_voyage_athens", "GUM_voyage_coron",
		  "GUM_whow_joke", "GUM_whow_overalls",
		  "GUM_bio_byron", "GUM_bio_emperor",
		  "GUM_fiction_lunre", "GUM_fiction_beast",
		  "GUM_academic_exposure", "GUM_academic_librarians",
		  "GUM_reddit_macroeconomics", "GUM_reddit_pandas",  # Reddit
		  "GUM_speech_impeachment", "GUM_textbook_labor",
		  "GUM_vlog_radiology", "GUM_conversation_grounded",
		  "GUM_textbook_governments", "GUM_vlog_portland",
		  "GUM_conversation_risk", "GUM_speech_inauguration",
		  "GUM_court_loan","GUM_essay_evolved",
		  "GUM_letter_arendt","GUM_podcast_wrestling"]
ud_test = ["GUM_interview_libertarian", "GUM_interview_hill",
		   "GUM_news_nasa", "GUM_news_sensitive",
		   "GUM_voyage_oakland", "GUM_voyage_vavau",
		   "GUM_whow_mice", "GUM_whow_cactus",
		   "GUM_fiction_falling", "GUM_fiction_teeth",
		   "GUM_bio_jespersen", "GUM_bio_dvorak",
		   "GUM_academic_eegimaa", "GUM_academic_discrimination",
		   "GUM_reddit_escape", "GUM_reddit_monsters",  # Reddit
		   "GUM_speech_austria", "GUM_textbook_chemistry",
		   "GUM_vlog_studying", "GUM_conversation_retirement",
		   "GUM_textbook_union", "GUM_vlog_london",
		   "GUM_conversation_lambada", "GUM_speech_newzealand",
		   "GUM_court_mitigation","GUM_essay_fear",
		   "GUM_letter_mandela","GUM_podcast_bezos"]


def make_data(corpus="gum",tag="dm"):
    if corpus == "gum":
        gum_target = script_dir + ".." +  os.sep + "target" + os.sep
    else:
        raise NotImplementedError("Only GUM supported for now")

    files = glob(gum_target + "rst" + os.sep + "disrpt" + os.sep +  "eng.pdtb*.conllu")
    output = {"train":[],"test":[],"dev":[]}

    docname = ""
    for file_ in files:
        lines = io.open(file_,encoding="utf8").readlines()
        for line in lines:
            if "newdoc" in line:
                docname = line.split("=")[1].strip()
            if line.startswith("#"):
                continue
            partition = "test" if docname in ud_test else "dev" if docname in ud_dev else "train"
            if "\t" in line:
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    continue
                label = "O"
                if "Seg=B-Conn" in fields[-1]:
                    label = "B-Conn"
                elif "Seg=I-Conn" in fields[-1]:
                    label = "I-Conn"
                output[partition].append(fields[1] + "\t" + label)
            elif len(line.strip()) == 0:
                if output[partition][-1] != "":
                    output[partition].append("")
    with io.open("dm-dependencies" + os.sep + corpus + "_"+tag+"_train.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write("\n".join(output["train"]).strip() + "\n\n")
    with io.open("dm-dependencies" + os.sep + corpus + "_"+tag+"_dev.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write("\n".join(output["dev"]).strip() + "\n\n")
    with io.open("dm-dependencies" + os.sep + corpus + "_"+tag+"_test.txt", 'w', encoding="utf8",newline="\n") as f:
        f.write("\n".join(output["test"]).strip() + "\n\n")

def train(corpus="gum",tag="dm"):
    # Prevent CUDA Launch Failure random error, but slower:
    import torch
    #torch.backends.cudnn.enabled = False
    # Or:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 1. get the corpus
    # this is the folder in which train, test and dev files reside
    data_folder = "dm-dependencies"  + os.sep

    # init a corpus using column format, data folder and the names of the train, dev and test files

    # define columns
    columns = {0: "text", 1: "dm"}

    make_data(corpus=corpus)

    corpus: Corpus = ColumnCorpus(
        data_folder, columns,
        train_file=corpus+"_"+tag+"_train.txt",
        test_file=corpus+"_"+tag+"_test.txt",
        dev_file=corpus+"_"+tag+"_dev.txt",
    )

    # 2. what tag do we want to predict?
    tag_type = 'dm'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    # 4. initialize embeddings
    electra = TransformerWordEmbeddings('google/electra-base-discriminator')
    embedding_types = [

        #WordEmbeddings('glove'),

        # comment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        #FlairEmbeddings('news-forward'),
        #FlairEmbeddings('news-backward'),
        #BertEmbeddings('distilbert-base-cased')
        electra
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger: SequenceTagger = SequenceTagger(hidden_size=128,
                                            embeddings=electra, #embeddings, #electra,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True,
                                            use_rnn=False)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train("dm-dependencies" + os.sep + 'flair_tagger',
                  learning_rate=0.1,
                  mini_batch_size=15,
                  max_epochs=30)

def predict(corpus="gum",tag="dm",in_format="flair",out_format="conllu", in_path=None, preloaded=None, as_text=False):

    if preloaded is None:
        model_name = "dm-dependencies" + os.sep +"flair_tagger/best-model_"+tag+"_"+corpus+".pt"
        model = SequenceTagger.load(model_name)
    else:
        model = preloaded

    if in_path is None:
        in_path = "dm-dependencies" + os.sep + "GUM_dm_dev.txt"
    if as_text:
        dev = in_path
    else:
        dev = io.open(in_path,encoding="utf8").read()
    sents = []
    words = []
    annos = []
    true_tags = []
    true_pos = []
    for line in dev.split("\n"):
        if len(line.strip())==0:
            if len(words) > 0:
                sents.append(Sentence(" ".join(words),use_tokenizer=lambda x:x.split(" ")))
                words = []
        else:
            if "\t" in line:
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    if in_format == "conllu":
                        continue
                if in_format == "flair":
                    words.append(line.split("\t")[0])
                    true_tags.append(line.split("\t")[1])
                else:
                    words.append(line.split("\t")[1])
                    if "Seg=B-Conn" in line:
                        true_tag = "B-Conn"
                    elif "Seg=I-Conn" in line:
                        true_tag = "I-Conn"
                    else:
                        true_tag = "O"
                    true_tags.append(true_tag)

    # predict tags and print
    model.predict(sents, all_tag_prob=True)

    preds = []
    scores = []
    words = []
    for i, sent in enumerate(sents):
        for tok in sent.tokens:
            pred = tok.labels[0].value
            score = str(tok.labels[0].score)
            preds.append(pred)
            scores.append(score)
            words.append(tok.text)

    do_postprocess = False
    if do_postprocess:
        preds, scores = post_process(words, preds, scores)

    toknum = 0
    output = []
    for i, sent in enumerate(sents):
        tid=1
        if i>0 and out_format=="conllu":
            output.append("")
        for tok in sent.tokens:
            pred = preds[toknum]
            score = str(scores[toknum])
            if len(score)>5:
                score = score[:5]
            if out_format == "conllu":
                fields = [str(tid),tok.text,"_",pred,"_","_","_","_","_"]
                output.append("\t".join(fields))
                tid+=1
            elif out_format == "xg":
                output.append("\t".join([pred, tok.text, score]))
            else:
                true_tag = true_tags[toknum]
                corr = "T" if true_tag == pred else "F"
                output.append("\t".join([pred, true_tag, corr, score, tok.text, true_pos[toknum]]))
            toknum += 1

    ext = tag + ".conllu" if out_format == "conllu" else "txt"
    partition = "test" if "test" in in_path else "dev"
    if as_text:
        return "\n".join(output)
    else:
        with io.open("dm-dependencies" +os.sep + "flair-"+corpus+"-" + tag + "-"+partition+"-pred." + ext,'w',encoding="utf8",newline="\n") as f:
            f.write("\n".join(output))


def post_process(word_list, pred_list, score_list, softmax_list=None):
    """
    Implement a subset of closed-class words that can only take one of their attested closed class POS tags
    """
    output = []

    closed = {"except":["IN"],
              "or":["CC"],
              "another":["DT"],
              "be":["VB"]
              }
    # case marking VVG can never be IN:
    vbg_preps = {("including","IN"):"VBG",("according","IN"):"VBG",("depending","IN"):"VBG",("following","IN"):"VBG",("involving","IN"):"VBG",
                 ("regarding","IN"):"VBG",("concerning","IN"):"VBG"}

    top100 = {",":",",".":".","of":"IN","is":"VBZ","you":"PRP","for":"IN","was":"VBD","with":"IN","The":"DT","are":"VBP",")":"-RRB-","(":"-LRB-","at":"IN","this":"DT","from":"IN","or":"CC","not":"RB","his":"PRP$","they":"PRP","an":"DT","we":"PRP","n't":"RB","he":"PRP","[":"-LRB-","]":"-RRB-","has":"VBZ","my":"PRP$","their":"PRP$","It":"PRP","were":"VBD","In":"IN","if":"IN","would":"MD","”":"''",";":":","into":"IN","when":"WRB","You":"PRP","also":"RB","she":"PRP","our":"PRP$","been":"VBN","who":"WP","We":"PRP","time":"NN","He":"PRP","This":"DT","its":"PRP$","did":"VBD","two":"CD","these":"DT","many":"JJ","And":"CC","!":".","should":"MD","because":"IN","how":"WRB","If":"IN","n’t":"RB","'re":"VBP","him":"PRP","'m":"VBP","city":"NN","could":"MD","may":"MD","years":"NNS","She":"PRP","really":"RB","now":"RB","new":"JJ","something":"NN","here":"RB","world":"NN","They":"PRP","life":"NN","But":"CC","year":"NN","us":"PRP","between":"IN","different":"JJ","those":"DT","language":"NN","does":"VBZ","same":"JJ","going":"VBG","United":"NNP","day":"NN","few":"JJ","For":"IN","every":"DT","important":"JJ","When":"WRB","things":"NNS","during":"IN","might":"MD","kind":"NN","How":"WRB","system":"NN","thing":"NN","example":"NN","another":"DT","small":"JJ","until":"IN","information":"NN","away":"RB"}

    scores = []

    #VBG must end in ing/in; VBN may not
    for i, word in enumerate(word_list):
        pred = pred_list[i]
        score = score_list[i]
        if word in top100:
            output.append(top100[word])
            scores.append("_")
        elif (word.lower(),pred) in vbg_preps:
            output.append(vbg_preps[(word.lower(),pred)])
            scores.append("_")
        else:
            output.append(pred)
            scores.append(score)

    # Also VB+RP/RB disambig from large list? PTB+ON?

    return output, scores


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-m","--mode",choices=["train","predict"],default="predict")
    p.add_argument("-f","--file",default=None,help="Blank for training, blank predict for eval, or file to run predict on")
    p.add_argument("-i","--input_format",choices=["flair","conllu"],default="flair",help="flair two column training format or conllu")
    p.add_argument("-o","--output_format",choices=["flair","conllu","xg"],default="conllu",help="flair two column training format or conllu")
    p.add_argument("-t","--tag",choices=["dm"],default="dm",help="tag to learn/predict")
    p.add_argument("-c","--corpus",default="gum",help="corpus name for model file name")
    p.add_argument("-s","--serialize",action="store_true",help="serialize BIO prediction")

    opts = p.parse_args()

    if opts.mode == "train":
        train(tag=opts.tag,corpus=opts.corpus)
    else:
        if "*" in opts.file:
            from glob import glob
            files = glob(opts.file)
            model_name = "dm-dependencies" + os.sep + "flair_tagger/best-model_" + opts.tag + "_" + opts.corpus + ".pt"
            model = SequenceTagger.load(model_name)
            print()
            for i,f in enumerate(files):
                sys.stderr.write(f"o Processing {os.path.basename(f)} ({i}/{len(files)})" + " "*20 + "\n")
                output = predict(corpus=opts.corpus, tag=opts.tag,
                    in_format=opts.file, out_format=opts.output_format,
                    in_path=open(f).read(),as_text=True,preloaded=model)
                with open("dm-dependencies" + os.sep + os.path.basename(f).replace(".conllu",".tab"),'w',encoding="utf8",newline="\n") as f:
                    f.write(output)
        else:
            predict(corpus=opts.corpus, tag=opts.tag,
                in_format=opts.input_format, out_format=opts.output_format,
                in_path=opts.file,as_text=not opts.serialize)

