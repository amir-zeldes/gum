import os, re, json
import pickle
import pandas as pd
import numpy as np
from depedit import DepEdit
from collections import defaultdict, Counter
from glob import glob
#from nltk.stem import SnowballStemmer
from argparse import ArgumentParser
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
conllu_dir = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep + "dep" + os.sep + "not-to-release" + os.sep
rsd_dir = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep + "rst" + os.sep + "dependencies" + os.sep
tsv_dir = script_dir + ".." + os.sep + ".." + os.sep + "src" + os.sep + "tsv" + os.sep

model = None

# All available features
header = ["mention", "docname", "genre", "doclen", "sumlen", "toklen", "position", "last_position", "s_prom", "upos",
          "xpos", "deprel", "etype", "identity", "cluster_size", "exact", "lower", "partial", "synonym",
          "head_in_summary", "rsdrel", "stype", "tense", "spk_overlap",
          "is_speaker", "llm", "llm2", "stanza", "stan_on", "stan_pre", "stan_onpre","stan_gum","stan_gumpre",
          "partition", "start", "end", "summary_number", "label"]

# Selected features for the ensemble
selected_feats = [f for f in header if f not in ["llm","stan_onpre","spk_overlap","label","xpos",#"stan_gum","stan_gumpre",#"llm",
                                                 "partition","mention","docname","start","end","summary_number"]]
#selected_feats = ["genre", "upos", "exact","lower","etype","cluster_size","partial","synonym","llm","stanza","stanza_on","stanza_pre"]
categorical_feats = ["genre","upos","etype","deprel","rsdrel","stype","tense"]

splits_lines = open(script_dir + ".." + os.sep + ".." + os.sep + ".." + os.sep + "splits.md").read().strip().split("\n")
ud_dev = []
ud_test = []
ud_gentle = []
partition = "train"
multi_dir = "C:\\Uni\\Corpora\\GUM\\salience\\gum_sum_salience\\out_summaries_merged\\corrected\\"
multi_gold = []
for line in splits_lines:
    if line.startswith("## "):
        partition = re.search("## ([^\s]+)", line).group(1)
    if "GUM_" in line or "GENTLE_" in line:
        if "GUM_" in line:
            docname = line.strip().split()[-1]
            if partition == "dev":
                ud_dev.append(docname)
            elif partition == "test":
                ud_test.append(docname)
        elif "GENTLE_" in line:
            docname = line.strip().split()[-1]
            ud_gentle.append(docname)
        if os.path.exists(multi_dir + docname + ".tsv"):
            multi_gold.append(docname)

gold_summaries = defaultdict(list)

freqs = Counter()
tsv_files = glob(tsv_dir + "*.tsv")
for f in tsv_files:
    tsv = open(f).read().strip().split("\n")
    for line in tsv:
        if line.startswith("#Summary"):
            summary = line.split("=",1)[-1].strip()
            # Use re \b to recognize whole words and add to frequencies
            for word in re.split(r'\W+', summary):
                freqs[word] += 1

# Transform frequencies to standard deviations from mean
freqs_sd = defaultdict(float)
mean = sum(freqs.values()) / len(freqs)
std = (sum([(f - mean) ** 2 for f in freqs.values()]) / len(freqs)) ** 0.5
for word in freqs:
    freqs_sd[word] = (freqs[word] - mean) / std

#stemmer = SnowballStemmer("english")

def get_json_lookup(filename, lower=True):
    output = defaultdict(list)
    cache = json.load(open(filename))
    for docname in cache:
        for mention in cache[docname]["human1"]:
            output[docname].append(mention.lower() if lower else mention)
    return output


stanza_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stan.json")
stanza_on_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stanon.json")
stanza_gum_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stangum.json")
stanza_pre_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stanpre.json")
stanza_onpre_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stanonpre.json")
stanza_gumpre_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_stangumpre.json")
llm_lookup = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_gpt4o.json")
llm_lookup2 = get_json_lookup(script_dir + "output" + os.sep + "alignment" + os.sep + "align_gpt4o_zeroshot.json")

d = DepEdit()


def contains(needle, haystack, lower=False):
    if lower:
        needle = needle.lower()
        haystack = haystack.lower()
    result = 0 if re.search(r'\b' + re.escape(needle) + r'\b', haystack) is None else 1
    return result


def uncamel(text):
    # Insert spaces into a camel-case string. Single cap letters get a period, e.g.:
    # HaroldMSmith -> Harold M. Smith
    text = re.sub(r'([A-Z])([A-Z])', r'\1.\2', text)
    output = []
    for c in text:
        if c.isupper() and output and output[-1] != " ":
            output.append(" ")
        output.append(c)
    return "".join(output)


def extract_features(docname, summary, summary_number, pos_filter=False):
    conllu = open(conllu_dir + docname + ".conllu").read().strip()
    rsd = open(rsd_dir + docname + ".rsd").read().strip()

    rsd_map = {}
    parents = {}
    rels = {}
    stypes = {}
    tenses = {}
    tok_num = 1
    for line in rsd.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            parents[fields[0]] = fields[6]
            rels[fields[0]] = fields[7]
    for line in rsd.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            relname = fields[7]
            while relname.startswith("same"):
                edu_id = parents[edu_id]
                relname = rels[edu_id]
            edu_id = fields[0]
            stype = re.search("stype=([^\s|]+)", fields[5]).group(1)
            tense = re.search("edu_tense=([^\s|]+)", fields[5]).group(1)
            if "Past" in tense:
                tense = "past"
            elif "Pres" in tense:
                tense = "present"
            else:
                tense = "other"
            stype = "decl" if stype == "decl" else "frag" if stype == "frag" else "other"
            relname = "attribute" if "attribute" in relname else "joint" if "joint" in relname else "orga" if "orga" in relname else "elab" if "elab" in relname else "other"
            for t in range(tok_num, tok_num + len(fields[1].split())):
                rsd_map[t] = relname.replace("_r","").replace("_m","")
                stypes[t] = stype
                tenses[t] = tense
            tok_num += len(fields[1].split())

    sal_string = re.search("salientEntities = ([^\n]+)", conllu).group(1)
    sal_clusters = sal_string.split(", ")
    if "*" in sal_string:  # 2 (5*), 3 (5*), 1 (4*), 7 (2) ...
        # New format, harvest only summary1 salient entities, marked by *, and remove score in brackets
        sal_clusters = [re.sub(r"\s+\([^()]*\)", "", c) for c in sal_clusters if "*" in c]
    tok_lines = [l.split("\t") for l in conllu.split("\n") if "\t" in l]
    tok_lines = [l[0] for l in tok_lines if "." not in l[0] and "-" not in l[0]]
    doclen = np.ceil(len(tok_lines)/100)  # bin document length
    genre = docname.split("_")[1]

    d.run_depedit(conllu, parse_entities=True)

    partition = "test" if docname in ud_test else "dev" if docname in ud_dev else "train"
    summary = summary.split(")", 1)[-1].strip()
    sumlen = np.ceil(len(summary)/20)  # bin summary length
    output = []
    if "gordon" in docname:
        a=4

    for cluster in sorted(d.entities, key=lambda x: int(x)):
        for mention in sorted(d.entities[cluster], key=lambda x: (float(x.start), -float(x.end))):
            spk_overlap = 0
            is_speaker = 0
            if mention.text.lower() in ["i","me","my","myself"]:
                if "speaker" in mention.sentence.input_annotations:
                    spk = mention.sentence.input_annotations["speaker"]
                    spk = uncamel(spk)
                    overlap = 0
                    for i in range(len(spk)):
                        if spk[:i] in summary:
                            overlap = i
                    spk_overlap = overlap / len(spk)
                    is_speaker = 1
            elif mention.text.lower() in ["you","your","yourself"]:
                if "addressee" in mention.sentence.input_annotations:
                    spk = mention.sentence.input_annotations["addressee"]
                    spk = uncamel(spk)
                    overlap = 0
                    for i in range(len(spk)):
                        if spk[:i] in summary:
                            overlap = i
                    spk_overlap = overlap / len(spk)
                    is_speaker = 2
            label = 1 if mention.annos["GRP"] in sal_clusters else 0
            etype = mention.annos["etype"]
            cluster_size = len(d.entities[cluster])
            llm = 1 if mention.text.lower() in llm_lookup[docname] else 0
            llm2 = 1 if mention.text.lower() in llm_lookup2[docname] else 0
            stan = 1 if mention.text.lower() in stanza_lookup[docname] else 0
            stan_on = 1 if mention.text.lower() in stanza_on_lookup[docname] else 0
            stan_gum = 1 if mention.text.lower() in stanza_gum_lookup[docname] else 0
            stan_pre = 1 if mention.text.lower() in stanza_pre_lookup[docname] else 0
            stan_onpre = 1 if mention.text.lower() in stanza_onpre_lookup[docname] else 0
            stan_gumpre = 1 if mention.text.lower() in stanza_gumpre_lookup[docname] else 0
            deprel = mention.head.func.split(":")[0]
            if deprel not in ["nsubj", "obj", "iobj", "obl", "nmod", "appos", "compound", "root", "conj", "expl"]:
                deprel = "other"
            upos = mention.head.pos if mention.head.pos in ["NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV", "DET",
                                                            "NUM"] else "_"
            xpos = mention.head.cpos
            toklen = len(mention.tokens) if len(mention.tokens) < 10 else 10
            position = np.ceil((float(mention.start) / doclen)*4)
            last_mention = sorted(d.entities[cluster], key=lambda x: (float(x.start), -float(x.end)))[-1]
            last_position = np.ceil((float(last_mention.start) / doclen)*4)
            exact = contains(mention.text, summary)
            lower = contains(mention.text, summary)
            partial = 0
            content_tokens = 0
            identity = 1 if "identity" in mention.annos else 0
            head_tok = "___" if mention.head.pos not in ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"] else mention.head.text.lower()
            head_in_summary = contains(head_tok, summary, lower=True)
            #stem = "___" if mention.head.pos not in ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"] else stemmer.stem(mention.head.text.lower())
            #stem_in_summary = contains(stem, summary, lower=True)
            for tok in mention.tokens:
                if pos_filter:
                    if tok.upos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"]:
                        content_tokens += 1
                    else:
                        continue
                if contains(tok.text, summary, lower=True):
                    # Add to partial weighted by frequency in standard deviations from mean
                    # partial += 1
                    partial += freqs_sd[tok.text]
            if pos_filter:
                partial = partial / content_tokens if content_tokens > 0 else 0
            else:
                partial = partial / len(mention.tokens)
            synonym = 0  # Alternative mentions in cluster
            s_prom = int(mention.sentence.input_annotations["s_prominence"])
            for mention2 in d.entities[cluster]:
                if mention2.text != mention.text:
                    if mention2.head.pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"]:
                        if contains(mention2.text, summary, lower=True):
                            synonym = 1
                    if int(mention2.sentence.input_annotations["s_prominence"]) < s_prom:
                        s_prom = int(mention2.sentence.input_annotations["s_prominence"])
            rsdrel = rsd_map[int(float(mention.start))]
            stype = stypes[int(float(mention.start))]
            tense = tenses[int(float(mention.start))]

            feats = [mention.text.replace("#", "_"), docname, genre, doclen, sumlen, toklen, position, last_position,
                     s_prom, upos, xpos, deprel, etype, identity, cluster_size, exact, lower, partial, synonym,
                     head_in_summary, rsdrel, stype, tense,
                     spk_overlap, is_speaker, llm, llm2, stan, stan_on, stan_pre, stan_onpre, stan_gum, stan_gumpre,
                     partition, mention.start, mention.end, summary_number, label]
            output.append(feats)
            break  # 1 instance per cluster

    return output


def make_train_data(pos_filter=False, use_five=False):

    conllu_files = glob(conllu_dir + "*.conllu")

    output = []
    for f in conllu_files:
        docname = os.path.basename(f).replace(".conllu","")
        targets = [0] if (docname not in multi_gold or docname in ud_test or not use_five) else list(range(5))
        conllu = open(f).read().strip()
        summaries = re.findall(r'# meta::summary[0-9]* = \(.*?\) ([^\n]+)', conllu)
        gold_summaries[docname] = summaries
        for k in targets:
            summary = gold_summaries[docname][k]  # First summary is the gold standard
            output.extend(extract_features(docname,summary,k,pos_filter))

    with open("facts.tab","w",newline="\n",encoding="utf8") as f:
        f.write("\t".join(header) + "\n")
        seen = set()
        filtered = []
        for feats in output:
            if tuple(feats[:-1]) in seen and use_five:
                if int(feats[-1]) == 0:
                    continue
                else:
                    negative_line = "\t".join([str(f) for f in feats[:-1]] + ["0"])
                    if negative_line in output:
                        filtered.remove(negative_line)
            line = "\t".join([str(f) for f in feats])
            filtered.append(line)
            seen.add(tuple(feats[:-1]))
        f.write("\n".join(filtered))

def convert_to_pandas(data, train=False):
    # Transform categorical features to one-hot encoding
    data = pd.DataFrame(data,columns=selected_feats)
    data = pd.get_dummies(data,columns=[f for f in categorical_feats if f in selected_feats])

    # Make sure all columns are present and add as 0 if not
    if not train:
        all_headers = open("headers_with_dummies.txt").read().strip().split("\n")
        for header in all_headers:
            if header not in data.columns:
                data[header] = 0
    else:
        all_headers = data.columns.tolist()

    # Alphabetize columns to ensure constant ordering
    data = data[all_headers]

    # Make sure all data types are int, float or bool
    for col in data.columns:
        data[col] = data[col].astype(float)

    return data


def train(partition="devtrain", use_gentle=False, hyperparams=None, use_five=False):
    # Train a random forest model using the facts.tab file with sklearn
    from sklearn.ensemble import RandomForestClassifier

    make_train_data(use_five=use_five)

    data = []
    labels = []

    lines = open("facts.tab").read().strip().split("\n")
    for line in lines[1:]:
        fields = line.split("\t")
        feats_dict = dict(zip(header,fields))
        if partition == "devtrain":
            if feats_dict["partition"] == "test":
                continue
        elif feats_dict["partition"] != partition:
            continue
        if "GENTLE" in feats_dict["docname"] and not use_gentle:
            continue
        label = feats_dict["label"]
        labels.append(label)
        data.append([feats_dict[f] for f in selected_feats])

    data = convert_to_pandas(data, train=True)

    all_headers_with_dummies = data.columns.tolist()
    with open("headers_with_dummies.txt","w",encoding="utf8",newline="\n") as f:
        f.write("\n".join(all_headers_with_dummies))

    #model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4, max_depth=20)

    # Try XGBoost
    from xgboost import XGBClassifier
    if hyperparams:
        print("Using hyperparameters:", hyperparams)
    else:
        # Use best hyperoptimized parameters
        hyperparams = {'colsample_bytree': 0.9202851298798715, 'gamma': 0.6729810569056707, 'learning_rate': 0.06873476785895469,
         'max_depth': 10, 'n_estimators': 150, 'subsample': 0.9812032459280908}
    model = XGBClassifier(random_state=42, n_jobs=4, use_label_encoder=False, **hyperparams)

    model.fit(data,labels)

    # Save the model
    with open("salience_ensemble.pkl","wb") as f:
        pickle.dump(model,f)


def evaluate(analysis=True, test_partition="test"):
    # Evaluate the model on the test set
    from sklearn.metrics import classification_report

    with open("salience_ensemble.pkl","rb") as f:
        model = pickle.load(f)

    data = []
    labels = []

    lines = open("facts.tab").read().strip().split("\n")
    locators = []
    for line in lines[1:]:
        fields = line.split("\t")
        feats_dict = dict(zip(header,fields))
        if feats_dict["partition"] != test_partition:# or feats_dict["summary_number"] != "0":
            continue
        label = feats_dict["label"]
        labels.append(label)
        locators.append((feats_dict["docname"],feats_dict["mention"]))
        data.append([feats_dict[f] for f in selected_feats])

    data = convert_to_pandas(data)

    preds = model.predict(data)

    if analysis:
        print(classification_report(labels,preds))

        # Print feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(len(data.columns)):
            print(f"{data.columns.tolist()[indices[i]]}: {importances[indices[i]]}")

        # Print some false positives and negatives
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        printed = 0
        for i in indices:
            if int(labels[i]) == 1 and int(preds[i]) == 0:
                print(f"Label: {labels[i]}, pred: {preds[i]}", end="\t")
                print(locators[i])
                printed += 1
            if printed > 5:
                break
        printed = 0
        for i in indices:
            if int(labels[i]) == 0 and int(preds[i]) == 1:
                print(f"Label: {labels[i]}, pred: {preds[i]}", end="\t")
                print(locators[i])
                printed += 1
            if printed > 5:
                break
    else:
        # Return positive class f-score
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, average="binary", pos_label='1')
        return f1


def predict(docname):
    # Predict salient entities for a document
    global model

    if model is None:
        with open("salience_ensemble.pkl","rb") as f:
            model = pickle.load(f)

    data = []
    ents = []
    gold = []
    for i, summary in enumerate(gold_summaries[docname]):
        if i == 0:
            feats = extract_features(docname, summary, 0)
            gold = [f[-1] for f in feats]
            continue  # We have gold data for summary 1
        else:
            feats = extract_features(docname,summary, 0)
            data += feats
            if i == 1:
                for j, ent in enumerate(data):
                    ents.append([ent[-3],ent[-2],gold[j]])  # Start, end and gold label of each entity

    # Filter data to have just the selected feature columns
    filtered = []
    for row in data:
        feats_dict = dict(zip(header, row))
        filtered.append([feats_dict[f] for f in selected_feats])

    data = convert_to_pandas(filtered)

    preds = model.predict(data)

    labels = ["s" if g == 1 else "n" for g in gold]
    for i in range(len(gold_summaries[docname][1:])):
        for j, ent in enumerate(ents):
            pred = "s" if preds[i*len(labels)+j] == '1' else "n"
            labels[j] += pred

    span2label = {}
    for i, row in enumerate(feats):
        span2label[(int(float(row[-3])),int(float(row[-2])))] = labels[i]

    tsv = open(tsv_dir + docname + ".tsv").read().strip().split("\n")
    span2eid = {}
    starts = {}
    ends = {}

    toknum = 1
    for line in tsv:
        if "\t" in line:
            fields = line.split("\t")
            if fields[5] != "_":
                sals = fields[5].split("|")
                for sal in sals:
                    eid = sal.split("[")[1].split("]")[0]
                    if eid not in starts:
                        starts[eid] = toknum
                    ends[eid] = toknum
            toknum += 1

    for eid in starts:
        span2eid[(starts[eid],ends[eid])] = eid

    output = []
    for line in tsv:
        if "\t" in line:
            fields = line.split("\t")
            if fields[5] != "_":
                sals = fields[5].split("|")
                out_sals = []
                for sal in sals:
                    eid = sal.split("[")[1].split("]")[0]
                    if (starts[eid],ends[eid]) in span2label:
                        out_sals.append(f"{span2label[(starts[eid],ends[eid])]}[{eid}]")
                    else:
                        out_sals.append(sal[0] + f"____[{eid}]")
                fields[5] = "|".join(out_sals)
                line = "\t".join(fields)
            output.append(line)
        else:
            if line.startswith("#Summary"):
                continue
            output.append(line)
            if line.startswith("#T_RL="):
                for i, summary in enumerate(gold_summaries[docname]):
                    output.append(f"#Summary{i+1}={summary}")

    return "\n".join(output).strip() + "\t\n"


def optimize(train_partition="train", test_partition="test"):

    # Hyperoptimize using tpe

    from hyperopt import fmin, tpe, hp, Trials

    def objective(params):
        # Train the model with the given hyperparameters
        train(hyperparams=params, partition=train_partition)

        # Evaluate the model
        score = evaluate(test_partition=test_partition, analysis=False)

        return -score

    # Define the search space
    space = {
        'n_estimators': hp.choice('n_estimators', [100, 150, 200]),
        'max_depth': hp.choice('max_depth', [3, 5, 10, 20]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma': hp.uniform('gamma', 0, 1),
    }

    trials = Trials()

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=60, trials=trials)

    print("Best hyperparameters:", best)

    return best


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--pos_filter",action="store_true")
    p.add_argument("-m","--mode",choices=["train","eval","traineval","predict","optimize"],default="eval")
    p.add_argument("-p","--partition",default="dev")
    p.add_argument("-t","--test",default="test")
    p.add_argument("-d","--docname",default="GUM_bio_marbles")
    p.add_argument("--five",action="store_true",help="Use 5 summaries for training when available")

    args = p.parse_args()

    if args.mode == "predict":
        output = predict(args.docname)
        print(output)
    elif args.mode == "train":
        train(partition=args.partition, use_five=args.five)
    elif args.mode == "eval":
        evaluate()
    elif args.mode == "traineval":
        train(partition=args.partition, use_five=args.five)
        evaluate()
    elif args.mode == "optimize":
        optimize(train_partition=args.partition,test_partition=args.test)
