"""
Make per-genre prompts for LLM alignment queries
"""

from glob import glob
from depedit import DepEdit
import os, re
from get_summary import extract_text_speaker_from_xml
from collections import defaultdict
from random import shuffle, seed
from argparse import ArgumentParser

prompt_template1 = (
    "For each of the entities mentioned in a document, please "
    "return the exact same entity as it appears in the phrase listed under 'Entities', if and only if it is mentioned in the summary. Otherwise, don't return anything and move on to the next. "
    "When matching, please also consider synonyms or alternative phrases that refer to the same entity. If a speaker says 'I' or is mentioned as 'you', then the speaker's name or label is considered mentioned (e.g. 'I' is a correct answer if 'Kim' is mentioned in the summary and says 'I' in the document)\n\n"
    "For example: \n\n"
    "Example document: \n\n{example-doc}\n\n"
    "Example summary: \n\n{example-summary}\n\n"
    "Example entities: \n\n{example-entities}\n\n"
    "Answer: \n\n{example-salient}\n\n"
    "Here is the actual document and summmary for entity alignment: \n\n"
    "Document: \n\n{doc_text} \n\n"
    "Summary: \n\n{summary} \n\n"
    "Entities: \n\n{entities} \n\n"
    "Which of the entities in the Entities list also appears in the Summary? Be very precise and only print entities that are mentioned in the summary, separated by new lines. Do not add extra or unrelated entities.\n\nAnswer: "
)

prompt_template2 = """Given the following document and summary, which noun phrases are referred to in both the document and the summary? References in the summary do not have to be identical, but must refer to the same entity. For example, if a document mentions "US President Barack Obama" and the summary mentions "President Obama", then the noun phrase "US President Barack Obama" has been referred to in the summary. Similarly if the document says "Jane Smith: I want to go." and the summary mentions Jane, then "Jane Smith" and "I" have been referred to. Also include nested mentions and abstract phrases - for example if "austerity measures" are mentioned in a document and summary, then both "austerity" and "austerity measures" are mentioned noun phrases.

Here is the document:

{doc_text}

Summary:

{summary}

Which noun phrases in the document are mentioned in the summary? Be very precise and exhaustive, going over each phrase in the summary and outputting the string of the equivalent noun phrase that it refers back to from the document, one phrase per line and nothing else.

List of document noun phrases referred to in the summary:
"""


seed(42)

max_selected_ents = 6
max_salient_ents = 3

def make_prompt(text_with_speakers, summary, entities, salient_ids, style=1):
    selected_ents = []
    salient_ents = []
    keys = list(entities.keys())
    shuffle(keys)
    for eid in keys:
        ent = entities[eid]
        if eid in salient_ids:
            if len(salient_ents) < max_salient_ents:
                if ent[0].text not in salient_ents:
                    salient_ents.append(ent[0].text)
                    selected_ents.append(ent[0].text)
        if len(salient_ents) == max_salient_ents:
            break
    for eid in keys:
        ent = entities[eid]
        if eid not in salient_ids:
            if len(selected_ents) < max_selected_ents:
                if ent[0].text not in selected_ents:
                    selected_ents.append(ent[0].text)
        if len(selected_ents) == max_selected_ents:
            break

    prompt = prompt_template1 if style == 1 else prompt_template2
    prompt = prompt.replace("{example-doc}", text_with_speakers)
    prompt = prompt.replace("{example-summary}", summary)
    prompt = prompt.replace("{example-entities}", "\n".join(selected_ents))
    prompt = prompt.replace("{example-salient}", "\n".join(salient_ents))
    return prompt.replace("\n","\\n")


p = ArgumentParser()
p.add_argument("-s","--style", type=int, default=1, choices=[1,2], help="Prompt style (1 or 2)")
args = p.parse_args()

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
gum_src = script_dir + ".." + os.sep + ".." + os.sep + "src" + os.sep
gum_trg = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep

splits_lines = open(".." + os.sep + ".." + os.sep + ".." + os.sep + "splits.md").read().strip().split("\n")
ud_dev = []
ud_test = []
ud_gentle = []
ud_train = []
partition = "train"
for line in splits_lines:
    if line.startswith("## "):
        partition = re.search("## ([^\s]+)",line).group(1)
    if "GUM_" in line:
        docname = line.strip().split()[-1]
        if partition == "dev":
            ud_dev.append(docname)
        elif partition == "test":
            ud_test.append(docname)
        else:
            ud_train.append(docname)
    elif "GENTLE_" in line:
        ud_gentle.append(line.strip().split()[-1])

xml_files = glob(gum_src + "xml" + os.sep + "*.xml")

doc2length = defaultdict(lambda : defaultdict(int))
doc2text = {}
doc2prompt = {}
d = DepEdit()

docnames = [os.path.basename(d).replace(".xml","") for d in xml_files]
with_speakers_list = extract_text_speaker_from_xml(gum_src + "xml")
with_speakers_dict = {docname: with_speakers for docname, with_speakers in zip(docnames, with_speakers_list)}

for file_ in xml_files:
    docname = os.path.basename(file_).replace(".xml","")
    if docname not in ud_train and docname not in ud_gentle:
        continue
    genre = docname.split("_")[1]
    xml = open(file_).read()
    length = int(xml.count("\t")/2)
    with_speakers = with_speakers_dict[docname]
    doc2length[genre][docname] = length
    doc2text[docname] = with_speakers
    conllu = open(gum_trg + "dep" + os.sep + "not-to-release" + os.sep + docname + ".conllu").read()
    summary = re.search(r'# meta::summary1 = ([^\n]+)', conllu).group(1).strip() if "summary1" in conllu else re.search(r'# meta::summary = ([^\n]+)', conllu).group(1).strip()
    summary = re.sub(r'^\([^)]+\)', "", summary)
    salient_ids = re.search(r'# meta::salientEntities = ([^\n]+)', conllu).group(1).strip().split(", ")
    d.run_depedit(conllu, parse_entities=True)
    doc2prompt[docname] = make_prompt(with_speakers, summary, d.entities, salient_ids, style=args.style)

output = []
for genre in doc2length:
    lengths = sorted(doc2length[genre].items(), key=lambda x: x[1])
    docnum = 0
    for docname, length in lengths:
        if docnum > 1:
            break
        output.append("\t".join([genre, docname, doc2prompt[docname]]))
        docnum += 1

with open("prompts"+str(args.style)+".tab", "w", encoding="utf8", newline="\n") as f:
    f.write("\n".join(output))
