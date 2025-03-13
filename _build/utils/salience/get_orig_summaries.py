from collections import defaultdict
from glob import glob
import os
from functools import cmp_to_key

def summary_sort(summary1, summary2):
    """
    Summary format: "(human1) summary text"

    Sort summaries: first human written summaries by source - humanA > humanB > humanC...
    then by model name case insensitive, so human1 > human2 > claude > gpt4 > Llama2 > ...
    """
    if summary1.startswith("(human") and summary2.startswith("(human"):
        return 0
    elif summary1.startswith("(human") and not summary2.startswith("(human"):
        return -1
    elif not summary1.startswith("(human") and summary2.startswith("(human"):
        return 1
    else:
        # Sort by model name
        model1 = summary1.split(" ")[0].lower()
        model2 = summary2.split(" ")[0].lower()
        if model1 < model2:
            return -1
        elif model1 > model2:
            return 1
        else:
            return 0



lines = open("final_summaries_original_model_version.txt").read().split("\n")

summaries = defaultdict(list)

for line in lines:
    if line.startswith("G"):
        docname = line.strip()
    elif line.startswith("("):
        if '(human' not in line:
            summaries[docname].append(line.strip())

tsv_files = glob(".." + os.sep + ".." + os.sep + "src" + os.sep + "tsv" + os.sep + "*.tsv")
for f in tsv_files:
    lines = open(f).read().split("\n")
    docname = os.path.basename(f).replace(".tsv", "")
    for line in lines:
        if line.startswith("#Summary") and "(human" in line:
            summaries[docname].append(line.split("=",1)[-1].strip())

for line in lines:
    if line.startswith("G"):
        docname = line.strip()
    elif line.startswith("("):
        summaries[docname].append(line.strip())


summary_files = glob("output\\summaries\\all\\**\\*.txt", recursive=True)

for f in summary_files:
    if "flan-" in f:
        continue
    model = f.split(os.sep)[-2]
    docname = f.split(os.sep)[-1].replace("0.txt", "").replace(model+"_","")
    summary = open(f).read().strip().replace("\n", "\\n")
    if len(summaries[docname]) ==5:
        continue
    if not any([model in s for s in summaries[docname]]):
        pass
        summaries[docname].append(f"({model}) {summary}")

# Output number of files with each length list
counter = 0
for docname in summaries:
    if len(summaries[docname]) < 5:
        counter += 1
        print(f"{docname}: {len(summaries[docname])}")
print(counter)

output = []

for docname in sorted(summaries):
    output.append(docname)
    # Sort summaries
    summaries[docname].sort(key=cmp_to_key(summary_sort))
    for summary in summaries[docname]:
        output.append(summary.replace('(human)','(human1)'))
    output.append("")

output = "\n".join(output)
with open("summaries_initial_outputs.txt", 'w', encoding="utf8", newline="\n") as f:
    f.write(output)