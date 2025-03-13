"""
Script for adding summaries and graded salience scores for new documents added to the corpus

To add new documents to the corpus, the following steps are required:
1. if you have underscored documents (e.g. reddit), make sure to restore the text in _build/src/ before running
2. add the docname(s) and human-written summary1 to summaries_final.txt
3. run this script (optional: if there are new gold documents to train on, retrain the ensemble before predicting final alignment)
"""
import sys
from argparse import ArgumentParser
import os, json
from copy import deepcopy
from glob import glob
from collections import defaultdict
from get_summary import get_summary, get_summary_gpt4o, get_summary_claude35, extract_gold_summaries_from_xml, read_documents, extract_text_speaker_from_xml
from score import get_sal_tsv, get_sal_mentions, sal_coref_cluster, extract_first_mentions, calculate_scores

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
gum_src = script_dir + ".." + os.sep + ".." + os.sep + "src" + os.sep
xml_src = gum_src + "xml" + os.sep

model_priorities = ["gpt4o", "claude-3-5-sonnet-20241022", "meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]

default_align_comps = ["LLM", "string_simple_lower", "string_simple", "stanza", "stanza_on", "stanza_pre", "stanza_onpre",
                       "stanza_gum","stanza_gumpre"]

aliases = {"stanza": "stan", "stanza_on":"stanon", "stanza_pre": "stanpre", "stanza_onpre": "stanonpre",
           "stanza_gum": "stangum", "stanza_gumpre": "stangumpre",
           "string_simple": "ss", "string_simple_lower": "ssl", "LLM": "llm",
           "string_match": "sm"}
align_llm = "gpt4o"


def main(n_summaries=5, alignment_components=None, overwrite_alignment=False, doclist=None,
         train_ensemble=False):

    # PART I: summaries
    # Diagnose which documents need how many summaries based on summaries_final.txt
    summaries = defaultdict(dict)

    docnames = glob(xml_src + "*.xml")
    if doclist is not None:
        docnames = open(doclist, "r").read().strip().split("\n")
        docnames = sorted(list(set(docnames)))
    else:
        docnames = sorted([os.path.basename(d).replace(".xml","") for d in docnames])

    #docnames = docnames[0:2]

    lines = open("summaries_final.txt", "r").read().split("\n")
    docname = ""
    for line in lines:
        if line[:4] in ["GUM_","GENT"]:
            docname = line.strip()
            human_idx = 1
        elif line.strip() != "":
            if docname not in docnames:
                continue
            model, summary = line.split(")",1)
            model = model.replace("(","").strip()  # Clean brackets
            if "human" in model:
                model = "human" + str(human_idx)
                human_idx += 1
            summaries[docname][model] = summary.strip()

    todo = defaultdict(list)
    for doc in summaries:
        if len(summaries[doc]) < n_summaries:
            for model in model_priorities:
                if not any([model.split("/")[-1].replace("; postedited","") in k for k in summaries[doc].keys()]):
                    if not ("qwen" in model.lower() and any(["qwen" in k.lower() for k in summaries[doc].keys()])):
                        todo[doc].append(model)
            while len(summaries[doc]) + len(todo[doc]) > n_summaries:
                todo[doc].pop()  # remove the last, less prioritized models if multiple human summaries exist

    texts = extract_text_speaker_from_xml(xml_src, docnames=docnames)

    doc2text = dict(zip(docnames, texts))
    model2docs = defaultdict(list)

    for doc in todo:
        for model in todo[doc]:
            model2docs[model].append(doc)

    for model in model2docs:
        docnames = []
        target_texts = []
        for doc in model2docs[model]:
            docnames.append(doc)
            target_texts.append(doc2text[doc])

        if "gpt" in model:
            # Return a dict of docname to list of new summary strings, which are also serialized to cache.
            # If summaries already exist, the model API will not be called.
            new_summaries = get_summary_gpt4o(target_texts, docnames, ".", "all", model_name="gpt4o", n=1, overwrite=False)
        elif "claude" in model:
            new_summaries = get_summary_claude35(target_texts, docnames, ".", "all", model_name="claude-3-5-sonnet-20241022", n=1, overwrite=False)
        elif "flan" in model:
            new_summaries = get_summary(target_texts, docnames, ".", "all", model_name=model, n=1, overwrite=False)
        else:
            new_summaries = get_summary_gpt4o(target_texts, docnames, ".", "all", model_name=model, n=1, overwrite=False)

        for doc in new_summaries:
            summaries[doc][model] = new_summaries[doc][0]

    with open("final_summaries_new.txt", "w", encoding="utf8", newline="\n") as f:
        for doc in summaries:
            f.write(doc + "\n")
            for model in sorted(summaries[doc],key=lambda x: ("human" not in x,x.lower())):
                slug = model.split('/')[-1]#.replace('; postedited','')
                if "qwen" in slug.lower() and "GENTLE" in doc:
                    slug = slug.replace("7B","3B")
                f.write(f"({slug}) {summaries[doc][model]}\n")
            f.write("\n")

    # PART II: graded salience alignment
    from align import align, get_entities_from_gold_tsv

    if alignment_components is None:
        alignment_components = default_align_comps

    summary_lists = []
    for doc in summaries:
        summary_lists.append(list(summaries[doc].values()))

    sal_mentions = get_sal_mentions(gum_src + "tsv")
    sc = sal_coref_cluster(sal_mentions)
    sc = {docname: sc[docname] for docname in docnames}

    all_entities_from_tsv = get_entities_from_gold_tsv(gum_src + "tsv", docnames=docnames, first=True)
    all_entities_from_tsv = {docnames[i]: all_entities_from_tsv[i] for i in range(len(docnames))}
    all_mentions_from_tsv = get_entities_from_gold_tsv(gum_src + "tsv", docnames=docnames, first=False)
    all_mentions_from_tsv = {docnames[i]: all_mentions_from_tsv[i] for i in range(len(docnames))}
    texts = {docnames[i]: texts[i] for i in range(len(docnames))}

    # Get all mentions from each document
    all_mentions = defaultdict(set)
    for docname in sc:
        for mention in all_mentions_from_tsv[docname]:
            all_mentions[docname].add(mention)

    all_alignments = []
    docnames = sorted(list(summaries.keys()))

    for component in alignment_components:
        alias = aliases[component] if "LLM" not in component else align_llm.split("/")[-1].replace("; postedited","")
        if component == "LLM_zero":
            alias += "_zeroshot"

        target_summaries = deepcopy(summaries)
        if os.path.exists(f"output/alignment/align_{alias}.json") and not overwrite_alignment:
            cache = json.load(open(f"output/alignment/align_{alias}.json", "r"))
            for doc in cache:
                if doc in summaries:
                    for source in cache[doc]:
                        if source in summaries[doc]:
                            del target_summaries[doc][source]
                    if len(target_summaries[doc]) == 0:
                        del target_summaries[doc]
            for doc in summaries:
                if doc not in texts and doc in target_summaries:
                    del target_summaries[doc]  # Can't obtain alignment for documents we don't have text for
        else:
            cache = defaultdict(lambda: defaultdict(list))

        target_summaries = {doc: target_summaries[doc] for doc in target_summaries if doc in docnames}
        if len(target_summaries) == 0:
            sys.stderr.write(f"o No new documents/summaries to align for {component}.\n")
            continue
        texts = {k: texts[k] for k in target_summaries}
        all_entities_from_tsv = {k: all_entities_from_tsv[k] for k in target_summaries}
        sc = {k: sc[k] for k in target_summaries}

        alignments = align(all_entities_from_tsv, target_summaries, all_mentions, texts, component=component)
        all_alignments.append(alignments)
        if component == "LLM_zero":
            # LLM_zero can mention any member of the chain, so provide all mentions, not just the first, for alignment
            preds = extract_first_mentions(all_mentions, alignments, target_summaries)
        else:
            # Use all_mentions instead of sc to allow any mention to match
            preds = extract_first_mentions(all_mentions, alignments, target_summaries, exact=True)
        for doc in preds:
            if doc not in cache:
                cache[doc] = {}
            for source in preds[doc]:
                cache[doc][source] = preds[doc][source]

        json.dump(cache, open(os.path.join(f"output/alignment/align_{alias}.json"), "w", encoding="utf8", newline="\n"), indent=4)

    if train_ensemble:
        from ensemble import train
        train(partition="devtrain", use_gentle=False)

    # PART III: serialize
    from ensemble import predict
    for i, docname in enumerate(docnames):
        sys.stderr.write(f"o Obtaining ensemble predictions for {docname} ({i}/{len(docnames)})")
        tsv = predict(docname)
        sys.stderr.write("\r")
        with open("output" + os.sep + "ensemble" + os.sep + docname + ".tsv", "w", encoding="utf8", newline="\n") as f:
            f.write(tsv)

    sys.stderr.write(f"\nSerialized predictions for {len(docnames)} documents to output/ensemble/\n")


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-n", "--n_summaries", type=int, default=5, help="Total number of summaries needed, including human ones")
    p.add_argument("--alignment_component", default=["LLM","LLM_zero","stanza","stanza_pre","stanza_on","stanza_onpre","stanza_gum","stanza_gumpre"],
                   choices=["LLM", "LLM_zero", "string_simple_lower", "string_simple", "string_match", "stanza", "stanza_on", "stanza_pre", "stanza_onpre",
                            "stanza_gum","stanza_gumpre"], nargs="+", help="Components to use for alignment. Note string components are deprecated, the ensemble computes string matches independently.")
    p.add_argument("--llm", default="gpt4o", choices=["gpt-4o-mini","gpt4o"], help="LLM to use for alignment. Default: gpt4o")
    p.add_argument("--doclist", default=None, help="Optional file with document names to process, one per line")
    p.add_argument("--train_ensemble", action="store_true", help="Train the ensemble model")

    opts = p.parse_args()

    if isinstance(opts.alignment_component, str):
        opts.alignment_component = [opts.alignment_component]

    main(opts.n_summaries,opts.alignment_component,doclist=opts.doclist,train_ensemble=opts.train_ensemble)
