import io, os, sys, re
from glob import glob
from collections import defaultdict
from argparse import ArgumentParser

def make_plain(conllu):
    tok_num = 1
    output = []
    for line in conllu.split("\n"):
        if len(line.strip()) == 0:
            continue
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
            if "BeginSeg=Yes" in fields[-1] or "Discourse=" in fields[-1]:
                misc = "BeginSeg=Yes"
            else:
                misc = "_"
            line = "\t".join([str(tok_num),fields[1],"_","_","_","_","_","_","_",misc])
            tok_num +=1
        if "# newdoc" in line:
            tok_num = 1
        elif line.startswith("#"):
            continue
        output.append(line)
    output = "\n".join(output)
    output = output.replace("# newdoc","\n# newdoc").strip() + "\n\n"
    return output


outmode = "standoff"
ellipsis_marker = "<*>"
if outmode == "standoff":
    header = ["doc", "unit1_toks", "unit2_toks", "unit1_txt", "unit2_txt", "s1_toks","s2_toks","unit1_sent","unit2_sent","dir", "orig_label","label"]
else:
    header = ["doc","start_toks","pre","arg1","mid","arg2","post","dir","label"]

rel_mapping = defaultdict(dict)
rel_mapping["eng.rst.rstdt"] = {"attribution":"attribution","attribution-e":"attribution","attribution-n":"attribution","attribution-negative":"attribution","background":"background","background-e":"background","circumstance":"background","circumstance-e":"background","cause":"cause","cause-result":"cause","result":"cause","result-e":"cause","consequence":"cause","consequence-n-e":"cause","consequence-n":"cause","consequence-s-e":"cause","consequence-s":"cause","comparison":"comparison","comparison-e":"comparison","preference":"comparison","preference-e":"comparison","analogy":"comparison","analogy-e":"comparison","proportion":"comparison","condition":"condition","condition-e":"condition","hypothetical":"condition","contingency":"condition","otherwise":"condition","contrast":"contrast","concession":"contrast","concession-e":"contrast","antithesis":"contrast","antithesis-e":"contrast","elaboration-additional":"elaboration","elaboration-additional-e":"elaboration","elaboration-general-specific-e":"elaboration","elaboration-general-specific":"elaboration","elaboration-part-whole":"elaboration","elaboration-part-whole-e":"elaboration","elaboration-process-step":"elaboration","elaboration-process-step-e":"elaboration","elaboration-object-attribute-e":"elaboration","elaboration-object-attribute":"elaboration","elaboration-set-member":"elaboration","elaboration-set-member-e":"elaboration","example":"elaboration","example-e":"elaboration","definition":"elaboration","definition-e":"elaboration","purpose":"enablement","purpose-e":"enablement","enablement":"enablement","enablement-e":"enablement","evaluation":"evaluation","evaluation-n":"evaluation","evaluation-s-e":"evaluation","evaluation-s":"evaluation","interpretation-n":"evaluation","interpretation-s-e":"evaluation","interpretation-s":"evaluation","interpretation":"evaluation","conclusion":"evaluation","comment":"evaluation","comment-e":"evaluation","evidence":"explanation","evidence-e":"explanation","explanation-argumentative":"explanation","explanation-argumentative-e":"explanation","reason":"explanation","reason-e":"explanation","list":"joint","disjunction":"joint","manner":"manner-means","manner-e":"manner-means","means":"manner-means","means-e":"manner-means","problem-solution":"topic-comment","problem-solution-n":"topic-comment","problem-solution-s":"topic-comment","question-answer":"topic-comment","question-answer-n":"topic-comment","question-answer-s":"topic-comment","statement-response":"topic-comment","statement-response-n":"topic-comment","statement-response-s":"topic-comment","topic-comment":"topic-comment","comment-topic":"topic-comment","rhetorical-question":"topic-comment","summary":"summary","summary-n":"summary","summary-s":"summary","restatement":"summary","restatement-e":"summary","temporal-before":"temporal","temporal-before-e":"temporal","temporal-after":"temporal","temporal-after-e":"temporal","temporal-same-time":"temporal","temporal-same-time-e":"temporal","sequence":"temporal","inverted-sequence":"temporal","topic-shift":"topic-change","topic-drift":"topic-change","textualorganization":"textual-organization"}

from .propagate import ud_test as gum_test, ud_dev as gum_dev


def get_rsd(dir_path, chars2toks, toks_by_doc, conll_data, add_missing_tokens=False, reddit=True):
    output = {}
    files = glob(dir_path + "*.rsd")
    if not reddit:
        files = [f for f in files if "_reddit" not in f]
    for file_ in files:
        tokenized = []
        text = io.open(file_,encoding="utf8").read()
        text = text.replace("​","")
        prev_tok = 0
        char_num = 0
        docname = os.path.basename(file_).replace(".rsd","")

        if add_missing_tokens:
            edu_map = defaultdict(list)
            edu_num = 0
            for line in conll_data[docname].strip().split("\n"):
                if "\t" in line:
                    fields = line.split("\t")
                    if "-" in fields[0] or "." in fields[0]:
                        continue
                    if "BeginSeg" in line:
                        edu_num += 1
                    edu_map[edu_num].append(fields[1])

            fixed = []
            for rsd_row in text.strip().split("\n"):
                fields = rsd_row.split("\t")
                fields[1] = " ".join(edu_map[int(fields[0])]).strip()
                fixed.append("\t".join(fields))
            output[docname] = "\n".join(fixed) + "\n"
            continue
        else:
            for line in text.split("\n"):
                if "\t" in line:
                    fields = line.split("\t")
                    raw_content_chars = re.sub(r'\s','',fields[1])
                    this_edu = []
                    for i,c in enumerate(raw_content_chars):
                        if chars2toks[docname][char_num] != prev_tok:
                            this_edu.append(" ")
                            prev_tok += 1
                        this_edu.append(c)
                        char_num +=1
                    fields[1] = "".join(this_edu).strip()
                    line = "\t".join(fields)
                if len(line) > 0:
                    tokenized.append(line)

        output[docname] = "\n".join(tokenized) + "\n"
    return output


def get_conll(dir_path, reddit=True):
    output = {}
    chars2toks = defaultdict(dict)
    toks = defaultdict(list)
    files = glob(dir_path + "*.conllu")
    if not reddit:
        files = [f for f in files if "_reddit" not in f]
    for file_ in files:
        text = io.open(file_,encoding="utf8").read().replace("​","")  # Remove invisible space
        parts = text.split("# newdoc")
        for i, part in enumerate(parts):
            if i == 0:
                continue
            part = part.strip()
            docname = re.search(r'^id ?= ?([^\s]+)',part).group(1)
            output[docname] = "# newdoc " + part.strip()
            tok_num = 0
            offset = 0
            for line in part.split("\n"):
                if "\t" in line:
                    fields = line.split("\t")
                    if "." not in fields[0] and "-" not in fields[0]:
                        for i, c in enumerate(fields[1]):
                            chars2toks[docname][i+offset] = tok_num
                        offset += len(fields[1])
                        toks[docname].append(fields[1])
                        tok_num += 1
    return output, chars2toks, toks


def format_range(tok_ids):
    # Takes a list of IDs and returns formatted string:
    # contiguous subranges of numbers are separated by '-', e.g. 5-24
    # discontinuous subranges are separated by ',', e.g. 2,5-24
    def format_subrange(subrange):
        if len(subrange) == 1:
            return str(subrange[0]+1)
        else:
            return str(min(subrange)+1) + "-" + str(max(subrange)+1)

    subranges = [[]]
    last = None
    for tid in sorted(tok_ids):
        if last is None:
            subranges[-1].append(tid)
        elif tid == last +1:
            subranges[-1].append(tid)
        else:
            subranges.append([tid])
        last = tid

    formatted = []
    for subrange in subranges:
        formatted.append(format_subrange(subrange))

    return ",".join(formatted)


def format_text(arg1_toks, toks):
    last = arg1_toks[0] - 1
    output = []
    for tid in sorted(arg1_toks):
        if tid != last + 1:
            output.append(ellipsis_marker)
        output.append(toks[tid])
        last = tid
    return " ".join(output)


def format_sent(arg1_sid, sents):
    sent = sents[arg1_sid]
    lines = sent.split("\n")
    output = []
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            if "." in fields[0] or "-" in fields[0]:  # supertok or ellipsis token
                continue
            output.append(fields[1])
    return " ".join(output)


def make_rels(rsd_data, conll_data, dev_set, test_set, corpus="eng.rst.gum"):
    err_docs = set()
    dev = ["\t".join(header)]
    test = ["\t".join(header)]
    train = ["\t".join(header)]

    for i, docname in enumerate(rsd_data):
        sent_map = {}
        toks = {}
        sents = conll_data[docname].split("\n\n")

        snum = 0
        toknum = 0
        s_starts = {}
        s_ends = {}

        for sent in sents:
            lines = sent.split("\n")
            for line in lines:
                if "\t" in line:
                    fields = line.split("\t")
                    if "-" in fields[0] or "." in fields[0]:
                        continue
                    if fields[0] == "1":
                        s_starts[snum] = toknum
                    sent_map[toknum] = snum
                    toks[toknum] = fields[1]
                    toknum += 1
            s_ends[snum] = toknum - 1
            snum += 1

        rsd_lines = rsd_data[docname].split("\n")

        parents = {}
        texts = {}
        tok_map = {}
        offset = 0
        rels = {}
        for line in rsd_lines:
            if "\t" in line:
                fields = line.split("\t")
                edu_id = fields[0]
                edu_parent = fields[6]
                relname = fields[7].replace("_m","").replace("_r","")
                text = fields[1].strip()
                texts[edu_id] = text
                tok_map[edu_id] = (offset, offset + len(text.split())-1)
                offset += len(text.split())
                if edu_parent == "0":  # Ignore root
                    continue
                parents[edu_id] = edu_parent
                rels[edu_id] = relname

        same_unit_components = defaultdict(set)
        same_unit_data = {}
        # set up same-unit storage
        for edu_id in parents:
            if rels[edu_id].lower().startswith("same"):
                # collect all intervening text inside same-unit children
                parent = parents[edu_id]
                start = int(parent)
                end = int(edu_id)
                unit_ids = [str(x) for x in range(start,end+1)]
                same_unit_components[parent].add(edu_id)
                if parent not in same_unit_data:
                    same_unit_data[parent] = (start,end," ".join([texts[t].strip() for t in unit_ids]))
                else:
                    start, end, text = same_unit_data[parent]
                    if int(edu_id) > start:  # This is a subsequent same-unit member on the right
                        unit_ids = [str(x) for x in range(end+1,int(edu_id)+1)]
                        more_text = " ".join([texts[t].strip() for t in unit_ids])
                        same_unit_data[parent] = (start,int(edu_id)," ".join([text,more_text]))
                    else:
                        raise IOError("LTR same unit!\n")

        output = []
        for edu_id in parents:
            if rels[edu_id].lower().startswith("same"):
                continue  # Skip the actual same-unit relation
            child_text = texts[edu_id]
            parent_id = parents[edu_id]
            if int(edu_id) < int(parent_id):
                direction = "1>2"
                arg1_start, arg1_end = tok_map[edu_id]
                arg2_start, arg2_end = tok_map[parent_id]
            else:
                direction = "1<2"
                arg1_start, arg1_end = tok_map[parent_id]
                arg2_start, arg2_end = tok_map[edu_id]

            parent_text = texts[parent_id]
            if parent_id in same_unit_data:
                start, end, text = same_unit_data[parent_id]
                if int(edu_id) < start or int(edu_id)> end:
                    parent_text = text
                    if int(edu_id) < int(parent_id):
                        arg2_start, _ = tok_map[str(start)]
                        _, arg2_end = tok_map[str(end)]
                    else:
                        arg1_start, _ = tok_map[str(start)]
                        _, arg1_end = tok_map[str(end)]

            if edu_id in same_unit_data:
                start, end, text = same_unit_data[edu_id]
                if int(parent_id) < start or int(parent_id)> end:
                    child_text = text
                    if int(edu_id) < int(parent_id):
                        arg1_start, _ = tok_map[str(start)]
                        _, arg1_end = tok_map[str(end)]
                    else:
                        arg2_start, _ = tok_map[str(start)]
                        _, arg2_end = tok_map[str(end)]

            arg1_sid = sent_map[arg1_start]
            arg2_sid = sent_map[arg2_start]

            s1_start = s_starts[arg1_sid]
            s1_end = s_ends[arg1_sid]
            s2_start = s_starts[arg2_sid]
            s2_end = s_ends[arg2_sid]

            pre = []
            pre_toks = []
            arg1 = []
            arg1_toks = []
            mid = []
            mid_toks = []
            arg2 = []
            arg2_toks = []
            post = []
            post_toks = []
            for i in sorted(list(set(list(range(s1_start,s1_end+1)) + list(range(s2_start, s2_end+1))))):
                tok = toks[i]
                if i < arg1_start:
                    pre.append(tok)
                    pre_toks.append(i)
                elif i >= arg1_start and i <= arg1_end:
                    arg1.append(tok)
                    arg1_toks.append(i)
                elif i > arg1_end and i < arg2_start:
                    mid.append(tok)
                    mid_toks.append(i)
                elif i >= arg2_start and i <= arg2_end:
                    arg2.append(tok)
                    arg2_toks.append(i)
                else:
                    post.append(tok)
                    post_toks.append(i)

            if outmode == "standoff":
                comp1 = edu_id if int(edu_id) < int(parent_id) else parent_id
                comp2 = parent_id if int(edu_id) < int(parent_id) else edu_id
                # Reduce EDUs to minimal span in standoff mode
                arg1_toks = list(range(tok_map[comp1][0], tok_map[comp1][1]+1))
                arg2_toks = list(range(tok_map[comp2][0], tok_map[comp2][1]+1))
                # Add explicit discontinuous spans
                if comp1 in same_unit_components:
                    for component in same_unit_components[comp1]:
                        component_toks = list(range(tok_map[component][0], tok_map[component][1]+1))
                        arg1_toks += component_toks
                if comp2 in same_unit_components:
                    for component in same_unit_components[comp2]:
                        component_toks = list(range(tok_map[component][0], tok_map[component][1]+1))
                        arg2_toks += component_toks
                arg1_txt = format_text(arg1_toks,toks)
                arg1_sent = format_sent(arg1_sid,sents)
                arg2_txt = format_text(arg2_toks,toks)
                arg2_sent = format_sent(arg2_sid,sents)
                arg1_toks = format_range(arg1_toks)
                arg2_toks = format_range(arg2_toks)
                s1_toks = format_range(list(range(s1_start,s1_end+1)))
                s2_toks = format_range(list(range(s2_start,s2_end+1)))

                mapped_rel = rels[edu_id]
                if corpus in rel_mapping:
                    if mapped_rel in rel_mapping[corpus]:
                        mapped_rel = rel_mapping[corpus][mapped_rel]
                    elif mapped_rel.lower() in rel_mapping[corpus]:
                        mapped_rel = rel_mapping[corpus][mapped_rel.lower()]
                    else:
                        if mapped_rel!="ROOT":
                            #mapped_rel = mapped_rel.lower()
                            raise IOError("no rel map "+mapped_rel)
                        elif mapped_rel == "ROOT":
                            raise IOError("found ROOT entry in " +corpus + ": "+docname)
                elif "-" in mapped_rel and "same-unit" not in mapped_rel.lower():
                    mapped_rel = mapped_rel.split("-")[0]
                if corpus.startswith("fas."):
                    mapped_rel = mapped_rel.lower()
                output.append("\t".join([docname,arg1_toks,arg2_toks,arg1_txt,arg2_txt,s1_toks,s2_toks,arg1_sent,arg2_sent,direction,rels[edu_id],mapped_rel]))
            else:
                pre = " ".join(pre) if len(pre) > 0 else "NULL"
                pre_toks = str(min(pre_toks)) if len(pre_toks) > 0 else "NA"
                arg1 = " ".join(arg1)
                arg1_toks = str(min(arg1_toks))
                mid = " ".join(mid) if len(mid) > 0 else "NULL"
                mid_toks = str(min(mid_toks)) if len(mid_toks) > 0 else "NA"
                arg2 = " ".join(arg2)
                arg2_toks = str(min(arg2_toks))
                post = " ".join(post) if len(post) > 0 else "NULL"
                post_toks = str(min(post_toks)) if len(post_toks) > 0 else "NA"

                indices = ";".join([pre_toks, arg1_toks, mid_toks, arg2_toks, post_toks])
                output.append("\t".join([docname,indices,pre,arg1,mid,arg2,post,direction,rels[edu_id]]))

        if docname in dev_set:
            dev += output
        elif docname in test_set:
            test += output
        else:
            train += output

    print("\n".join(sorted(list(err_docs))))
    dev = "\n".join(dev) + "\n"
    train = "\n".join(train) + "\n"
    test = "\n".join(test) + "\n"

    return dev, train, test


def main(make_tok_files=True, reddit=False):
    dev_set = gum_dev
    test_set = gum_test

    corpus = "eng.gum.rst"
    target_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." +os.sep + "target" + os.sep
    conllu_dir = target_dir + "dep" + os.sep + "not-to-release" + os.sep
    rsd_dir = target_dir + "rst" + os.sep + "dependencies" + os.sep
    disrpt_dir = target_dir + "rst" + os.sep + "disrpt" + os.sep
    if not os.path.exists(disrpt_dir):
        os.makedirs(disrpt_dir)

    conll_data, chars2toks, toks_by_doc = get_conll(conllu_dir, reddit=reddit)
    add_missing = False  # True if corpus == "eng.rst.rstdt" else False
    rsd_data = get_rsd(rsd_dir, chars2toks, toks_by_doc, conll_data, add_missing_tokens=add_missing, reddit=reddit)

    dev, train, test = make_rels(rsd_data, conll_data, dev_set, test_set, corpus=corpus)

    if make_tok_files:
        plain_dev = ""
        plain_test = ""
        plain_train = ""
        for docname in conll_data:
            if docname in dev_set:
                plain_dev += make_plain(conll_data[docname].strip() + "\n\n")
            elif docname in test_set:
                plain_test += make_plain(conll_data[docname].strip() + "\n\n")
            else:
                plain_train += make_plain(conll_data[docname].strip() + "\n\n")
        with io.open(disrpt_dir + corpus + "_dev.tok", 'w', encoding="utf8", newline="\n") as f:
            f.write(plain_dev)
        with io.open(disrpt_dir + corpus + "_test.tok", 'w', encoding="utf8", newline="\n") as f:
            f.write(plain_test)
        with io.open(disrpt_dir + corpus + "_train.tok", 'w', encoding="utf8", newline="\n") as f:
            f.write(plain_train)

    with io.open(disrpt_dir + corpus + "_dev.rels",'w',encoding="utf8",newline="\n") as f:
        f.write(dev)
    with io.open(disrpt_dir + corpus + "_test.rels",'w',encoding="utf8",newline="\n") as f:
        f.write(test)
    with io.open(disrpt_dir + corpus + "_train.rels", 'w', encoding="utf8", newline="\n") as f:
        f.write(train)


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-p","--plain",action="store_true",help="also write plain .tok files")
    p.add_argument("-r","--reddit",action="store_true",help="include reddit files")
    opts = p.parse_args()

    main(make_tok_files=opts.plain,reddit=opts.reddit)
