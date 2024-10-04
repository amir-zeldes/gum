import os, sys
from modules.explicit import Explicit
from modules.implicit import Implicit
from modules.altlex import Altlex
from modules.altlexC import AltlexC
from process import read_file
from argspan import make_span_contain_dm
from collections import defaultdict

# PDTB senses that have a level-3 information in the form of ARGX-AS-SOMETHING
senses2adjust_dict = {
    "contingency.condition": "as-cond",
    "contingency.negative-condition": "as-negcond",
    "contingency.purpose": "as-goal",
    "comparison.concession": "as-denier",
    "expansion.exception": "as-except",
    "expansion.instantiation": "as-instance",
    "expansion.level-of-detail": "as-detail",
    "expansion.manner": "as-manner",
    "expansion.substitution": "as-subst"
}


def order_rel_args(rel, doc_state, filter=None):
    global senses2adjust_dict

    if len(rel.pdtb_rels) == 0:  # Nothing to do
        return None

    # meta information
    doc_name = doc_state.docname

    rst = rel.relname

    output = []

    for orig_rel_type in rel.pdtb_rels:
        if filter is not None and orig_rel_type != filter:
            continue
        for pdtb_rel in rel.pdtb_rels[orig_rel_type]:
            rel_type = orig_rel_type
            note = "_"
            dm = "_"
            if orig_rel_type == "cache":
                rel_type = pdtb_rel[0]  # Retrieve actual cached reltype if this is a cached rel
            if rel_type == "hypophora":
                source_ids = [edu.edu_id for edu in rel.source.edus]
                target_ids = [edu.edu_id for edu in rel.target.edus]
                source_ids = sorted(list(set(source_ids)))
                target_ids = sorted(list(set(target_ids)))

                source_sent_id = rel.source.sent_ids[-1]
                target_sent_id = rel.target.sent_ids[0]

                source_sent = doc_state.sents[source_sent_id]
                target_sent = doc_state.sents[target_sent_id]

                # fixed, should be entire sentence text now
                source_text = source_sent.plain_text
                target_text = target_sent.plain_text

                # If cache says to take more we take more:
                if orig_rel_type == "cache" and "|" in pdtb_rel[-1]:
                    src, trg = pdtb_rel[-1].split("|")
                    trg = eval(trg)
                    src = eval(src)
                    pdtb_rel = list(pdtb_rel)
                    pdtb_rel[-1] = "_"
                    source_edus = [doc_state.edus[edu] for edu in doc_state.edus if edu in src]
                    target_edus = [doc_state.edus[edu] for edu in doc_state.edus if edu in trg]
                    source_text = ' '.join([edu.text for edu in source_edus])
                    target_text = ' '.join([edu.text for edu in target_edus])

                # Arg2 is the later one in the text
                arg1 = source_text if source_sent_id < target_sent_id else target_text
                arg2 = target_text if source_sent_id < target_sent_id else source_text
                arg1_ids = source_ids if source_sent_id < target_sent_id else target_ids
                arg2_ids = target_ids if source_sent_id < target_sent_id else source_ids
                out_rel_type = sense = rel_type

            elif rel_type in ["implicit","entrel","norel"] and rel.source.head_edu[0].sent_id != rel.target.head_edu[0].sent_id:

                # JL: Arg2 is the later one in the text
                _, sense, tok_ids, dm = pdtb_rel

                if min(rel.source.sent_ids) <= max(rel.target.sent_ids):  # Implicit/EntRel/NoRel must be in text order
                    target_sent_id = max(rel.source.sent_ids)
                    source_sent_id = min(rel.target.sent_ids)
                else:
                    source_sent_id = min(rel.source.sent_ids)
                    target_sent_id = max(rel.target.sent_ids)

                if source_sent_id == target_sent_id:
                    if source_sent_id - 1 in rel.source.sent_ids:
                        source_sent_id -= 1
                    else:
                        continue

                source_sent = doc_state.sents[source_sent_id]
                target_sent = doc_state.sents[target_sent_id]

                if rel_type in ["entrel","norel"]:
                    # Check same paragraph and consecutive sentences
                    if source_sent.par_id != target_sent.par_id or source_sent.sent_id - 1 != target_sent.sent_id:
                        continue

                # fixed, should be entire sentence text now
                source_text = source_sent.plain_text
                target_text = target_sent.plain_text

                source_ids = sorted([edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == source_sent_id])
                target_ids = sorted([edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == target_sent_id])

                # Arg2 is the later one in the text
                if source_sent_id < target_sent_id:
                    arg1 = source_text
                    arg2 = target_text

                    if sense in senses2adjust_dict:
                        sense = f"{sense}.arg1-{senses2adjust_dict[sense]}"
                    arg1_ids = source_ids
                    arg2_ids = target_ids

                else:
                    arg1 = target_text
                    arg2 = source_text

                    if sense in senses2adjust_dict:
                        sense = f"{sense}.arg2-{senses2adjust_dict[sense]}"
                    arg1_ids = target_ids
                    arg2_ids = source_ids

                out_rel_type = rel_type if sense != "NoRel" or rel_type == "entrel" else "norel"

            else:  # explicit, altlex, altlexc
                note, sense, tok_ids, dm = pdtb_rel

                # Level 3 is determined by RST source/target status except cause/result
                # Note that concessions are reversed (target is the 'denier')
                # And second senses of manner are also reversed (sense2_explicit, have 'explicit' in note field)
                # e.g. 'by' can have source as the manner arg, but then target is the purpose arg for two sense cases
                if "." in sense and "cause" not in sense and "asynchronous" not in sense:
                    sense = ".".join(sense.split(".")[:2])

                # source
                if len(rel.source.sent_ids) > 1:  # if multi-sentential
                    sent_id = rel.source.head_edu[0].sent_id  # head_edu should be a list of length 1
                    # getting all edu ids whose sentence id is the same as head_edu's sentence id
                    # if the target overlaps the same sentence, we limit it to descendants of the head EDU
                    if rel.target.head_edu[0].sent_id == rel.source.head_edu[0].sent_id:
                        rel_source_edus = [e.edu_id for e in rel.source.edus]
                        source_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id and edu in rel_source_edus]
                    else:
                        source_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id]
                else:
                    source_ids = [edu.edu_id for edu in rel.source.edus]

                for edu_id in source_ids:
                    source_ids.extend(list(doc_state.edus[edu_id].same_unit))

                # target
                if len(rel.target.sent_ids) > 1:  # if multi-sentential
                    sent_id = rel.target.head_edu[0].sent_id  # head_edu should be a list of length 1
                    if sent_id == rel.source.head_edu[0].sent_id:  # This is actually intersentential - same head sents
                        source_ids = [edu.edu_id for edu in rel.source.edus if edu.sent_id == sent_id]
                        target_ids = [edu.edu_id for edu in rel.target.edus if edu.sent_id == sent_id]
                    else:
                        # getting all edu ids whose sentence id is the same as head_edu's sentence id
                        # if the source overlaps the same sentence, we limit it to descendants of the head EDU
                        if rel.target.head_edu[0].sent_id == rel.source.head_edu[0].sent_id:
                            rel_target_edus = [e.edu_id for e in rel.target.edus]
                            target_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id and edu in rel_target_edus and edu not in source_ids]
                        else:
                            target_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id and edu not in source_ids]
                else:
                    target_ids = [edu.edu_id for edu in rel.target.edus]

                for edu_id in target_ids:
                    target_ids.extend(list(doc_state.edus[edu_id].same_unit))

                source_ids = [e for e in source_ids if e not in target_ids]  # no overlap

                # obtain sentence id of each involved EDU to determine whether this pair is inter-sentential or not
                source_edu_sent_ids = sorted(list(set([doc_state.edus[eduid].sent_id for eduid in source_ids])))
                target_edu_sent_ids = sorted(list(set([doc_state.edus[eduid].sent_id for eduid in target_ids])))

                is_src_attribution = True #if any(["attribution" in doc_state.edus[x].rel for x in source_ids]) else False
                is_trg_attribution = True #if any(["attribution" in doc_state.edus[x].rel for x in target_ids]) else False

                if source_edu_sent_ids != target_edu_sent_ids:
                    if is_src_attribution:  # Include entire sentence including attribution satellites
                        source_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == source_edu_sent_ids[0]]
                    else:  # Include entire sentence except attribution satellites
                        source_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == source_edu_sent_ids[0] and "attribution" not in doc_state.edus[edu].rel]
                    if is_trg_attribution:
                        target_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == target_edu_sent_ids[0]]
                    else:
                        target_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == target_edu_sent_ids[0] and "attribution" not in doc_state.edus[edu].rel]

                # EDU IDs of SOURCE AND TARGET EDUs
                source_ids = sorted(list(set(source_ids)))
                target_ids = sorted(list(set(target_ids)))

                # Change from assert to warning - AZ
                if len(source_edu_sent_ids) != 1 or len(target_edu_sent_ids) != 1:
                    sys.stderr.write(f"WARN: Relation spanning multiple sentences found in {doc_state.docname} for relation {rel.relname} with head EDU {rel.head_edu} - argspan may be incorrect\n")

                # updated from TA's argspan.py on April 16th: to include spans that contain the DM
                # uncomment the line below if the EDU span expansion causes any issues
                if dm not in ["_","","BLOCK"] and note != "implicit":
                    source_ids, target_ids = make_span_contain_dm(doc_state, rel, tok_ids, dm, source_ids, target_ids)

                # obtain span texts
                source_text = ' '.join([doc_state.edus[edu].text for edu in source_ids])
                target_text = ' '.join([doc_state.edus[edu].text for edu in target_ids])

                # +1 is used because some form of the token ids is zero-indexed
                source_token_ids = [tokid + 1 for eduid in source_ids for tokid in doc_state.edus[eduid].tok_ids]
                target_token_ids = [tokid + 1 for eduid in target_ids for tokid in doc_state.edus[eduid].tok_ids]

                # ARG ORDERING #
                tok_id2head_id_dict = doc_state.id2head
                tok_id2deprel_dict = doc_state.id2deprel

                if source_edu_sent_ids[0] != target_edu_sent_ids[0]:  # inter-sentential; text order
                    # Arg2 is the later one in the text
                    if source_token_ids[0] < target_token_ids[0]:
                        arg1 = source_text
                        arg2 = target_text
                        argn = "arg1" if "concession" not in sense and note != "explicit" else "arg2"
                        # check sense label to complete the level-3 information, if applicable
                        if sense in senses2adjust_dict:
                            sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                        arg1_ids = source_ids
                        arg2_ids = target_ids
                    else:
                        arg1 = target_text
                        arg2 = source_text
                        argn = "arg2" if "concession" not in sense and note != "explicit" else "arg1"
                        # check sense label to complete the level-3 information, if applicable
                        if sense in senses2adjust_dict:
                            sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                        arg1_ids = target_ids
                        arg2_ids = source_ids

                else:  # intra-sentential: loop through head to find out which span is arg1/arg2
                    # rename SOURCE and TARGET to reflect ordering
                    # condition 1: when source and target conform to text ordering
                    if source_token_ids[0] < target_token_ids[0]:
                        clause_early_token_ids = source_token_ids
                        clause_late_token_ids = target_token_ids
                        clause_early_text = source_text
                        clause_late_text = target_text

                        # get token ids: need to exclude punctuations
                        clause_early_token_head_ids = [int(tok_id2head_id_dict[int(tokid)]) for tokid in
                                                       clause_early_token_ids if tok_id2deprel_dict[tokid] != "punct"]
                        clause_late_token_head_ids = [int(tok_id2head_id_dict[tokid]) for tokid in
                                                      clause_late_token_ids if tok_id2deprel_dict[tokid] != "punct"]

                        # if there's a token in clause early whose parent is in clause late,
                        # then clause_early is ARG 2, clause_late is ARG1
                        # else clause_early is ARG1, clause_late is ARG2
                        # Check each token in each arg and see if it is a syntactic child of a token in the other arg.
                        # [f(x) if condition else g(x) for x in sequence]
                        temp_check_clause_early = [1 if item in clause_late_token_ids else 0 for item in
                                                   clause_early_token_head_ids]
                        temp_check_clause_late = [1 if item in clause_early_token_ids else 0 for item in
                                                  clause_late_token_head_ids]
                        if temp_check_clause_early.count(1) != 0:
                            arg1 = clause_late_text
                            arg2 = clause_early_text
                            argn = "arg2" if "concession" not in sense and note != "explicit" else "arg1"
                            if sense in senses2adjust_dict:  # always the SOURCE (from RST notation)
                                sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                            arg1_ids = target_ids
                            arg2_ids = source_ids
                        else:
                            if temp_check_clause_late.count(1) != 0:
                                arg1 = clause_early_text
                                arg2 = clause_late_text
                            else:
                                arg1 = clause_early_text
                                arg2 = clause_late_text
                            argn = "arg1" if "concession" not in sense and note != "explicit" else "arg2"
                            if sense in senses2adjust_dict:  # always the SOURCE (from RST notation)
                                sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                            arg1_ids = source_ids
                            arg2_ids = target_ids

                    else:
                        # condition 2: when source and target DO NOT conform to text ordering
                        clause_early_token_ids = target_token_ids
                        clause_late_token_ids = source_token_ids
                        clause_early_text = target_text
                        clause_late_text = source_text

                        # get token ids
                        clause_early_token_head_ids = [int(tok_id2head_id_dict[int(tokid)]) for tokid in
                                                       clause_early_token_ids if
                                                       tok_id2deprel_dict[tokid] != "punct"]
                        clause_late_token_head_ids = [int(tok_id2head_id_dict[tokid]) for tokid in
                                                      clause_late_token_ids if
                                                      tok_id2deprel_dict[tokid] != "punct"]

                        # if there's a token in clause early whose parent is in clause late,
                        # then clause_early is ARG 2, clause_late is ARG1
                        # else clause_early is ARG1, clause_late is ARG2
                        # Check each token in each arg and see if it is a syntactic child of a token in the other arg.
                        # [f(x) if condition else g(x) for x in sequence]
                        temp_check_clause_early = [1 if item in clause_late_token_ids else 0 for item in
                                                   clause_early_token_head_ids]
                        temp_check_clause_late = [1 if item in clause_early_token_ids else 0 for item in
                                                  clause_late_token_head_ids]
                        if temp_check_clause_early.count(1) != 0:
                            arg1 = clause_late_text
                            arg2 = clause_early_text
                            argn = "arg1" if "concession" not in sense and note != "explicit" else "arg2"
                            if sense in senses2adjust_dict:  # always the SOURCE (from RST notation)
                                sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                            arg1_ids = source_ids
                            arg2_ids = target_ids
                        else:
                            if temp_check_clause_late.count(1) != 0:
                                arg1 = clause_early_text
                                arg2 = clause_late_text
                            else:
                                arg1 = clause_early_text
                                arg2 = clause_late_text
                            argn = "arg2" if "concession" not in sense and note != "explicit" else "arg1"
                            if sense in senses2adjust_dict:  # always the SOURCE (from RST notation)
                                sense = f"{sense}.{argn}-{senses2adjust_dict[sense]}"
                            arg1_ids = target_ids
                            arg2_ids = source_ids

                out_rel_type = rel_type
                if note == "implicit":
                    note = "intra_implicit"
                elif note == "explicit":
                    note = "sense2_explicit"

                if len(tok_ids) > 0 and tok_ids != [-1]:
                    note += "||conn_tok_ids=" + ",".join([str(tok_id) for tok_id in tok_ids])

            if note == "":
                note = "_"
            key = rel.key if rel.key != "" else "_"
            output.append([doc_name, out_rel_type, dm, sense, rst, arg1, arg2, arg1_ids, arg2_ids, key, note])

    return output


def remove_duplicates(outlist):
    def rank_types(s):
        if s == "norel":
            return 5
        elif s == "entrel":
            return 4
        elif s == "implicit":
            return 3
        elif s == "hypophora":
            return 2
        elif "altlex" in s:
            return 1
        elif s == "explicit":
            return 0

    seen = defaultdict(list)
    explicit = defaultdict(list)
    for rel_row in outlist:
        doc_name, rel_type, dm, sense, rst, arg1, arg2 = rel_row[:7]
        if "." in sense:
            sense = ".".join(sense.split(".")[:2])
        key = (doc_name, arg1, arg2)
        if rel_type not in ["explicit","altlex","altlexc"]:
            if key in seen:
                prev = seen[key][-1]
                if rank_types(rel_type) < rank_types(prev[1]):
                    seen[key] = [rel_row]
            else:
                if 'intra_implicit' in rel_row:  # Allowed to coexist with explicit
                    if (doc_name, arg1, arg2, "intra") not in seen:
                        seen[(doc_name, arg1, arg2, "intra")].append(rel_row)
                else:
                    seen[key].append(rel_row)
        else:
            if 'sense2_explicit' in rel_row:  # Allowed to have two explicit senses for manner (by = manner + purpose, .. etc.)
                explicit[(doc_name, arg1, arg2, "sense2_explicit")].append(rel_row)
            else:
                if key in explicit:
                    prev = explicit[key][-1]
                    prev_sense = prev[3]
                    if "." in prev_sense:
                        prev_sense = ".".join(prev_sense.split(".")[:2])
                    # Do not allow altlex(c)+explicit for same level 2 sense, but allow for different level 2 senses or multiple explicit entries
                    if rank_types(rel_type) < rank_types(prev[1]) and sense == prev_sense:
                        explicit[key] = [rel_row]
                        continue
                explicit[key].append(rel_row)

    output = []
    for key in explicit:
        output.extend(explicit[key])
    for key in seen:
        if key not in explicit:
            output.extend(seen[key])
    final = []
    for item in output:
        if item[2] == "BLOCK":
            continue
        final.append(item)
    return final


def arg_ordering_algo(rel_type):
    """
    Arg span ordering algorithm:
    - If the relation is implicit (intra- or inter-sentential) or entrel:
        - Arg2 is the later one in the text
    - Else if it is explicit (includes altlex, altlexc):
        - Check each token in each arg and see if it is a syntactic child of a token in the other arg.
        - If there is a tok in arg X that is a child of a tok in arg Y, but not the other way around, then arg Y is arg2
        - If there is no such token (or if there are crosswise tokens in BOTH args):
            - If there are connective tokens only in one of the two args, that is arg2
            - If there are connective tokens in BOTH args, arg2 is the later one in text order

    :param rel_type: a PDTB relation type: explicit, implicit, altlex, and altlexc
    :return: ordered argument spans
    """
    import pandas as pd

    # set up module object for each type
    if rel_type == "explicit":
        module = Explicit(data_dir=data_dir, direct_mappings_dir=mappings_dir, disco_pred_dir=disco_pred_dir)
    elif rel_type == "implicit":
        module = Implicit(data_dir=data_dir, direct_mappings_dir=mappings_dir, probs_mappings_dir=disco_pred_dir,
                          conn_preds_dir=conn_preds_dir)
    elif rel_type == "altlex":
        module = Altlex(data_dir=data_dir, direct_mappings_dir=mappings_dir)
    else:  # altlexc
        module = AltlexC(data_dir=data_dir, direct_mappings_dir=mappings_dir)

    df = [['DOCNAME', 'CONN', 'SENSE', 'RST', 'ARG1', 'ARG2', 'ARG1_IDS', 'ARG2_IDS', 'NOTES', 'KEY']]

    # start processing text units
    for doc_name in doc_names:
        if doc_name.startswith("GUM"):
            doc_state = read_file(os.path.join(data_dir, os.path.join("dep", doc_name+".conllu")),
                                  os.path.join("data", "rst", "rstweb", doc_name + ".rs4"), doc_name)
            bad_keys = 0
            total_keys = 0
            for rel_id in doc_state.rels:
                rel = doc_state.rels[rel_id]
                # order the arguments
                bad_key = module.convert(doc_state, rel)
                if bad_key:
                    bad_keys += bad_key
                total_keys += 1
                output = order_rel_args(rel, doc_state, filter=rel_type)
                if output is not None:
                    df += output

    df = pd.DataFrame(df)
    df.to_excel(os.path.join(output_dir, f"{rel_type}_discoARGs.xlsx"), index=False, header=False)


if __name__ == "__main__":
    # set up directories
    data_dir = os.path.join('data')
    output_dir = data_dir.replace("data", "output")
    disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')
    conn_preds_dir = os.path.join("implicit", "gum_test_data")
    mappings_dir = os.path.join(data_dir, 'mappings.json')

    doc_names = ['GUM_academic_librarians', 'GUM_bio_byron', "GUM_essay_dividends", 'GUM_fiction_rose',
                 'GUM_interview_cocktail', "GUM_letter_mandela", 'GUM_news_nasa', 'GUM_reddit_introverts',
                 'GUM_speech_albania', 'GUM_textbook_union', 'GUM_vlog_lipstick', 'GUM_voyage_cleveland', 'GUM_whow_elevator']

    # obtain ordered output
    #arg_ordering_algo("explicit")
    arg_ordering_algo("implicit")
