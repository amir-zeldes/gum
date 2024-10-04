import sys

from modules.explicit import Explicit
from modules.implicit import Implicit
import os
from process import read_file


script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

data_dir = os.path.join('data')
disco_pred_dir = os.path.join(data_dir, 'discodisco_preds')
mappings_dir = os.path.join(data_dir, 'mappings.json')
conllu_dir = script_dir + ".." + os.sep + ".." + os.sep + "target" + os.sep + "dep" + os.sep + "not-to-release" + os.sep
doc_names = [file.split('.')[0] for file in os.listdir(conllu_dir) if file.endswith('.conllu')]

conn_preds_dir = os.path.join(data_dir, 'connector_preds')


def make_span_contain_dm(doc_state, rel, tok_ids, dm, source_ids, target_ids, verbose=False):
    def expand(lone_dm, edu_ids):
        old_edu_ids = edu_ids
        if any(lone_dm):
            lone_dm_tok_id = tok_ids[lone_dm.index(1)]  # assuming only 1 such EDU
            lone_dm_edu_id = [edu_id for edu_id, edu in doc_state.edus.items() if lone_dm_tok_id in edu.tok_ids][0]
            if lone_dm_edu_id < edu_ids[0]:
                edu_ids = list(range(lone_dm_edu_id, edu_ids[0])) + edu_ids
            elif lone_dm_edu_id > edu_ids[-1]:
                edu_ids = edu_ids + list(range(edu_ids[-1]+1, lone_dm_edu_id+1))
            else:  # EDU that contains the DM is somehow in the middle of current EDU IDs
                edu_ids.append(lone_dm_edu_id)
            if verbose:
                # print(f"Sentence clipping resulted in Argspans *not* containing the DMs, so we've added EDU [{lone_dm_edu_id}] to make sure the DM '{dm}' is included. docname: {doc_state.docname}.")
                print(f"{doc_state.docname}: added EDU [{lone_dm_edu_id}] to make sure the DM '{dm}' is included.")
                print(f"Old: {old_edu_ids}")
                print(f"-> New: {edu_ids}")
                print(f"Old: {' '.join([doc_state.edus[edu_id].text for edu_id in old_edu_ids])}")
                print(f"-> New: {' '.join([doc_state.edus[edu_id].text for edu_id in edu_ids])}\n")
        return sorted(edu_ids)

    # expand edus until DM is included in the span
    tok_ids.sort()  # make sure these tok_ids are all in the span
    # first determine which span contains the DM (could be both, e.g. if... || then...)
    source_tok_ids, target_tok_ids = [], []  # current source and target IDs in terms of TOKEN
    tok_ids = tok_ids
    for edu_id in source_ids:
        source_tok_ids.extend(doc_state.edus[edu_id].tok_ids)
    for edu_id in target_ids:
        target_tok_ids.extend(doc_state.edus[edu_id].tok_ids)
    in_source = set(tok_ids).intersection(set(rel.source.tok_ids))  # DM tok IDs that are present in the *entire* source span
    in_target = set(tok_ids).intersection(set(rel.target.tok_ids))  # DM tok IDs that are present in the *entire* target span
    lone_dm_source = [int(tok not in source_tok_ids) for tok in sorted(list(in_source))]
    lone_dm_target = [int(tok not in target_tok_ids) for tok in sorted(list(in_target))]
    if in_source and in_target:  # if both source and target contain the DM
        source_ids = expand(lone_dm_source, source_ids)
        target_ids = expand(lone_dm_target, target_ids)
    elif in_source:
        source_ids = expand(lone_dm_source, source_ids)
    # TODO expand
    elif in_target:
        target_ids = expand(lone_dm_target, target_ids)
    # TODO expand
    else:  # if dm in neither
        # Attempt to find which unit's sentence contains the DM
        source_sent_id = rel.source.sent_ids[0]
        target_sent_id = rel.target.sent_ids[0]
        source_sent_edus = [edu_id for edu_id, edu in doc_state.edus.items() if edu.sent_id == source_sent_id]
        target_sent_edus = [edu_id for edu_id, edu in doc_state.edus.items() if edu.sent_id == target_sent_id]
        source_sent_tok_ids = [tok_id for edu_id in source_sent_edus for tok_id in doc_state.edus[edu_id].tok_ids]
        target_sent_tok_ids = [tok_id for edu_id in target_sent_edus for tok_id in doc_state.edus[edu_id].tok_ids]
        in_source_sent = set(tok_ids).intersection(set(source_sent_tok_ids))
        in_target_sent = set(tok_ids).intersection(set(target_sent_tok_ids))
        if in_source_sent and in_target_sent:
            source_ids = expand([1], source_ids)
            target_ids = expand([1], target_ids)
        elif in_source_sent:
            source_ids = expand([1], source_ids)
        elif in_target_sent:
            target_ids = expand([1], target_ids)
        else:
            source_text = " ".join([doc_state.edus[x].text for x in source_ids]).lower()
            target_text = " ".join([doc_state.edus[x].text for x in target_ids]).lower()
            if tok_ids != [] or dm.lower() not in source_text and dm.lower() not in target_text:  # Check if the string is actually in there and we had empty token IDs
                sys.stderr.write(f"WARN: DM '{dm}' not found in either source or target spans in {doc_state.docname} for relation {rel.relname} with head EDU {rel.head_edu} - argspan may be incorrect\n")

    return source_ids, target_ids


def main():
    explicit = Explicit(data_dir=data_dir, direct_mappings_dir=mappings_dir, disco_pred_dir=disco_pred_dir)
    """
    1. for rel object, if pdtb rel exists:
    2. list all the EDU ids by: [edu.edu_id for edu in rel.source.edus] for source & target
    3. if multi-sentence, take the head EDU and return a single sentence that contains the head EDU
    4. for same-unit -> for each, if same unit exists (EDU.same_unit
    5. get ' '.join([edu.text for edu in rel.source.edus]) for source/target (.raw_text does the same, but only for head_edu)
    6. get doc_name, conn, sense_label
    """

    df = [['DOCNAME', 'CONN', 'SENSE', 'RST', 'SOURCE', 'TARGET', 'SOURCE_IDS', 'TARGET_IDS', 'same-unit?', 'note']]
    for doc_name in doc_names:
        try:
            doc_state = read_file(os.path.join("data", "dep", doc_name + '.conllu'),
                                  os.path.join("data", "rst", "rstweb", doc_name + ".rs4"), doc_name)
            for rel_id in doc_state.rels:
                rel = doc_state.rels[rel_id]
                rst = rel.relname
                explicit.convert(doc_state, rel)
                same_unit_t = ''  # if target contains same unit
                same_unit_s = ''  # if source contains same unit
                for pdtb_rel in rel.pdtb_rels['explicit']:
                    note, sense, tok_ids, dm = pdtb_rel
                    # source
                    if len(rel.source.sent_ids) > 1:  # if multi-sentencial
                        sent_id = rel.source.head_edu[0].sent_id  # head_edu should be a list of length 1
                        # getting all edu ids whose sentence id is the same as head_edu's sentence id
                        source_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id]
                    else:
                        source_ids = [edu.edu_id for edu in rel.source.edus]
                    # target
                    if len(rel.target.sent_ids) > 1:  # if multi-sentencial
                        sent_id = rel.target.head_edu[0].sent_id  # head_edu should be a list of length 1
                        # getting all edu ids whose sentence id is the same as head_edu's sentence id
                        target_ids = [edu for edu in doc_state.edus if doc_state.edus[edu].sent_id == sent_id]
                    else:
                        target_ids = [edu.edu_id for edu in rel.target.edus]
                    for edu_id in source_ids:
                        if doc_state.edus[edu_id].same_unit: same_unit_s = 'source'
                        source_ids.extend(list(doc_state.edus[edu_id].same_unit))
                    for edu_id in target_ids:
                        if doc_state.edus[edu_id].same_unit: same_unit_t = 'target'
                        target_ids.extend(list(doc_state.edus[edu_id].same_unit))
                    source_ids = sorted(list(set(source_ids)))
                    target_ids = sorted(list(set(target_ids)))
                    # uncomment the line below if the EDU span expansion causes any issues
                    if dm != "_" and dm != "":
                        source_ids, target_ids = make_span_contain_dm(doc_state, rel, tok_ids, dm,
                                                                  source_ids, target_ids)
                    source_text = ' '.join([doc_state.edus[edu].text for edu in source_ids])
                    target_text = ' '.join([doc_state.edus[edu].text for edu in target_ids])
                    doc_name = doc_state.docname
                    same_unit = '-'.join([same_unit_s, same_unit_t])
                    df.append([doc_name, dm, sense, rst, source_text, target_text, source_ids, target_ids, same_unit, note])

        except:
            continue


#    df = pd.DataFrame(df)
#    df.to_excel("explicit.xlsx", index=False, header=False)


def main_implicit():
    implicit = Implicit(data_dir=data_dir, direct_mappings_dir=mappings_dir, probs_mappings_dir=disco_pred_dir, conn_preds_dir=conn_preds_dir)
    """
    1. for rel object, if pdtb rel exists:
    2. list all the EDU ids by: [edu.edu_id for edu in rel.source.edus] for source & target
    3. for same-unit -> for each, if same unit exists (EDU.same_unit
    4. get ' '.join([edu.text for edu in rel.source.edus]) for source/target (.raw_text does the same, but only for head_edu)
    5. get doc_name, conn, sense_label
    """
    df = [['DOCNAME', 'CONN', 'SENSE', 'ARG1', 'ARG2', 'ARG1_IDS', 'ARG2_IDS', "RST_rel"]]
    bad_keys = 0
    total_keys = 0
    for doc_name in doc_names:
        doc_state = read_file(os.path.join("data", "dep", doc_name + '.conllu'),
                          os.path.join("data", "rst", "rstweb", doc_name + ".rs4"), doc_name)
        for rel_id in doc_state.rels:
            total_keys += 1
            rel = doc_state.rels[rel_id]
            rst_rel = rel.relname
            bad_key = implicit.convert(doc_state, rel)
            if bad_key:
                bad_keys += bad_key
            for pdtb_rel in rel.pdtb_rels['implicit']:
                _, sense, tok_ids, dm = pdtb_rel
                source_ids = [edu.edu_id for edu in rel.source.edus]
                target_ids = [edu.edu_id for edu in rel.target.edus]
                source_ids = sorted(list(set(source_ids)))
                target_ids = sorted(list(set(target_ids)))

                source_sent_id = rel.source.sent_ids[0]
                target_sent_id = rel.target.sent_ids[0]

                source_sent = doc_state.sents[source_sent_id]
                target_sent = doc_state.sents[target_sent_id]

                # fixed, should be entire setnence text now
                source_text = source_sent.plain_text
                target_text = target_sent.plain_text

                if source_sent_id < target_sent_id:
                    arg1 = source_text
                    arg1_ids = source_ids
                    arg2 = target_text
                    arg2_ids = target_ids
                elif target_sent_id < source_sent_id:
                    arg1 = target_text
                    arg1_ids = target_ids
                    arg2 = source_text
                    arg2_ids = source_ids

                # source_text = ' '.join([doc_state.edus[edu].text for edu in source_ids])
                # target_text = ' '.join([doc_state.edus[edu].text for edu in target_ids])

                doc_name = doc_state.docname
                df.append([doc_name, dm, sense, arg1, arg2, source_ids, target_ids, rst_rel])
    print("Bad Keys:", bad_keys)
    print("Total Keys:", total_keys)
    # df = pd.DataFrame(df)
    # df.to_excel("implicit_discoV2.xlsx", index=False, header=False)


if __name__ == "__main__":
    main()
