import io, os, sys, re
from collections import defaultdict
from copy import deepcopy


class Coref(object):
    """
    the class of coref units (span mentions), each mention contains text id, tokens, entity types, current coref,
    next coref, coref type (coref or bridging), head dep function, head pos tag, span's dep functions,
    span's pos tags, span length
    """
    def __init__(self):
        self.tsv_line = list()
        self.text_id = str()
        self.tok = str()
        self.lemma = str()
        self.e_type = str()
        self.cur = str()
        self.next = str()
        self.coref = {}
        self.coref_type = str()
        self.head_func = str()
        self.head_pos = str()
        self.func = str()
        self.pos = str()
        self.span_len = 1
        self.child_cop = bool()
        self.dep_appos = bool()
        self.appos = bool()
        self.acl_children = list()
        self.appos_father = str()
        self.seen = str()
        self.delete = bool()
        self.definite = bool()
        self.nmod_poss = bool()
        self.new_e = bool()
        self.replaced_by = str()
        self.appos_point_to = str()
        self.expanded = False
        self.verb_head = bool()
        self.verb_head_aux = list()
        self.sent = list()
        self.num = bool()


def count(doc):
    entities = defaultdict(int)
    prp, nnp, nn = 0, 0, 0
    total = 0
    COUNT_FLAG = False
    for k, v in doc.items():
        if v.cur:
            if v.next and v.tok:
                total += 1
                COUNT_FLAG = True
            else:
                for prev_k, prev_v in doc.items():
                    if prev_v.next == k:
                        total += 1
                        COUNT_FLAG = True
                        break

            if COUNT_FLAG:
                if 'NNP' in v.head_pos:
                    nnp += 1
                elif 'PRP' in v.head_pos:
                    prp += 1
                else:
                    nn += 1
                e = v.e_type
                entities[e] += 1
            COUNT_FLAG = False

    return prp, nnp, nn, entities


def coref_(fields: list) -> dict:
    """
    find the coref relations between the current entity and next entity
    example: 15-8[81_3]
    :param fields: elements in a line
    """
    coref = {}
    line_id = fields[0]
    all_mentions = {x.strip(']').split('[')[1] for x in fields[3].split('|')}
    seen = set()

    # when the entity has not coref relations except that it is pointed to by other entities
    if fields[-1] != '_':
        coref_types = fields[-2].split('|')
        for i, x in enumerate(fields[-1].split('|')):
            point_to = x.split('[')[0]
            cur_e = ''
            next_e = ''
            coref_type = ''
            if ']' in x:
                e_info = x.strip(']').split('[')[1].split('_')
                cur_e = e_info[1] if e_info[1] != '0' else ''
                next_e = e_info[0]
                if next_e == '0':
                    next_e = f'0_{point_to}'

                coref_type = coref_types[i]
                seen.add(cur_e)

            # e.g.  academic_art
            #       7-3	397-403	people	person	new	ana	8-7
            elif fields[-2] != '_':
                coref_type = coref_types[i]

            if 'bridge' in coref_type or 'cata' in coref_type:
                continue

            if cur_e in coref.keys():
                raise ValueError(f'The coref type {coref_type} has not been added into conversion at line {fields[0]}.')
            coref[cur_e] = (point_to, next_e, coref_type)

    # keep singletons
    for mention in all_mentions:
        if mention not in seen:
            coref[mention] = ('', '', '')

    # when the entity exists and it is mentioned before, but does not have next coref
    # e.g.  fiction_veronique
    #       23-2	939-941	He	person	giv	_	_
    #       45-1	2047-2048	I	person	giv	_	_
    # elif fields[3] != '_' and fields[4] == 'giv':
    #     coref[''] = ('', '', '')
    #
    # # to keep singletons
    # elif '[' in fields[3] and fields[-1] == '_':
    #     entities = fields[3].split('|')
    #     for entity in entities:
    #         cur_e = entity.strip(']').split('[')[1]
    #         coref[cur_e] = ('', '', '')

    return coref


def add_tsv(entities: dict, fields: list, coref_r: dict, doc: dict, entity_group: list, antecedent_dict):
    """
    add entities (unique id) in the current line to the doc
    :param entities: entity id and type
    :param fields: elements in a line
    :param coref_r: coref relations
    :param doc: coref in an article
    """
    for id, e in entities.items():
        text_id = fields[0]
        k = id

        if k in doc.keys(): # if the named entity id is more than one word, add to the existed id
            doc[k].tok += f" {fields[2]}"
            doc[k].span_len += 1

        else:   # if the word is the first token of an named entity, create a new Coref class
            if k in coref_r.keys() or '' in coref_r.keys():
                if_fake = True if '' in coref_r.keys() else False   # if the NE does not have id, create a fake id
                new_id = f'0_{text_id}' if if_fake else k
                if f'0_{text_id}' in doc.keys():
                    del doc[new_id]

                doc[new_id] = Coref()
                doc[new_id].text_id = text_id
                doc[new_id].tok = fields[2]
                doc[new_id].e_type = e
                doc[new_id].cur = new_id
                doc[new_id].coref = coref_r[id][0]
                doc[new_id].seen = fields[4].split('[')[0]
                doc[new_id].tsv_line = fields

                # if the coref does not a named entity
                # 7-3	397-403	people	person	new	ana	8-7
                # 8-7	472-476	they	person	giv	coref	9-9
                if coref_r[id][1] == '' and coref_r[id][0]:
                    doc[new_id].next = f'0_{coref_r[id][0]}'
                else:
                    doc[new_id].next = coref_r[id][1]

                doc[new_id].coref_type = coref_r[id][2]

                next_e = coref_r[id][1]

                # if the coref.next is not a named entity, create a fake entity id
                if (coref_r[id][1].startswith('0') or coref_r[id][1] == ''):
                    next_e = f'0_{coref_r[id][0]}'
                    if f'0_{coref_r[id][0]}' not in doc.keys():
                        fake_id = next_e
                        doc[fake_id] = Coref()
                        doc[fake_id].text_id = coref_r[id][0]
                        doc[fake_id].seen = fields[4]
                        doc[fake_id].tsv_line = fields

                # combine coref entities in group
                if_new_group = True
                for idx, g in enumerate(entity_group):
                    if new_id in g:
                        if_new_group = False
                        entity_group[idx].append(next_e)
                        break
                    if next_e in g:
                        if_new_group = False
                        entity_group[idx].append(new_id)
                if if_new_group:
                    entity_group.append([new_id, next_e])

                # antecedent info
                antecedent_dict[new_id] = next_e

            # if the current line does not have coref but it's the coref of the previous entity, add token info
            elif f'0_{text_id}' in doc.keys():
                fake_id = f'0_{text_id}'
                doc[fake_id].e_type = e
                doc[fake_id].tok = fields[2]
                doc[fake_id].seen = fields[4]
                doc[fake_id].tsv_line = fields

            # if no next coref, but has antecedent
            elif k in antecedent_dict.values():
                doc[k] = Coref()
                doc[k].text_id = text_id
                doc[k].tok = fields[2]
                doc[k].e_type = e
                doc[k].cur = id
                doc[k].coref = ''
                doc[k].next = ''
                doc[k].coref_type = ''
                doc[k].seen = [x.split('[')[0] for x in fields[4].split('|') if k in x][0]
                doc[k].tsv_line = fields

    return entity_group, antecedent_dict


def break_dep_doc(doc):
    sent_id = 0
    sents = defaultdict(list)
    for i, line in enumerate(doc):
        if line[0].startswith('# sent_id'):
            sent_id += 1
        elif len(line) > 1 and not line[0].startswith('#'):
            sents[sent_id].append(line)
    return sents


def find_acl_child(sent, head):
    """
    This function finds the dependency child whose dep func is acl (or acl:rel???) given the dep head as the function
    input. One head can have more than one acl children
    """
    acl_children = []
    for row in sent:
        if row[6] == head and 'acl' in row[7]:
            acl_children += find_all_dep_children(sent, [row[0]])
        elif row[6] == head and row[7] == 'acl:relcl':
            acl_children += find_all_dep_children(sent, [row[0]])

    return acl_children


def find_all_dep_children(sent, head):
    """
    This function recursively finds all children given a head token id
    :param sent: a list of tokens with dependency information
    :param head: the token id (the first column in a conllu format)
    :return: all token ids under the given head
    """
    for row in sent:
        if row[6] in head and row[0] not in head:
            head.append(row[0])
            find_all_dep_children(sent, head)
    return head


def find_direct_dep_children(sent, head):
    """
    This function recursively finds all children given a head token id
    :param sent: a list of tokens with dependency information
    :param head: type: string, the token id (the first column in a conllu format)
    :return: all token ids under the given head
    """
    head = head[0]  # to have the same input as the function "find_all_dep_children"
    children = []
    for row in sent:
        if row[6] == head:
            children.append(row[0])
    return children


def check_appos(sent: list, head_row: list, dep_sent_id) -> bool:
    """
    check GUM_fiction_veronique, sent 31, the word "history"
    """
    appos = []
    head_id, head_head_id, head_func = head_row[0], head_row[6], head_row[7]
    for row in sent:
        if row[6] == head_id and row[7] == 'appos':
            appos = sorted([f'{dep_sent_id}-{x}' for x in find_all_dep_children(sent, [row[0]])], reverse=True)

            # if the last token is a period or comma, do not count it as part of an apposition
            # NOT_COUNT = [',', '.', ';', '-', '!', '?']
            last_tok_id = str(sorted([int(x.split('-')[-1]) for x in appos])[-1])
            last_tok = [y[1] for y in sent if y[0] == last_tok_id]
            # if last_tok in NOT_COUNT:
            #     appos.remove(f'{dep_sent_id}-{last_tok_id}')
            return appos
    return appos


def check_dep_cop_child(sent: list, head_range: list, head_row: list) -> bool:
    """
    check GUM_academic_art, sent 25 and GUM_academic_games, sent 6. correct √

    check GUM_fiction_veronique, sent 41-6, "Earth" in "we are from [Earth]" is not copula
    """
    head_id, head_head_id, head_func = head_row[0], head_row[6], head_row[7]
    for row in sent:
        if row[0] not in head_range and row[7] == 'cop':
            if row[6] == head_id:
                children = find_direct_dep_children(sent, head_id)
                for r in sent:
                    if r[0] in children and r[7] == 'case':
                        return False
                return True
            elif head_func == 'conj' and row[6] == head_head_id:
                return True
    return False


def process_doc(dep_doc, coref_doc):
    """
    align dep lines in conllu and coref lines in tsv, extract coref-related information for each entity id
    """
    doc = {}
    antecedent_dict = {}
    entity_group = [[]]
    tokens = []

    for coref_line in coref_doc:
        if coref_line.startswith('#'):
            continue
        elif coref_line:
            coref_fields = coref_line.strip().split('\t')
            line_id, token = coref_fields[0], coref_fields[2]

            # test
            if line_id == '32-25':
                a = 1
            if coref_fields[5] == 'appos':
                a = 1

            if coref_fields[3] == '_' and coref_fields[4] == '_':
                tokens.append((token, line_id, [], {}))
                continue

            # entity info
            entities = {'' if '[' not in x else x.strip(']').split('[')[1]: x.split('[')[0]
                        for x in coref_fields[3].split('|')}

            # coref info
            coref = coref_(coref_fields)
            tokens.append((token, line_id, list(entities.keys()), coref))

            # tsv
            add_tsv(entities, coref_fields, coref, doc, entity_group, antecedent_dict)

            a = 1

    # map text_id and entity
    id_e = {}
    for k, v in doc.items():
        if v.text_id not in id_e.keys():
            id_e[v.text_id] = []
        id_e[v.text_id].append((v.span_len, k))
    # id_e = {v.text_id: (v.span_len, k) for k,v in doc.items()}

    # break the dep conllu into sents
    dep_sents = break_dep_doc(dep_doc)

    # new ids, e.g. appos
    new_id2entity = {}

    # dep
    dep_sent_id = 0
    for i, dep_line in enumerate(dep_doc):
        if dep_line[0].startswith('# sent_id'):
            dep_sent_id += 1
        elif dep_line[0].startswith('#'):
            continue
        elif len(dep_line) > 1:

            # match dep_text_id to the format in coref tsv
            dep_text_id = f'{dep_sent_id}-{dep_line[0]}'
            cur_dep_sent = dep_sents[dep_sent_id]
            if dep_text_id == '6-27':
                a = 1

            # TODO: 将ide替换为doc.keys()，避免18-26 "these techniques"没有任何dep的信息
            if dep_text_id in id_e:
                for e in id_e[dep_text_id]:
                    span_len, entity = e[0], e[1]

                    # check dep appos, if there is another e having appositions, don't count the current one in dep appos
                    another_appos = False
                    for ee in id_e[dep_text_id]:
                        if ee[1] != entity and doc[ee[1]].coref_type == 'appos':
                            another_appos = True
                            break

                    # if the entity is deleted, check if it is replaced by another entity
                    if doc[entity].replaced_by:
                        span_len, entity = doc[doc[entity].replaced_by].span_len, doc[entity].replaced_by

                    # find dep information and index for each word in heads
                    heads = [dep_doc[x] for x in range(int(i), int(i)+span_len) if len(dep_doc[x]) == 10]
                    head_range = [dep_doc[x][0] for x in range(int(i), int(i)+span_len) if len(dep_doc[x]) == 10]
                    head_of_the_phrase = ''

                    # loop each word in the head to find the head_func/head_pos/if_cop_in_dep for the entity
                    for row in heads:

                        # if find the ROOT
                        if row[6] == '0':
                            doc[entity].head = row[1]
                            doc[entity].lemma = row[2]
                            doc[entity].head_func = row[7]
                            doc[entity].head_pos = row[4]
                            doc[entity].head_id = f'{dep_sent_id}-{row[0]}'
                            # check if the head has a copula child
                            doc[entity].child_cop = check_dep_cop_child(cur_dep_sent, head_range, row)
                            # check whether appos is a child of the current head
                            doc[entity].dep_appos = True if row[7] == 'appos' and not another_appos else False
                            # if doc[entity].coref_type == 'appos' or doc[entity].dep_appos:
                            #     doc[entity].appos = check_appos(cur_dep_sent, row, dep_sent_id)
                            doc[entity].acl_children = find_acl_child(cur_dep_sent, row[0])
                            head_of_the_phrase = row[0]

                            # verbal span contraction
                            if row[4].startswith('V'):
                                doc[entity].verb_head = True

                        # if the head is outside the range, it's the head of the entity
                        elif doc[entity].head_func == '' and row[6] not in head_range:
                            doc[entity].head = row[1]
                            doc[entity].lemma = row[2]
                            doc[entity].head_func = row[7]
                            doc[entity].head_pos = row[4]
                            doc[entity].head_id = f'{dep_sent_id}-{row[0]}'
                            doc[entity].child_cop = check_dep_cop_child(cur_dep_sent, head_range, row)
                            # check whether appos is a child of the current head
                            doc[entity].dep_appos = True if row[7] == 'appos' and not another_appos else False
                            # if doc[entity].coref_type == 'appos' or doc[entity].dep_appos:
                                # doc[entity].appos = check_appos(cur_dep_sent, row, dep_sent_id)
                            doc[entity].acl_children = find_acl_child(cur_dep_sent, row[0])
                            head_of_the_phrase = row[0]

                            # verbal span contraction
                            if row[4].startswith('V'):
                                doc[entity].verb_head = True

                        doc[entity].func += f' {row[7]}'
                        doc[entity].pos += f' {row[4]}'
                    doc[entity].func = doc[entity].func.strip()
                    doc[entity].pos = doc[entity].pos.strip()
                    doc[entity].sent = cur_dep_sent

                    # check if the span is headed by numbers. (a conservative way is to only consider the span with length of 1)
                    if len(heads) == 1 and heads[0][4] == 'CD':
                        doc[entity].num = True

                    # double check the verbal head
                    # this function is to check copula that are ignored by the previous checking step
                    # Example: if a cat is a good mouse hunter -> is, verbal head: hunter
                    head_id = doc[entity].head_id.split('-')[-1]
                    for row in heads:
                        row_head = row[6]
                        if row_head == head_id and row[2] == 'be':
                            doc[entity].verb_head_aux = [True, f'{dep_sent_id}-{row[0]}']

                    # check acl children
                    '''
                    If there is a gap within the current entity's acl children, fill it
                    '''
                    cur_span = [str(int(doc[entity].text_id.split('-')[-1])+x) for x in range(doc[entity].span_len)]
                    for x in doc[entity].acl_children:
                        if x in cur_span:
                            doc[entity].acl_children.remove(x)
                    # PUNCT = ['-', ',', '–', '[', '4']
                    if doc[entity].acl_children:
                        GAP_FLAG = True
                        min_acl = min([int(x) for x in doc[entity].acl_children])
                        max_acl = max([int(x) for x in doc[entity].acl_children])
                        for acl_id in range(min_acl+1, max_acl):
                            if str(acl_id) not in cur_span:
                                GAP_FLAG = False
                            if str(acl_id) not in doc[entity].acl_children:
                                doc[entity].acl_children.append(str(acl_id))
                                # print(cur_dep_sent[int(acl_id)-1])

                        if GAP_FLAG:
                            doc[entity].expanded = True

                        '''
                        Some punctuaction marks such as '-' are not included in acl children, expand it to make sure the
                        span is continuous
                        '''
                        min_expand = min([int(x) for x in doc[entity].acl_children])
                        max_tok = max([int(x) for x in head_range])
                        if min_expand - max_tok > 1:
                            for l in dep_sents[dep_sent_id]:
                                if int(l[0]) > max_tok and int(l[0]) < min_expand:
                                    # if l[1] in PUNCT:
                                    # ignore the PUNCT to avoid some outliers
                                    doc[entity].acl_children.append(l[0])
                                    # else:
                                    #     raise ValueError('The acl children gap is not correctly recognized.')

                    # check definiteness
                    """
                    If it's in the following cases, definite √:
                        - NP/NPS
                        - Pronouns (PP, PP$, DT as a head)
                        - Anything which has deprel 'det' with a lemma 'the/this/that/those/these/all/every..' Maybe take
                        a look at all lemmas with tok_func="det" in GUM to make a good list of the definite determiners
                        - Anything possessed (either by 's/' or by a possessive determiner, my, your…, or a genitive 
                        pronoun like like 'its')
                    """
                    HEAD_POS = ['PRP', 'PRP$', 'DT']
                    LEMMA = ['the', 'this', 'that', 'those', 'these', 'all', 'every', 'any', 'no', 'such']
                    MISC = ['Everyone', 'everyone', 'Anyone', 'anyone']
                    if entity == '186':
                        a = 1
                    if doc[entity].head_pos == 'FW':
                        doc[entity].definite = True
                    elif 'NP' in doc[entity].head_pos:
                        doc[entity].definite = True
                    elif doc[entity].head_pos in HEAD_POS:
                        doc[entity].definite = True
                    elif heads[0][6] == head_of_the_phrase and heads[0][2] == 'such':
                        doc[entity].definite = True
                    elif doc[entity].pos.startswith('PRP$'):
                        doc[entity].definite = True
                    elif doc[entity].lemma in MISC:
                        doc[entity].definite = True
                    elif doc[entity].head_func == 'xcomp':
                        for prev_k, prev_v in doc.items():
                            if prev_v.next == entity and prev_v.text_id.split('-')[0] == doc[entity].text_id.split('-')[0] and prev_v.tok in ['himself', 'myself', 'herself']:
                                doc[entity].definite = True
                                break
                    else:
                    # elif 'det' in doc[entity].func:
                        for row in heads:
                            if row[7] == 'det' and row[2] in LEMMA and row[6] == head_of_the_phrase:
                                doc[entity].definite = True
                    # elif 'POS' in doc[entity].pos:
                        for row in heads:
                            if row[4] == 'POS' and row[6] == doc[entity].head_id.split('-')[-1] and row[6] == head_of_the_phrase:
                                doc[entity].definite = True

                    if doc[entity].head_func == '':
                        # raise ValueError('The head feature is empty.')
                        print('Warning: The head feature is empty.')

    # group dict
    entity_group = [x for x in entity_group if x]
    group_id = 0
    group_dict = {}
    for lst in entity_group:
        group_id += 1
        for x in lst:
            group_dict[x] = group_id

    return doc, tokens, group_dict, antecedent_dict, new_id2entity, dep_sents