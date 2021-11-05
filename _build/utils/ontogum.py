import io
import os
import sys
import re
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
            #       7-3 397-403 people  person  new ana 8-7
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
    #       23-2    939-941 He  person  giv _   _
    #       45-1    2047-2048   I   person  giv _   _
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
                # 7-3   397-403 people  person  new ana 8-7
                # 8-7   472-476 they    person  giv coref   9-9
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


def check_dep_cop_child(sent: list, head_range: list, head_row: list, starting_heads_id: str) -> bool:
    """
    check GUM_academic_art, sent 25 and GUM_academic_games, sent 6. correct √

    check GUM_fiction_veronique, sent 41-6, "Earth" in "we are from [Earth]" is not copula
    """
    head_id, head_head_id, head_func = head_row[0], head_row[6], head_row[7]
    for row in sent:
        if row[0] not in head_range and row[7] == 'cop':
            if row[6] == head_id:
                # if the token is right before the span, and the token is a preposition, the span should not be removed
                prev_pos = sent[int(starting_heads_id) - 2][3]
                if prev_pos == 'ADP':
                    return False
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
                    starting_heads_id = head_range[0]
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
                            doc[entity].child_cop = check_dep_cop_child(cur_dep_sent, head_range, row, starting_heads_id)
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
                            doc[entity].child_cop = check_dep_cop_child(cur_dep_sent, head_range, row, starting_heads_id)
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
                        # print('Warning: The head feature is empty.')
                        a = 1

    # group dict
    entity_group = [x for x in entity_group if x]
    group_id = 0
    group_dict = {}
    for lst in entity_group:
        group_id += 1
        for x in lst:
            group_dict[x] = group_id

    return doc, tokens, group_dict, antecedent_dict, new_id2entity, dep_sents


class Convert(object):
    def __init__(self, doc: dict, antecedent_dict, group_dict, if_appos, if_singletons):
        self.antecedent_dict = antecedent_dict
        self.group_dict = group_dict
        self.doc = doc
        self.if_appos = if_appos
        self.if_singletons = if_singletons

    def _remove_junk(self):
        if '0_' in self.doc.keys():
            del self.doc['0_']

    def _expand_acl(self):
        """
        If the acl contains a coref, expand the span to its head if the head does not indicate a span
        :return: update self.docs

        to restore the conll: '\n'.join(['\t'.join(x) for x in doc[i]])
        """
        loop_doc = deepcopy(self.doc)
        for k, _coref in loop_doc.items():
            sent_id = self.doc[k].text_id.split('-')[0]
            if _coref.acl_children and not _coref.expanded:
                if k.startswith('0_'):  # if the original entity has only one token, create a new fake id to handle the expanded acl

                    # create a new fake id
                    new_k = str(self.last_e + 1)
                    self.last_e += 1
                    self.doc[new_k] = deepcopy(self.doc[k])
                    self.doc[new_k].cur = new_k

                    # find prev
                    for prev, prev_coref in self.doc.items():
                        if prev_coref.next and prev_coref.next == k:
                            self.doc[prev].next = new_k
                            break

                    # reset the original k
                    self.doc[k] = Coref()
                    self.doc[k].delete = True

                    # update new_id2entity
                    cur_id = self.doc[new_k].text_id
                    ori_e = k.split('_')[-1]
                    ids = [f'{sent_id}-{tok_id}' for tok_id in self.doc[new_k].acl_children] + [cur_id]

                    # update last_e
                    self.last_e += 1

                else:
                    new_k = k
                    cur_id = self.doc[new_k].text_id
                    ori_e = k.split('_')[-1]
                    ids = [f'{sent_id}-{tok_id}' for tok_id in self.doc[new_k].acl_children] + [cur_id]

                # find the beginning id of the entity
                '''
                Sometimes the acl span precedes the entity
                - Example:
                    1   Born    bear    VERB    VBN Tense=Past|VerbForm=Part    5   acl _   _
                    2   in  in  ADP IN  _   3   case    _   _
                    3   England England PROPN   NNP Number=Sing 1   obl _   Entity=(place-12)|SpaceAfter=No
                    4   ,   ,   PUNCT   ,   _   1   punct   _   _
                    5   Norton  Norton  PROPN   NNP Number=Sing 6   nsubj   _   Entity=(person-1)
                '''
                min_id = min([int(x.split('-')[-1]) for x in ids])
                if f'{sent_id}-{min_id}' != self.doc[new_k].text_id:
                    self.doc[new_k].text_id = f'{sent_id}-{min_id}'

                for id in ids:
                    if id in self.new_id2entity.keys():
                        if ori_e in self.new_id2entity[id]:
                            self.new_id2entity[id] = [new_k if i == ori_e else i for i in self.new_id2entity[id]]
                        else:
                            self.new_id2entity[id].append(new_k)
                    else:
                        self.new_id2entity[id] = [new_k]
                cur_span = [str(int(self.doc[new_k].text_id.split('-')[-1]) + x) for x in
                            range(self.doc[new_k].span_len)]
                self.doc[new_k].span_len += len([x for x in _coref.acl_children if x not in cur_span])

    def _verb_contract(self):
        """
        Contract the verbal markables to the verbs

        example:
            [I want to go to Rome.] [This] angered Kim.  -> ON: [want]
            I want [to go to Rome]. Kim wants to do [that] too. -> ON: [go]
        """
        loop_doc = deepcopy(self.doc)
        for k, _coref in loop_doc.items():
            if _coref.verb_head_aux:
                kept, kept_head = _coref.verb_head_aux
            elif _coref.verb_head:
                kept, kept_head = _coref.verb_head, _coref.head_id
            else:
                kept, kept_head = False, ''

            if k in self.doc.keys() and kept and _coref.next in self.doc.keys() and _coref.next != '':

                # (1) redirect the current markable and (2) create a new id to shorten the markable span
                new_id = str(self.last_e + 1)
                self.last_e += 1

                self.doc[new_id] = deepcopy(_coref)
                self.doc[new_id].cur = new_id
                self.doc[new_id].span_len = 1
                self.doc[new_id].text_id = kept_head
                self.doc[new_id].tok = ''
                if kept_head not in self.new_id2entity.keys():
                    self.new_id2entity[kept_head] = []
                self.new_id2entity[kept_head].append(new_id)

                # modify the antecedent
                for prev_k, prev_v in loop_doc.items():
                    if prev_k in self.doc.keys() and prev_v.next == k:
                        self.doc[prev_k].next = new_id

                # delete the original markable
                self.doc[k].delete = True

    def _remove_compound(self):
        """
        if the func is 'compound' and (1) not NNP/NNPS, (2) has coref, remove the current coref relation
            # if the current coref relation has previous and next coref, connect the two

        example: Allergan Inc. said it received approval to sell the PhacoFlex
                intraocular lens, the first foldable silicone lens available
                for [cataract surgery]*. The lens' foldability enables it to be
                inserted in smaller incisions than are now possible for
                [cataract surgery]*.
        * means two mention spans are not annotated as coreferences.

        Question: [a collaborative pilot project] ... [this (det) pilot project]
        """
        for k, _coref in self.doc.items():
            # currently remove the question example. To avoid deleting such examples, add another condition
            # "_coref.pos not startswith('det')"
            if 'compound' == _coref.head_func and ('NNP' not in _coref.pos and 'NNPS' not in _coref.pos):

                # find prev and next
                prev_k = ''
                for prev, _prev_coref in self.doc.items():
                    if _prev_coref.next and _prev_coref.next == k:
                        prev_k = prev
                        break
                if prev_k and _coref.next:
                    self.doc[prev_k].next = ''
                    self.doc[prev_k].coref = ''
                    self.doc[prev_k].coref_type = ''

                self.doc[k] = Coref()
                self.doc[k].delete = True

    def _remove_bridge(self):
        """
        if the coref type contains "bridge" (also include "bridge:aggr"), remove that coref relation
        """
        for k, _coref in self.doc.items():
            if 'bridge' in _coref.coref_type:
                self.doc[k].coref = ''
                self.doc[k].next = ''
                self.doc[k].coref_type = ''

    def _remove_cop(self):
        """
        if (1) two coref mentions are in the same sentence, (2) a copula dep func is the child of the first one,
        remove the coref relations between the two and move the second coref to the first one

        A -> B & B -> C ====> A -> C
        remove B
        Also, the function should handle multiple copula, e.g. copula in coordination
        Example: [He] was [one of the first opponents of the Vietnam War] , and is [a self-described Libertarian Socialist] .
        """
        for k, _coref in self.doc.items():
            # A -> B1..B2 -> C
            # remove B1, B2, ... Bn
            cur_v = deepcopy(self.doc[k])
            while cur_v.head_func != '' and cur_v.next in self.doc.keys() and self.doc[cur_v.next].child_cop == True:
                cur_v = deepcopy(self.doc[cur_v.next])

                self.doc[cur_v.cur].coref = ''
                self.doc[cur_v.cur].next = ''
                self.doc[cur_v.cur].coref_type = ''

                if cur_v.next:
                    self.doc[k].coref = cur_v.coref
                    self.doc[k].next = cur_v.next
                    self.doc[k].coref_type = cur_v.coref_type
                else:
                    self.doc[k].coref = ''
                    self.doc[k].next = ''
                    self.doc[k].coref_type = ''

                self.doc[cur_v.cur] = Coref()
                self.doc[cur_v.cur].delete = True


    def _break_chain(self):
        """
        If two mentions do not obey the specificity scale, break the coref chain

        simple way to break currently:
        the > a

        """
        for k, _coref in self.doc.items():
            if _coref.appos:
                continue
            if _coref.next and _coref.next != '0' and _coref.next in self.doc.keys() and self.doc[_coref.next].definite == False:
                """
                WARNING: This is not a valid operation but avoids some annotation errors
                """
                # If the chain's next is a definite entity and there are in the same sentence, do not break the chain
                next_next = deepcopy(self.doc[_coref.next].next)
                if next_next and (self.doc[next_next].lemma in ['he', 'she'] or self.doc[next_next].pos == 'NNP'):
                    next_sent_id = self.doc[_coref.next].text_id.split('-')[0]
                    next_next_sent_id = self.doc[next_next].text_id.split('-')[0]
                    if int(next_next_sent_id) <= int(next_sent_id) + 2:
                        print(f'Warning: Skip breaking chains in Line {next_sent_id}.')
                        continue

                break_group = max(self.group_dict.values()) + 1
                next_coref = self.doc[_coref.next]
                while next_coref.cur in self.antecedent_dict.keys():
                    # avoid cataphora that the coref points to itself
                    # E.g. GUM_fiction_pag
                    #      55-1 3945-3954   Something   person  giv cata    55-1
                    if next_coref.text_id == next_coref.next or f'0_{next_coref.text_id}' == next_coref.next:
                        break

                    self.group_dict[self.doc[next_coref.cur].cur] = break_group

                    # it is a repeated operation, but help with the last coref, which will not appear in the antecedent_dict.keys()
                    # if self.doc[next_coref.next].cur in self.group_dict.keys():
                    if next_coref.next in self.doc.keys():
                        self.group_dict[self.doc[next_coref.next].cur] = break_group
                        next_coref = self.doc[next_coref.next]
                    else:
                        break

                self.doc[k].coref = ''
                self.doc[k].next = ''
                self.doc[k].coref_type = ''

                # break the coref chain between cur and next
                if k in self.antecedent_dict.keys():
                    del self.antecedent_dict[k]


    def _appos_merge(self):
        """
        If coref_type==appos, merge the appos into the current coref

        Example:
            from :
                27-40   3795-3798   the object[220] new[220]    appos   27-42[0_220]
                27-41   3799-3803   13th    object[220] new[220]    _   _
                27-42   3804-3812   Benjamin    object  giv coref   27-45[221_0]
            to:
                27-40   3795-3798   the object[220]|object[999] new[220]|new[999]   coref|appos 27-45[221_220]|27-42[0_999]
                27-41   3799-3803   13th    object[220]|object[999] new[220]|new[999]   _   _
                27-42   3804-3812   Benjamin    object||object[220] giv|new[220]    _   _
        """
        loop_doc = deepcopy(self.doc)
        for k1, v in loop_doc.items():
            k1_sent_id = v.text_id.split('-')[0]
            for k2, next_v in loop_doc.items():
                if v.delete or next_v.delete:
                    continue
                # assign the appos value to be True if it is missed in the earlier step
                # (the missing may be due to that "coref_type" does not correlate with "appos")

                k2_sent_id = next_v.text_id.split('-')[0]
                if k1_sent_id == k2_sent_id and v.next == k2 and v.coref_type == 'appos':
                    self.doc[k1].appos = True
                    k1_tok_start_id = int(v.text_id.split('-')[-1])

                    prev_start = int(v.text_id.split('-')[-1])
                    prev_last = int(v.text_id.split('-')[-1]) + v.span_len - 1
                    next_start = int(next_v.text_id.split('-')[-1])
                    next_last = int(next_v.text_id.split('-')[-1]) + next_v.span_len - 1

                    # search the antecedent of k1
                    ante = ''
                    for ante_k, ante_v in self.doc.items():
                        if ante_v.next and ante_v.next in self.doc.keys() and ante_v.next == k1:
                            ante = ante_k
                            break

                    # create a new e as the new big span for the apposition
                    new_k1 = str(self.last_e+1)
                    self.last_e += 1
                    self.doc[new_k1] = deepcopy(self.doc[k1])
                    self.doc[new_k1].cur = new_k1
                    self.doc[new_k1].coref = next_v.coref
                    self.doc[new_k1].next = next_v.next
                    self.doc[new_k1].coref_type = next_v.coref_type
                    self.doc[new_k1].e_type = v.e_type
                    self.doc[new_k1].new_e = True

                    if prev_last <= next_last:
                        self.doc[new_k1].func += ' ' + next_v.func
                        self.doc[new_k1].pos += ' ' + next_v.pos
                        self.doc[new_k1].tok += ' ' + next_v.tok

                    self.last_e += 1

                    self.doc[k1].dep_appos = True
                    # self.doc[k1].appos_point_to = new_k1

                    # add k1 span to new_k1
                    k1_span = [f'{k1_sent_id}-{i}' for i in range(prev_start, prev_last+1)]
                    for i in k1_span:
                        if i not in self.new_id2entity.keys():
                            self.new_id2entity[i] = []
                        self.new_id2entity[i].append(new_k1)

                    # fill the gap
                    gap = [f'{k1_sent_id}-{i}' for i in range(prev_last+1, next_start)]
                    next_span_len = 0 if prev_last > next_last else next_v.span_len
                    self.doc[new_k1].span_len += next_span_len + len(gap)
                    for i in gap:
                        # self.doc[new_k1].tok += f' {i}'
                        if i not in self.new_id2entity.keys():
                            self.new_id2entity[i] = []
                        self.new_id2entity[i].append(new_k1)

                    # check the token right after the appositive, if it is in '|)|", etc., expand the larger span
                    if next_last <= len(self.doc[k1].sent) - 1 and prev_last <= next_last:
                        next_tok = self.doc[k1].sent[next_last][1]
                        next_tok_id = self.doc[k1].sent[next_last][0]
                        # print(next_tok)
                        if next_tok in ["'", '"', ')', ']', '’', '”', '）', '}']:
                            self.doc[new_k1].span_len += 1
                            self.doc[new_k1].tok += ' ' + next_tok
                            id = f'{k1_sent_id}-{next_tok_id}'
                            if id not in self.new_id2entity.keys():
                                self.new_id2entity[id] = []
                            self.new_id2entity[id].append(new_k1)

                    # check the token between the two markables of the appositive construction, if it is in '|-|",
                    # move it to the second markable
                    prev_tok = self.doc[k1].sent[next_start-1][1]
                    prev_tok_id = int(self.doc[k1].sent[next_start-1][0])
                    id = f'{k1_sent_id}-{prev_tok_id}'
                    if prev_tok in ['"', "'", '-'] and prev_tok_id <= prev_last:
                        print(prev_tok)
                        self.doc[k1].span_len -= 1

                        self.doc[k2].span_len += 1
                        self.doc[k2].tok = prev_tok + ' ' + self.doc[k2].tok
                        if id not in self.new_id2entity.keys():
                            self.new_id2entity[id] = []
                        self.new_id2entity[id].append(k2)

                    # add k2 span to new_k1
                    k2_span = [f'{k1_sent_id}-{i}' for i in range(next_start, next_start+self.doc[k2].span_len)]
                    for i in k2_span:
                        if i not in self.new_id2entity.keys():
                            self.new_id2entity[i] = []
                        self.new_id2entity[i].append(new_k1)

                    if ante:
                        self.doc[ante].next = new_k1

                    # add appos ids to new_k1 if the original k2 has only one token
                    ids = self.doc[new_k1].appos
                    if k2.startswith('0_') and len(ids) > 1:
                        new_k2 = str(self.last_e + 1)
                        self.last_e += 1
                        self.doc[new_k2] = deepcopy(self.doc[k2])
                        self.doc[new_k2].cur = new_k2

                        # points prev
                        self.doc[k1].next = new_k2

                        # reset the original k
                        self.doc[k2] = Coref()
                        self.doc[k2].delete = True

                        # update new_id2entity
                        ori_e = k2.split('_')[-1]
                        for id in ids:
                            if id not in self.new_id2entity.keys():
                                self.new_id2entity[id] = []
                            self.new_id2entity[id].append(new_k2)

                        # update last_e
                        self.last_e += 1
                    else:
                        new_k2 = k2

                    self.doc[new_k2].coref = ''
                    self.doc[new_k2].next = ''
                    self.doc[new_k2].coref_type = ''

                    if not self.if_appos:
                        self.doc[k1].delete = True
                        self.doc[new_k2].delete = True

    def _remove_nested_coref(self):
        for k, _coref in self.doc.items():
            if not _coref.text_id:
                continue
            if self.doc[k].delete:
                continue
            sent_id, tok_id = _coref.text_id.split('-')[0], int(_coref.text_id.split('-')[1])
            ids = [f'{sent_id}-{tok_id+i}' for i in range(_coref.span_len)]
            if self.doc[k].next and self.doc[k].coref in ids:
                next_k = self.doc[k].next
                if self.doc[next_k].next:
                    self.doc[k].next = self.doc[next_k].next
                    self.doc[k].coref = self.doc[next_k].coref
                    self.doc[k].coref_type = self.doc[next_k].coref_type
                else:
                    self.doc[k].next = ''
                    self.doc[k].coref = ''
                    self.doc[k].coref_type = ''

                self.doc[next_k] = Coref()
                self.doc[next_k].delete = True

        # revise the deleted coref
        for del_k, del_coref in self.doc.items():
            if del_coref.next and del_coref.next in self.doc.keys() and self.doc[del_coref.next].delete:
                self.doc[del_k].next = ''
                self.doc[del_k].coref = ''
                self.doc[del_k].coref_type = ''


    def _remove_nmodposs(self):
        """
        Ok, this is wrong. Double check the function to see if this one conflicts with others and why this affects the
        final output.

        Example: [Zurbarán ’s] cycle of Jacob and his Sons
        """
        for k,v in self.doc.items():
            if k in self.doc.keys() and v.next != '' and self.doc[v.next].nmod_poss:
                cur_v = v
                while cur_v.next in self.doc.keys() and cur_v.next != '' and self.doc[cur_v.next].nmod_poss:
                    cur_v = deepcopy(self.doc[cur_v.next])

                    self.doc[cur_v.cur].coref = ''
                    self.doc[cur_v.cur].next = ''
                    self.doc[cur_v.cur].coref_type = ''

            # if k in self.doc.keys() and v.next in self.doc.keys() and v.next != '' and self.doc[v.next].nmod_poss:
                if not cur_v.nmod_poss:
                    self.doc[k].coref = cur_v.coref
                    self.doc[k].coref_type = cur_v.coref_type
                    self.doc[k].next = cur_v.cur
                else:
                    if cur_v.next:
                        self.doc[k].coref = cur_v.coref
                        self.doc[k].coref_type = self.doc[cur_v.next].coref_type
                        self.doc[k].next = cur_v.next
                    else:
                        self.doc[k].coref = ''
                        self.doc[k].coref_type = ''
                        self.doc[k].next = ''


    def _change_cata(self):
        """
        some cataphora affects the coref chain, move the coref chain to its cataphoric entity

        Example:
            45-1    2047-2048   I   person  giv _   _
            45-2    2049-2051   'm  _   _   _   _
            45-3    2052-2053   a   person[145] giv[145]    cata|coref  45-1[0_145]|49-8[0_145]
            45-4    2054-2062   graduate    person[145] giv[145]    _   _
            45-5    2063-2070   student person[145] giv[145]    _   _
            45-6    2071-2072   .   _   _   _   _
            45-7    2073-2074   "   _   _   _   _
        """
        for k1, v1 in self.doc.items():
            # If the cataphora does not have an antecedent
            if v1.tsv_line and 'cata' in v1.tsv_line[5]:
                for i, x in enumerate(v1.tsv_line[5].split('|')):
                    if 'cata' in x:
                        coref_rel = v1.tsv_line[6].split('|')[i]
                        cata_to = coref_rel.split('[')[0]
                        coref_id = f'0_{cata_to}'

                        coref_entity = coref_rel.split('[')[-1].split('_')[0]
                        if coref_entity != '0':
                            """
                            Example: academic_games (probably an annotation error)
                                14-19   2696-2705   attention   abstract    giv cata    17-18[176_0]    
                            """
                            cata_to = coref_entity
                            coref_id = coref_entity

                        if coref_id not in self.doc.keys():
                            continue

                        if v1.next:
                            if v1.next != coref_id:
                                self.doc[coref_id].next = self.doc[k1].next
                                self.doc[coref_id].coref = self.doc[k1].coref
                                self.doc[coref_id].coref_type = self.doc[k1].coref_type

                            self.doc[k1].next = ''
                            self.doc[k1].coref = ''
                            self.doc[k1].coref_type = ''

            # If it's the case that the cataphora has an antecedent
            if v1.next and v1.next in self.doc.keys() and self.doc[v1.next].tsv_line and 'cata' in self.doc[v1.next].tsv_line[5]:
                v_cata = deepcopy(v1.next)
                fields = self.doc[v_cata].tsv_line
                for i, x in enumerate(fields[5].split('|')):
                    if 'cata' in x:
                        coref_rel = fields[6].split('|')[i]
                        cata_to = coref_rel.split('[')[0]
                        coref_id = f'0_{cata_to}'

                        coref_entity = coref_rel.split('[')[-1].split('_')[0]
                        if coref_entity != '0':
                            """
                            Example: academic_games (probably an annotation error)
                                14-19   2696-2705   attention   abstract    giv cata    17-18[176_0]    
                            """
                            cata_to = coref_entity
                            coref_id = coref_entity

                        if coref_id not in self.doc.keys():
                            continue

                        if self.doc[v_cata].next == cata_to:
                            self.doc[k1].coref = ''
                            self.doc[k1].next = ''

                        # revise the antecedent
                        self.doc[k1].coref = cata_to
                        self.doc[k1].next = coref_id

                        # revise the cataphoric entity only if the current cataphoric entity has a next coref
                        if self.doc[v_cata].next and self.doc[v_cata].next != cata_to:
                            self.doc[coref_id].next = self.doc[v_cata].next
                            self.doc[coref_id].coref = self.doc[v_cata].coref
                            self.doc[coref_id].coref_type = self.doc[v_cata].coref_type

                        # revise the current entity with the cataphoric coref type
                        self.doc[v_cata].next = ''
                        self.doc[v_cata].coref = ''
                        self.doc[v_cata].coref_type = ''

    def _remove_excluded_heads(self):
        """
        If the head is not eligible, delete it

        Example:
            You 'll probably have more basil than you could possibly eat fresh , so plan on storing [some] in the fridge .
        """
        NOT_INCLUDED = ['some']
        for k_ante, v_ante in self.doc.items():
            if v_ante.next:
                k_next = deepcopy(v_ante.next)
                if k_next in self.doc.keys() and self.doc[k_next].tok in NOT_INCLUDED:
                    if self.doc[k_next].next:
                        v_ante.next = self.doc[k_next].next
                        v_ante.coref = self.doc[k_next].coref
                        v_ante.coref_type = self.doc[k_next].coref_type

                    self.doc[k_next] = Coref()
                    self.doc[k_next].delete = True

    def _remove_singleton(self):
        """
        Remove the entities that have no coref relations with other mentions
        """
        coref_next = [v.next for v in self.doc.values()]
        valid_coref = []
        for k,v in self.doc.items():
            if self.if_singletons:
                if v.cur and not v.num:
                    valid_coref.append(k)
            else:
                if k in coref_next or v.next != '':
                    valid_coref.append(k)

        return valid_coref

    def _remove_deleted_relations(self):
        """
        Some coref relations are removed by previous functions. This function is to remove those relations.
        """
        for k, _coref in self.doc.items():
            if self.doc[_coref.next] == 1:
                a = 1

    def process(self, new_id2entity):
        self._remove_junk()
        self.new_id2entity = new_id2entity
        self.last_e = sorted([int(x) for x in self.doc.keys() if not x.startswith('0_')], reverse=True)[0] + 30
        self._expand_acl()
        self._verb_contract()
        self._appos_merge()
        self._change_cata()
        self._remove_compound()
        self._remove_bridge()
        self._remove_cop()
        self._remove_nmodposs()
        self._break_chain()
        self._remove_excluded_heads()
        self._remove_nested_coref()
        valid_coref = self._remove_singleton()

        # gum output for acl_span
        # valid_coref = list(self.doc.keys())
        return self.doc, valid_coref, self.new_id2entity



def remove_singleton(e, non_singleton, coref_fields, line_id):
    if e:
        coref_fields[3] = '|'.join([x for x in coref_fields[3].split('|') if e != x.split('[')[-1].strip(']')])
        coref_fields[4] = '|'.join([x for x in coref_fields[4].split('|') if e != x.split('[')[-1].strip(']')])
        coref_fields[5] = '|'.join([x for x in coref_fields[5].split('|') if e != x.split('[')[-1].strip(']')])
    else:
        if f'0_{line_id}' not in non_singleton:
            coref_fields[3], coref_fields[4], coref_fields[5] = '', '', ''
    return coref_fields[3], coref_fields[4], coref_fields[5]


def to_tsv(doc, coref_article, non_singleton, new_id2entity):
    converted_article = ''
    added_coref = []
    last_coref = ''

    new_entities = {k:v for k,v in doc.items() if v.new_e}
    for k, v in new_entities.items():
        tok_id = int(v.text_id.split('-')[-1])
        ids = {f'{v.text_id.split("-")[0]}-{tok_id+i}': k for i in range(v.span_len)}
        # for id, e in ids.items():
        #     if id not in new_id2entity.keys():
        #         new_id2entity[id] = []
        #     new_id2entity[id].append(e)
        #     a = 1

    for line in coref_article:
        if line.startswith('#') or line == '':
            converted_article += line + '\n'
            continue

        coref_fields = line.strip().split('\t')
        line_id, token = coref_fields[0], coref_fields[2]
        coref_fields[-2], coref_fields[-1] = '', ''

        # test
        if line_id == '44-14':
            a = 1

        # entity info
        entities = ['' if '[' not in x else x.strip(']').split('[')[1] for x in coref_fields[3].split('|')]
        # deal with new entities, such as appositions
        if line_id in new_id2entity.keys():
            new_id = new_id2entity[line_id]
            entities += new_id

        # loop every possible entities in a single line
        for e in set(entities):
            if e in doc.keys():
                id = e
            elif f'0_{line_id}' in doc.keys():
                id = f'0_{line_id}'
            elif f'0_{e}' in doc.keys():
                id = f'0_{e}'
            else:
                id = ''

            # remove deleted entities in func.py
            if id in doc.keys() and doc[id].delete == True:
                if e:
                    coref_fields[3] = '|'.join([x for x in coref_fields[3].split('|') if e != x.split('[')[-1].strip(']')])
                    coref_fields[4] = '|'.join([x for x in coref_fields[4].split('|') if e != x.split('[')[-1].strip(']')])
                    coref_fields[5] = '|'.join([x for x in coref_fields[5].split('|') if e != x.split('[')[-1].strip(']')])
                else:
                    if '|' in coref_fields[3]:
                        raise ValueError('The line with fake id has deleted entities. Revise the code.')
                    coref_fields[3] = ''
                    coref_fields[4] = ''
                    coref_fields[5] = ''
                continue

            # remove singleton
            if id not in non_singleton:
                coref_fields[3], coref_fields[4], coref_fields[5] = remove_singleton(e, non_singleton, coref_fields, line_id)
                continue

            if e in doc.keys():
                cur, next = doc[id].cur, doc[id].next
                if doc[id].acl_children:
                    last_coref = id

                # check appositions, handle coref_fields[3, 4]
                if e not in coref_fields[3] and e not in coref_fields[4]:
                    if doc[e].span_len == 1 and e.startswith('0_'):
                        coref_fields[3] += f'|{doc[e].e_type}'
                        coref_fields[4] += f'|{doc[e].seen}'
                    else:
                        coref_fields[3] += f'|{doc[e].e_type}[{e}]'
                        coref_fields[4] += f'|{doc[e].seen}[{e}]'

                if next == '':
                    # if next originally exists but deleted by func.py, remove it in coref_fields
                    if doc[e].nmod_poss:
                        coref_fields[3] = '|'.join([x for x in coref_fields[3].split('|') if cur != x.split('[')[-1].strip(']')])
                        coref_fields[4] = '|'.join([x for x in coref_fields[4].split('|') if cur != x.split('[')[-1].strip(']')])
                        coref_fields[5] = '|'.join([x for x in coref_fields[5].split('|') if cur != x.split('[')[-1].strip(']')])
                    continue

                if next.startswith('0_'):
                    next = '0'

                coref = f'{doc[id].coref}[{next}_{cur}]'

                # check if it's the beginning of a mention, if not, continue
                if coref in added_coref:
                    continue

                coref_fields[-2] += f'|{doc[id].coref_type}'
                coref_fields[-1] += f'|{coref}'
                added_coref.append(coref)

            # if the current token is not an entity, but has coref
            elif f'0_{line_id}' in doc.keys():

                # if next coref is removed
                if doc[id].next == '' and doc[id].coref_type == '':
                    if doc[id].cur and '0_' not in doc[id].cur and doc[id].cur not in coref_fields[3] and coref_fields[3] != '_':
                        coref_fields[-3] += f'[{doc[id].cur}]'
                        coref_fields[3] += f'[{doc[id].cur}]'
                    pass

                # if the current word is not an named entity and the coref next is neither an named entity
                # but has coref relations
                elif doc[id].cur.startswith('0_') and doc[id].next.startswith('0_'):
                    coref_fields[-2] += doc[id].coref_type
                    coref_fields[-1] += doc[id].coref

                # if the current word is not an named entity while the next coref is an named entity
                # and has coref relation
                elif doc[id].cur.startswith('0_'):
                    coref_fields[-2] += doc[id].coref_type
                    coref_fields[-1] += f'{doc[id].coref}[{doc[id].next}_0]'

                # if changed in func, such in function "remove_appos"
                else:
                    if doc[id].cur:
                        coref_fields[-3] += f'[{doc[id].cur}]'
                        coref_fields[3] += f'[{doc[id].cur}]'

            # if the current token is added to the first part of an apposition, i.e. ","
            elif f'0_{e}' in doc.keys():
                cur_id = f'0_{e}'
                coref_fields[3] += f'|{doc[cur_id].e_type}'
                coref_fields[4] += f'|{doc[cur_id].seen}'
                print(f'Warning: A token outside the coref span is added in Line {e}. This should not happen too often.')

        # if the expanded acl is not added, add it to the lines that are originally not included in the mention
        # cur_sent_id, cur_tok_id = line_id.split('-')[0], line_id.split('-')[1]
        # if last_coref in doc.keys() and cur_sent_id == doc[last_coref].text_id.split('-')[0] and \
        #         cur_tok_id in doc[last_coref].acl_children and doc[last_coref].cur not in coref_fields[3]:
        #     coref_fields[3] += '|' + doc[last_coref].e_type + f'[{doc[last_coref].cur}]'
        #     coref_fields[4] += '|' + doc[last_coref].seen + f'[{doc[last_coref].cur}]'

        # if the line does not contain coref info, do not change it
        if f'0_{line_id}' not in doc.keys() and coref_fields[3] == '_':
            converted_article += line.strip() + '\n'
            continue

        # format revise
        if coref_fields[-1] == '' and coref_fields[-2] == '':
            coref_fields[-2], coref_fields[-1] = '_', '_'
        if coref_fields[3] == '' and coref_fields[4] == '':
            coref_fields[3], coref_fields[4] = '_', '_'
        if coref_fields[5] == '':
            coref_fields[5] = '_'

        coref_fields = [x.strip('|') for x in coref_fields]

        converted_article += '\t'.join(coref_fields) + '\n'

    return converted_article


def to_conll(docname, doc, converted_tsv_article, dep_sents):
    count = 0
    chain, group_chain, group = {}, {}, 0
    end_coref = defaultdict(list)
    seen = []
    conll_article = f'# begin document {docname}\n'

    for i, line in enumerate(converted_tsv_article.split('\n')):
        if line.startswith('#') or line == '':
            # converted_tsv_article += line + '\n'
            continue

        fields = line.strip().split('\t')
        line_id, token = fields[0], fields[2]
        cur_sent_last_id = len(dep_sents[int(line_id.split('-')[0])])

        cur_line = str(count) + '\t' + token + '\t'
        count += 1
        coref_part = ''

        if line_id == '20-11':
            a = 1

        if fields[3] == '_':
            cur_line += '_\n'
            conll_article += cur_line
            continue

        for entity_info in fields[3].split('|'):
            cur_e = entity_info.rstrip(']').split('[')
            cur_e = f'0_{line_id}' if len(cur_e) == 1 else cur_e[-1]
            next_e = doc[cur_e].next

            if line_id != doc[cur_e].text_id and line_id not in end_coref.keys():
                continue

            cur_end_id = f'{doc[cur_e].text_id.split("-")[0]}-{int(doc[cur_e].text_id.split("-")[1]) + doc[cur_e].span_len - 1}'

            if cur_e not in seen:
                group += 1
                seen.append(cur_e)
                chain[group] = [cur_e]
                group_chain[cur_e] = group
                end_coref[cur_end_id].append(cur_e)

                if next_e and next_e in doc.keys():
                    seen.append(next_e)
                    chain[group].append(next_e)
                    group_chain[next_e] = group

                    next_sent_id = doc[next_e].text_id.split('-')[0]
                    next_tok_id = doc[next_e].text_id.split('-')[-1]

                    end = int(next_tok_id) + doc[next_e].span_len - 1
                    end_id = f'{next_sent_id}-{int(next_tok_id) + doc[next_e].span_len - 1}'
                    end_coref[end_id].append(next_e)

            elif cur_e in seen and next_e and next_e not in seen and next_e in doc.keys():
                cur_group = group_chain[cur_e]
                seen.append(next_e)
                chain[cur_group].append(next_e)
                group_chain[next_e] = cur_group

                next_sent_id = doc[next_e].text_id.split('-')[0]
                next_tok_id = doc[next_e].text_id.split('-')[-1]
                end_id = f'{next_sent_id}-{int(next_tok_id) + doc[next_e].span_len - 1}'
                end_coref[end_id].append(next_e)

            if cur_end_id == doc[cur_e].text_id:
                coref_part += f'({group_chain[cur_e]})'
            elif line_id == doc[cur_e].text_id:
                coref_part += f'({group_chain[cur_e]}'
            elif line_id in end_coref.keys():
                for e in set(end_coref[line_id]):
                    if e == cur_e:
                        coref_part = f'{group_chain[e]})' + coref_part

        if coref_part == '':
            coref_part += '_'
        cur_line += coref_part + '\n'
        conll_article += cur_line
    conll_article += '# end document\n'
    return conll_article


def build_ontogum(dep_string, tsv_string):
    dep_article = dep_string.split('\n')

    dep_doc = []
    for l in dep_article:
        l = l.split('\t')
        if len(l) == 10:
            # deal with the meta words such as (1) 34-35 world's or (2) 14.1 was
            if '-' in l[0] or '.' in l[0]:
                continue
        dep_doc.append(l)
    if "# newdoc id = " in dep_doc[0]:
        docname = dep_doc[0].split('=')[-1].strip()
    else:
        docname = ""

    tsv_article = tsv_string.split('\n')

    if_appos = False
    if_singletons = False

    doc, tokens, group_dict, next_dict, new_id2entity, dep_sents = process_doc(dep_doc, tsv_article)

    # make GUM as same as OntoNotes
    convert = Convert(doc, next_dict, group_dict, if_appos, if_singletons)
    converted_doc, non_singleton, new_id2entity = convert.process(new_id2entity)

    # generate tsv format
    converted_tsv_article = to_tsv(doc, tsv_article, non_singleton, new_id2entity)

    # generate conll bracket format
    converted_conll_article = to_conll(docname, doc, converted_tsv_article, dep_article)

    return converted_tsv_article, converted_conll_article


if __name__ == '__main__':
    corefDir = os.path.join('gum', 'coref', 'tsv')
    depDir = os.path.join('gum', 'dep')

    for f in os.listdir(corefDir):
        if '.tsv' not in f:
            continue

        filename = f.split('.')[0]
        name_fields = filename.split('_')
        new_name = f'GUM_{name_fields[1]}_{name_fields[2]}'

        print(filename)
        if filename != 'GUM_fiction_teeth':
            continue

        tsv_string = io.open(os.path.join(corefDir, f), encoding='utf-8').read()
        dep_string = io.open(os.path.join(depDir, new_name + '.conllu'), encoding='utf-8').read()

        converted_tsv_article, converted_conll_article = build_ontogum(dep_string, tsv_string)
