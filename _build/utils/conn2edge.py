import re, os, sys
from copy import deepcopy
from collections import defaultdict
from glob import glob
from argparse import ArgumentParser
from rst2dep import read_rst, make_rsd, rsd2rs3
from browse_rs4 import read_doc
from random import shuffle, seed, uniform

seed(42)

from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings, DocumentRNNEmbeddings, TransformerWordEmbeddings
from flair.models import TextClassifier, SequenceTagger

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

unsignaled_rels = {"context-background", "evaluation-comment", "organization-preparation", "topic-question",
                   "topic-solutionhood", "organization-heading", "organization-phatic", "same-unit"}

sigtypes = """		<sigtypes>
			<sig type="dm" subtypes="dm"/>
			<sig type="graphical" subtypes="colon;dash;items_in_sequence;layout;parentheses;question_mark;quotation_marks;semicolon"/>
			<sig type="lexical" subtypes="alternate_expression;indicative_phrase;indicative_word"/>
			<sig type="morphological" subtypes="mood;tense"/>
			<sig type="numerical" subtypes="same_count"/>
			<sig type="orphan" subtypes="orphan"/>
			<sig type="reference" subtypes="comparative_reference;demonstrative_reference;personal_reference;propositional_reference"/>
			<sig type="semantic" subtypes="antonymy;attribution_source;lexical_chain;meronymy;negation;repetition;synonymy"/>
			<sig type="syntactic" subtypes="infinitival_clause;interrupted_matrix_clause;modified_head;nominal_modifier;parallel_syntactic_construction;past_participial_clause;present_participial_clause;relative_clause;reported_speech;subject_auxiliary_inversion"/>
			<sig type="unsure" subtypes="unsure"/>
		</sigtypes>"""


class Connective:
    def __init__(self,text,tokens,rel,doc,sigtype):
        self.text = text
        self.tokens = sorted(tokens)
        self.rel = rel
        self.doc = doc
        self.sigtype = sigtype

    def __repr__(self):
        return self.text + f" ({self.rel.relname.replace('_r','').replace('_m','')}, {str(self.tokens)}, {self.doc})"


def doc2instances(rs4, conllu, docname, conn2allowed):
    rs4_no_signals = re.sub(r'<signals>.*?</signals>\n?','',rs4,flags=re.DOTALL)  # Remove signals
    rsd = make_rsd(rs4_no_signals, "", as_text=True)

    # Get EDU to children
    children = defaultdict(list)
    same_unit_children = defaultdict(list)
    for line in rsd.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            children[fields[0]].append(fields[6])
            if fields[7].startswith("same-unit"):
                same_unit_children[fields[6]].append(fields[0])

    # Get corresponding conllu data and map token indices to sentences
    tok2sent = {}
    tok2edu = {}
    edu2tok = defaultdict(list)
    sid = 1
    edu_id = 0
    toknum = 0
    for line in conllu.split("\n"):
        if "Discourse=" in line:
            edu_id += 1
        if len(line) == 0:
            sid += 1
        elif not line.startswith("#"):
            if "\t" in line:
                fields = line.split("\t")
                if "-" in fields[0] or "." in fields[0]:
                    continue
                toknum += 1
                tok2sent[toknum] = sid
                tok2edu[toknum] = edu_id
                edu2tok[edu_id].append(toknum - 1)

    edu2rel = {}
    connectives = []

    relations, conllu_annos, nodes = read_doc(rs4, rsd, conllu, docname, allow_sourceless_signals=True)
    forms = conllu_annos[0]
    # Flag left leg multinuc relations
    for rel in relations.values():
        rel.is_left_leg = False
        if rel.relkind == "multinuc":
            left = rel.node.left
            parent = rel.node.parent
            if parent in relations:
                if relations[parent].node.left == left:
                    rel.is_left_leg = True
        rel.source_sent = tok2sent[rel.source["head"][0]+1]
        rel.target_sent = tok2sent[rel.target["head"][0]+1]

    for rel in relations.values():
        if rel.head_edu == "26":
            a = 4
        rel.head_edus = set(rel.head_edus)
        edu2rel[rel.head_edu] = rel
        if rel.head_edu in same_unit_children:
            for child in same_unit_children[rel.head_edu]:
                if child not in rel.head_edus:
                    rel.head_edus.add(child)
                toks = edu2tok[int(child)]
                rel.source["head"] = tuple(sorted(list(set(list(rel.source["head"]) + toks))))
        if rel.dep_parent in same_unit_children:
            for child in same_unit_children[rel.dep_parent]:
                toks = edu2tok[int(child)]
                rel.target["head"] = tuple(sorted(list(set(list(rel.target["head"]) + toks))))
        rel.node.dep_children = rel.dep_children = children[rel.head_edu]
        rel.node.dep_parent = rel.dep_parent
        nodes[rel.nid] = rel.node
        for sig in rel.signals:
            doc, sigtype, subtype, tokens = sig
            tokens = [t + 1 for t in tokens]  # 1 indexing
            if sigtype in ["dm", "orphan"]:
                text = " ".join([forms[t - 1] for t in sorted(tokens)])
                conn = Connective(text, tokens, rel, doc, sigtype)
                conn.edu = tok2edu[tokens[0]]
                connectives.append(conn)
                relname = rel.relname.replace("_r", "").replace("_m", "")
                direction = "rtl" if int(rel.dep_parent) < int(rel.head_edu) else "ltr"
                conn_position = "target" if conn.tokens[0] -1 in rel.target["head"] else "source"
                if relname not in unsignaled_rels and docname not in ud_test:
                    conn2allowed[conn.text.lower()].add(relname + ":" + conn_position + ":" + direction)

    # Flag sentences nodes
    sent2node = {}

    for node in nodes.values():
        if "-" in node.id:
            continue
        if node.id == "153":
            a = 4
        first_tok = min(t for t in tok2edu if tok2edu[t] == node.left)
        last_tok = max(t for t in tok2edu if tok2edu[t] == node.right)
        first_sent = tok2sent[first_tok]
        last_sent = tok2sent[last_tok]

        is_sent = True
        if first_sent != last_sent:
            is_sent = False
        else:
            sent_first = min([t for t in tok2sent if tok2sent[t] == first_sent])
            sent_last = max([t for t in tok2sent if tok2sent[t] == first_sent])
            if sent_first != first_tok or sent_last != last_tok:
                is_sent = False

        if is_sent:
            sent2node[first_sent] = node

    return (docname,connectives,relations,nodes,forms,sent2node, tok2sent), conn2allowed


def read_data(rs4_glob,conllu_dir,test=False):
    rs4_files = glob(rs4_glob)
    if test:
        rs4_files = [f for f in rs4_files if os.path.basename(f).replace(".rs4","") in ud_test]

    sys.stderr.write("o Predicting DM associations/secedges for " + str(len(rs4_files)) + " files\n")

    conn2allowed = defaultdict(set)

    docs = []

    for file_ in rs4_files:
        docname = os.path.basename(file_).replace(".rs4","")
        rs4 = open(file_,encoding="utf8").read()
        conllu = open(conllu_dir + docname + ".conllu",encoding="utf8").read()

        doc, conn2allowed = doc2instances(rs4, conllu, docname, conn2allowed)

        docs.append(doc)

    return docs, conn2allowed


def make_samples(conn,relations,nodes,conn2allowed,forms, sent2node, tok2sent, method="train"):

    def rel_possible(rel, conn, conn2allowed):
        if not rel.is_left_leg:
            relname = rel.relname.replace("_r", "").replace("_m", "")
            if conn.text.lower() in conn2allowed:
                direction = "rtl" if int(rel.dep_parent) < int(rel.head_edu) else "ltr"
                conn_position = "target" if conn.tokens[0] - 1 in rel.target["head"] else "source"
                structure = relname + ":" + conn_position + ":" + direction
                if structure in conn2allowed[conn.text.lower()]:
                    return True
            else:  # Unknown connective, allow all signalable rels
                if relname not in unsignaled_rels:
                    return True
        return False

    def make_sent_rels(conn, sent2node, tok2sent, conn2allowed):
        # Check if the connective EDU has a sentence node above it
        sent_rels = []
        sent2 = tok2sent[conn.tokens[0]]
        if sent2 in sent2node:
            node2 = sent2node[sent2]
            sent1 = sent2 - 1
            if sent1 in sent2node:
                node1 = sent2node[sent1]
                for structure in conn2allowed[conn.text.lower()]:
                    relname, conn_position, direction = structure.split(":")
                    sent_rel = deepcopy(conn.rel)
                    if direction == "rtl" and conn_position == "source":
                        sent_rel.node = node2
                        sent_rel.head_edu = node2.left
                        sent_rel.head_edus = [node2.left]
                        sent_rel.nid = node2.id + "-" + node1.id
                        sent_rel.relname = relname
                        sent_rel.source["head"] = sorted([t-1 for t in tok2sent if tok2sent[t] == sent2])
                        sent_rel.source["all"] = (sent_rel.source["head"][0],sent_rel.source["head"][-1])
                        sent_rel.target["head"] = sorted([t-1 for t in tok2sent if tok2sent[t] == sent1])
                        sent_rel.target["all"] = (sent_rel.target["head"][0],sent_rel.target["head"][-1])
                    elif direction == "ltr" and conn_position == "target":
                        sent_rel.node = node1
                        sent_rel.head_edu = node1.left
                        sent_rel.head_edus = [node1.left]
                        sent_rel.nid = node1.id + "-" + node2.id
                        sent_rel.relname = relname
                        sent_rel.target["head"] = sorted([t-1 for t in tok2sent if tok2sent[t] == sent2])
                        sent_rel.target["all"] = (sent_rel.source["head"][0],sent_rel.source["head"][-1])
                        sent_rel.source["head"] = sorted([t-1 for t in tok2sent if tok2sent[t] == sent1])
                        sent_rel.source["all"] = (sent_rel.target["head"][0],sent_rel.target["head"][-1])
                    sent_rel.signals = []
                    sent_rels.append(sent_rel)
        return sent_rels

    def rel2sample(rel,conn,forms,same_path_rel,extra_tokens=None):
        relname = rel.relname.replace("_m","").replace("_r","")
        label = "True" if rel.nid == conn.rel.nid and relname == conn.rel.relname.replace("_m","").replace("_r","") else "False"
        relname = relname.replace("-"," ")
        direction = "left"
        if int(rel.dep_parent) < int(rel.head_edu):
            direction = "right"
        distance = abs(int(rel.dep_parent) - int(rel.head_edu))
        source_words = []
        source_tokens = rel.source["head"]
        if extra_tokens is not None:
            source_tokens += extra_tokens
        for t in sorted(source_tokens):
            if t == conn.tokens[0] - 1:
                source_words.append("**")
            source_words.append(forms[t])
            if t == conn.tokens[-1] - 1:
                source_words.append("**")
        target_words = []
        for t in rel.target["head"]:
            if t == conn.tokens[0] - 1:
                target_words.append("**")
            target_words.append(forms[t])
            if t == conn.tokens[-1] - 1:
                target_words.append("**")
        source = " ".join(source_words)
        target = " ".join(target_words)
        if direction == "right":
            sample = f"__label__{label}\t{relname} ( {same_path_rel} ) {direction} {distance} : {target} << {source}"
        else:
            sample = f"__label__{label}\t{relname} ( {same_path_rel} ) {direction} {distance} : {source} >> {target}"
        return sample

    samples = []
    sources = []
    # Get positive samples
    same_path_rel = "_"
    if conn.sigtype == "orphan":
        if method == "test_prim":  # Do not attempt to predict secondary edges
            sources.append((conn, conn.rel))
            return None, sources
        for rel in relations.values():
            if "-" not in rel.nid and ((rel.head_edu == conn.rel.head_edu and rel.dep_parent == conn.rel.dep_parent) or (rel.head_edu == conn.rel.dep_parent and rel.dep_parent == conn.rel.head_edu)):
                same_path_rel = rel.relname
                break
    for rel in relations.values():
        if "-" in rel.nid and method != "train":
            continue
        if rel_possible(rel, conn, conn2allowed):
            if str(conn.edu) in rel.dep_children or str(conn.edu) in rel.head_edus or str(conn.edu) == rel.dep_parent:
                sample = rel2sample(rel,conn,forms,same_path_rel)
                if sample not in samples:
                    samples.append(sample)
                    sources.append((conn,rel))

    if method == "train":
        # Get negative samples
        same_path_rel = "_"
        n = 4
        negative_samples = []
        for rel in relations.values():
            if rel_possible(rel, conn, conn2allowed):
                if str(conn.edu) in rel.dep_children or str(conn.edu) in rel.head_edus or str(conn.edu) == rel.dep_parent:
                    sample = rel2sample(rel, conn, forms, same_path_rel)
                    if sample not in negative_samples and sample not in samples:
                        negative_samples.append(sample)
                        sources.append((conn, rel))
            if len(negative_samples) > n:
                break

        if len(negative_samples) == 0 and method == "train" and uniform(0.0,1.0) > 0.5:
            # Make a negative sent sample at 50% probability
            sent_rels = make_sent_rels(conn, sent2node, tok2sent, conn2allowed)
            shuffle(sent_rels)
            for sent_rel in sent_rels:
                # Make a rel connecting the connective EDU sentence with the preceding sentence
                sample = rel2sample(sent_rel, conn, forms, "_")
                if sample not in samples and rel_possible(sent_rel, conn, conn2allowed):
                    negative_samples.append(sample)
                    sources.append((conn, sent_rel))
                    break

        if conn.sigtype == "orphan":
            sent_rels = make_sent_rels(conn, sent2node, tok2sent, conn2allowed)
            shuffle(sent_rels)
            i = 0
            for sent_rel in sent_rels:
                if sent_rel.target["all"] == conn.rel.target["all"] and sent_rel.source["all"] == conn.rel.source["all"]:  # Don't duplicate the true relation
                    pass
                    continue
                if i >= n:
                    break
                # Make a rel connecting the connective EDU sentence with the preceding sentence
                samples.append(rel2sample(sent_rel, conn, forms, "_"))
                i += 1

    if len(samples) == 0:
        this_rel_options = [r for r in relations.values() if r.head_edu == str(conn.edu)]
        if len(this_rel_options) > 0:
            this_rel = this_rel_options[0]
            node = this_rel.node
            if node.dep_rel.startswith("attribution"):
                parent_edu = node.dep_parent
                parent_rel = [r for r in relations.values() if r.head_edu == parent_edu]
                if len(parent_rel) > 0:
                    parent_rel = parent_rel[0]
                    if rel_possible(parent_rel, conn, conn2allowed):
                        sample = rel2sample(parent_rel, conn, forms, "attribution",extra_tokens=this_rel.source["head"])
                        if sample not in samples:
                            samples.append(sample)
                            sources.append((conn, parent_rel))

    if len(samples) == 0 and method == "test_all":
        # orphan, generate potential secedge samples
        for rel in relations.values():
            if str(conn.edu) in rel.dep_children or str(conn.edu) in rel.head_edus or str(conn.edu) == rel.dep_parent:
                if "-" in rel.nid:
                    if method == "train":  # This is already the true secedge, should already be in samples
                        continue
                    else:  # True secedge in eval mode, follows targeted paths, add it and continue
                        if rel_possible(rel, conn, conn2allowed):
                            same_path_rel = "_"
                            for rel2 in relations.values():
                                if "-" not in rel2.nid and ((rel2.head_edu == rel.head_edu and rel2.dep_parent == rel.dep_parent) or (rel2.head_edu == rel.dep_parent and rel2.dep_parent == rel.head_edu)):
                                    same_path_rel = rel2.relname
                                    break
                            sample = rel2sample(rel, conn, forms, same_path_rel)
                            if sample not in samples:
                                samples.append(sample)
                                sources.append((conn, rel))
                                continue
                same_path_rel = rel.relname
                for structure in conn2allowed[conn.text.lower()]:
                    relname, conn_position, direction = structure.split(":")
                    clone_rel = deepcopy(rel)
                    clone_rel.relname = relname
                    clone_rel_direction = "rtl" if int(clone_rel.dep_parent) < int(clone_rel.head_edu) else "ltr"
                    if clone_rel.source_sent != clone_rel.target_sent and clone_rel.source_sent in sent2node and clone_rel.target_sent in sent2node:
                        # Reroute edge to be head sentence to head sentence
                        source_sent_toks = sorted([t - 1 for t in tok2sent if tok2sent[t] == clone_rel.source_sent])
                        target_sent_toks = sorted([t - 1 for t in tok2sent if tok2sent[t] == clone_rel.target_sent])
                        if clone_rel_direction == direction:  # Relation is facing the correct way
                            clone_rel.source["head"] = source_sent_toks
                            clone_rel.source["all"] = (source_sent_toks[0], source_sent_toks[-1])
                            clone_rel.target["head"] = target_sent_toks
                            clone_rel.target["all"] = (target_sent_toks[0], target_sent_toks[-1])
                            source_node = sent2node[clone_rel.source_sent]
                            target_node = sent2node[clone_rel.target_sent]
                            clone_rel.nid = source_node.id + "-" + target_node.id
                            clone_rel.node = deepcopy(source_node)
                        else: # Invert
                            clone_rel.target["head"] = source_sent_toks
                            clone_rel.target["all"] = (source_sent_toks[0], source_sent_toks[-1])
                            clone_rel.source["head"] = target_sent_toks
                            clone_rel.source["all"] = (target_sent_toks[0], target_sent_toks[-1])
                            target_node = sent2node[clone_rel.source_sent]
                            source_node = sent2node[clone_rel.target_sent]
                            clone_rel.nid = source_node.id + "-" + target_node.id
                            clone_rel.node = deepcopy(source_node)
                        clone_rel.node.id = clone_rel.nid
                    else:
                        if rel.node.relname.endswith("_m"):  # Multinuc, do not use parent; use sibling instead
                            multinuc = nodes[rel.node.parent]
                            siblings = [nid for nid in multinuc.children if nid != rel.nid]
                            parent = siblings[0]
                            if direction == "ltr":
                                siblings = sorted([nid for nid in siblings if nodes[nid].left > rel.node.left],key=lambda x: nodes[x].left)
                                if len(siblings) > 0:
                                    parent = siblings[0]
                            else:
                                siblings = sorted([nid for nid in siblings if nodes[nid].left < rel.node.left],key=lambda x: nodes[x].left)
                                if len(siblings) > 0:
                                    parent = siblings[-1]
                        else:
                            parent = rel.node.parent
                        if clone_rel_direction == direction:  # Relation is facing the correct way
                            clone_rel.nid = rel.nid + "-" + parent
                        else:
                            clone_rel.nid = parent + "-" + rel.nid
                    if rel_possible(clone_rel, conn, conn2allowed):
                        clone_rel.signals = []
                        sample = rel2sample(clone_rel, conn, forms, same_path_rel)
                        if sample not in samples:
                            samples.append(sample)
                            sources.append((conn, clone_rel))
        sent_rels = make_sent_rels(conn, sent2node, tok2sent, conn2allowed)
        for sent_rel in sent_rels:
            # Make a rel connecting the connective EDU sentence with the preceding sentence
            samples.append(rel2sample(sent_rel, conn, forms, "_"))
            sources.append((conn, sent_rel))

    if method == "train":
        samples += negative_samples

    return samples, sources


def train():

    corpus = ClassificationCorpus(
        data_folder=".",
        train_file="conn2edge_train.tab",
        dev_file="conn2edge_dev.tab",
        test_file="conn2edge_test.tab",
    )

    label_dictionary = corpus.make_label_dictionary()

    doc_emb = DocumentRNNEmbeddings([TransformerWordEmbeddings("google/electra-base-discriminator")],bidirectional=True)
    # Alternatives for paper:
    # doc_emb = TransformerDocumentEmbeddings("google/electra-base-discriminator")
    # doc_emb = DocumentRNNEmbeddings([TransformerWordEmbeddings("microsoft/deberta-v3-base")],bidirectional=True)

    tagger = TextClassifier(
        document_embeddings=doc_emb,
        label_dictionary=label_dictionary,
    )

    from flair.trainers import ModelTrainer

    trainer = ModelTrainer(tagger, corpus)

    trainer.train('flair',
                  learning_rate=0.1,
                  mini_batch_size=64,
                  anneal_factor=0.5,
                  patience=3,
                  max_epochs=30,
                  num_workers=1)


def serialize(rs4, preds):
    xml_out = rs4

    # Remove old secedges and dm signals which look like this:
    # <signal source="64" type="dm" subtype="dm" tokens="598"/>
    # <secedges> ... </secedges>
    xml_out = re.sub(r'<signal source="(\d+)" type="(dm|orphan)" subtype="(dm|orphan)" tokens="(\d*)"\/>\n?\t*', "", xml_out)
    xml_out = re.sub(r'[ \t]*<secedges>.*<\/secedges>\n?\t*', "", xml_out, flags=re.DOTALL)

    # Add new secedges and signals
    secedges = []
    seen_nids = set()
    signals = list(re.findall(r'(<signal source="[\d-]+" type="[^"]+" subtype="[^"]+" tokens="[\d,]*"\/>)', xml_out))
    for pred in preds:
        tokens, nid, relname = pred
        sigtype = "dm"
        if "-" in nid:  # Make secedge
            if nid not in seen_nids:
                source, target = nid.split("-")
                secedges.append(f'\t\t\t<secedge id="{nid}" source="{source}" target="{target}" relname="{relname}"/>')
                sigtype = "orphan"
                seen_nids.add(nid)
        signals.append(f'\t\t\t<signal source="{nid}" type="{sigtype}" subtype="{sigtype}" tokens="{tokens}"/>')
    secedges = sorted(list(set(secedges)))
    if len(secedges) > 0:
        secedges = ['		<secedges>'] + secedges + ['		</secedges>']
    secedges = "\n".join(secedges)
    signals = "\n".join(sorted(signals))
    xml_out = re.sub(r'[\t ]*<signals>.*<\/signals>\n?\t*', "", xml_out, flags=re.DOTALL)
    xml_out = xml_out.replace('</body>', f'{secedges}\n\t\t<signals>\n{signals}\n\t\t</signals>\n\t</body>')
    return xml_out


def predict(docs, conn2allowed, model=None, test=False, method="test_all"):
    if model is None:
        model = TextClassifier.load("dm-dependencies\\best-model_assoc_gum.pt")

    samples = []
    all_sources = []
    if test:
        docs = [d for d in docs if d[0] in ud_test]

    ground_truth = []
    ground_truth_orphan = []
    for doc in docs:
        docname, connectives, relations, nodes, forms, sent2node, tok2sent = doc
        for conn in connectives:
            conn_samples, sources = make_samples(conn, relations, nodes, conn2allowed, forms, sent2node, tok2sent, method=method)
            if conn_samples is not None:
                samples += conn_samples
            all_sources += sources
            ground_truth.append(str(conn) + ":" + str(conn.rel))
            if conn.sigtype == "orphan":
                ground_truth_orphan.append(str(conn) + ":" + str(conn.rel))

    inputs = []
    for i, sample in enumerate(samples):
        conn, rel = all_sources[i]
        label, text = sample.split("\t")
        sent = Sentence(text, use_tokenizer=lambda x: x.split(" "))
        inputs.append(sent)

    model.predict(inputs)
    best_scores = defaultdict(float)
    best_rels = {}
    best_orphan_rels = {}
    for i, sample in enumerate(inputs):
        conn, rel = all_sources[i]
        pred_lab = sample.labels[0].value
        proba = sample.labels[0].score
        proba = 1 - proba if pred_lab == 'False' else proba
        if proba > best_scores[str(conn)]: # and pred_lab == 'True':
            best_rels[str(conn)] = rel
            best_scores[str(conn)] = proba

    for conn_str in best_rels:
        if "-" in best_rels[conn_str].nid:  # System has predicted a secedge
            best_orphan_rels[conn_str] = best_rels[conn_str]

    final_preds = []
    final_orphan_preds = []
    for conn_str in best_rels:
        final_preds.append(conn_str + ":" + str(best_rels[conn_str]))
        if conn_str in best_orphan_rels:
            final_orphan_preds.append(conn_str + ":" + str(best_orphan_rels[conn_str]))

    # Serialization info
    serialized = defaultdict(list)
    for conn_str, rel in best_rels.items():
        tokens = re.split(r'[\[\]]',conn_str)[1]
        source = rel.nid
        relname = rel.relname.replace("_r","").replace("_m","")
        serialized[rel.docname].append((tokens,source,relname))

    return final_preds, ground_truth, final_orphan_preds, ground_truth_orphan, serialized


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-m","--mode",default="predict",choices=["train","eval","predict"])
    p.add_argument("-r","--regenerate",action="store_true",help="Regenerate train and eval data")
    p.add_argument("-e","--edges",default="test_all",choices=["test_prim","test_all"])
    p.add_argument("-f","--files",default=None, help="Files to predict on as glob expression")
    p.add_argument("-c","--conllu",default=None, help="Directory with conllu files matching prediction file base names")
    p.add_argument("-p","--predict_dms",action="store_true", help="Predict DMs on plain rs3 before associating signals")
    opts = p.parse_args()

    rs4_glob = script_dir + ".." + os.sep + "src" + os.sep + "rst"  + os.sep + "*.rs4" if opts.files is None else opts.files
    conllu_dir = opts.conllu if opts.conllu is not None else script_dir + ".." + os.sep + "target" + os.sep + "dep"  + os.sep + "not-to-release" + os.sep

    if opts.mode == "train":
        if opts.regenerate:
            docs, conn2allowed = read_data(rs4_glob,conllu_dir,test=False)

            samples = defaultdict(list)
            for doc in docs:
                docname, connectives, relations, nodes, forms, sent2node, tok2sent = doc
                partition = "test" if docname in ud_test else "train"
                if docname in ud_dev:
                    partition = "dev"

                for conn in connectives:
                    conn_samples, sources = make_samples(conn,relations,nodes,conn2allowed,forms, sent2node, tok2sent, method="train")
                    if conn_samples is not None:
                        samples[partition] += conn_samples

            with open("conn2edge_train.tab",'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(samples["train"]))
            with open("conn2edge_dev.tab",'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(samples["dev"]))
            with open("conn2edge_test.tab",'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(samples["test"]))

        train()
    elif opts.mode == "eval":
        # Get test data just for test file
        if opts.regenerate:
            docs, conn2allowed = read_data(rs4_glob,conllu_dir,test=False)
            output = []
            for dm in conn2allowed:
                output.append(dm + "\t" + ";".join(conn2allowed[dm]))
            with open("conn2allowed.tab",'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(output))
        else:
            docs, _ = read_data(rs4_glob, conllu_dir, test=True)
            conn2allowed = defaultdict(set)
            with open("conn2allowed.tab",encoding="utf8") as f:
                for line in f.read().strip().split("\n"):
                    dm, rels = line.split("\t")
                    conn2allowed[dm] = set(rels.split(";"))

        final_preds, ground_truth, final_orphan_preds, ground_truth_orphan, _ = predict(docs, conn2allowed, test=True, method=opts.edges)

        precision = len(set(final_preds).intersection(set(ground_truth))) / len(set(final_preds))
        recall = len(set(final_preds).intersection(set(ground_truth))) / len(set(ground_truth))
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-Score: {f1}")

        precision = len(set(final_orphan_preds).intersection(set(ground_truth_orphan))) / len(set(final_orphan_preds))
        recall = len(set(final_orphan_preds).intersection(set(ground_truth_orphan))) / len(set(ground_truth_orphan))
        f1 = 2 * precision * recall / (precision + recall)
        print(f"Secedge Precision: {precision}")
        print(f"Secedge Recall: {recall}")
        print(f"Secedge F-Score: {f1}")

    elif opts.mode == "predict":

        rs4_glob = script_dir + "dm-dependencies" + os.sep +"dm_preds" + os.sep + "*.rs4"
        if opts.predict_dms:
            trash = glob(rs4_glob)
            for f in trash:
                os.remove(f)

        if opts.files is not None:
            rst_input = glob(opts.files)
        else:
            rst_input = glob(rs4_glob)

        if opts.predict_dms:

            from flair_dm_tagger import predict as predict_dm

            model_name = "dm-dependencies" + os.sep + "flair_tagger/best-model_dm_gum.pt"
            dm_tagger = SequenceTagger.load(model_name)

            sys.stderr.write("o Predicting DMs for " + str(len(rst_input)) + " files\n")
            for file_ in rst_input:
                docname = os.path.basename(file_).split(".")[0]
                conllu = open(conllu_dir + docname + ".conllu").read()
                tokens = []
                for line in conllu.split("\n"):
                    if "\t" in line:
                        fields = line.split("\t")
                        if "-" in fields[0] or "." in fields[0]:
                            continue
                        tokens.append(fields[1] + "\tO")
                    elif len(line.strip())==0:
                        tokens.append("")
                pred_conllu = predict_dm(in_path="\n".join(tokens).strip()+"\n",preloaded=dm_tagger,as_text=True)
                preds = [l.split("\t")[3] for l in pred_conllu.split("\n") if "\t" in l]

                rs4 = open(file_,encoding="utf8").read()
                rsd = make_rsd(rs4, "", as_text=True, docname=docname)
                rs4 = rsd2rs3(rsd,default_rels=True)  # Make deterministic/sequential

                edu2dep_parent = {}
                for line in rsd.split("\n"):
                    if "\t" in line:
                        edu2dep_parent[line.split("\t")[0]] = line.split("\t")[6]

                # Get running token ID to EDU ID mapping in rs3 format
                tok2edu = {}
                counter = 0
                edu_id = 1
                for line in rs4.split("\n"):
                    if '<segment' in line:
                        toks = line.split(">")[1].split("<")[0].split(" ")
                        for tok in toks:
                            tok2edu[counter] = edu_id
                            counter += 1
                        edu_id +=1

                if '<sigtypes>' not in rs4:
                    rs4 = rs4.replace("</relations>\n","</relations>\n"+sigtypes+"\n")

                signals = []
                sigs = defaultdict(lambda : defaultdict(list))
                last_b = 0
                for i, pred in enumerate(preds):
                    if "I" in pred:
                        sigs[tok2edu[i]][last_b].append(i+1)
                    elif "B" in pred:
                        last_b = i
                        sigs[tok2edu[i]][last_b].append(i+1)

                out_sigs = ["		<signals>"]
                for edu_id in sigs:
                    for sig in sigs[edu_id]:
                        toks = ",".join([str(x) for x in sigs[edu_id][sig]])
                        out_sigs.append(f'			<signal source="{edu_id}" type="dm" subtype="dm" tokens="{toks}"/>')
                out_sigs.append("		</signals>\n")
                rs4 = rs4.replace("	</body>","\n".join(out_sigs) +"     </body>")
                with open('dm-dependencies' + os.sep + 'dm_preds' + os.sep + docname + ".rs4", 'w', encoding="utf8") as f:
                    f.write(rs4)

        docs, _ = read_data(rs4_glob, conllu_dir, test=False)
        conn2allowed = defaultdict(set)
        with open("conn2allowed.tab",encoding="utf8") as f:
            for line in f.read().strip().split("\n"):
                dm, rels = line.split("\t")
                conn2allowed[dm] = set(rels.split(";"))

        final_preds, _, _, _, serialization = predict(docs, conn2allowed, test=False, method=opts.edges)

        for i, doc in enumerate(docs):
            docname, connectives, relations, nodes, forms, sent2node, tok2sent = doc
            rs4 = open(os.path.dirname(rs4_glob) + os.sep + docname + ".rs4",encoding="utf8").read()
            xml_out = serialize(rs4, serialization[docname])
            with open("dm-dependencies" + os.sep + "final_preds" + os.sep + docname + ".rs4", 'w', encoding="utf8") as f:
                f.write(xml_out)
