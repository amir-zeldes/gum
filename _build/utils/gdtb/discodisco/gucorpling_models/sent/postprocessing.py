from glob import glob
from argparse import ArgumentParser
import json, sys
from diaparser.parsers import Parser
import stanza


def get_stanza_model(lang):
    if lang == 'eng':
        stanza.download('en')
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'deu':
        stanza.download('de')
        nlp = stanza.Pipeline(lang='de', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'fra':
        stanza.download('fr')
        nlp = stanza.Pipeline(lang='fr', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'nld':
        stanza.download('nl')
        nlp = stanza.Pipeline(lang='nl', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'por':
        stanza.download('pt')
        nlp = stanza.Pipeline(lang='pt', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'rus':
        stanza.download('ru')
        nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'spa':
        stanza.download('es')
        nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'zho':
        stanza.download('zh')
        nlp = stanza.Pipeline(lang='zh', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'eus':
        stanza.download('eu')
        nlp = stanza.Pipeline(lang='eu', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == 'tur':
        stanza.download('tr')
        nlp = stanza.Pipeline(lang='tr', processors='tokenize,pos,lemma', tokenize_pretokenized=True)
    elif lang == "fas":
        stanza.download('fa')
        nlp = stanza.Pipeline(lang='fa', processors='tokenize,pos,lemma', tokenize_pretokenized=True)

    return nlp


def get_diaparser_model(lang, model_dir):
    if lang == 'eng':
        parser = Parser.load('en_ewt-electra')
    elif lang == 'deu':
        parser = Parser.load('de_hdt.dbmdz-bert-base')
    elif lang == 'fra':
        parser = Parser.load('fr_sequoia.camembert')
    elif lang == 'nld':
        parser = Parser.load('nl_alpino_lassysmall.wietsedv')
    elif lang == 'rus':
        parser = Parser.load('ru_syntagrus.DeepPavlov')
    elif lang == 'spa':
        parser = Parser.load('es_ancora.mbert')
    elif lang == 'zho':
        parser = Parser.load('zh_ptb.hfl')
    elif lang == 'tur':
        parser = Parser.load('tr_boun.electra-base')
    elif lang == 'por':
        parser = Parser.load(model_dir + '/diaparser.pt_bosque.bert-base-portuguese-cased.pt')
    elif lang == 'eus':
        parser = Parser.load(model_dir + '/diaparser.eu_bdt.distilbert-multilingual-cased.pt')
    elif lang == "fas":
        parser = Parser.load(model_dir + '/diaparser.fa_seraji.parsbert-base-cased.pt')

    return parser


def get_tags(sentences, lang):
    nlp = get_stanza_model(lang)
    tg = {'lemma': [], 'pos1': [], 'pos2': []}
    doc = nlp(sentences)
    for i, sentence in enumerate(doc.sentences):
        tg['lemma'].append([])
        tg['pos1'].append([])
        tg['pos2'].append([])
        for token in sentence.words:
            if token.lemma:
                tg['lemma'][-1].append(token.lemma)
            else:
                tg['lemma'][-1].append("_")
            if token.xpos:
                tg['pos1'][-1].append(token.xpos)
            else:
                tg['pos1'][-1].append("_")
            if token.upos:
                tg['pos2'][-1].append(token.upos)
            else:
                tg['pos2'][-1].append("_")

    return tg


def dependency_parser(sentences, lang, model_dir):
    parser = get_diaparser_model(lang, model_dir)
    dataset = parser.predict(sentences, prob=True)
    return dataset


if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-f", "--file", help="dir containing subdirectories for each languages, including sent_test.pred "
                                        "file")
    p.add_argument("-i", "--inf", help="dir containing doc info, output of get_docs_inof.py")
    p.add_argument("-d", "--model_dir", help="directory containing diaparser.eu_bdt.distilbert-multilingual-cased.pt "
                                             "and diaparser.pt_bosque.bert-base-portuguese-cased.pt for diaparser.")
    p.add_argument("-m", "--mode", help="train/test/dev.",
                   default="test")
    opts = p.parse_args()

    folders = glob(opts.file + '/' + '*/')
    for data_dir in folders:
        with open(data_dir + '/sent_' + opts.mode + '.predV2', 'r') as inp:
            lang = data_dir.split('/')[-2].split('.')[0]
            print('\n\n******************************************************\n\n')
            print(lang)
            print(data_dir)
            print('\n\n******************************************************\n\n')
            sentences = []
            for line in inp:
                if line.startswith('<s>'):
                    sentences.append([])
                elif line.startswith('</s>'):
                    continue
                else:
                    if len(sentences) == 0:
                        print('issue with first sentence in predictions')
                        sentences.append([])
                    sentences[-1].append(line.rstrip())

            tags = get_tags(sentences, lang)
            data = dependency_parser(sentences, lang, opts.model_dir)
            name = data_dir.split('/')[-2]
            with open(opts.inf + '/' + name + '/docs_tokens_' + opts.mode + '.json') as f:
                inf = json.load(f)

            with open(data_dir + '/' + data_dir.split('/')[-2] + '_' + opts.mode + '_silver.conllu', 'w') as of:
                start = 0
                st = '# newdoc id = ' + inf['docs'][0] + '\n'
                tok_index = 0
                doc_index = 0
                for i in range(start, len(sentences)):
                    t_n = 1
                    # import pdb; pdb.set_trace();
                    lns = str(data.sentences[i]).split('\n')
                    for j in range(len(sentences[i])):
                        if tok_index == len(inf['toks'][doc_index]):
                            if not st.endswith('\n\n'):
                                st += '\n'
                            st += '# newdoc id = ' + inf['docs'][doc_index + 1] + '\n'
                            tok_index = 0
                            doc_index += 1
                            t_n = 1
                        ann = lns[j].split('\t')
                        if ann[1] != inf['toks'][doc_index][tok_index]:
                            if not (ann[1] == '&' and inf['toks'][doc_index][tok_index] == '&amp;'):
                                print("tokens not matching")
                                print(ann[1] + '\n' + inf['toks'][doc_index][tok_index])
                                print(inf['toks'][doc_index][tok_index + 1])
                                if inf['toks'][doc_index][tok_index + 1] == ann[1]:
                                    tok_index += 1
                                    print("+1")
                        # import pdb; pdb.set_trace();
                        res = ann[0] + '\t' + inf['toks'][doc_index][tok_index] + '\t' + tags['lemma'][i][j] + '\t' + \
                              tags['pos1'][i][
                                  j] + '\t' + tags['pos2'][i][j] + '\t' + ann[5] + '\t' + ann[6] + '\t' + ann[
                                  7] + '\t' + \
                              ann[8] + '\t' + ann[9] + '\n'
                        tok_index += 1
                        st += res
                        t_n += 1
                    st += '\n'
                of.write(st)