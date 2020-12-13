import KM_parser as KM_parser
import torch
import os
import gdown
import nltk # used for its tree formatting

from configuration import BEST_MODEL_PATH,BEST_MODEL_REMOTE_PATH,CONFIG_CALC_HEAD_CONTRIBUTIONS

class LALConstituentParser():

    def __init__(self,targetdir,usecpu=True):

        self.targetdir = targetdir
        self.reversetokenmap = dict([(value, key) for key, value in KM_parser.BERT_TOKEN_MAPPING.items()])
        self.parser = None
        self.usecpu = usecpu  # Forces CPU usage if the GPU ram isnt enough

        self.load_best_model()

    def load_best_model(self):
        try:
            print("Loading model from {}...".format(BEST_MODEL_PATH))
            assert BEST_MODEL_PATH.endswith(".pt"), "Only pytorch savefiles supported"

            if not os.path.isfile(BEST_MODEL_PATH):
                print ('Downloading the best model from the server. This can take a while, recommend a coffee break..')
                gdown.download(BEST_MODEL_REMOTE_PATH,BEST_MODEL_PATH,quiet=False)

            info = self.torch_load(BEST_MODEL_PATH)
            assert 'hparams' in info['spec'], "Older savefiles not supported"

            self.parser = KM_parser.ChartParser.from_spec(info['spec'], info['state_dict'])
            self.parser.contributions = (CONFIG_CALC_HEAD_CONTRIBUTIONS == 1)
            self.parser.eval()

        except Exception as e:
            print('Could not download or initialize the best model for the LAL constituent parser')
            raise e # terminate processing as need to retry

    def torch_load(self,load_path):

        if KM_parser.use_cuda and not self.usecpu:
            return torch.load(load_path)
        else:
            return torch.load(load_path, map_location=lambda storage, location: storage)

    def run_parse(self,sentences,ptbfile):

        def save_data(syntree_pred):

            for tree in syntree_pred:
                treestr = "{}\n".format(tree.convert().linearize())
                # add the ROOT
                treestr = "(ROOT " + treestr + ")"
                tree = nltk.Tree.fromstring(treestr)
                output_file.write(tree.pformat())
                output_file.write('\n\n')


        # sentences are in the form tag0\tword0 tag1\tword1
        sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

        syntree_pred = []
        tagged_sentences = [[(self.reversetokenmap.get(word.split('\t')[0], word.split('\t')[0]),
                              self.reversetokenmap.get(word.split('\t')[1], word.split('\t')[1])) for word in
                             sentence.split(' ')] for sentence in sentences]

        syntree, _ = self.parser.parse_batch(tagged_sentences)
        syntree_pred.extend(syntree)

        with open(ptbfile, 'w') as output_file:
            save_data(syntree_pred)


def main():
    targetdir = "<gum_home_dir>_build/target/const/"
    test = LALConstituentParser(targetdir) # loads model automatically

if __name__ == "__main__":
    main()