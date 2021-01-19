import nltk
import glob
import os
from argparse import ArgumentParser

class PTBTreeChecker():

    """
    Created to run some tests on the new LAL PTB trees versus the old coreNLP ones
    First test here is to check that tokens are the same
    Can Add others!
    """

    def __init__(self,oldtreedir,newtreedir):
        self.oldtreedir = oldtreedir
        self.newtreedir = newtreedir

    def check_token_parity(self):

        print ('Starting token parity test')

        totalfiles = len(glob.glob(self.newtreedir + '*.ptb'))
        filecounter = 0

        for file in glob.glob(self.newtreedir + '*.ptb'):

            filename = file.split(os.sep)[-1]

            oldtrees = []
            newtrees = []

            # Get old trees
            with open(self.oldtreedir + filename, 'r') as r:
                fulltree = ''
                for line in r.readlines():

                    if fulltree != '' and 'ROOT' in line:
                        oldtrees.append(fulltree.strip())
                        fulltree = ''

                    temp = line.strip()
                    if temp != '':
                        fulltree += temp
                        fulltree += ' '

                oldtrees.append(fulltree.strip())

            # get new trees
            with open(self.newtreedir + filename, 'r') as r:
                fulltree = ''
                for line in r.readlines():
                    if fulltree != '' and 'ROOT' in line:
                        newtrees.append(fulltree.strip())
                        fulltree = ''

                    temp = line.strip()
                    if temp != '':
                        fulltree += temp
                        fulltree += ' '

                newtrees.append(fulltree.strip())

            try:
                assert len(oldtrees) == len(newtrees)
            except AssertionError:
                print ('Number of sentences dont match between the old and new file versions %s. If this is a reddit file, consider adding the tokens to the file' % filename)
                raise

            for i in range(0,len(newtrees)):
                oldsent = nltk.Tree.fromstring(oldtrees[i]).flatten().leaves()
                newsent = nltk.Tree.fromstring(newtrees[i]).flatten().leaves()

                try:
                    assert ' '.join(oldsent) == ' '.join(newsent)
                except AssertionError:
                    print ('Check failed for sentence %s of file %s. If this is a reddit file, consider adding the tokens back' % (str(i + 1),filename))
                    raise

            filecounter += 1

        assert filecounter == totalfiles
        print ('Token parity test successful')

def main(old,new):


    ptbcheck = PTBTreeChecker(old,new)
    ptbcheck.check_token_parity()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", dest="old", action="store", help="Directory with old PTB trees", default='..' + os.sep + 'target' + os.sep + 'const' + os.sep)
    parser.add_argument("-n", dest="new", action="store", help="Directory with new PTB trees, default is target/const", default='..' + os.sep + 'target' + os.sep + 'const' + os.sep)

    options = parser.parse_args()

    main(options.old,options.new)