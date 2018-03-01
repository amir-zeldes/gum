# Data from reddit

For one of the text types in this corpus, reddit forum discussions, plain text data is not supplied in this repository. To obtain this data, please follow the instructions below.

## Annotations

Documents in the reddit subcorpus are named GUM_reddit_* (e.g. GUM_reddit_superman) and are *not* included in the root folder with all annotation layers. The annotations for the reddit subcorpus can be found together with all other document annotations in `_build/src/`. Token representations in these files are replaced with underscores, while the annotations themselves are included in the files. To compile the corpus including reddit data, you must obtain the underlying texts.

## Obtaining underlying reddit text data

To recover reddit data, use the API provided by the script `_build/process_reddit.py`. If you have your own credentials for use with the Python reddit API wrapper (praw) and Google bigquery, you should include them in two files, `praw.txt` and `key.json` in `_build/utils/get_reddit/`. For this to work, you must have the praw and bigquery libraries installed for python (e.g. via pip). You can then run `python _build/process_reddit.py` to recover the data, and proceed to the next step, re-building the corpus.

Alternatively, if you can't use praw/bigquery, the script `_build/process_reddit.py` will offer to download the data for you by proxy. To do this, run the script and confirm that you will only use the data according to the terms and conditions determined by reddit, and for non-commercial purposes. The script will then download the data for you - if the download is successful, you can continue to the next step and re-build the corpus.

## Rebuilding the corpus with reddit data

To compile all projected annotations and produce all formats not included in `_build/src/`, you will need to run the GUM build bot: `python _build/build_gum.py`. This process is described in detail at https://corpling.uis.georgetown.edu/gum/build.html, but summarized instructions follow.

At a minumum, you can run `python _build/build_gum.py` with no options. This will produce basic formats in `_build/target/`, but skip generating fresh constituent parses, CLAWS5 tags and the Universal Dependencies version of the dependency data. To include these you will need:

  * CLAWS5: use option -c and ensure that utils/paths.py points to an executable for the TreeTagger (http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/). The CLAWS 5 parameter file is already included in utils/treetagger/lib/, and tags are auto-corrected by the build bot based on gold PTB tags.
  * Constituent parses: option -p; ensure that paths.py correctly points your installation of the Stanford Parser/CoreNLP
  * Universal Dependencies: option -u; ensure the paths.py points to CoreNLP, and that you have installed udapi and depedit (pip install udapi; pip install depedit). Note that this only works with Python 3.

If you run into problems building the corpus, feel free to report an issue via GitHub or contact us via e-mail.