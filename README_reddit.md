# Data from Reddit

For one of the text types in this corpus, Reddit forum discussions, plain text data is not supplied in this repository. To obtain this data, please follow the instructions below.

## Annotations

Documents in the Reddit subcorpus are named `GUM_reddit_*` (e.g. GUM_reddit_superman) and are included in the root folder with all annotation layers but with underscores instead of text. To compile the corpus including Reddit data, you must obtain the underlying texts, and either regenerate the files in the top level folders (works for all formats except `PAULA` and `annis`), or rebuild the corpus (see below).

## Obtaining underlying Reddit text data

To recover Reddit data, use the API provided by the Python script `get_text.py`, which will restore text in all top-level folders except for `PAULA` and `annis`. If you do not have credentials for the Python Reddit API wrapper (praw) and Google bigquery, the script can attempt to download data for you from a proxy. Otherwise you can also use your own credentials for praw etc. and include them in two files, `praw.txt` and `key.json`. For this to work, you must have the praw and bigquery libraries installed for python (e.g. via pip). 

If you also require the `PAULA` and `annis` formats, you must rebuild the corpus from `_build/src/`. To do this,  run `_build/process_reddit.py`, which again requires either running a proxy download or using your own credentials and placing them in `_build/utils/get_reddit/`. Once the download completes successfully, you will need to rebuild the corpus as explained in the next step.

## Rebuilding the corpus with Reddit data

To compile all projected annotations and produce all formats not included in `_build/src/`, you will need to run the GUM build bot: `python build_gum.py` in `_build/`. This process is described in detail at https://gucorpling.org/gum/build.html, but summarized instructions follow.

At a minumum, you can run `build_gum.py` with no options. This will produce basic formats in `_build/target/`, but skip generating fresh constituent parses and CLAWS5 tags. To include these you will need:

  * CLAWS5: use option -c and ensure that utils/paths.py points to an executable for the TreeTagger (http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/). The CLAWS 5 parameter file is already included in utils/treetagger/lib/, and tags are auto-corrected by the build bot based on gold PTB tags.
  * Constituent parses: option -p; requires dependencies for the included neural parser

After the build bot runs, data including `PAULA` and `annis` versions will be generated in the specified `target/` folder. If you run into problems building the corpus, feel free to report an issue via GitHub or contact us via e-mail.