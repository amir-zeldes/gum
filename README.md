# GUM

Repository for the Georgetown University Multilayer Corpus (GUM)

This repository contains release versions of the Georgetown University Multilayer Corpus (GUM), a corpus of English texts from twelve written and spoken text types:

  * interviews
  * news
  * travel guides
  * how-to guides
  * academic writing
  * biographies
  * fiction
  * online forum discussions
  * spontaneous face to face conversations
  * political speeches
  * textbooks
  * vlogs

The corpus is created as part of the course LING-367 (Computational Corpus Linguistics) at Georgetown University. For more details see: https://corpling.uis.georgetown.edu/gum.

## A note about reddit data

For one of the twelve text types in this corpus, reddit forum discussions, plain text data is not supplied. To obtain this data, please run `_build/process_reddit.py`, then `run _build/build_gum.py`. This, and all data, is provided with absolutely no warranty; users agree to use the data under the license with which it is provided, and reddit data is subject to reddit's terms and conditions. See [README_reddit.md](README_reddit.md) for more details.

## Citing

To cite this corpus, please refer to the following article:

Zeldes, Amir (2017) "The GUM Corpus: Creating Multilayer Resources in the Classroom". Language Resources and Evaluation 51(3), 581–612. 

```
@Article{Zeldes2017,
  author    = {Amir Zeldes},
  title     = {The {GUM} Corpus: Creating Multilayer Resources in the Classroom},
  journal   = {Language Resources and Evaluation},
  year      = {2017},
  volume    = {51},
  number    = {3},
  pages     = {581--612},
  doi       = {http://dx.doi.org/10.1007/s10579-016-9343-x}
}
```

For a full list of contributors please see [the corpus website](https://corpling.uis.georgetown.edu/gum).

## Directories

The corpus is downloadable in multiple formats. Not all formats contain all annotations: The most accessible format is probably CoNLL-U dependencies (in `dep/`), but the most complete XML representation is in [PAULA XML](https://www.sfb632.uni-potsdam.de/en/paula.html), and the easiest way to search in the corpus is using [ANNIS](http://corpus-tools.org/annis). Here is [an example query](https://corpling.uis.georgetown.edu/annis/#_q=ZW50aXR5IC0-YnJpZGdlIGVudGl0eSAmICMxIC0-aGVhZCBsZW1tYT0ib25lIg&_c=R1VN&cl=5&cr=5&s=0&l=10) for phrases headed by 'one' bridging back to a different, previously mentioned entity. Other formats may be useful for other purposes. See website for more details.

**NB: reddit data is not included in top folders - consult README_reddit.md to add it**

  * _build/ - The [GUM build bot](https://corpling.uis.georgetown.edu/gum/build.html) and utilities for data merging and validation
  * annis/ - The entire merged corpus, with all annotations, as a relANNIS 3.3 corpus dump, importable into [ANNIS](http://corpus-tools.org/annis)
  * const/ - Constituent trees with function labels and PTB POS tags in the PTB bracketing format (automatic parser output)
  * coref/ - Entity and coreference annotation in two formats: 
    * conll/ - CoNLL shared task tabular format (with Wikification but no bridging or split antecedent annotations)
    * tsv/ - WebAnno .tsv format, including entity and information status annotations, Wikification, bridging, split antecedent and singleton entities
  * dep/ - Dependency trees using Universal Dependencies, enriched with sentence types, entities, coreference, bridging, Wikification, automatic morphological tags and Universal POS tags according to the UD standard
  * paula/ - The entire merged corpus in standoff [PAULA XML](https://www.sfb632.uni-potsdam.de/en/paula.html), with all annotations
  * rst/ - Rhetorical Structure Theory analyses in .rs3 format as used by RSTTool and rstWeb, as well as binary and n-ary lisp trees (.dis) and an RST dependency representation (.rsd)
  * xml/ - vertical XML representations with 1 token or tag per line and tab delimited lemmas and POS tags (extended, VVZ style, vanilla, UPOS and CLAWS5, as well as dependency functions), compatible with the IMS Corpus Workbench (a.k.a. TreeTagger format).