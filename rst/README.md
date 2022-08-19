# RST - Discourse Parses in Rhetorical Structure Theory

This directory contains discourse parses according to Rhetorical Structure Theory, in multiple formats. The recommended native format for the discourse parses is the .rs3 XML format in the rstweb directory. The remaining formats are automatically converted from .rs3 by the GUM build bot.

  * dependencies: RST dependency representation according to the algorithm described in Li et al. (2014). The format resembles the 10 column tab-delimited conllu format. Each row is a single discourse unit, with an ID in column 1, text in column 2 (tokens separated by space) and the parent unit and relation name in columns 7-8. Note that multinuclear vs. satellite-nucleus relations can be distinguished by the suffixes `_m` and `_r` respectively. Other columns give additional information on attachment depth in the constituent tree, the head word of each EDU, POS tags, sentence types, etc., which are encoded in other formats of the corpus.
  * lisp_binary: binary branching consituent trees, with head unit indicated using `SN` (satellite-nucleus), `NS` (nucleus-satellite) or `NN` (multinuclear). Only terminal EDU nodes have text content, with tokens separated by space and surrounded by `text _!..._!`
  * lisp_nary: same as lisp_binary, but trees are not guaranteed to be binary branching: multinuclear nodes may have n children, where n > 1. Corresponds more directly to the source data in rstweb/ .rs3, but cannot be used to train parsers which require binary trees
  * rstweb: source format for the RST annotations, compatible with rstWeb (Zeldes 2016) and RSTTool. The format natively distinguishes multinuclear nodes and satellite-nucleus, nested hierarchy and n-ary nodes. Can be used to visualize RST trees (see the website for rstWeb at https://gucorpling.org/rstweb/info/ for examples)

## References

  * Li, Sujian, Liang Wang, Ziqiang Cao & Wenjie Li (2014) Text-level discourse dependency parsing. In Proceedings of ACL 2014. Baltimore, MD, 25–35.
  * Zeldes, Amir (2016) rstWeb - A Browser-based Annotation Interface for Rhetorical Structure Theory and Discourse Relations. In Proceedings of NAACL 2016 System Demonstrations. San Diego, CA, 1-5.