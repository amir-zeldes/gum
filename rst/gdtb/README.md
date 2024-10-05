# GDTB - GUM Discourse Treebank data

This directory contains files for the GDTB shallow discourse relation annotations in GUM. Files are available in two formats, though note that our .conllu dependency files also contain PDTB annotations in the MISC field:

  * pdtb/ - original PDTB Annotator standoff files in two folders: raw/00/ for the raw text files and gold/00/ for standoff annotations separated by pipes, indexed via character offsets
  * disrpt/ - standard DISRPT shared task .rels format for relation classification

Note that Reddit data in the release does not contain text and must be reconstructed according to the instructions in the main README.md and under `_build/utils/gdtb/`. For more information and for citing please refer to this paper:

  * Yang Janet Liu, Tatsuya Aoyama, Wesley Scivetti, Yilun Zhu, Shabnam Behzad, Lauren Elizabeth Levine, Jessica Lin, Devika Tiwari, and Amir Zeldes (2024), "GDTB: Genre Diverse Data for English Shallow Discourse Parsing across Modalities, Text Types, and Domains". In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics: Miami, USA.

```bibtex
@inproceedings{liu-etal-2024-GDTB,
    title = "GDTB: Genre Diverse Data for English Shallow Discourse Parsing across Modalities, Text Types, and Domains",
    author = "Yang Janet Liu and Tatsuya Aoyama and Wesley Scivetti and Yilun Zhu and Shabnam Behzad and Lauren Elizabeth Levine and Jessica Lin and Devika Tiwari and Amir Zeldes",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, USA",
    publisher = "Association for Computational Linguistics",
    abstract = "Work on shallow discourse parsing in English has focused on the Wall Street Journal corpus, the only large-scale dataset for the language in the PDTB framework. However, the data is not openly available, is restricted to the news domain, and is by now 35 years old. In this paper, we present and evaluate a new open-access, multi-genre benchmark for PDTB-style shallow discourse parsing, based on the existing UD English GUM corpus, for which discourse relation annotations in other frameworks already exist. In a series of experiments on cross-domain relation classification, we show that while our dataset is compatible with PDTB, substantial out-of-domain degradation is observed, which can be alleviated by joint training on both datasets.",
}
```