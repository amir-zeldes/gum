# GDTB connective prediction

Connective prediction for implicit relations in GDTB as described in:

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

First run `../main.py` to generate `connpred/missing.jsonl` with one line per implicit relation that needs connective predictions. Verify that this file contains your new implicit relations, and that you have the model inside `model/` (e.g. `model.safetensors` etc.), or download the model from https://gucorpling.org/.

Now run prediction by executing `> python predict.py` - this should generate `data/connector_preds/gum_implicit_add_preds.jsonl`.