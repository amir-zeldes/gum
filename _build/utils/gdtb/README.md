# GDTB build utils

This directory contains build utilities for compiling the GDTB version of the discourse relations in GUM, following the PDTB v3 guidelines.

## When do I need this?

Normally, never, since GUM is distributed with pre-compiled GDTB relations. These utilities are only used if data is added to the GUM corpus for which no GDTB style relation predictions exist. This can happen for two reasons:

  * New documents have been added to GUM for which we need initial GDTB predictions
  * eRST annotations have changed significantly, leading to new argument spans for which we need GDTB predictions

## Running GDTB compilation

To run a fresh build of GDTB, go to `utils/gdtb/` and run:

`> python main.py`

Alternatively if you want to run compilation for a single document, run for example:

`> python main.py -d GUM_academic_art`

Resulting files will be serialized to `_build/target/rst/gdtb/` in all formats. Note that if you have new data (documents or new eRST relations/spans spawning new GDTB relations), the system will need fresh predictions to assess the probability of each GDTB label for each relation/span, and for implicit relations, fresh predictions of implicit connectives. To obtain these, first run `main.py` as is - this will allow the system to identify new relations for which predictions do not yet exist. Then run the following two steps.

Also note that the scripts assume all relevant data for GDTB generation lives in `_build/target/`, so you have to have a fresh build of GUM itself first, including reconstruction of Reddit text if needed. A GDTB document will be generated for every eRST file in `_build/target/rst/rstweb/`.

### 1. Running relation label probability predictions

Relation label probabilities inform system decisions, though note that they are just one source of information, as described in the GDTB paper. For example, if an eRST relations has an explicit "but" marking it, the only labels for which "but" is a possible connective will be considered. To obtain probabilities for new data:

  * First run main.py without the new probabilities. This will generate three files:
    * `discodisco/eng.pdtb.missing_test.rels` - Relations to predict labels for in DISRPT2023 format
    * `eng.pdtb.missing_test.conllu` - Syntax trees for the relevant document(s)
    * `eng.pdtb.missing_test_keys.tab` - Relation IDs (eRST source-target-label)
  * Make sure you have the discodisco model `eng.pdtb.pdtb_flair_clone` in models/ (download from https://gucorpling.org/) and the dependencies in requirements.txt are installed
  * Now run `predict_missing.bat`/`predict_missing.sh`

This process will populate `data/discodisco_preds/eng.rst.gum_add.rels` and `data/discodisco_preds/eng.rst.gum_add_predictions.json`. You can now re-run `main.py` using fresh relation label probabilities.

### 2. Running implicit connective predictions

If you have already run `main.py` once, then `connpred/missing.jsonl` will be populated with one line per implicit relation that needs connective predictions. Verify that this file contains your new implicit relations, and that you have the model inside `model/` (e.g. `model.safetensors` etc.), or download the model from https://gucorpling.org/.

Now run prediction by executing `> python predict.py` - this should generate `data/connector_preds/gum_implicit_add_preds.jsonl`.

## Citing

To cite GDTB and for more information, please refer to this paper:

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