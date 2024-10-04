# Introduction
DisCoDisCo (**Dis**trict of **Co**lumbia **Dis**course **Co**gnoscente) is [GU Corpling](http://corpling.uis.georgetown.edu/corpling/)'s submission to the [DISRPT 2021 shared task](https://sites.google.com/georgetown.edu/disrpt2021). 
DisCoDisCo [placed first](https://sites.google.com/georgetown.edu/disrpt2021/results) among all systems submitted to the 2021 shared task across all five subtasks.
Consult [the official repo](https://github.com/disrpt/sharedtask2021) for more information on the shared task.

See our paper here: https://aclanthology.org/2021.disrpt-1.6/

Citation:

```
@inproceedings{gessler-etal-2021-discodisco,
    title = "{D}is{C}o{D}is{C}o at the {DISRPT}2021 Shared Task: A System for Discourse Segmentation, Classification, and Connective Detection",
    author = "Gessler, Luke  and
      Behzad, Shabnam  and
      Liu, Yang Janet  and
      Peng, Siyao  and
      Zhu, Yilun  and
      Zeldes, Amir",
    booktitle = "Proceedings of the 2nd Shared Task on Discourse Relation Parsing and Treebanking (DISRPT 2021)",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.disrpt-1.6",
    pages = "51--62"
}
```

# Usage

## Setup
1. Create a new environment:

```bash
conda create --name disrpt python=3.8
conda activate disrpt
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure [the 2021 shared task data](https://github.com/disrpt/sharedtask2021) is at `data/2021/`.

## Experiments

Gold segmentation:

```bash
bash seg_scripts/single_corpus_train_and_test_ft.sh zho.rst.sctb
```

Silver segmentation:

```bash
bash seg_scripts/silver_single_corpus_train_and_test_ft.sh zho.rst.sctb
```

Relation classification:

```bash
bash rel_scripts/run_single_flair_clone.sh zho.rst.sctb
```

### Troubleshooting
Batch size may be modified, if necessary, using the `batch_size` parameter in:

* [`configs/seg/baseline/bert_baseline_ft.jsonnet`](configs/seg/baseline/bert_baseline_ft.jsonnet)
* [`configs/seg/baseline/bert_baseline_ft_silver.jsonnet`](configs/seg/baseline/bert_baseline_ft_silver.jsonnet)
* [`configs/rel/flair_clone.jsonnet`](configs/rel/flair_clone.jsonnet)
