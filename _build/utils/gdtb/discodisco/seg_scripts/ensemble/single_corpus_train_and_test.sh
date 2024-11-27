#!/bin/bash
#set -o errexit
if [ $# -eq 0 ]; then
	echo "Supply the name of a corpus"
	exit 1
fi
CORPUS="$1"
CORPUS_DIR="data/2019/${1}"
if [[ ! -d $CORPUS_DIR ]]; then
	echo "Corpus \"$CORPUS_DIR\" not found"
	exit 1
fi

python gucorpling_models/seg/subtree_segmenter.py -c "$1" -d sharedtask2019/data --mode train --eval_test
python gucorpling_models/seg/subtree_segmenter.py -c "$1" -d sharedtask2019/data --mode test --eval_test
python gucorpling_models/seg/flair_edu_segmenter.py -c "$1" --mode train
python gucorpling_models/seg/flair_edu_segmenter.py -c "$1" --mode test
bash seg_scripts/single_corpus_train_and_test.sh "$1" "models" "models/seg_ensemble_jsons/allennlp"
