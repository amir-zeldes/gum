#!/bin/bash

if [ $# -eq 0 ]; then
	echo "Supply the name of a corpus"
	exit 1
fi

CORPUS="$1"
CORPUS_DIR="sharedtask2021/data/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_rel_lm

if [[ ! -d $CORPUS_DIR ]]; then
	echo "Corpus \"$CORPUS_DIR\" not found"
	exit 1
fi

if [[ -d $MODEL_DIR ]]; then
   # echo "\"$MODEL_DIR\" already exists. Ignore it"
   # exit 1
	echo "\"$MODEL_DIR\" already exists. Removing it now..."
	rm -rf "$MODEL_DIR"
fi

# use language-specific berts if we can
export EMBEDDING_DIMS=768
if [[ "$CORPUS" == "eng.sdrt.stac"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-uncased"
elif [[ "$CORPUS" == "eng"* ]]; then
	export EMBEDDING_MODEL_NAME="SpanBERT/spanbert-base-cased"
#    export EMBEDDING_MODEL_NAME="bert-base-cased"
elif [[ "$CORPUS" == "deu"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-german-cased"
elif [[ "$CORPUS" == "fra"* ]]; then
	export EMBEDDING_MODEL_NAME="dbmdz/bert-base-french-europeana-cased"
elif [[ "$CORPUS" == "nld"* ]]; then
	export EMBEDDING_MODEL_NAME="GroNLP/bert-base-dutch-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
	export EMBEDDING_MODEL_NAME="bert-base-chinese"
elif [[ "$CORPUS" == "eus"* ]]; then
	export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "spa"* ]]; then
	export EMBEDDING_MODEL_NAME="dccuchile/bert-base-spanish-wwm-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
	export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
elif [[ "$CORPUS" == "rus"* ]]; then
	export EMBEDDING_MODEL_NAME="blinoff/roberta-base-russian-v0"
elif [[ "$CORPUS" == "fas"* ]]; then
	export EMBEDDING_MODEL_NAME="HooshvareLab/bert-fa-zwnj-base"
else
	export EMBEDDING_MODEL_NAME="bert-base-multilingual-cased"
fi

# export EMBEDDING_MODEL_NAME="bert-base-multilingual-cased"

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.rels"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.rels"
echo $TRAIN_DATA_PATH
allennlp train \
	configs/rel/e2e/e2e.jsonnet \
	-s "$MODEL_DIR" \

# echo ""
# echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "# Predicting on ${CORPUS}"
# echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo ""
# export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.rels"
# echo $VALIDATION_DATA_PATH
# allennlp predict \
#         $MODEL_DIR \
#         $VALIDATION_DATA_PATH \
#         --use-dataset-reader \
#         --output-file tmp/dev_rel_e2e_renovated/predictions_e2e_rel_${CORPUS}.json

# echo ""
# echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo "# Evaluating on ${CORPUS}"
# echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
# echo ""
# python utils/e2e_metrics.py tmp/dev_rel_e2e_renovated/predictions_e2e_rel_${CORPUS}.json


echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Predicting on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_test.rels"
echo $VALIDATION_DATA_PATH
allennlp predict \
        $MODEL_DIR \
        $VALIDATION_DATA_PATH \
        --use-dataset-reader \
        --output-file tmp/test_rel_e2e_multilingualBERT_renovated/predictions_e2e_rel_${CORPUS}.json

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Evaluating on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python utils/e2e_metrics.py tmp/test_rel_e2e_multilingualBERT_renovated/predictions_e2e_rel_${CORPUS}.json
