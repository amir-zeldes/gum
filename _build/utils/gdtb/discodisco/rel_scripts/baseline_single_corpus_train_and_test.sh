#!/bin/bash
#set -o errexit
if [ $# -eq 0 ]; then
  echo "Supply the name of a corpus"
  exit 1
fi
CORPUS="$1"
CORPUS_DIR="data/2021/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_rel_bert_baseline
if [[ ! -d $CORPUS_DIR ]]; then
  echo "Corpus \"$CORPUS_DIR\" not found"
  exit 1
fi
if [[ -d $MODEL_DIR ]]; then
  echo "\"$MODEL_DIR\" already exists. Removing it now..."
  rm -rf "$MODEL_DIR"
fi

# use language-specific berts if we can
export EMBEDDING_DIMS=768
if [[ "$CORPUS" == "eng"* ]]; then
  export EMBEDDING_DIMS=1024
  export EMBEDDING_MODEL_NAME="roberta-large"
elif [[ "$CORPUS" == "zho"* ]]; then
  export EMBEDDING_MODEL_NAME="bert-base-chinese"
elif [[ "$CORPUS" == "eus"* ]]; then
  export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
else
  export EMBEDDING_DIMS=1024
  export EMBEDDING_MODEL_NAME="xlm-roberta-large"
fi

# use fastText embeddings
if [[ "$CORPUS" == "eng"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.en.300.vec"
elif [[ "$CORPUS" == "deu"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.de.300.vec"
elif [[ "$CORPUS" == "eus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.eu.300.vec"
elif [[ "$CORPUS" == "fra"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.fr.300.vec"
elif [[ "$CORPUS" == "nld"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.nl.300.vec"
elif [[ "$CORPUS" == "por"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.pt.300.vec"
elif [[ "$CORPUS" == "rus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.ru.300.vec"
elif [[ "$CORPUS" == "spa"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.es.300.vec"
elif [[ "$CORPUS" == "tur"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.nl.300.vec"
elif [[ "$CORPUS" == "zho"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="embeddings/cc.zh.300.vec"
else
  echo "Couldn't find a fasttext embedding for \"$CORPUS\"" >&2
  exit 1
fi

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.rels"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.rels"
echo $TRAIN_DATA_PATH
allennlp train \
  configs/rel/baseline/bert_baseline.jsonnet \
  -s "$MODEL_DIR"
echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Testing on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
JSON_PRED_PATH="${MODEL_DIR}/output_test.jsonl"
RELS_GOLD_PATH="${CORPUS_DIR}/${CORPUS}_test.rels"
allennlp predict \
  "${MODEL_DIR}/model.tar.gz" \
  "$RELS_GOLD_PATH" \
  --silent \
  --use-dataset-reader \
  --cuda-device 0 \
  --output-file "$JSON_PRED_PATH"

echo "Removing model files..."
rm $MODEL_DIR/*.th

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Scoring on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python gucorpling_models/rel/e2e_metrics.py $JSON_PRED_PATH
