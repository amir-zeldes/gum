#!/bin/bash
if [ $# -eq 0 ]; then
  echo "Supply the name of a corpus"
  exit 1
fi
if [[ ! -d "data/" ]]; then
        echo "Data not found--please download it from https://drive.google.com/file/d/1wDmv6TzZqUwnw1Csn4Yz66uF1UYV-K-L/view?usp=sharing"
        exit 1
fi

CORPUS="$1"
export CORPUS
CORPUS_DIR="data/2021/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_flair_clone

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
if [[ "$CORPUS" == "deu"* ]]; then
  export EMBEDDING_MODEL_NAME="bert-base-german-cased"
elif [[ "$CORPUS" == "eng.sdrt.stac"* ]]; then
  export EMBEDDING_MODEL_NAME="bert-base-uncased"
elif [[ "$CORPUS" == "eng"* ]]; then
  export EMBEDDING_MODEL_NAME="bert-base-cased"
elif [[ "$CORPUS" == "eus"* ]]; then
  export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "fas"* ]]; then
  export EMBEDDING_MODEL_NAME="HooshvareLab/bert-fa-base-uncased"
elif [[ "$CORPUS" == "fra"* ]]; then
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-french-europeana-cased"
elif [[ "$CORPUS" == "nld"* ]]; then
  export EMBEDDING_MODEL_NAME="GroNLP/bert-base-dutch-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "rus"* ]]; then
  export EMBEDDING_MODEL_NAME="DeepPavlov/rubert-base-cased-sentence"
elif [[ "$CORPUS" == "spa"* ]]; then
  export EMBEDDING_MODEL_NAME="dccuchile/bert-base-spanish-wwm-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
  export EMBEDDING_MODEL_NAME="hfl/chinese-bert-wwm-ext"
else
  echo "No LM configuration!"
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
  configs/rel/flair_clone.jsonnet \
  -s "$MODEL_DIR" \

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Predicting on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TEST_DATA_PATH="${CORPUS_DIR}/${CORPUS}_test.rels"
export OUTPUT_FILE_PATH="$MODEL_DIR/test_predictions.json"
echo $TEST_DATA_PATH
allennlp predict \
        $MODEL_DIR \
        $TEST_DATA_PATH \
        --silent \
	--cuda-device 0 \
        --use-dataset-reader \
        --output-file $OUTPUT_FILE_PATH

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Evaluating on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
python utils/e2e_metrics.py $OUTPUT_FILE_PATH
cat $MODEL_DIR/predicti.res
