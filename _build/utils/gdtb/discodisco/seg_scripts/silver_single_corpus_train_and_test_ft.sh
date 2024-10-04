#!/bin/bash
#set -o errexit
if [ $# -eq 0 ]; then
  echo "Supply the name of a corpus"
  exit 1
fi
CORPUS="$1"
export CORPUS
CORPUS_DIR="data/2021_silver/${1}"
MODEL_DIR=${2:-models}/${CORPUS}_seg_bert_baseline_ft_silver
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
  #export EMBEDDING_MODEL_NAME="roberta-large"
  export EMBEDDING_MODEL_NAME="google/electra-large-discriminator"
elif [[ "$CORPUS" == "fas"* ]]; then
  export EMBEDDING_MODEL_NAME="HooshvareLab/bert-fa-base-uncased"
#elif [[ "$CORPUS" == "deu"* ]]; then
#  #export EMBEDDING_DIMS=1024
#  #export EMBEDDING_MODEL_NAME="deepset/gelectra-large"
#  export EMBEDDING_DIMS=1024
#  export EMBEDDING_MODEL_NAME="deepset/gbert-large"
#elif [[ "$CORPUS" == "fra"* ]]; then
#  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-french-europeana-cased"
elif [[ "$CORPUS" == "zho"* ]]; then
  #export EMBEDDING_DIMS=1024
  #export EMBEDDING_MODEL_NAME="hfl/chinese-electra-180g-large-discriminator"
  export EMBEDDING_MODEL_NAME="bert-base-chinese"
elif [[ "$CORPUS" == "nld"* ]]; then
  #export EMBEDDING_MODEL_NAME="GroNLP/bert-base-dutch-cased"
  export EMBEDDING_MODEL_NAME="pdelobelle/robbert-v2-dutch-base"
elif [[ "$CORPUS" == "eus"* ]]; then
  export EMBEDDING_MODEL_NAME="ixa-ehu/berteus-base-cased"
elif [[ "$CORPUS" == "spa"* ]]; then
  #export EMBEDDING_MODEL_NAME="mrm8488/electricidad-base-discriminator"
  export EMBEDDING_MODEL_NAME="dccuchile/bert-base-spanish-wwm-cased"
elif [[ "$CORPUS" == "por"* ]]; then
  export EMBEDDING_MODEL_NAME="neuralmind/bert-base-portuguese-cased"
elif [[ "$CORPUS" == "tur"* ]]; then
  #export EMBEDDING_MODEL_NAME="dbmdz/electra-base-turkish-cased-discriminator"
  export EMBEDDING_MODEL_NAME="dbmdz/bert-base-turkish-cased"
elif [[ "$CORPUS" == "rus"* ]]; then
  export EMBEDDING_MODEL_NAME="DeepPavlov/rubert-base-cased"
else
  export EMBEDDING_DIMS=1024
  export EMBEDDING_MODEL_NAME="xlm-roberta-large"
fi

# do not use CRF on RST datasets
export USE_CRF=0
if [[ "$CORPUS" == *".pdtb."* ]]; then
  export USE_CRF=1
fi

# use fastText embeddings
if [[ "$CORPUS" == "eng"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
elif [[ "$CORPUS" == "deu"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz"
elif [[ "$CORPUS" == "eus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.eu.300.vec.gz"
elif [[ "$CORPUS" == "fas"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz"
elif [[ "$CORPUS" == "fra"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
elif [[ "$CORPUS" == "nld"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz"
elif [[ "$CORPUS" == "por"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
elif [[ "$CORPUS" == "rus"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz"
elif [[ "$CORPUS" == "spa"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz"
elif [[ "$CORPUS" == "tur"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz"
elif [[ "$CORPUS" == "zho"* ]]; then
  export FASTTEXT_EMBEDDING_FILE="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"
else
  echo "Couldn't find a fasttext embedding for \"$CORPUS\"" >&2
  exit 1
fi

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Training on $CORPUS"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
export TRAIN_DATA_PATH="${CORPUS_DIR}/${CORPUS}_train.conllu"
export VALIDATION_DATA_PATH="${CORPUS_DIR}/${CORPUS}_dev.conllu"
echo $TRAIN_DATA_PATH
allennlp train \
  configs/seg/baseline/bert_baseline_ft_silver.jsonnet \
  -s "$MODEL_DIR"

echo ""
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "# Testing on ${CORPUS}"
echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ""
JSON_PRED_PATH="${MODEL_DIR}/output_test.jsonl"
CONLL_PRED_PATH="${MODEL_DIR}/output_test.conllu"
CONLL_GOLD_PATH="${CORPUS_DIR}/${CORPUS}_test.conllu"

allennlp predict \
  "${MODEL_DIR}/model.tar.gz" \
  "$CONLL_GOLD_PATH" \
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
python seg_scripts/format_output.py "$JSON_PRED_PATH" "$CONLL_PRED_PATH"
python seg_scripts/seg_eval_2019_modified.py "$CONLL_GOLD_PATH" "$CONLL_PRED_PATH" | tee "$MODEL_DIR/score.txt"
printf "#!/bin/sh\npython seg_scripts/seg_eval_2019_modified.py $CONLL_GOLD_PATH $CONLL_PRED_PATH\n" > "$MODEL_DIR/calc_score.sh"

# Accumulate a record of scores
TSV_PATH="all_scores_finetuning_silver.tsv"
cat "$MODEL_DIR/score.txt" | cut -d " " -f2 | head -n 1 | cut -d "_" -f1 | tr "\n" "\t" >> "$TSV_PATH"
cat "$MODEL_DIR/score.txt" | cut -d " " -f3 | tail -n 3 | tr "\n" "\t" | sed 's/\t$/\n/g' >> "$TSV_PATH"
