for CORPUS_NAME in `ls sharedtask2021/data/`; do
  echo "#@#@ $CORPUS_NAME"
  bash rel_scripts/train_single.sh $CORPUS_NAME ${1:-models}
done
