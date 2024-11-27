for CORPUS_NAME in `ls data/2021 | sort | tac`; do
  echo "#@#@ $CORPUS_NAME"
  bash rel_scripts/run_single_flair_clone.sh $CORPUS_NAME ${1:-models}
done
