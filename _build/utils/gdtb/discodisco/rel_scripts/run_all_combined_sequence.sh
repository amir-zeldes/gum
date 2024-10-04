for CORPUS_NAME in `ls data/2021 | sort | tac`; do
  echo "#@#@ $CORPUS_NAME"
  bash rel_scripts/run_single_combined_sequence.sh $CORPUS_NAME ${1:-models}
done
