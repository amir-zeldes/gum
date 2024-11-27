for CORPUS_NAME in  `ls data/2021_silver | sort | tac`; do 
  echo "#@#@ $CORPUS_NAME"
  bash seg_scripts/silver_single_corpus_train_and_test_ft.sh $CORPUS_NAME ${1:-models}
done
