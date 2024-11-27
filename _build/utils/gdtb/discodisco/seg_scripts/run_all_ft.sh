for CORPUS_NAME in  `ls data/2021/ | sort | tac`; do 
  echo "#@#@ $CORPUS_NAME"
  bash seg_scripts/single_corpus_train_and_test_ft.sh $CORPUS_NAME ${1:-models}
done
