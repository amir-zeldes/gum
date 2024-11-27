for CORPUS_NAME in  `ls sharedtask2019/data/ | sort | tac`; do 
  echo "#@#@ $CORPUS_NAME"
  bash seg_scripts/single_corpus_train_and_test.sh $CORPUS_NAME ${1:-models}
done
