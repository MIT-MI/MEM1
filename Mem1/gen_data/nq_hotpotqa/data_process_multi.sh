WORK_DIR=train
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_multi_3

## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size 3

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size 3
