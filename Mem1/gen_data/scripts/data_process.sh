WORK_DIR=.
LOCAL_DIR=$WORK_DIR/data/deepdive

## process multiple dataset search format train file
# DATA=nq,hotpotqa
# python $WORK_DIR/gen_data/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
python $WORK_DIR/gen_data/data_process/deepdive.py --local_dir $LOCAL_DIR

## process multiple dataset search format test file
# DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python $WORK_DIR/gen_data/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
