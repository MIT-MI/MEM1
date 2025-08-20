WORK_DIR=.

# Default batch size
BATCH_SIZE=3

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

LOCAL_DIR=$WORK_DIR/train/data/nq_hotpotqa_train_multi_${BATCH_SIZE}

## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/gen_data/data_process/qa_search_train_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size $BATCH_SIZE

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/gen_data/data_process/qa_search_test_merge_multi.py --local_dir $LOCAL_DIR --data_sources $DATA --batch_size $BATCH_SIZE
