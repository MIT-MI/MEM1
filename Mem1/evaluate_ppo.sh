#!/usr/bin/env bash

# Default values
CHECKPOINT_DIR="/raid/shared/mem1/models/nq-search-r1-ppo-qwen2.5-7b-base"
CHECKPOINT_STEP=300  # Set to specific step or leave empty for latest
TEST_FILE="data/nq_search/test.parquet"
NUM_EXAMPLES=100  # Set to a smaller number for faster testing
SEARCH_URL="http://127.0.0.1:8011/retrieve"

# # Set VLLM_ATTENTION_BACKEND for Qwen models
# if [ -n "$CHECKPOINT_DIR" ] && [ "$(echo "$CHECKPOINT_DIR" | grep -c "qwen")" -gt 0 ]; then
#   export VLLM_ATTENTION_BACKEND=XFORMERS
# fi

# Make actor model path from checkpoint
if [ -n "$CHECKPOINT_STEP" ]; then
  ACTOR_MODEL="$CHECKPOINT_DIR/actor/global_step_$CHECKPOINT_STEP"
else
  # Try a simpler approach:
  LATEST_STEP=""
  for dir in "$CHECKPOINT_DIR"/actor/global_step_*; do
    if [ -d "$dir" ]; then
      LATEST_STEP="$dir"
    fi
  done
  if [[ -n "$LATEST_STEP" ]]; then
    ACTOR_MODEL="$LATEST_STEP"
    echo "Using latest checkpoint: $ACTOR_MODEL"
  else
    echo "No checkpoint found in $CHECKPOINT_DIR/actor/"
    exit 1
  fi
fi

# Verify the actor model path exists
if [[ ! -d "$ACTOR_MODEL" ]]; then
  echo "Error: Actor model directory not found: $ACTOR_MODEL"
  exit 1
fi

echo "Starting evaluation with the following parameters:"
echo "Actor model: $ACTOR_MODEL"
echo "Test file: $TEST_FILE"
echo "Number of examples: $NUM_EXAMPLES"
echo "Search URL: $SEARCH_URL"

# Run the Python script with the specified parameters
python eval/evaluate_ppo.py \
  --model_id "$ACTOR_MODEL" \
  --test_file "$TEST_FILE" \
  --search_url "$SEARCH_URL" \
  --num_examples "$NUM_EXAMPLES"

echo "Evaluation complete."