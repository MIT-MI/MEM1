export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WAND_PROJECT='MEM1'

# nq_hotpotqa
export DATA_DIR='/home/project/11714281/zijian/tmp/MEM1/Mem1/data/deepdive'
# export BASE_MODEL="Qwen/Qwen2.5-7B"
export BASE_MODEL="Mem-Lab/DeepDive-SFT"
export EXPERIMENT_NAME=DeepDive-MEM1-SFT-MAXTURNS-10
export PROGRAM_ENTRY=verl.trainer.main_ppo
export MAX_TURNS=10

ACTOR_MODEL=$BASE_MODEL
CRITIC_MODEL=$BASE_MODEL

# webshops
# export DATA_DIR='data/webshop'
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=webshop-search-r1-ppo-qwen2.5-7b-it-em
# export PROGRAM_ENTRY=verl.trainer.main_ppo_webshop
# export MAX_TURNS=10

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# ray
export RAY_TMPDIR=/tmp/ray-$USER
mkdir -p $RAY_TMPDIR
# ray start --head
export RAY_memory_usage_threshold=0.9

PYTHONUNBUFFERED=1 python3 -m $PROGRAM_ENTRY \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1000 \
    data.max_start_length=2048 \
    data.max_obs_length=1000 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$ACTOR_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$CRITIC_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=150 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME-resume50 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME-resume50 \
    max_turns=$MAX_TURNS \
    retriever.url="http://127.0.0.1:8013/retrieve" \
    retriever.topk=3 \
