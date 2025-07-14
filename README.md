# Mem1: Train your LLMs to reason and call a search engine with reinforcement learning


## Installation

### Mem1 environment
```bash
conda create -n mem1 python=3.9
conda activate mem1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# Install requirements
pip install -r requirements.txt

# verl
pip install -e Mem1/

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a separate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

## Quick Start

1. Download the necessary data:
```bash
python download.py
```

2. Launch a local retrieval server:
```bash
bash Mem1/retrieval_launch.sh
```

3. Train the Mem1 model:
```bash
bash Mem1/train_ppo.sh
```

## Evaluation
You can evaluate the model using:
```bash
bash run_eval.sh
```

## Acknowledgement

The codebase is built upon
[veRL](https://github.com/volcengine/verl/tree/main), [search-R1](https://github.com/PeterGriffinJin/Search-R1/tree/main), [AgenticMemory](https://github.com/WujiangXu/AgenticMemory)