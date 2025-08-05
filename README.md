# Mem1: Train your LLMs to reason and consolidate memory at the same time!
> This repo is the official code implementation of **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents** <br>
> [[arXiv]](https://arxiv.org/abs/2506.15841) | [[Project Site]](https://mit-mi.github.io/mem1-site/) <br>
> **Zijian Zhou^, Ao Qu^, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, and Bryan Kian Hsiang Low, Paul Pu Liang**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
![GitHub Repo stars](https://img.shields.io/github/stars/MIT-MI/MEM1)

⭐️ Please star this repository if you find it helpful!

---

> **Abstract**: Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized.

## Demo Video

[![Demo Video](https://github.com/user-attachments/assets/8e6bdedf-f4ff-40ff-9e08-ae41248fca57)](https://github.com/user-attachments/assets/8e6bdedf-f4ff-40ff-9e08-ae41248fca57)

## News
- (Aug 2025) MEM1 has been accepted at COLM 2025 Workshop RAM2 for **oral** presentation!

## Installation

### Mem1 environment
```bash
# Install requirements. This may take a few minutes to complete.
conda env create -f environment.yml

# verl
conda activate mem1
pip install -e .
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
cd setup/
python download.py
```

2. Pre-process the multi-objective QA dataset (change the batch size for varying number of objectives)
```bash
cd Mem1/
bash gen_data/scripts/data_process_multi.sh --batch_size 2
```

3. Launch a local retrieval server:
```bash
cd Mem1/train/
bash retrieval_launch.sh
```

4. Train the Mem1 model:
```bash
cd Mem1/train/
bash train_ppo.sh
```

## Evaluation
You can evaluate the model using:
```bash
bash run_eval.sh
```

## Trained Model Details

- wandb: https://api.wandb.ai/links/Mem1/vl5osiui
- 🤗 HF Checkpoint: https://huggingface.co/Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release

## Reference

Please use the following bibtex citation format for your reference.

```
@misc{2025mem1learningsynergizememory,
  title        = {MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents},
  author       = {Zhou, Zijian and Qu, Ao and Wu, Zhaoxuan and Kim, Sunghwan and Prakash, Alok and Rus, Daniela and Zhao, Jinhua and Low, Bryan Kian Hsiang and Liang, Paul Pu},
  year         = {2025},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2506.15841},
}
```

## Acknowledgement

The codebase has referred to the following repositories:

- [veRL](https://github.com/volcengine/verl/tree/main)
- [search-R1](https://github.com/PeterGriffinJin/Search-R1/tree/main)
- [AgenticMemory](https://github.com/WujiangXu/AgenticMemory)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MIT-MI/MEM1&type=Date)](https://www.star-history.com/#MIT-MI/MEM1&Date)
