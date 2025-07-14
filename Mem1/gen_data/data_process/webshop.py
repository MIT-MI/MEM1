# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the webshop data to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rollout.env.webshop.env import WebShopEnv

env_instruction = """
You are browsing an online shop. Your goal is to find a product that matches the given description. You will interact with the site step-by-step. Each step gives you a <state>...</state> representing the current webpage. You must decide what action to take next until you identify the correct product.

Available actions (shown in the <state> tag) depend on the page:
- On the search page: search[<keywords>]
- On search result pages: click[<item url>] to view a product, or click[next >] to go to the next results page
- On product pages: click[description], click[features], click[color], click[size], click[buy now]
- To return to search: click[back to search]

Example goal: "Find a gingko light and 20x20 pillow cover that is hand painted." 
Example first action: <answer>search[gingko light 20x20 pillow cover hand painted]</answer>
Only respond with valid actions formatted as: search[...], click[...], etc.
After you navigate and find the product that best fits the user goal, you should click[buy now] to buy the product at the product page when the buy now button is available.
"""

# response_instruction = """
# At every step, you must iteratively update the <think>...</think> section, which serves as your only memory throughout the task. It must contain a self-contained, growing summary of all useful information—*including user goals, actions taken, search results, product details, and reasoning about relevance*. You must overwrite and improve the previous <think> block with the new one in every step.

# CRITICAL: You cannot rely on any information other than what is inside the current <state> and your previous <think>. Each new <think> must integrate the most useful content from both sources to help guide the next action without remembering earlier steps.

# Your output must always follow this strict format:
# <think>...</think>
# <answer>...</answer>

# Guidelines for <think>:
# - It must accumulate useful observations, decisions, and insights from all previous and current steps.
# - Remove redundant or outdated details. Keep only what helps you move closer to finding and buying the correct product.
# - It should read like a growing internal monologue or scratchpad used to solve the task.

# Guidelines for <answer>:
# - Choose the best next action from the Available Actions listed in the current <state>.
# - Valid formats: search[<keywords>] or click[<option>] where <option> is an exact clickable item from the state.

# The task begins with the following user instruction:
# {user_instruction}
# """

response_instruction = """
At each step, you will receive the current webpage state within <state></state>, the previous cumulative memory within <summary></summary> (except for the first step), and your last action within <answer></answer> (except for the first step). NEVER generate or modify the <state>; use it exactly as provided.

Respond strictly using the following format:
<think>...</think>
<summary>...</summary>
<answer>...</answer>

Guidelines:
- <think>: Clearly reason about relevant details and briefly plan your next step.
- <summary>: Maintain and update a concise, cumulative memory of all essential information by integrating current <state> with previous <summary>. This is your only persistent memory. It should serve as a growing internal monologue that accumulates useful observations, actions taken, and insights from all previous and current steps, while removing redundant or outdated details to stay focused on finding and buying the correct product.
- <answer>: Select your next action strictly from the Available Actions explicitly listed within the provided <state>. Use ONLY valid action formats like search[<keywords>] or click[<option>]. Click buy now when the right product is found and the buy now button is available.

Task begins with:
{user_instruction}
"""


def make_prefix(observation):
    prefix = env_instruction + "\n" + response_instruction.format(user_instruction=observation)
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/webshop')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    FILE_PATH = "file path to the webshop data"
    env = WebShopEnv()
    
    num_envs = len(env.server.goals)
    test_envs = []
    train_envs = []
    for idx in tqdm(range(num_envs)):
        observation = env.reset(session=idx)
        state = env.render_cache
        if idx < 500:

            test_envs.append({"observation": state, "index": idx})
        else:
            train_envs.append({"observation": state, "index": idx})
    # Convert lists to HuggingFace Datasets
    train_dataset = datasets.Dataset.from_list(train_envs)
    test_dataset = datasets.Dataset.from_list(test_envs)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            observation = example['observation']
            environment_id = example['index']
            prompt = make_prefix(observation)
            data = {
                "data_source": "webshop",
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "webagent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None
                },
                "extra_info": {
                    'split': split,
                },
                "environment_ids": environment_id,
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
