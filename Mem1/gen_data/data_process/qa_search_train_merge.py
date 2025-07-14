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
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""You will answer complex questions through iterative summary and web search.

# Your response must include:

# <think>
# - Keep information from the current information that is potentially relevant and useful for answering the question.
# - The current information will be discarded in the next step and the think part will be the only information you have to complete the task.
# - You should also summarize previous searches you have made to avoid repetitive searches.
# - You will be told how many turns you have left inside the information given to you after you have made a search. You should keep track of the number of turns you have left.
# </think>

# Then either:
# <search>
# QUERY (only if you have turns left)
# </search>

# Or:
# <answer>
# FINAL ANSWER ONLY (no explanation, if you have gained all the information you need or you have no turns left)
# </answer>

# Follow this format strictly for your response so that it's either <think>...</think><search>...</search> or <think>...</think><answer>...</answer>.

# Question: {question}\n"""
        prefix = f"""You will answer a complex question through iterative reasoning, summarization, and web searches.

At each step, you can see the question, previous summary in <think> ... </think>, search query in <search> ... </search>, and the returned information in <information> ... </information> (except the first step where you will be given only the question). Then, you should:

1. Conduct reasoning, and then update a concise, cumulative summary with essential information inside <think> </think>. This is your persistent memory and should include all important information from previous <think> </think> and <information> </information> (i.e. information and answers already found for questions).
2. Then choose one:
   - Issue a query (i.e., key words / phrases for search) inside <search> </search> (you may search repeatedly until the answer is clear). This query will be used to conduct search and return the results in <information> results </information>
   - Provide the final concise answer (no explanations) if no additional information is needed inside <answer> </answer>. The answer should be concise and only contain the words necessary to answer the question.

After <information> </information> (or question at the beginning), you should always follow the order: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
   
Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    # data_source = 'nq'
    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)

        train_dataset = dataset['train']

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
