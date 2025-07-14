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

# from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    questions = dp['questions']  # Now expects a list of questions

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""You will answer multiple complex questions through iterative reasoning, summarization, and web searches.

# At each step, you can see the questions, previous summary of related information, search query, and the returned information (except the first step where you will be given only the questions). Then, you should:

# 1. Conduct reasoning, and then update a concise, cumulative summary with essential information inside <think> </think>. This is your persistent memory and should include all important information from previous <think> </think> and <information> </information> (i.e. information and answers already found for questions).
# 2. Then choose one:
#    - Issue a query (i.e., key words / phrases for search) inside <search> </search> (you may search repeatedly until the answer is clear). This query will be used to conduct search and return the results in <information> results </information>
#    - Provide the final concise answers to each question separated by semicolons in the format of <answer> answer1; answer2; ... </answer> (no explanations) if no additional information is needed inside <answer> </answer>. The answer should be concise and only contain the words necessary to answer the questions.

# After <information> </information> (or questions at the beginning), you should always follow the order: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
   
# Answer all of the following two questions: {questions}\n"""
        prefix = f"""You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--batch_size', type=int, default=3, help='Number of questions to process together')

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:
        if data_source != 'strategyqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="/home/peterjin/mnt/data/strategyqa/train_correct.jsonl")

        train_dataset = dataset['train']

        def process_batch(examples, indices):
            # Process questions in batches of k
            processed_data = []
            
            # Format questions with semicolons
            questions = []
            golden_answers = []
            
            questions = examples['question']
            new_questions = []
            for question in questions:
                if question[-1] != '?':
                    question += '?'
                new_questions.append(question)
            questions = new_questions

            golden_answers = examples['golden_answers']
            
            questions_str = '; '.join(questions)
            
            # Create the prompt with multiple questions
            prompt_data = {
                'questions': questions_str
            }
            question = make_prefix(prompt_data, template_type=args.template_type)
            
            solution = {
                "target": golden_answers
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
                    'split': 'train',
                    'indices': indices,
                }
            }

            processed_data.append(data)
            
            return processed_data

        # Process the dataset in batches
        processed_data = []
        if len(train_dataset) % args.batch_size != 0:
            train_dataset = train_dataset.select(range(len(train_dataset) - (len(train_dataset) % args.batch_size)))
        print("length of train_dataset: ", len(train_dataset))

        for i in range(0, len(train_dataset), args.batch_size):
            batch = train_dataset[i:i + args.batch_size]
            batch_indices = list(range(i, i + args.batch_size))
            processed_data.extend(process_batch(batch, batch_indices))
        
        # Convert processed data to dataset
        train_dataset = datasets.Dataset.from_list(processed_data)
        all_dataset.append(train_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.concatenate_datasets(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)
