import pandas as pd
import argparse
import os
import datasets


def make_prefix(questions, template_type='base'):
    if template_type == 'base':
        prefix = f"""You will answer multiple complex questions through iterative reasoning, summarization, and web searches.

At each step, you can see the questions, previous summary of related information, search query, and the returned information (except the first step where you will be given only the questions). Then, you should:

1. Conduct reasoning, and then update a concise, cumulative summary with essential information inside <think> </think>. This is your persistent memory and should include all important information from previous <think> </think> and <information> </information> (i.e. information and answers already found for questions).
2. Then choose one:
   - Issue a query (i.e., key words / phrases for search) inside <search> </search> (you may search repeatedly until the answer is clear). This query will be used to conduct search and return the results in <information> results </information>
   - Provide the final concise answers to each question separated by semicolons in the format of <answer> answer1; answer2; ... </answer> (no explanations) if no additional information is needed inside <answer> </answer>. The answer should be concise and only contain the words necessary to answer the questions.

After <information> </information> (or questions at the beginning), you should always follow the order: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
   
Answer all of the following two questions: {questions}\n"""
    else:
        raise NotImplementedError
    return prefix


def process_websearch_data(data_path_dir, batch_size=2, template_type='base'):
    # load parquet file
    train_file_name = f"{data_path_dir}/orig_train.parquet"
    test_file_name = f"{data_path_dir}/orig_test.parquet"
    train_data = pd.read_parquet(train_file_name)
    test_data = pd.read_parquet(test_file_name)

    def process_batch(data_batch, indices):
        processed_data = []
        
        # Format questions with semicolons
        questions = data_batch["prompt"].apply(lambda x: x[0]["content"]).tolist()
        new_questions = []
        for question in questions:
            if question[-1] != '?':
                question += '?'
            new_questions.append(question)
        questions = new_questions
        questions_str = '; '.join(questions)
        
        # Get golden answers
        golden_answers = data_batch["reward_model"].apply(lambda x: x["ground_truth"].split("<|answer_split|>")).tolist()
        
        # Create the prompt with multiple questions
        prompt_data = {
            'questions': questions_str
        }
        question = make_prefix(prompt_data['questions'], template_type=template_type)
        
        solution = {
            "target": golden_answers
        }

        data = {
            "data_source": "websearch",
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "websearch",
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
    processed_train_data = []
    for i in range(0, len(train_data), batch_size):
        batch = train_data.iloc[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(train_data))))
        processed_train_data.extend(process_batch(batch, batch_indices))

    processed_test_data = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data.iloc[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(test_data))))
        processed_test_data.extend(process_batch(batch, batch_indices))

    # Convert processed data to dataset
    train_dataset = datasets.Dataset.from_list(processed_train_data)
    test_dataset = datasets.Dataset.from_list(processed_test_data)

    # save to parquet file
    train_dataset.to_parquet(f"{data_path_dir}/train.parquet")
    test_dataset.to_parquet(f"{data_path_dir}/test.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_dir', default='./data/websearch_multi')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of questions to process together')
    parser.add_argument('--template_type', type=str, default='base')
    args = parser.parse_args()

    process_websearch_data(args.data_path_dir, args.batch_size, args.template_type)