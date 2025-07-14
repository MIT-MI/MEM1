import pandas as pd


def process_websearch_data(data_path_dir):
    # load parquet file
    train_file_name = f"{data_path_dir}/orig_train.parquet"
    test_file_name = f"{data_path_dir}/orig_test.parquet"
    train_data = pd.read_parquet(train_file_name)
    test_data = pd.read_parquet(test_file_name)

    # ['id', 'question', 'golden_answers', 'data_source', 'prompt', 'ability',
    #    'reward_model', 'extra_info']
    # reward_model: {'ground_truth': '20 July 1851', 'style': 'unknown'} => {'ground_truth': {'target': array(['Wilhelm Conrad Röntgen'], dtype=object)}, 'style': 'rule'}

    def map_data(data):
        ground_truth = data["reward_model"]["ground_truth"]
        golden_answers = ground_truth.split("<|answer_split|>")
        data["data_source"] = "websearch"
        data["golden_answers"] = golden_answers
        data["question"] = data["prompt"][0]["content"]
        data["prompt"] = [{"role": "user", "content": make_prefix(data["question"])}]
        data["ability"] = "websearch"
        data["reward_model"] = {"ground_truth": {"target": golden_answers}, "style": "rule"}
        return data

    def make_prefix(question):
        prefix = f"""You will answer a complex question through iterative reasoning, summarization, and web searches.

At each step, you can see the question, previous summary, search query, and the returned information (except the first step where you will be given only the question). Then, you should:

1. Conduct reasoning, and then update a concise, cumulative summary with essential information inside <think> </think>. This is your persistent memory and should include all important information from previous <think> </think> and <information> </information>.
2. Then choose one:
   - Issue a query (i.e., key words / phrases for search) inside <search> </search> (you may search repeatedly until the answer is clear). This query will be used to conduct search and return the results in <information> results </information>
   - Provide the final concise answer (no explanations) if no additional information is needed inside <answer> </answer>. The answer should be concise and only contain the words necessary to answer the question.

After <information> </information> (or question at the beginning), you should always follow the order: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
   
Question: {question}\n"""  
        return prefix

    train_data = train_data.apply(map_data, axis=1)
    test_data = test_data.apply(map_data, axis=1)

    # save to parquet file
    train_data.to_parquet(f"{data_path_dir}/train.parquet")
    test_data.to_parquet(f"{data_path_dir}/test.parquet")


if __name__ == "__main__":
    data = process_websearch_data("./data/websearch")