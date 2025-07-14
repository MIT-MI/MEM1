import pandas as pd
# set seed
import random
import numpy as np
random.seed(42)
np.random.seed(42)

pd.options.display.max_columns = 100
from Mem1.inference.models import LiteLLMClient, AMemClient, VLLMOpenAIClient
import argparse
import json
import numpy as np
from Mem1.inference.data_pipelines import Mem1Pipeline, model_estimated_match
from Mem1.rollout.env.webshop_manager import WebShopEnvManager
from tqdm import tqdm
import logging
import hashlib


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# turn off logger
logging.getLogger().setLevel(logging.WARNING)

########################################################
#  utils for reading the test data
########################################################
def read_nq_search_data(data_file):
    """
    Reads and returns the test data from the NQ search dataset.
    
    Returns:
        pd.DataFrame: A pandas DataFrame containing the test data
    """
    file_path = data_file
    df = pd.read_parquet(file_path)
    size = len(df)
    # we only want 1000 rows
    frac = 1000 / size
    df = df.sample(frac=frac, random_state=42)
    
    # add hash to df
    df['hash'] = df['prompt'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())

    return df

def read_webshop_data(data_file):
    file_path = data_file
    df = pd.read_parquet(file_path)
    env_manager = WebShopEnvManager()
    df['hash'] = df['prompt'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    return df, env_manager

# JSON serialization helper
def json_serialize_helper(obj):
    """Helper function to make objects JSON serializable"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM loop with OpenAI")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use: 'openai'")
    parser.add_argument("--use_amem", action="store_true", default=False,
                        help="Use Agentic Memory client")
    parser.add_argument("--use_mem1", action="store_true", default=False,
                        help="Use mem1 inference style")
    parser.add_argument("--use_litellm", action="store_true", default=False,
                        help="Use LiteLLM client")
    parser.add_argument("--resume_file", type=str, default=None,
                        help="Output file name")
    parser.add_argument("--data_file", type=str, default="Mem1/data/websearch_multi_3/test.parquet",
                        help="Data file to use")
    parser.add_argument("--task_type", type=str, default="rag", choices=["rag", "websearch", "webshop"], 
                        help="Task type")
    args = parser.parse_args()
    
    reconstruction_dicts = []

    if args.use_mem1:
        inference_type = "mem1"
    elif args.use_amem:
        inference_type = "amem"
    else:
        inference_type = "normal"

    if args.resume_file:
        file_path = args.resume_file
        print(f"Resuming from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                reconstruction_dicts.append(json.loads(line))
        print(f"Loaded {len(reconstruction_dicts)} reconstruction dicts from {file_path}")
    else:
        file_path = f'websearch_multi_3_train_reconstruction_dicts_{args.model.replace("/", "_")}_{inference_type}.jsonl'

    # Read the test data
    if args.task_type == "rag" or args.task_type == "websearch":
        train_data = read_nq_search_data(args.data_file)
    elif args.task_type == "webshop":
        train_data, env_manager = read_webshop_data(args.data_file)
    original_len = len(train_data)

    if len(reconstruction_dicts) > 0:
        all_hashes = set()
        for row in reconstruction_dicts:
            all_hashes.add(row['hash'])
        train_data = train_data[~train_data['hash'].isin(all_hashes)]
        print(f"Filtered {len(train_data)} rows from {original_len} rows")

    if args.use_mem1:
        assert not args.use_amem, "Cannot use Agentic memory while mem1 style inference is on"

    # Initialize the appropriate client
    if args.use_amem:
        llm_client = AMemClient()
    elif args.use_litellm:
        llm_client = LiteLLMClient()
    else:
        llm_client = VLLMOpenAIClient()


    # Run the LLM loop for each row in the test data
    reconstruction_dicts = []

    all_hashes = set()
    for row in reconstruction_dicts:
        all_hashes.add(row['hash'])

    import concurrent.futures
    import threading
    
    # Create a thread-safe list for results
    results_lock = threading.Lock()
    
    def process_row(func_args):
        index, row, client, model = func_args
        client.reset()
        try:
            # prompt = row['question']
            prompt = row["prompt"][0]["content"]
            pipeline = Mem1Pipeline(client, inference_type=inference_type)
            answer, results_dict = pipeline.run_llm_loop(prompt, model=model, use_mem1=args.use_mem1)

            print(f"Generated answer: {answer}, Golden answer: {row['reward_model']['ground_truth']}")

            if "multi" in args.data_file:
                answers = str(answer).split(";")
                ground_truths = row['reward_model']['ground_truth']['target']
                exact_match = 0
                for idx, gt in enumerate(ground_truths):

                    gt = gt[0]
                    try:
                        if str(answers[idx]).lower().strip() in str(gt).lower().strip() or str(gt).lower().strip() in str(answers[idx]).lower().strip():
                            exact_match += 1
                        else:
                            exact_match += 0
                    except Exception as e:
                        exact_match = 0
                        break
                if exact_match == len(ground_truths):
                    print(f"Test {index} passed")
                else:
                    print(f"Test {index} failed")
            else:
                if str(answer).lower().strip() in str(row['reward_model']['ground_truth']).lower().strip():
                    print(f"Test {index} passed")
                    exact_match = True
                else:
                    print(f"Test {index} failed")
                    exact_match = False
            results_dict["index"] = index
            results_dict["hash"] = row["hash"]
            if "multi" in args.data_file:
                results_dict['Golden_answer'] = row['reward_model']['ground_truth']['target']
            else:
                results_dict['Golden_answer'] = row['golden_answers']
            results_dict['Exact_match'] = exact_match

            if client.has_memory:
                results_dict["memories"] = client.memories
            
            try:
                if "multi" in args.data_file:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['reward_model']['ground_truth']['target'], prompt, client)
                else:
                    results_dict['Model_estimated_match'] = model_estimated_match(answer, row['golden_answers'], prompt, client)
            except Exception as e:
                logger.error(f"Error in model_estimated_match for index {index}: {str(e)}")
                results_dict['Model_estimated_match'] = False
            
            # Thread-safe append to the results list
            with results_lock:
                # add the entry
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(results_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            
            return index
        except Exception as e:
            logger.error(f"Error processing row at index {index}: {str(e)}", exc_info=True)
            # Add minimal error information to results
            result_dict = {
                    'index': index,
                    'hash': row["hash"],
                    'error': str(e),
                    'question': row.get('question', 'Unknown'),
                    'Golden_answer': row.get('golden_answers', 'Unknown'),
                    'Exact_match': False,
                    'Model_estimated_match': False
                }
            with results_lock:
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=None, ensure_ascii=False, default=json_serialize_helper)
                    f.write('\n')
            return index
    
    # Convert DataFrame to list of (index, row) tuples for parallel processing
    row_data = [(index, row, llm_client, args.model) for index, row in train_data.iterrows()]
    
    # Use ThreadPoolExecutor to process rows in parallel
    if args.use_amem:
        # we must run in a single thread
        # otherwise chromadb will clash
        for row in row_data:
            process_row(row)
    else:
        # otherwise we can use parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            list(tqdm(executor.map(process_row, row_data), total=len(row_data)))

