import json
import string
import re
import tiktoken
import argparse


def preprocess_text(text: str) -> str:
    text = text.lower()

    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    return text

def check_tags_balance(solution_str: str) -> bool:
    tags_to_check = ['tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        if start_count != end_count:
            return False
            
        last_pos = -1
        while True:
            start_pos = solution_str.find(start_tag, last_pos + 1)
            if start_pos == -1:
                break
                
            end_pos = solution_str.find(end_tag, start_pos)
            if end_pos == -1:
                return False
                
            last_pos = end_pos
            
    return True


def compute_peak_token_non_compression(data) -> float:
    if data["Exact_match"] == 0:
        return None
    solution_str = ""
    for i in range(10):
        if f"t{i}" in data:
            solution_str += data[f"t{i}"]
        if f"r{i}" in data:
            solution_str += data[f"r{i}"]
        if f"i{i}" in data:
            solution_str += data[f"i{i}"]
        if f"t{i}" not in data and f"r{i}" not in data and f"i{i}" not in data:
            break
    return len(encoding.encode(solution_str))

def compute_peak_token_compression(data) -> float:
    if data["Exact_match"] == 0:
        return None
    peak_count = 0
    solution_str = ""
    # prompt_len = len(encoding.encode(data["q"]))
    for i in range(10):
        new_solution_str = ""
        if f"t{i}" in data:
            solution_str += data[f"t{i}"]
            new_solution_str += data[f"t{i}"]
        if f"r{i}" in data:
            solution_str += data[f"r{i}"]
            new_solution_str += data[f"r{i}"]
        
        peak_count = max(peak_count, len(encoding.encode(solution_str)))
        solution_str = new_solution_str
        if f"i{i}" in data:
            solution_str += data[f"i{i}"]

    if peak_count == 0:
        return None

    return peak_count


def compute_dependency_compression(data) -> float:
    dependency_count = 0
    total_length = 0
    solution_str = ""
    for i in range(10):
        new_solution_str = ""
        if f"t{i}" in data:
            lt = len(encoding.encode(data[f"t{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + lt // 2) * lt
            total_length += lt
            solution_str += data[f"t{i}"]
            new_solution_str += data[f"t{i}"]
        if f"r{i}" in data:
            lr = len(encoding.encode(data[f"r{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + lr // 2) * lr
            total_length += lr
            solution_str += data[f"r{i}"]
            new_solution_str += data[f"r{i}"]
        solution_str = new_solution_str
        if f"i{i}" in data:
            li = len(encoding.encode(data[f"i{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + li // 2) * li
            total_length += li
            solution_str += data[f"i{i}"]
        if f"t{i}" not in data and f"r{i}" not in data and f"i{i}" not in data:
            break
    
    if total_length == 0:
        return None
    
    dependency_count = dependency_count / total_length
    return dependency_count


def compute_compression_ratio(data) -> float:
    total_length = 0
    solution_str = ""
    length = 0
    for i in range(10):
        cur_solution_str = ""
        if f"t{i}" in data:
            lt = len(encoding.encode(data[f"t{i}"]))
            total_length += lt
            cur_solution_str += data[f"t{i}"]
            solution_str += data[f"t{i}"]
        if f"r{i}" in data:
            lr = len(encoding.encode(data[f"r{i}"]))
            total_length += lr
            cur_solution_str += data[f"r{i}"]
            solution_str += data[f"r{i}"]
            length = len(encoding.encode(solution_str))
        solution_str = cur_solution_str
        if f"i{i}" in data:
            li = len(encoding.encode(data[f"i{i}"]))
            total_length += li
            cur_solution_str += data[f"i{i}"]
            solution_str += data[f"i{i}"]
        if f"t{i}" not in data and f"r{i}" not in data and f"i{i}" not in data:
            break
    
    if total_length == 0:
        return None
    
    compression_ratio = length / total_length
    return compression_ratio


def compute_dependency_non_compression(data) -> float:
    dependency_count = 0
    total_length = 0
    solution_str = ""
    for i in range(10):
        if f"t{i}" in data:
            lt = len(encoding.encode(data[f"t{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + lt // 2) * lt
            total_length += lt
            solution_str += data[f"t{i}"]
        if f"r{i}" in data:
            lr = len(encoding.encode(data[f"r{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + lr // 2) * lr
            total_length += lr
            solution_str += data[f"r{i}"]
        if f"i{i}" in data:
            li = len(encoding.encode(data[f"i{i}"]))
            dependency_count += (len(encoding.encode(solution_str)) + li // 2) * li
            total_length += li
            solution_str += data[f"i{i}"]
        if f"t{i}" not in data and f"r{i}" not in data and f"i{i}" not in data:
            break
    
    if total_length == 0:
        return None

    dependency_count = dependency_count / total_length
    return dependency_count


def compute_score(question,solution_str, ground_truths, val_type='f1',cot=False) -> float:
    solution_str = solution_str.lower()
    if cot == True:
        solution_str = solution_str + "</answer>"
    solution_str = solution_str.split("<|im_start|>assistant")[-1]

    if not check_tags_balance(solution_str):
        return -0.0
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
        else:
            return -0.0
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -0.0
    
    answers = answer_content.split(";")
    answers = [preprocess_text(answer) for answer in answers]

    max_score = 0.0

    if len(answers) != len(ground_truths):
        return max_score

    for idx, answer in enumerate(answers):
        cur_max_score = 0.0
        gts = ground_truths[idx]
        gts = [preprocess_text(gt) for gt in gts]
    
        if val_type == 'em' or val_type == "mbe":
            if answer in gts:
                cur_max_score = 1.0
        else:
            for gt in gts:
                pred_tokens = set(answer.split())
                gt_tokens = set(gt.split())
                
                if not gt_tokens:
                    continue
                if not pred_tokens:
                    continue
                
                common_tokens = pred_tokens & gt_tokens
                
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
                
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    cur_max_score = max(cur_max_score, f1)

        max_score += cur_max_score
        if val_type == "mbe":
            # max_score = get_mbe_result(question,ground_truths,answer_content)
            return 0

    return max_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file_path", required=True, help="Enter the path to your eval file contatining all trajectories.")

    args = parser.parse_args()

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    all_data = []

    with open(args.eval_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)

    all_scores_f1 = []
    all_scores_em = []
    all_scores_mbe = []
    all_compression_ratios = []
    all_peak_token_compressions = []
    all_dependency_compressions = []
    all_peak_token_non_compressions = []
    all_dependency_non_compressions = []

    for data in all_data:
        question = data["q"]
        solution_str = ""

        for i in range(10):
            if f"t{i}" in data:
                solution_str += data[f"t{i}"]
            if f"r{i}" in data:
                solution_str += data[f"r{i}"]
            if f"i{i}" in data:
                solution_str += data[f"i{i}"]
            if f"t{i}" not in data and f"r{i}" not in data and f"i{i}" not in data:
                break

        ground_truths = data["Golden_answer"]
        if isinstance(ground_truths[0], str):
            ground_truths = [ground_truths]
        score_f1 = compute_score(question,solution_str,ground_truths,val_type="f1")
        score_em = compute_score(question,solution_str,ground_truths,val_type="em")
        all_scores_f1.append(score_f1)
        all_scores_em.append(score_em)
        all_scores_mbe.append(data["Model_estimated_match"])
        compression_ratio = compute_compression_ratio(data)
        peak_token_compression = compute_peak_token_compression(data)
        dependency_compression = compute_dependency_compression(data)
        peak_token_non_compression = compute_peak_token_non_compression(data)
        dependency_non_compression = compute_dependency_non_compression(data)
        all_compression_ratios.append(compression_ratio)
        all_peak_token_compressions.append(peak_token_compression)
        all_dependency_compressions.append(dependency_compression)
        all_peak_token_non_compressions.append(peak_token_non_compression)
        all_dependency_non_compressions.append(dependency_non_compression)

    # filter out None
    all_compression_ratios = [x for x in all_compression_ratios if x is not None]
    all_peak_token_compressions = [x for x in all_peak_token_compressions if x is not None]
    all_dependency_compressions = [x for x in all_dependency_compressions if x is not None]
    all_peak_token_non_compressions = [x for x in all_peak_token_non_compressions if x is not None]
    all_dependency_non_compressions = [x for x in all_dependency_non_compressions if x is not None]

    print(f"F1: {sum(all_scores_f1) / len(all_scores_f1)}")
    print(f"EM: {sum(all_scores_em) / len(all_scores_em)}")
    print(f"MBE: {sum(all_scores_mbe) / len(all_scores_mbe)}")
    print(f"Compression Ratio: {sum(all_compression_ratios) / len(all_compression_ratios)}")
    print(f"Peak Token Compression: {sum(all_peak_token_compressions) / len(all_peak_token_compressions)}")
    print(f"Dependency Compression: {sum(all_dependency_compressions) / len(all_dependency_compressions)}")
    print(f"Peak Token Non-Compression: {sum(all_peak_token_non_compressions) / len(all_peak_token_non_compressions)}")
    print(f"Dependency Non-Compression: {sum(all_dependency_non_compressions) / len(all_dependency_non_compressions)}")
