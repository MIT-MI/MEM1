import json
import argparse
import os
import random
import time
from tqdm import tqdm
import re
import pandas as pd
import string
import sys

from openai.types import Completion as OpenAICompletion
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from openai import Timeout as OpenAITimeout

import requests

def call_gpt_4o_mini(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": os.environ["OPENAI_API_KEY"],
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"


def check_tags_balance(solution_str: str) -> bool:
    """检查标签是否正确配对
    
    Args:
        solution_str: 需要检查的字符串
    
    Returns:
        bool: 标签是否都正确配对
    """
    # 需要检查的标签对
    tags_to_check = ['tool_call', 'think', 'answer']
    
    for tag in tags_to_check:
        # 计算开始和结束标签的数量
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        
        start_count = solution_str.count(start_tag)
        end_count = solution_str.count(end_tag)
        
        # 如果开始和结束标签数量不相等，返回False
        if start_count != end_count:
            return False
            
        # 检查标签的嵌套顺序（确保结束标签不会在开始标签之前出现）
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

def preprocess_text(text: str) -> str:
    """预处理文本，用于NQ数据集的评分
    
    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    return text

PROMPT='''You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list.
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:

The output should in the following json format:
\'\'\'json
{
\"rationale\": \"your rationale for the judgement, as a text\",
\"judgement\": \"your judgement result, can only be \'correct\' or \'incorrect\'\"
}
\'\'\'
Your output:
'''

def get_json(json_str):
    import json
    import re

    # 使用正则提取花括号中的 JSON 部分
    try:
        match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            return data
        else:
            return {}
    except:
        return {}

def get_mbe_result(question,gts,pred_answer):
    judgement = ""
    try_cnt = 0
    while True:
        prompt = PROMPT.replace("{question}",question).replace("{gt_answer}",str(gts)).replace("{pred_answer}",pred_answer)
        try:
            batch_responses = call_gpt_4o_mini(prompt)
            judgement = get_json(batch_responses)
            print(judgement)
            if "judgement" in judgement:
                judgement = judgement["judgement"]
            if judgement in ["correct", "incorrect"]:
                if judgement == "correct":
                    return 1.0
                else:
                    return 0.0
        except:
            try_cnt += 1
            if try_cnt > 100:
                return 0.0


def compute_score(question,solution_str, ground_truths, val_type='f1',cot=False) -> float:
    solution_str = solution_str.lower()
    # 首先检查标签是否配对正确(格式是否正确)
    if cot == True:
        solution_str = solution_str + "</answer>"
    solution_str = solution_str.split("<|im_start|>assistant")[-1]
    if not check_tags_balance(solution_str):
        return -0.0
    # 使用正则提取第一个<answer>标签中的内容
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # 对答案进行预处理
            answer_content = preprocess_text(answer_content)
        else:
            return -0.0  # 如果没有answer标签，返回-1.0表示格式错误
    except Exception as e:
        print(f"Error extracting answer content: {e}")
        return -0.0
    
    max_score = 0.0
    
    for gt in ground_truths:
        # 对ground truth进行预处理
        gt = preprocess_text(gt)
        
        if val_type == 'em' or val_type == "mbe":
            if gt == answer_content:
                return 1.0
        else:
            # 将答案和参考答案分词
            pred_tokens = set(answer_content.split())
            gt_tokens = set(gt.split())
            
            if not gt_tokens:  # 避免除零错误
                continue
            if not pred_tokens:
                continue
            
            # 计算共同的词数
            common_tokens = pred_tokens & gt_tokens
            
            # 计算精确率和召回率
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
            
            # 计算F1分数
            if precision + recall > 0:  # 避免除零错误
                f1 = 2 * (precision * recall) / (precision + recall)
                max_score = max(max_score, f1)
    if val_type == "mbe":
        max_score = get_mbe_result(question,ground_truths,answer_content)
    
    
    return max_score


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    return compute_score("", solution_str, ground_truth["target"], "f1", cot=False)


# method = sys.argv[1]
# file_path = "../data/test.parquet"
# df = pd.read_parquet(file_path)
# gts = json.loads(df.to_json(orient="records", force_ascii=False))


# with open(f"./{method}_result.json","r",encoding="utf-8") as f:
#     answers = json.load(f)
# result = {}
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# from collections import defaultdict

# result = defaultdict(lambda: {"f1": [], "em": [], "mbe": []})

# def compute_metrics(gt, answer, method):
#     question = gt["prompt"][0]["content"]
#     gt_answer = gt["reward_model"]["ground_truth"]
#     data_source = gt["data_source"]
#     mbe = 0.0
#     if method in ["rag", "cot"]:
#         f1 = compute_score(question, answer["response"], gt_answer, "f1", cot=True)
#         em = compute_score(question, answer["response"], gt_answer, "em", cot=True)
#         mbe = compute_score(question, answer["response"], gt_answer, "mbe", cot=True)
#     elif method in ["search_r1_wo_ir","search_r1"]:
#         data_source = answer["data_source"]
#         question =  answer["question"]
#         gt_answer = answer["gt_answer"]
#         f1 = compute_score(question, answer["r1_answer"], gt_answer, "f1", cot=False)
#         em = compute_score(question, answer["r1_answer"], gt_answer, "em", cot=False)
#         mbe = compute_score(question, answer["r1_answer"], gt_answer, "mbe", cot=False)
#     elif method in ["r1_searcher"]: 
#         data_source = answer["data_source"]
#         question =  answer["question"]
#         gt_answer = answer["answer"]
#         an = f"<answer>{answer['pred_ans']}</answer>"
#         f1 = compute_score(question, an, gt_answer, "f1", cot=False)
#         em = compute_score(question, an, gt_answer, "em", cot=False)
#         mbe = compute_score(question, an, gt_answer, "mbe", cot=False)
#     else:
#         f1 = compute_score(question, answer["message_str"], gt_answer, "f1", cot=False)
#         em = compute_score(question, answer["message_str"], gt_answer, "em", cot=False)
#         mbe = compute_score(question, answer["message_str"], gt_answer, "mbe", cot=False)
   
#     return data_source, f1, em, mbe

# with ThreadPoolExecutor(max_workers=16) as executor:
#     futures = [executor.submit(compute_metrics, gt, answer, method) for gt, answer in zip(gts, answers)]
    
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         data_source, f1, em, mbe = future.result()
#         result[data_source]["f1"].append(f1)
#         result[data_source]["em"].append(em)
#         result[data_source]["mbe"].append(mbe)

# # 平均分计算
# for data_source in result:
#     result[data_source]["f1"] = sum(result[data_source]["f1"]) / len(result[data_source]["f1"])
#     result[data_source]["em"] = sum(result[data_source]["em"]) / len(result[data_source]["em"])
#     result[data_source]["mbe"] = sum(result[data_source]["mbe"]) / len(result[data_source]["mbe"])

# with open(f"./{method}_score.json","w" ,encoding="utf-8") as f:
#     answers = json.dump(result,f,indent=4)
