import json
import os
import random
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import argparse
import re

"""
def extract_answer(response):
    # 定义正则表达式，使用捕获组来提取 Answer 后的所有内容
    pattern = r"^Solution:\s*.*\s*Answer:\s*(.+)\s*$"

    # 使用正则表达式匹配输入字符串
    match = re.match(pattern, response, re.DOTALL)

    # 如果匹配成功，返回提取的Answer:之后的内容
    if match:
        answer = match.group(1).strip()  # 提取捕获的内容并去除首尾空格
        return answer
    else:   # 如果匹配不成功，说明是直接回答，直接返回答案
        return response.strip()
"""
def extract_answer(response):
    # 定义正则表达式，只匹配 "Answer:" 之后的内容
    pattern = r"Answer:\s*(.+)\s*$"

    # 使用正则表达式匹配输入字符串
    match = re.search(pattern, response, re.DOTALL)

    # 如果匹配成功，返回提取的 Answer: 之后的内容
    if match:
        answer = match.group(1).strip()  # 提取捕获的内容并去除首尾空格
        return answer
    else:   # 如果匹配不成功，直接返回原始响应
        return response.strip()


def is_ans_correct(answer, label):
    # 尝试将 answer 和 label 转换为浮点数
    try:
        answer_float = float(answer.strip())
        label_float = float(label.strip())

        # 如果转换成功，判断 answer 是否在 label 的正负 5% 范围内
        if abs(answer_float - label_float) <= 0.05 * abs(label_float):
            return True
    except ValueError:
        # 如果不能转换为浮点数，继续下面的逻辑
        pass

    # 如果不是数值，按照字符串严格匹配
    if answer.lower().strip() == label.lower().strip():
        return True
    else:
        return False


def main(result_file_path, tab_test_path):
    """读取模型生成的结果文件，整理为CLEVR评测所需的格式并计算各类指标"""
    # 检查文件是否存在
    if not os.path.exists(result_file_path):
        raise FileNotFoundError(f"Result file not found at path: {result_file_path}")

    # 读取结果文件
    with open(result_file_path, 'r') as file:
        results_gen = json.load(file)
    print(f"cal {result_file_path}")

    # 加载测试集
    tab_test = json.load(open(tab_test_path, 'r'))

    total_num = 0
    correct_num = 0
    for item in tqdm(results_gen):
        total_num += 1
        answer = item["answer"]
        prediction = extract_answer(item["prediction"])

        if is_ans_correct(prediction, answer):
            item["correct"] = True
            correct_num += 1
        else:
            item["correct"] = False

    print("Acc: {}".format(correct_num/total_num))

    json.dump(results_gen, open(result_file_path, 'w'))
    print(f"Total: {total_num}, Correct: {correct_num}, Acc: {correct_num/total_num}, saved to {result_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model results")
    parser.add_argument('--result_file', type=str, required=True, help='Path to the result file')
    parser.add_argument('--tab_test_file', type=str, default="/home/nfs04/chengkz/VL-RLHF/chartqa_qwenvl_data/chartqa_test.json")

    args = parser.parse_args()
    main(args.result_file, args.tab_test_file)