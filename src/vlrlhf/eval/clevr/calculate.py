import json
import os
import random
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import argparse
import re


def extract_number(text):
    # 检查输入是否为一个数字（0-10）
    if re.fullmatch(r'[0-9]|10', text.strip()):
       return text.strip()

    # 匹配 "Solution:\nxxxxx\nAnswer:\n{y}" 形式并提取最后的数字
    solution_answer_match = re.search(r'Answer:\s*(-?\d+)', text, re.DOTALL)
    if solution_answer_match:
        return solution_answer_match.group(1)

    # 如果不满足上述两种情况，输出提取失败
    return "unknown"


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
        prediction = extract_number(item["prediction"])
        if prediction == "unknown":
            print(item["prediction"])
        if str(answer) == str(prediction):
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
    parser.add_argument('--tab_test_file', type=str, default="/home/nfs04/chengkz/VL-RLHF/clevr_qwenvl_data/clevr_test.json")

    args = parser.parse_args()
    main(args.result_file, args.tab_test_file)