# 计算TabMWP的评测指标，主要是按照官方的实现
# 在正则表达式匹配中增加了r'Answer: ([\s\S]+)'以适配我们的prompt
import json
import os
import random
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import argparse


def print_scores(scores):
    latex_output = ""
    print("")
    for key, score in scores.items():
        print(f"{key}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print("")
    print(latex_output)


def get_acc_with_condition(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_files, data_file):

    if not isinstance(result_files, list):
        result_files = [result_files]

    res_pds = []
    for result_file in result_files:
        # read result file
        results = json.load(open(result_file, 'r'))
        test_pids = list(results.keys())

        # read data file
        prob_data = json.load(open(data_file))

        # construct pandas data
        prob_data = pd.DataFrame(prob_data).T
        res_pd = prob_data[prob_data.index.isin(test_pids)]  # test set

        # update data
        for index, row in res_pd.iterrows():
            res_pd.loc[index, 'true_false'] = results[index]['true_false']

        # append result pd
        res_pds.append(res_pd)

    # merge all result pds
    res_pd = pd.concat(res_pds)
    num = len(res_pd)
    #assert num == 7686
    print("number of questions:", num)

    # accuracy scores
    acc_average = round(len(res_pd[res_pd['true_false'] == True]) / num * 100, 3)
    #assert acc_average == round(result_data["acc"], 3)

    scores = {
        'acc_free': get_acc_with_condition(res_pd, 'ques_type', 'free_text'),
        'acc_mc': get_acc_with_condition(res_pd, 'ques_type', 'multi_choice'),
        'acc_integer': get_acc_with_condition(res_pd, 'ans_type', 'integer_number'),
        'acc_decimal': get_acc_with_condition(res_pd, 'ans_type', 'decimal_number'),
        'acc_extractive': get_acc_with_condition(res_pd, 'ans_type', 'extractive_text'),
        'acc_boolean': get_acc_with_condition(res_pd, 'ans_type', 'boolean_text'),
        'acc_other': get_acc_with_condition(res_pd, 'ans_type', 'other_text'),
        'acc_grade_1_6': get_acc_with_condition(res_pd, 'grade', [1, 2, 3, 4, 5, 6]),
        'acc_grade_7_8': get_acc_with_condition(res_pd, 'grade', [7, 8]),
        'acc_average': "{:.2f}".format(acc_average),
    }

    return scores


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)

    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)

        if result is not None:
            number = int(text)
        elif "/" in text:
            nums = text.split("/")
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r"\.[0]+$", "", number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output, options, option_inds):
    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)

    output = re.sub(r"(?<![AP]\.M)\.$", "", output)
    output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
    output = re.sub(r"\u2212", "-", output)

    ## Multi-choice questions
    if options:
        patterns = [r'Answer: ([A-Za-z])',  # "Answer: B"
                    r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
                    r'^([A-Za-z])$',  # "b", "B"
                    r'^([A-Za-z]). ',  # "b", "B"
                    r'[Th]he answer is ([A-Z])',  # "The answer is B"
                    r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
                    r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$',  # "The answer is (B) XXXXX."
                    ]

        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper()  # e.g., "B"
                if pred in option_inds:
                    ind = option_inds.index(pred)  # 1
                    if ind >= len(options):
                        ind = random.choice(range(len(options)))
                    prediction = options[ind]
                    return prediction

        # find the most similar options
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction


    else:
        ## free_text QA problems, numeric answer
        patterns = [
            # r'^\([A-Za-z]\) ([\s\S]+)$', # "(A) XXXXX"
            # r'[Th]he answer is \([A-Za-z]\) ([\s\S]+)$', # "The answer is (B) XXXXX."
            # r'Answer: ([^\n]+?)(?=\.\D|\n|\(|\)|\bor\b| |$)'
            r'Answer: ([\s\S]+)',  # "Answer: XXXXX"
            r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
            r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
            r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
            r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
            r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                if prediction.endswith(".") and ".M." not in prediction:
                    prediction = prediction[:-1]
                return prediction

    return output


def main(result_file_path, tab_test_path):
    """读取模型生成的结果文件，整理为TabMWP评测所需的格式并计算各类指标"""
    # 检查文件是否存在
    if not os.path.exists(result_file_path):
        raise FileNotFoundError(f"Result file not found at path: {result_file_path}")

    # 读取结果文件
    with open(result_file_path, 'r') as file:
        results_gen = json.load(file)

    result_name = result_file_path.split('/')[-1][:-5]

    #print(len(results_gen))
    #results_gen = results_gen[4000:5000]

    print(f"cal {result_file_path}")

    # 加载测试集
    tab_test = json.load(open(tab_test_path, 'r'))

    results = {}
    correct = 0
    total = len(results_gen)
    option_inds = ["A", "B", "C", "D", "E", "F"]
    print("Total Evaluation: "+str(total))
    for i, item in enumerate(results_gen):
        test_id = item["index"]
        output = item["prediction"]
        unit = tab_test[test_id]["unit"]
        answer = item["answer"]
        answer_norm = normalize_answer(answer, unit)
        prediction = extract_prediction(output, tab_test[test_id]['choices'], option_inds)
        try:
            prediction_norm = normalize_answer(prediction, unit)
        except:
            prediction_norm = "0"

        # 保存
        results[test_id] = {}
        results[test_id]["answer"] = answer
        results[test_id]["answer_norm"] = answer_norm
        results[test_id]["output"] = output
        results[test_id]["prediction"] = prediction
        results[test_id]["prediction_norm"] = prediction_norm

        # correct or not
        if answer_norm.lower() == prediction_norm.lower():
            correct += 1
            results[test_id]["true_false"] = True
        else:
            results[test_id]["true_false"] = False

        acc = correct / (i + 1) * 100

    result_cal_path = os.path.join(os.path.split(result_file_path)[0], 'cal_'+result_name+'.json')
    json.dump(results, open(result_cal_path, 'w'))
    print(f"Total: {total}, Correct: {correct}, Acc: {round(acc, 2)}%, saved to {result_cal_path}")

    scores = get_scores(result_cal_path, tab_test_path)
    print_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model results")
    parser.add_argument('--result_file', type=str, required=True, help='Path to the result file')
    parser.add_argument('--tab_test_file', type=str, default="/home/nfs04/chengkz/datasets/tabmwp/problems_test1k.json")

    args = parser.parse_args()
    main(args.result_file, args.tab_test_file)
