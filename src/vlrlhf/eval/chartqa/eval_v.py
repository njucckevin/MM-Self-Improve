# ChartQA评测
import json
from accelerate import Accelerator
import argparse
from torch.utils.data import Dataset
from ..utils import run_vqa, VLCollator
from collections import Counter
import os
import re

accelerator = Accelerator(mixed_precision="bf16")
model_path = None


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


def get_question_text_qa(question):
    """将问题formulate成输入的prompt（QA）"""
    question = question + "\nAnswer the question with a numerical value, Yes/No, or a word or phrase."
    return question


def get_question_text_cot(question):
    """将问题formulate成输入的prompt（QA）"""
    if "qwen" in model_path.lower():
        question += "\nFirst generate a step-by-step solution, then answer the question with a numerical value, Yes/No, or a word or phrase."
    elif "llava" in model_path.lower():
        question += "\nLet's think step by step."
    else:
        print("undefined model name")
        input()

    return question


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tab_test_file", type=str, required=True)
    parser.add_argument("--tab_test_sample1", type=str, required=True)
    parser.add_argument("--tab_test_sample2", type=str, required=True)
    parser.add_argument("--tab_test_sample3", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_sample", type=bool, default=False)
    return parser.parse_args()


class ChartDataset_Select(Dataset):
    def __init__(self, data_root, tab_test_path, tab_test_sample1, tab_test_sample2, tab_test_sample3) -> None:
        super().__init__()
        self.data_root = data_root
        self.tab_test_path = tab_test_path
        with open(tab_test_path, "r") as f:
            tab_test = json.load(f)

        sample1 = json.load(open(tab_test_sample1, 'r'))
        sample2 = json.load(open(tab_test_sample2, 'r'))
        sample3 = json.load(open(tab_test_sample3, 'r'))

        results = [sample1, sample2, sample3]
        num = 0
        oracle_correct_num = 0
        major_vote_correct_num = 0
        self.data = []
        for index in range(len(sample1)):
            model_preds = {"pred_1": sample1[index]["prediction"],
                           "pred_2": sample2[index]["prediction"],
                           "pred_3": sample3[index]["prediction"]}
            self.data.append((index, model_preds))

            num += 1
            if sum([item[index]['correct'] for item in results]) >= 1:
                oracle_correct_num += 1

            pred1 = extract_answer(sample1[index]["prediction"])
            pred2 = extract_answer(sample2[index]["prediction"])
            pred3 = extract_answer(sample3[index]["prediction"])
            counter = Counter([pred1, pred2, pred3])
            pred_maj, _ = counter.most_common(1)[0]
            if is_ans_correct(pred_maj, sample1[index]["answer"]):
                major_vote_correct_num += 1

        print("Oracle Acc: " + str(oracle_correct_num / num))
        print("Major Vote Acc: " + str(major_vote_correct_num / num))

        self.tab_test = tab_test
        self.data = self.data[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "id": index,
            "image": os.path.join(self.data_root, self.tab_test[index]["imgname"]),
            "question": self.tab_test[index]["query"],
            "answer": self.tab_test[index]["label"],
            "model_preds": self.data[index][1]
        }


class Collator(VLCollator):
    def __call__(self, batch):
        system_message = "Which prediction is correct? Give the final answer with a numerical value, Yes/No, or a word or phrase."
        ids = [b["id"] for b in batch]
        answers = [b["answer"] for b in batch]
        questions = [get_question_text_cot(b["question"]) for b in batch]
        images = [b["image"] for b in batch]
        model_preds = [b["model_preds"] for b in batch]
        user_prompts = [question+"\n"+"Model\'s prediction 1:\n"+model_pred["pred_1"]+"\n"+"Model\'s prediction 2:\n"+model_pred["pred_2"]+"\n"+"Model\'s prediction 3:\n"+model_pred["pred_3"]+"\n"+system_message for question, model_pred in zip(questions, model_preds)]
        prompt = [
            self.processor.format_multimodal_prompt(user_prompt, img).replace("Picture 1: ", "")
            for user_prompt, img in zip(user_prompts, images)
        ]
        inputs = self.processor(texts=prompt, images_path=images, padding_side="left", check_format=False)
        others = [
            dict(index=index, answer=answer, question=question)
            for index, answer, question, prompt in zip(ids, answers, questions, user_prompts)
        ]
        return inputs, others


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    model_path = args.model_path
    tab_test_file = args.tab_test_file
    tab_test_sample1 = args.tab_test_sample1
    tab_test_sample2 = args.tab_test_sample2
    tab_test_sample3 = args.tab_test_sample3
    dataset = ChartDataset_Select(data_root, tab_test_file, tab_test_sample1, tab_test_sample2, tab_test_sample3)
    print("Num of generate: "+str(len(dataset)))
    results = run_vqa(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size, args.do_sample)
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)