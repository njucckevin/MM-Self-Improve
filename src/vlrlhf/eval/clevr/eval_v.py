# CLEVR-MATH评测
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


def get_question_text_qa(question):
    """将问题formulate成输入的prompt（QA）"""
    question = question + "\nAnswer the question using a single numerical value (0-10)."
    return question


def get_question_text_cot(question):
    """将问题formulate成输入的prompt（QA）"""
    if "qwen" in model_path.lower():
        question += "\nFirst generate a step-by-step solution, then answer the question using a single numerical value (0-10)."
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


class CLEVRDataset_Select(Dataset):
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

            pred1 = extract_number(sample1[index]["prediction"])
            pred2 = extract_number(sample2[index]["prediction"])
            pred3 = extract_number(sample3[index]["prediction"])
            counter = Counter([pred1, pred2, pred3])
            pred_maj, _ = counter.most_common(1)[0]
            if str(pred_maj) == str(sample1[index]["answer"]):
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
            "image": os.path.join(self.data_root, self.tab_test[index]["filename"]),
            "question": self.tab_test[index]["question"],
            "answer": self.tab_test[index]["answer"],
            "model_preds": self.data[index][1]
        }


class Collator_Select(VLCollator):
    def __call__(self, batch):
        system_message = "Which prediction is correct? Give the final answer with a single numerical value (0-10)."
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
    dataset = CLEVRDataset_Select(data_root, tab_test_file, tab_test_sample1, tab_test_sample2, tab_test_sample3)
    print("Num of generate: "+str(len(dataset)))
    results = run_vqa(model_path, dataset, Collator_Select, accelerator, args.processor_path, args.batch_size, args.do_sample)
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)