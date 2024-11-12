# 整理tabmwp CoT评测
import json
from accelerate import Accelerator
import argparse
from torch.utils.data import Dataset
from ..utils import run_vqa, VLCollator
import os
import sys
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

accelerator = Accelerator(mixed_precision="bf16")
model_path = None


def get_question_text_solution(problem, option_inds):
    """将问题formulate成输入的prompt"""
    question = problem['question']

    unit = problem['unit']
    if unit and len(unit) > 0:
        question = f"{question} (Unit: {unit})"

    choices = problem['choices']
    if choices and len(choices) > 0:
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(option_inds[i], c))
        options = "\n".join(choice_list)
        question = f"{question}\nOptions:\n{options}"

    # 根据model_path中的关键字选择不同的提示方式
    if "qwen" in model_path:
        if choices and len(choices) > 0:
            question += "\nFirst generate a step-by-step solution, then answer with the option’s letter from the given choices directly."
        else:
            question += "\nFirst generate a step-by-step solution, then answer the question using a single word or phrase."
    elif "llava" in model_path:
        question += "\nLet's think step by step."

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


class TabMWPDataset_Select(Dataset):
    def __init__(self, data_root, tab_test_path, tab_test_sample1, tab_test_sample2, tab_test_sample3) -> None:
        super().__init__()
        self.data_root = data_root
        self.tab_test_path = tab_test_path
        with open(tab_test_path, "r") as f:
            tab_test = json.load(f)

        # 构建select测试集，同时计算Test@3 Oracle和major voting
        sample1 = json.load(open(tab_test_sample1, 'r'))
        sample2 = json.load(open(tab_test_sample2, 'r'))
        sample3 = json.load(open(tab_test_sample3, 'r'))
        results = [sample1, sample2, sample3]
        num = 0
        oracle_correct_num = 0
        major_vote_correct_num = 0
        self.data = []
        for index in sample1.keys():
            model_preds = {"pred_1": sample1[index]["output"],
                           "pred_2": sample2[index]["output"],
                           "pred_3": sample3[index]["output"]}
            self.data.append((index, model_preds))

            num += 1
            if sum([item[index]['true_false'] for item in results]) >= 1:
                oracle_correct_num += 1
            answer_norm = sample1[index]['answer_norm']
            pred_norm_list = [item[index]['prediction_norm'] for item in results]
            pred_norm_list = [item.lower() for item in pred_norm_list]
            counter = Counter(pred_norm_list)
            pred_norm, _ = counter.most_common(1)[0]
            if answer_norm.lower() == pred_norm.lower():
                major_vote_correct_num += 1

        print("Oracle Acc: " + str(oracle_correct_num / num))
        print("Major Vote Acc: " + str(major_vote_correct_num / num))

        self.tab_test = tab_test
        self.data = self.data[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_id = self.data[index][0]
        return {
            "id": data_id,
            "data_origin": self.tab_test[data_id],
            "image": os.path.join(self.data_root, "tables", str(data_id)+'.png'),
            "question": self.tab_test[data_id]["question"],
            "choices": self.tab_test[data_id]["choices"],
            "unit": self.tab_test[data_id]["unit"],
            "answer": self.tab_test[data_id]["answer"],
            "model_pred": self.data[index][1],
        }


class Collator_Select(VLCollator):
    def __call__(self, batch):
        system_message = "Which prediction has the right OCR, reasoning and calculations? Give the final answer."
        option_inds = ["A", "B", "C", "D", "E", "F"]
        ids = [b["id"] for b in batch]
        answers = [b["answer"] for b in batch]
        model_preds = [b["model_pred"] for b in batch]
        # questions = [get_question_text_solution_noprompt(b["data_origin"], option_inds) for b in batch]
        # user_prompts = [system_message+"\n"+"Model\'s prediction:\n"+model_pred+"\nAre the model's prediction correct?" for model_pred in model_preds]
        questions = [get_question_text_solution(b["data_origin"], option_inds) for b in batch]
        user_prompts = [question+"\n"+"Model\'s prediction 1:\n"+model_pred["pred_1"]+"\n"+"Model\'s prediction 2:\n"+model_pred["pred_2"]+"\n"+"Model\'s prediction 3:\n"+model_pred["pred_3"]+"\n"+system_message for question, model_pred in zip(questions, model_preds)]
        images = [b["image"] for b in batch]
        prompt = [
            self.processor.format_multimodal_prompt(user_prompt, img).replace("Picture 1: ", "")
            for user_prompt, img in zip(user_prompts, images)
        ]
        inputs = self.processor(texts=prompt, images_path=images, padding_side="left", check_format=False)
        others = [
            dict(index=index, answer=answer, question=question)
            for index, answer, question in zip(ids, answers, questions)
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
    dataset = TabMWPDataset_Select(data_root, tab_test_file, tab_test_sample1, tab_test_sample2, tab_test_sample3)
    print("Num of generate: "+str(len(dataset)))
    results = run_vqa(model_path, dataset, Collator_Select, accelerator, args.processor_path, args.batch_size, args.do_sample)
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
