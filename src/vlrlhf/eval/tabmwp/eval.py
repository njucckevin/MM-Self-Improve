# 整理tabmwp评测
import json
from accelerate import Accelerator
import argparse
from torch.utils.data import Dataset
from ..utils import run_vqa, VLCollator
import os

accelerator = Accelerator(mixed_precision="bf16")
model_path = None


def get_question_text(problem, option_inds):
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
        question = f"{question}\nOptions:\n{options}\nAnswer with the option’s letter from the given choices directly."
    else:
        question += "\nAnswer the question using a single word or phrase."

    return question


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
    if "qwen" in model_path.lower():
        if choices and len(choices) > 0:
            question += "\nFirst generate a step-by-step solution, then answer with the option’s letter from the given choices directly."
        else:
            question += "\nFirst generate a step-by-step solution, then answer the question using a single word or phrase."
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
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_sample", type=bool, default=False)
    return parser.parse_args()


class TabMWPDataset(Dataset):
    def __init__(self, data_root, tab_test_path) -> None:
        super().__init__()
        self.data_root = data_root
        self.tab_test_path = tab_test_path
        with open(tab_test_path, "r") as f:
            data = json.load(f)
        self.data = [(k, v) for k, v in data.items()]
        self.data = self.data[:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "id": self.data[index][0],
            "image": os.path.join(self.data_root, "tables", str(self.data[index][0])+'.png'),
            "question": self.data[index][1]["question"],
            "choices": self.data[index][1]["choices"],
            "unit": self.data[index][1]["unit"],
            "answer": self.data[index][1]["answer"],
        }


class Collator(VLCollator):
    def __call__(self, batch):
        option_inds = ["A", "B", "C", "D", "E", "F"]
        ids = [b["id"] for b in batch]
        answers = [b["answer"] for b in batch]
        questions = [get_question_text_solution(b, option_inds) for b in batch]
        images = [b["image"] for b in batch]
        prompt = [
            self.processor.format_multimodal_prompt(q, img).replace("Picture 1: ", "")
            for q, img in zip(questions, images)
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
    dataset = TabMWPDataset(data_root, tab_test_file)
    print("Num of generate: "+str(len(dataset)))
    results = run_vqa(model_path, dataset, Collator, accelerator, args.processor_path, args.batch_size, args.do_sample)
    results = [r for r in results if r.update(prediction=r.pop("response")) is None]
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
