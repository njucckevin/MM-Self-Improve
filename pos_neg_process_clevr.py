import json
import random
import re
import os
from tqdm import tqdm
from collections import Counter


def get_question_text_cot(question, model_name):
    """formulate question to input prompt"""
    if model_name == "qwenvl":
        question += "\nFirst generate a step-by-step solution, then answer the question using a single numerical value (0-10)."
    elif model_name == "llava":
        question += "\nLet's think step by step."
    else:
        print("undefined model name")
        input()

    return question

def extract_number(text):
    # extract number（0-10）
    if re.fullmatch(r'[0-9]|10', text.strip()):
       return text.strip()

    # match "Solution:\nxxxxx\nAnswer:\n{y}" and extract the final answer
    solution_answer_match = re.search(r'Answer:\s*(-?\d+)', text, re.DOTALL)
    if solution_answer_match:
        return solution_answer_match.group(1)

    return "unknown"


def contruct_selftrain_data_clevr(gen_result_files, sft_path, model_name, clevr_dir, data_self_train_dir):
    clevr_train_origin = json.load(open(os.path.join(clevr_dir, 'clevr_train.json'), 'r'))

    clevr_qa = json.load(open(os.path.join(data_self_train_dir, 'sft_qa.json'), 'r'))
    if model_name == "qwenvl":
        clevr_cot_gpt = json.load(open(os.path.join(data_self_train_dir, 'qwenvl_sft_gpt.json'), 'r'))
    elif model_name == "llava":
        clevr_cot_gpt = json.load(open(os.path.join(data_self_train_dir, 'llava_sft_gpt.json'), 'r'))
    else:
        print("undefined model name")
        input()

    if len(gen_result_files) == 0:  # iter0
        sft_train_new = clevr_qa+clevr_cot_gpt
        random.shuffle(sft_train_new)
        print("New Train Num: " + str(len(sft_train_new)))
        json.dump(sft_train_new, open(sft_path, 'w'))
        return

    result_all = []
    for file_path in gen_result_files:
        with open(file_path, 'r') as f:
            result_all.append(json.load(f))

    total_num = len(result_all[0])
    correct_num = len([item for item in result_all[0] if item["correct"]])
    print(f"Single Total: {total_num}, Correct: {correct_num}, Acc: {correct_num/total_num}")

    total_num = len(result_all[0])
    correct_num = 0
    for i in range(len(result_all[0])):
        true_false = [result_item[i]["correct"] for result_item in result_all]
        true = true_false.count(True)
        if true > 0:
            correct_num += 1
    print(f"Test@N Total: {total_num}, Correct: {correct_num}, Acc: {correct_num/total_num}")


    # construct self-CoT data
    clevr_train_sft_CoT = []
    clevr_imgs_dir = os.path.join(clevr_dir, 'clevr_math_imgs')
    pos_num = 0
    for i in tqdm(range(len(result_all[0]))):

        question = clevr_train_origin[i]["question"]
        img_filename = clevr_train_origin[i]["filename"]
        img_path = os.path.join(clevr_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
            input()
        question = get_question_text_cot(question, model_name)
        if question != result_all[0][i]["question"]:
            print("unmatch")
            input()

        true_false = [result_item[i]["correct"] for result_item in result_all]
        if sum(true_false) == 0:
            continue
        pos_num += 1
        true_indexes = []
        for j, item in enumerate(true_false):
            if item:
                true_indexes.append(j)
        pos_preds = [result_all[k][i]["prediction"] for k in true_indexes]
        pos_pred_selected = pos_preds[-1]

        conversations = []
        conversations.append({"from": "user", "value": question})
        conversations.append({"from": "assistant", "value": pos_pred_selected})
        sft_item = {"image": img_filename, "conversations": conversations}
        clevr_train_sft_CoT.append(sft_item)

    random.shuffle(clevr_train_sft_CoT)
    print("Num of Pos: "+str(pos_num))
    print("Num of SFT train: "+str(len(clevr_train_sft_CoT)))


    # construct self-refine data
    clevr_train_sft_refine = []
    clevr_imgs_dir = os.path.join(clevr_dir, 'clevr_math_imgs')
    
    num_pos_neg = 0
    num_pos = 0
    num_neg = 0
    system_message = "Judge the correctness of the model's prediction and refine it."
    for i in tqdm(range(len(result_all[0]))):
    
        question = clevr_train_origin[i]["question"]
        img_filename = clevr_train_origin[i]["filename"]
        img_path = os.path.join(clevr_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
            input()
        question = get_question_text_cot(question, model_name)
        if question != result_all[0][i]["question"]:
            print("unmatch")
            input()
    
        true_false = [result_item[i]["correct"] for result_item in result_all]
        if len(set(true_false)) == 1:
            continue
    
        num_pos_neg += 1
        true_indexes = []
        false_indexes = []
        for j, item in enumerate(true_false):
            if item:
                true_indexes.append(j)
        for j, item in enumerate(true_false):
            if not item:
                false_indexes.append(j)
        pos_preds = [result_all[k][i]["prediction"] for k in true_indexes]
        pos_pred_before = pos_preds[0]
        pos_pred_after = pos_preds[-1]
        neg_pred = result_all[false_indexes[-1]][i]["prediction"]
    
        # neg
        user_prompt = question + "\n" + "Model\'s prediction:\n" + neg_pred + "\n" + system_message
        model_answer = pos_pred_after
        conversations = []
        conversations.append({"from": "user", "value": user_prompt})
        conversations.append({"from": "assistant", "value": model_answer})
        sft_item = {"image": img_filename, "conversations": conversations}
        clevr_train_sft_refine.append(sft_item)
        num_neg += 1
    
    random.shuffle(clevr_train_sft_refine)
    print("Num of pos/neg pair: "+str(num_pos_neg))
    print("Num of pos: "+str(num_pos))
    print("Num of neg: "+str(num_neg))
    print("Num of SFT train: "+str(len(clevr_train_sft_refine)))


    # construct self-select data
    clevr_train_sft_select = []
    clevr_imgs_dir = os.path.join(clevr_dir, 'clevr_math_imgs')
    
    num_pos_neg = 0
    num_2pos_1neg = 0
    num_1pos_2neg = 0
    system_message = "Which prediction is correct? Give the final answer with a single numerical value (0-10)."
    for i in tqdm(range(len(result_all[0]))):
    
        question = clevr_train_origin[i]["question"]
        img_filename = clevr_train_origin[i]["filename"]
        img_path = os.path.join(clevr_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
            input()
        question = get_question_text_cot(question, model_name)
        if question != result_all[0][i]["question"]:
            print("unmatch")
            input()
    
        true_false = [result_item[i]["correct"] for result_item in result_all]
        if len(set(true_false)) == 1:
            continue
    
        num_pos_neg += 1
        true_indexes = []
        false_indexes = []
        for j, item in enumerate(true_false):
            if item:
                true_indexes.append(j)
        for j, item in enumerate(true_false):
            if not item:
                false_indexes.append(j)
    
        if len(true_indexes) >= 2:  # 2pos 1neg
            pos_solution_1 = result_all[true_indexes[-1]][i]["prediction"]
            pos_solution_2 = result_all[true_indexes[-2]][i]["prediction"]
            neg_solution_1 = result_all[false_indexes[-1]][i]["prediction"]
            solutions_cand = [pos_solution_1, pos_solution_2, neg_solution_1]
            random.shuffle(solutions_cand)
            if extract_number(pos_solution_1) != extract_number(pos_solution_2):
                print(pos_solution_1)
                print(pos_solution_2)
                input()
                continue
            pos_answer = extract_number(pos_solution_1)
            if pos_answer == "unknown":
                continue
            user_prompt = question + "\n" + "Model\'s prediction 1:\n" + solutions_cand[
                0] + "\n" + "Model\'s prediction 2:\n" + solutions_cand[1] + "\n" + "Model\'s prediction 3:\n" + \
                          solutions_cand[2] + "\n" + system_message
    
            conversations = []
            conversations.append({"from": "user", "value": user_prompt})
            conversations.append({"from": "assistant", "value": pos_answer})
            sft_item = {"image": img_filename, "conversations": conversations}
            clevr_train_sft_select.append(sft_item)
            num_2pos_1neg += 1
    
        if len(false_indexes) >= 2:  # 1pos 2neg
            pos_solution_1 = result_all[true_indexes[-1]][i]["prediction"]
            neg_solution_1 = result_all[false_indexes[-1]][i]["prediction"]
            neg_solution_2 = result_all[false_indexes[-2]][i]["prediction"]
            solutions_cand = [pos_solution_1, neg_solution_1, neg_solution_2]
            random.shuffle(solutions_cand)
            pos_answer = extract_number(pos_solution_1)
            if pos_answer == "unknown":
                continue
            user_prompt = question + "\n" + "Model\'s prediction 1:\n" + solutions_cand[
                0] + "\n" + "Model\'s prediction 2:\n" + solutions_cand[1] + "\n" + "Model\'s prediction 3:\n" + \
                          solutions_cand[2] + "\n" + system_message
    
            conversations = []
            conversations.append({"from": "user", "value": user_prompt})
            conversations.append({"from": "assistant", "value": pos_answer})
            sft_item = {"image": img_filename, "conversations": conversations}
            clevr_train_sft_select.append(sft_item)
            num_1pos_2neg += 1
    
    random.shuffle(clevr_train_sft_select)
    print("Num of pos/neg pair: "+str(num_pos_neg))
    print("Num of 2pos_1neg: "+str(num_2pos_1neg))
    print("Num of 1pos_2neg: "+str(num_1pos_2neg))
    print("Num of SFT train: "+str(len(clevr_train_sft_select)))

    print("Num QA: {}".format(len(clevr_qa)))
    print("Num GPT CoT: {}".format(len(clevr_cot_gpt)))
    print("Self-CoT Num: "+str(len(clevr_train_sft_CoT)))
    print("Self-Refine Num: "+str(len(clevr_train_sft_refine)))
    print("Self-Select Num: "+str(len(clevr_train_sft_select)))

    clevr_sft = clevr_qa+clevr_cot_gpt+clevr_train_sft_CoT+clevr_train_sft_refine+clevr_train_sft_select
    random.shuffle(clevr_sft)
    print("Num SFT: {}".format(len(clevr_sft)))
    json.dump(clevr_sft, open(sft_path, 'w'))


