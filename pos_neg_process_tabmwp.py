import json
import random
import re
import os
from tqdm import tqdm


def get_question_text_solution(problem, option_inds, model_name):
    """formulate question to input prompt"""
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

    if model_name == "qwenvl":
        if choices and len(choices) > 0:
            question += "\nFirst generate a step-by-step solution, then answer with the option’s letter from the given choices directly."
        else:
            question += "\nFirst generate a step-by-step solution, then answer the question using a single word or phrase."
    elif model_name == "llava":
        question += "\nLet's think step by step."
    else:
        print("undefined model name")
        input()

    return question


def contruct_selftrain_data_tabmwp(gen_result_files, sft_path, model_name, tabmwp_dir, data_self_train_dir):
    # sampling on training set

    qa_train = json.load(open(os.path.join(data_self_train_dir, 'sft_qa.json'), 'r'))
    if model_name == "qwenvl":
        gptcot_train = json.load(open(os.path.join(data_self_train_dir, 'qwenvl_sft_gpt.json'), 'r'))
    elif model_name == "llava":
        gptcot_train = json.load(open(os.path.join(data_self_train_dir, 'llava_sft_gpt.json'), 'r'))
    else:
        print("undefined model name")
        input()

    if len(gen_result_files) == 0:  # iter0, constructing dataset with qa and gpt warmup
        sft_train_new = qa_train+gptcot_train
        random.shuffle(sft_train_new)
        print("New Train Num: " + str(len(sft_train_new)))
        json.dump(sft_train_new, open(sft_path, 'w'))
        return

    result_all = []
    for file_path in gen_result_files:
        with open(file_path, 'r') as f:
            result_all.append(json.load(f))

    tab_test_path = os.path.join(tabmwp_dir, "problems_train.json")
    tab_imgs_dir = os.path.join(tabmwp_dir, "tables")
    tab_test = json.load(open(tab_test_path, 'r'))

    # construct self-CoT data
    tab_train_sft_CoT = []

    gen_result = list(result_all[0].items())
    option_inds = ["A", "B", "C", "D", "E", "F"]
    pos_num = 0
    for i in tqdm(range(23059)):
        index = gen_result[i][0]

        true_false = [item[index]['true_false'] for item in result_all]
        if sum(true_false) == 0:
            continue

        pos_num += 1
        true_indexes = []
        for j, item in enumerate(true_false):
            if item:
                true_indexes.append(j)
        pos_solution = result_all[true_indexes[-1]][index]["output"]

        img_filename = str(index) + '.png'
        img_path = os.path.join(tab_imgs_dir, img_filename)
        question = get_question_text_solution(tab_test[index], option_inds, model_name)

        answer = pos_solution
        conversations = []
        conversations.append({"from": "user", "value": question})
        conversations.append({"from": "assistant", "value": answer})
        sft_item = {"image": img_filename, "conversations": conversations}
        tab_train_sft_CoT.append(sft_item)

    print("Num of Pos: "+str(pos_num))
    print("Num of CoT train: "+str(len(tab_train_sft_CoT)))


    # construct self-refine data
    tab_train_sft_refine = []

    num_pos_neg = 0
    gen_result = list(result_all[0].items())
    option_inds = ["A", "B", "C", "D", "E", "F"]
    system_message = "Judge the correctness of the model's prediction and refine it."
    for i in tqdm(range(23059)):
        index = gen_result[i][0]

        true_false = [item[index]['true_false'] for item in result_all]

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

        pos_solution = result_all[true_indexes[-1]][index]["output"]
        neg_solution = result_all[false_indexes[-1]][index]["output"]

        img_filename = str(index) + '.png'
        img_path = os.path.join(tab_imgs_dir, img_filename)
        question = get_question_text_solution(tab_test[index], option_inds, model_name)

        # neg
        user_prompt = question+"\n"+"Model\'s prediction:\n"+neg_solution+"\n"+system_message
        model_answer = pos_solution
        conversations = []
        conversations.append({"from": "user", "value": user_prompt})
        conversations.append({"from": "assistant", "value": model_answer})
        data_item = {"image": img_filename, "conversations": conversations}
        tab_train_sft_refine.append(data_item)


    random.shuffle(tab_train_sft_refine)
    print("Num of pos/neg pair: "+str(num_pos_neg))
    print("Num of refine train: "+str(len(tab_train_sft_refine)))

    """
    # construct self-select data（with multi-candidates)
    tab_train_sft_select = []

    num_examples = 0
    num_samples_per_n = {n: 0 for n in range(2, 7)}
    gen_result = list(result_all[0].items())
    option_inds = ["A", "B", "C", "D", "E", "F"]
    system_message = "Which prediction has the right OCR, reasoning and calculations? Give the final answer."

    for i in tqdm(range(len(gen_result))):
        index = gen_result[i][0]

        true_false = [item[index]['true_false'] for item in result_all]

        if len(set(true_false)) == 1:
            continue

        num_examples += 1
        true_indexes = [j for j, item in enumerate(true_false) if item]
        false_indexes = [j for j, item in enumerate(true_false) if not item]

        img_filename = str(index) + '.png'
        img_path = os.path.join(tab_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
            input()
        question = get_question_text_solution(tab_test[index], option_inds, model_name)

        if len(true_indexes) == 0:
            continue

        desired_n_list = list(range(2, 7))
        for n in desired_n_list:
            k = 2 
            samples_generated = 0
            max_trials = 3 
            trials = 0
            while samples_generated < k and trials < max_trials:
                trials += 1
                
                if len(true_indexes) + len(false_indexes) < n:
                    break  
                
                num_pos_samples = random.randint(1, min(len(true_indexes), n - 1))
                num_neg_samples = n - num_pos_samples
                if len(false_indexes) < num_neg_samples:
                    continue
                selected_pos_indexes = random.sample(true_indexes, num_pos_samples)
                selected_neg_indexes = random.sample(false_indexes, num_neg_samples)
                selected_indexes = selected_pos_indexes + selected_neg_indexes
                random.shuffle(selected_indexes)
                
                solutions_cand = [result_all[j][index]["output"] for j in selected_indexes]
                
                pos_answers = []
                for j in selected_pos_indexes:
                    match = re.search(r'Answer:\s*(.*)', result_all[j][index]["output"])
                    if match:
                        pos_answers.append(match.group(1).strip())
                if len(set(pos_answers)) != 1:
                    continue  
                if len(pos_answers) == 0:
                    continue 
                model_answer = "Answer: " + pos_answers[0]
                
                user_prompt = question + "\n"
                for idx, solution in enumerate(solutions_cand):
                    user_prompt += f"Model's prediction {idx + 1}:\n{solution}\n"
                user_prompt += system_message
                
                conversations = []
                conversations.append({"from": "user", "value": user_prompt})
                conversations.append({"from": "assistant", "value": model_answer})
                data_item = {"image": img_filename, "conversations": conversations}
                tab_train_sft_select.append(data_item)
                samples_generated += 1
                num_samples_per_n[n] += 1

    random.shuffle(tab_train_sft_select)
    print("Num of examples with pos/neg samples: " + str(num_examples))
    for n in num_samples_per_n:
        print(f"Num of samples with {n} candidates: {num_samples_per_n[n]}")
    print("Total Num of Select train samples: " + str(len(tab_train_sft_select)))
    """

    # construct self-select data
    tab_train_sft_select = []

    num_pos_neg = 0
    num_2pos_1neg = 0
    num_1pos_2neg = 0
    num_1pos_2neg_hard = 0
    gen_result = list(result_all[0].items())
    option_inds = ["A", "B", "C", "D", "E", "F"]
    system_message = "Which prediction has the right OCR, reasoning and calculations? Give the final answer."
    for i in tqdm(range(23059)):
        index = gen_result[i][0]

        true_false = [item[index]['true_false'] for item in result_all]

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

        img_filename = str(index) + '.png'
        img_path = os.path.join(tab_imgs_dir, img_filename)
        if not os.path.exists(img_path):
            print("img not exist")
            input()
        question = get_question_text_solution(tab_test[index], option_inds, model_name)

        if len(true_indexes) >= 2:     # 2pos 1neg
            pos_solution_1 = result_all[true_indexes[-1]][index]["output"]
            pos_solution_2 = result_all[true_indexes[-2]][index]["output"]
            neg_solution_1 = result_all[false_indexes[-1]][index]["output"]
            solutions_cand = [pos_solution_1, pos_solution_2, neg_solution_1]
            random.shuffle(solutions_cand)
            match1 = re.search(r'Answer:\s*(.*)', pos_solution_1)
            match2 = re.search(r'Answer:\s*(.*)', pos_solution_2)
            if match1 or match2:
                model_answer = "Answer: " + (match1.group(1) if match1 else match2.group(1))

                user_prompt = question + "\n" + "Model\'s prediction 1:\n" + solutions_cand[
                    0] + "\n" + "Model\'s prediction 2:\n" + solutions_cand[1] + "\n" + "Model\'s prediction 3:\n" + \
                              solutions_cand[2] + "\n" + system_message

                conversations = []
                conversations.append({"from": "user", "value": user_prompt})
                conversations.append({"from": "assistant", "value": model_answer})
                data_item = {"image": img_filename, "conversations": conversations}

                tab_train_sft_select.append(data_item)
                num_2pos_1neg += 1

            else:
                print("no match")

        if len(false_indexes) >= 2:     # 1pos 2neg
            pos_solution_1 = result_all[true_indexes[-1]][index]["output"]
            neg_solution_1 = result_all[false_indexes[-1]][index]["output"]
            neg_solution_2 = result_all[false_indexes[-2]][index]["output"]
            solutions_cand = [pos_solution_1, neg_solution_1, neg_solution_2]
            random.shuffle(solutions_cand)
            match1 = re.search(r'Answer:\s*(.*)', pos_solution_1)
            if match1:
                model_answer = "Answer: "+match1.group(1)
                try:
                    if re.search(r'Answer:\s*(.*)', neg_solution_1).group(1) == re.search(r'Answer:\s*(.*)',
                                                                                          neg_solution_2).group(1):
                        num_1pos_2neg_hard += 1
                except:
                    pass
                user_prompt = question + "\n" + "Model\'s prediction 1:\n" + solutions_cand[
                    0] + "\n" + "Model\'s prediction 2:\n" + solutions_cand[1] + "\n" + "Model\'s prediction 3:\n" + \
                              solutions_cand[2] + "\n" + system_message
                conversations = []
                conversations.append({"from": "user", "value": user_prompt})
                conversations.append({"from": "assistant", "value": model_answer})
                data_item = {"image": img_filename, "conversations": conversations}
                tab_train_sft_select.append(data_item)
                num_1pos_2neg += 1
            else:
                print("no match")
    random.shuffle(tab_train_sft_select)
    print("Num of pos/neg pair: "+str(num_pos_neg))
    print("Num of 2pos_1neg: "+str(num_2pos_1neg))
    print("Num of 1pos_2neg: "+str(num_1pos_2neg))
    print("Num of 1pos_2neg_hard: "+str(num_1pos_2neg_hard))
    print("Num of Select train: "+str(len(tab_train_sft_select)))


    print("QA Num: "+str(len(qa_train)))
    print("GPT CoT Num: "+str(len(gptcot_train)))
    print("Self-CoT Num: "+str(len(tab_train_sft_CoT)))
    print("Self-Refine Num: "+str(len(tab_train_sft_refine)))
    print("Self-Select Num: "+str(len(tab_train_sft_select)))
    sft_train_new = qa_train+gptcot_train+tab_train_sft_CoT+tab_train_sft_refine+tab_train_sft_select
    random.shuffle(sft_train_new)
    print("New Train Num: "+str(len(sft_train_new)))
    json.dump(sft_train_new, open(sft_path, 'w'))