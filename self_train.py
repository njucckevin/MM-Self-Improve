import os
import sys
import json
import re
import subprocess
import threading
from pos_neg_process_tabmwp import contruct_selftrain_data_tabmwp
from pos_neg_process_chartqa import contruct_selftrain_data_chartqa
from pos_neg_process_clevr import contruct_selftrain_data_clevr
import time
import argparse
import logging

logger = None

# keep the original stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr


class StreamToLogger(object):
    """save log files"""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def start_reading_threads(process, logger):
    def read_output(stream, log_func):
        for line in iter(stream.readline, ''):
            if line:
                log_func(line.rstrip())
        stream.close()
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, logger.info))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, logger.error))
    stdout_thread.start()
    stderr_thread.start()
    return stdout_thread, stderr_thread


def init_logger(dataset_name, model_name):
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"selftraining_{dataset_name}_{model_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('self_training')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    console_handler = logging.StreamHandler(original_stdout)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filepath, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def find_checkpoint_dir(path):
    # find checkpoint for given model path
    items = os.listdir(path)
    pattern = re.compile(r'checkpoint-\d+')
    for item in items:
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path) and pattern.match(item):
            return full_path
    return None


def run_eval_bash_script(dataset_name, result_path, model_path, do_sample, dataset_type, data_root, gpu_ids):
    # evaluation/sampling
    global logger
    num_gpus = str(len(gpu_ids.split(',')))  # gpu num
    if dataset_name == "tabmwp":
        script = './scripts/eval/tabmwp.sh'
    elif dataset_name == "clevr":
        script = './scripts/eval/clevr.sh'
    elif dataset_name == "chartqa":
        script = './scripts/eval/chartqa.sh'
    else:
        logger.error("Undefined dataset")
        raise Exception("Undefined dataset")

    command = [
        'bash', script,
        result_path,  # result saved
        model_path,  # model ckpt
        do_sample,  # true/false
        dataset_type,  # test/train
        data_root,  # dataset root
        num_gpus,
        gpu_ids
    ]
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_thread, stderr_thread = start_reading_threads(process, logger)
        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            logger.error(f"Bash script exited with return code {process.returncode}")
        else:
            logger.info("Bash script executed successfully.")
    except Exception as e:
        logger.error(f"Error occurred while executing bash script: {e}")


def run_eval_select_bash_script(dataset_name, result_path, sample1_path, sample2_path, sample3_path, model_path, do_sample, data_root, gpu_ids):
    # evaulation with test-time selection
    global logger
    num_gpus = str(len(gpu_ids.split(',')))
    if dataset_name == "tabmwp":
        script = './scripts/eval/tabmwp_select.sh'
    elif dataset_name == "clevr":
        script = './scripts/eval/clevr_select.sh'
    elif dataset_name == "chartqa":
        script = './scripts/eval/chartqa_select.sh'
    else:
        logger.error("Undefined dataset")
        raise Exception("Undefined dataset")

    command = [
        'bash', script,
        result_path,   # result saved
        sample1_path,   # sample1
        sample2_path,   # sample2
        sample3_path,   # sample3
        model_path,   # model ckpt
        do_sample,   # true/false
        data_root,   # dataset root
        num_gpus,
        gpu_ids
    ]
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_thread, stderr_thread = start_reading_threads(process, logger)
        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            logger.error(f"Bash script exited with return code {process.returncode}")
        else:
            logger.info("Bash script executed successfully.")
    except Exception as e:
        logger.error(f"Error occurred while executing bash script: {e}")


def run_training_bash_script(data_json, sft_iter, images_dir, ckpts_dir, gpu_ids, model_name, model_ckpt):
    # launch training
    global logger
    num_gpus = str(len(gpu_ids.split(',')))
    if 32 % int(num_gpus) != 0:
        logger.error(f"{num_gpus} GPUs cannot train with bs=64")
        raise Exception(f"{num_gpus} GPUs cannot train with bs=64")
    else:
        grad_acc = str(int(32/int(num_gpus)))
    if model_name == "qwenvl":
        script = './scripts/sft_qwenvl.sh'
    elif model_name == "llava":
        script = './scripts/sft_llava.sh'
    else:
        logger.error("Undefined model name")
        raise Exception("Undefined model name")

    command = [
        'bash', script,
        data_json,  # data_json
        sft_iter,  # sft_iter
        images_dir,
        ckpts_dir,
        model_ckpt,
        gpu_ids,
        grad_acc
    ]
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout_thread, stderr_thread = start_reading_threads(process, logger)
        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode != 0:
            logger.error(f"Training script exited with return code {process.returncode}")
        else:
            logger.info(f"Training script for {sft_iter} executed successfully.")
    except Exception as e:
        logger.error(f"Error occurred while executing training script: {e}")


def main():
    global logger
    parser = argparse.ArgumentParser(description="Self-training script")
    parser.add_argument('--num_iter', type=int, default=5, help='Number of self-training iterations')
    parser.add_argument('--model_prefix', type=str, default='bs64_ep3_lr_3e-5_vlquery_json_sft_iter', help='Model prefix')
    parser.add_argument('--data_self_train_dir', type=str, default='./data/data_self_train', help='Data self-training directory')
    parser.add_argument('--model_name', type=str, choices=['qwenvl', 'llava'], required=True, help='Model name (must be "qwenvl" or "llava")')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Model ckpt')
    parser.add_argument('--dataset_name', type=str, choices=['tabmwp', 'chartqa', 'clevr'], required=True, help='Dataset name (must be "tabmwp", "chartqa", or "clevr")')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--gpu_ids', type=str, required=True, help='GPU IDs to use')
    args = parser.parse_args()

    # redirect stdout and stderr to logger
    logger = init_logger(args.dataset_name, args.model_name)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    num_iter = args.num_iter
    model_prefix = args.model_prefix
    data_self_train_dir = args.data_self_train_dir
    model_name = args.model_name
    model_ckpt = args.model_ckpt
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    if not os.path.exists(data_self_train_dir):
        os.mkdir(data_self_train_dir)
    if not os.path.exists("./ckpts"):
        os.mkdir("./ckpts")
    results_dir = os.path.join(data_self_train_dir, dataset_name)
    ckpts_dir = os.path.join("./ckpts", dataset_name+'-'+model_name)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    if not os.path.exists(ckpts_dir):
        os.mkdir(ckpts_dir)

    # ======================================================================== #
    #                     Start the self-training loops
    # ======================================================================== #
    for cur_iter in range(0, num_iter):
        logger.info(f"Current iteration: {cur_iter}")

        train_sample_pool = []
        # Check that the trained model in previous iteration and that the training set sampling was completed
        time.sleep(3)
        if cur_iter != 0:
            for past_iter in range(cur_iter):
                logger.info(f"Check if iter {past_iter} has completed")
                model_path = os.path.join(ckpts_dir, model_prefix+str(past_iter))
                if os.path.exists(model_path):
                    ckpt_path = find_checkpoint_dir(model_path)
                    if ckpt_path == None:
                        logger.error("Checkpoint not found")
                        raise Exception("Checkpoint not found")
                    # training set sampling
                    sample_1 = f"result_sft_iter{past_iter}_train_sample1.json"
                    sample_2 = f"result_sft_iter{past_iter}_train_sample2.json"
                    sample_3 = f"result_sft_iter{past_iter}_train_sample3.json"
                    for sample in [sample_1, sample_2, sample_3]:
                        filename = model_name+"_"+sample
                        sample_path = os.path.join(results_dir, filename)
                        if not os.path.exists(sample_path):     # sampling
                            logger.info(f"{sample} does not exist, performing train sample ...")
                            run_eval_bash_script(dataset_name, sample_path, ckpt_path, "true", "train", dataset_dir, args.gpu_ids)
                        else:
                            logger.info(f"{sample} exists")
                        if dataset_name == "tabmwp":
                            sample_path = os.path.join(results_dir, "cal_"+filename)
                        train_sample_pool.append(sample_path)   # add sampled solutions to sample pool
                else:
                    logger.error(f"Iter {past_iter} model does not exist")
                    raise Exception(f"Iter {past_iter} model does not exist")

        # construct training dataset for this iteration
        time.sleep(3)
        logger.info(f"Iteration {cur_iter} Sample Pool:")
        logger.info(train_sample_pool)
        new_sft_path = os.path.join(results_dir, model_name+f"_sft_iter{cur_iter}.json")
        if dataset_name == "tabmwp":
            contruct_selftrain_data_tabmwp(train_sample_pool, new_sft_path, model_name, dataset_dir, results_dir)
        elif dataset_name == "clevr":
            contruct_selftrain_data_clevr(train_sample_pool, new_sft_path, model_name, dataset_dir, results_dir)
        elif dataset_name == "chartqa":
            contruct_selftrain_data_chartqa(train_sample_pool, new_sft_path, model_name, dataset_dir, results_dir)
        else:
            logger.error("Undefined dataset")
            raise Exception("Undefined dataset")

        # training
        time.sleep(3)
        data_json = new_sft_path
        sft_iter = f"sft_iter{cur_iter}"
        if dataset_name == "tabmwp":
            images_dir = os.path.join(dataset_dir, "tables")
        elif dataset_name == "chartqa":
            images_dir = os.path.join(dataset_dir, "train/png")
        elif dataset_name == "clevr":
            images_dir = os.path.join(dataset_dir, "clevr_math_imgs")
        else:
            logger.error("Undefined dataset")
            raise Exception("Undefined dataset")
        run_training_bash_script(data_json, sft_iter, images_dir, ckpts_dir, args.gpu_ids, model_name, model_ckpt)

        # evaluation
        # Test@1
        model_path = os.path.join(ckpts_dir, model_prefix + str(cur_iter))
        ckpt_path = find_checkpoint_dir(model_path)
        logger.info(f"Model checkpoint: {ckpt_path}")
        logger.info("Calculating Test@1")
        test_file = model_name + f"_result_sft_iter{cur_iter}.json"
        test_path = os.path.join(results_dir, test_file)
        run_eval_bash_script(dataset_name, test_path, ckpt_path, "false", "test", dataset_dir, args.gpu_ids)

        if cur_iter >= 1:
            # Test@3 and self-select
            # sample 3 times
            logger.info("Sampling for Test@3 and Select")
            test_file_sample1 = os.path.join(results_dir, model_name + f"_result_sft_iter{cur_iter}_sample1.json")
            test_file_sample2 = os.path.join(results_dir, model_name + f"_result_sft_iter{cur_iter}_sample2.json")
            test_file_sample3 = os.path.join(results_dir, model_name + f"_result_sft_iter{cur_iter}_sample3.json")
            for test_path in [test_file_sample1, test_file_sample2, test_file_sample3]:
                run_eval_bash_script(dataset_name, test_path, ckpt_path, "true", "test", dataset_dir, args.gpu_ids)

            # Select
            logger.info("Selecting")
            if dataset_name == "tabmwp":
                test_file_sample1 = os.path.join(results_dir, "cal_"+test_file_sample1)
                test_file_sample2 = os.path.join(results_dir, "cal_"+test_file_sample2)
                test_file_sample3 = os.path.join(results_dir, "cal_"+test_file_sample3)
            test_path_select = os.path.join(results_dir, model_name + f"_result_sft_iter{cur_iter}_select.json")
            run_eval_select_bash_script(dataset_name, test_path_select, test_file_sample1, test_file_sample2, test_file_sample3, ckpt_path, "false", dataset_dir, args.gpu_ids)

if __name__ == "__main__":
    main()