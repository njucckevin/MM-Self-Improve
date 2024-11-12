#!/bin/bash
source scripts/config.sh

export CUDA_VISIBLE_DEVICES=${6:-0,1,2,3,4,5,6,7}

export WANDB_MODE=offline

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the data JSON file as the first argument."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Please provide the iteration identifier (e.g., sft_iter0) as the second argument."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: Please provide the image root as the third argument."
    exit 1
fi

if [ -z "$4" ]; then
    echo "Error: Please provide the output dir as the fourth argument."
    exit 1
fi

if [ -z "$5" ]; then
    echo "Error: Please provide the model ckpt as the fifth argument."
    exit 1
fi

per_device_train_batch_size=2
gradient_accumulation_steps=${7:-4}
epoch=3
lr=3e-5
lm_lora_modules="c_attn,attn.c_proj,w1,w2"
dataset="vlquery_json"
data_json=$1
data_images=$3
# gpu_number=$(nvidia-smi --list-gpus | wc -l)
gpu_number=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Number of GPUs (gpu_number): $gpu_number"
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
sft_iter=$2
ckpts_dir=$4
model_ckpt=$5
name="${ckpts_dir}/bs${global_bs}_ep${epoch}_lr_${lr}_${dataset}_${sft_iter}"

accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number --main_process_port 29502 \
        src/vlrlhf/sft.py \
        --model_name_or_path ${model_ckpt} \
        --output_dir ${name} \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_path $data_json \
        --image_root $data_images \
        --dataset_num_proc 16 \
        --freeze_vision_tower True \
        --use_flash_attention_2 False \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules $lm_lora_modules \
        --lora_bias "none" \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_train_epochs $epoch \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate $lr \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "constant_with_warmup" \
        --gradient_checkpointing False \
        --bf16 True \
        --tf32 True \
        --remove_unused_columns False \
        --max_length 2048 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --logging_first_step False \
        --logging_steps 5 \
        --report_to wandb \
        --run_name  $name\
        --project_name "VL-RLHF" \
        --group_name "QwenVL" \
