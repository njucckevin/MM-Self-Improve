#!/bin/bash
source scripts/config.sh

export CUDA_VISIBLE_DEVICES=${6:-0,1,2,3,4,5,6,7}

export WANDB_MODE=offline

if [ -z "$1" ]; then
    echo "Error: Please provide the path to the data JSON file as the first argument."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Please provide the iteration identifier (e.g., dpo_iter0) as the second argument."
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
    echo "Error: Please provide the pretrained model."
    exit 1
fi

# Training parameters
per_device_train_batch_size=4
gradient_accumulation_steps=${7:-2}
epoch=3
margin=-1
beta=0.1
lr=1e-5
lm_lora_modules="auto"
dataset="rlaifv"
data_json=$1
data_images=$3
gpu_number=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Number of GPUs (gpu_number): $gpu_number"
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
dpo_iter=$2
ckpts_dir=$4
pretrain_path=$5
name="${ckpts_dir}/bs${global_bs}_ep${epoch}_mg${margin}_bt${beta}_lr${lr}_${dataset}_${dpo_iter}"

accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number\
        src/vlrlhf/dpo.py \
        --model_name_or_path ${pretrain_path} \
        --output_dir ${name} \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_path $data_json \
        --image_root $data_images \
        --freeze_vision_tower True \
        --use_lora True \
        --use_flash_attention_2 True \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --lora_bias none \
        --lora_target_modules $lm_lora_modules \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_train_epochs $epoch \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate $lr \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --score_margin $margin \
        --remove_unused_columns False \
        --beta $beta \
        --max_length 2048 \
        --max_prompt_length 1024 \
        --max_target_length 512 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --logging_first_step False \
        --logging_steps 5 \
        --report_to wandb \
        --run_name  $name\
        --project_name "VL-RLHF" \
        --group_name "llava-1.5-7b-dpo" \
