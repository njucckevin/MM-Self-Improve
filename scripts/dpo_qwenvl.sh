source scripts/config.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WANDB_MODE=offline

per_device_train_batch_size=8
gradient_accumulation_steps=2
epoch=3
margin=0
beta=0.1
lr=1e-5
dr=1.0
lm_lora_modules="c_attn,attn.c_proj,w1,w2"
vision_lora_modules="in_proj,out_proj,c_fc"
full_lora_modules="${lm_lora_modules},${vision_lora_modules}"
dataset="rlaifv"
data_json="/home/nfs04/chengkz/VL-RLHF/tabmwp_qwenvl_data/sft_iter3_dpo.json"
data_images="/home/nfs04/chengkz/datasets/tabmwp/tables"
# gpu_number=$(nvidia-smi --list-gpus | wc -l)
gpu_number=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
global_bs=$((per_device_train_batch_size * gradient_accumulation_steps * gpu_number))
name="bs${global_bs}_ep${epoch}_mg${margin}_bt${beta}_lr${lr}_${dataset}_dr${dr}_star_dpo_iter3"
accelerate launch --config_file accelerate_config/zero2.yaml --num_processes $gpu_number\
        src/vlrlhf/dpo.py \
        --model_name_or_path /home/nfs04/chengkz/VL-RLHF/ckpts/TabMWP-QwenVL/bs64_ep3_lr_3e-5_vlquery_json_star_sft_iter3/checkpoint-2103 \
        --output_dir ckpts/TabMWP-QwenVL/$name \
        --dataset_name ${dataset_name_map[$dataset]} \
        --data_path $data_json \
        --data_ratio $dr \
        --image_root $data_images \
        --freeze_vision_tower True \
        --use_flash_attention_2 False \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules $lm_lora_modules \
        --lora_bias "none" \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_train_epochs $epoch \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate $lr \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --score_margin $margin \
        --remove_unused_columns False \
        --beta $beta \
        --max_length 1024 \
        --max_prompt_length 512 \
        --max_target_length 512 \
        --save_strategy "epoch" \
        --save_total_limit 3 \
        --logging_first_step False \
        --logging_steps 5 \
        --report_to wandb \
        --run_name  $name\
        --project_name "VL-RLHF" \
        --group_name "TabMWP-QwenVL"
