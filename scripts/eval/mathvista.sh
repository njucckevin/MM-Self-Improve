#export PYTHONPATH=$PYTHONPATH:$PWD
#source scripts/eval/config.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p ./eval/mathvista/results
batch_size=${BATCH_SIZE:-6}
gpu_number=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
result=./eval/mathvista/results/result_qwenvl_cot_gpt_ours.xlsx
model=/home/nfs04/chengkz/VL-RLHF/ckpts/OOD-QwenVL/bs64_ep3_lr_3e-5_vlquery_json_ood_gpt_ours/checkpoint-981
DATA_DIR=data_dir/MathVista_MINI.tsv

accelerate launch --config_file accelerate_config/infer.yaml --num_processes $gpu_number --main_process_port 29502 \
    -m vlrlhf.eval.mathvista.eval \
    --data_root $DATA_DIR \
    --model_path $model \
    --output_path $result \
    --batch_size $batch_size
