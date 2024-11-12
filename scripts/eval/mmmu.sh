#export PYTHONPATH=$PYTHONPATH:$PWD
#source scripts/eval/config.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p ./eval/mmmu/results
batch_size=${BATCH_SIZE:-6}
gpu_number=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
result=./eval/mmmu/results/result_qwenvl_cot_ours_ep1.xlsx
model=/home/nfs04/chengkz/VL-RLHF/ckpts/OOD-QwenVL/bs64_ep3_lr_3e-5_vlquery_json_ood_ours/checkpoint-780
DATA_DIR=data_dir/MMMU_DEV_VAL.tsv

workdir=./eval/mmmu/results

accelerate launch --config_file accelerate_config/infer.yaml --num_processes $gpu_number --main_process_port 29501 \
    -m vlrlhf.eval.mmmu.eval \
    --data_root $DATA_DIR \
    --model_path $model \
    --output_path $result \
    --batch_size $batch_size
