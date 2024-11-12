#!/bin/bash
# evaluation on tabmwp
# export PYTHONPATH=$PYTHONPATH:$PWD
# source scripts/eval/config.sh
export CUDA_VISIBLE_DEVICES=${7:-0,1,2,3,4,5,6,7}

if [ -z "$1" ]; then
  echo "Error: Missing result file path."
  exit 1
fi

if [ -z "$2" ]; then
  echo "Error: Missing model path."
  exit 1
fi

if [ -z "$3" ]; then
  echo "Error: Missing do_sample parameter (true/false)."
  exit 1
fi

if [ -z "$4" ]; then
  echo "Error: Missing dataset type (test/train)."
  exit 1
fi

if [ -z "$5" ]; then
  echo "Error: Missing data root path."
  exit 1
fi

result=$1
model=$2
do_sample=$3
dataset_type=$4
data_root=$5
batch_size=${BATCH_SIZE:-12}
num_processes=${6:-8}

if [ "$dataset_type" = "test" ]; then
  tab_test="$data_root/problems_test.json"
elif [ "$dataset_type" = "train" ]; then
  tab_test="$data_root/problems_train.json"
else
  echo "Error: Dataset type must be 'test' or 'train'."
  exit 1
fi

if [ "$do_sample" = "true" ]; then
    do_sample_flag="--do_sample true"
else
    do_sample_flag=""
fi

accelerate launch --config_file accelerate_config/infer.yaml --num_processes $num_processes --main_process_port 29502 \
    -m vlrlhf.eval.tabmwp.eval \
    --data_root $data_root \
    --tab_test_file $tab_test \
    --model_path $model \
    --output_path $result \
    --batch_size $batch_size \
    $do_sample_flag

python -m vlrlhf.eval.tabmwp.calculate \
        --result_file $result \
        --tab_test_file $tab_test