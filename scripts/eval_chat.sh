#!/bin/bash

model=$1 
task_list=$2
template=$3
num_shot=$4
retriever=$5
data_seed=$6
output_dir=$7

python $PROJECT_DIR/eval_chat.py \
  --model_name $model\
  --task_list $task_list \
  --output_dir $output_dir \
  --cache_dir $TRANSFORMERS_CACHE \
  --dataset_cache_dir $HF_DATASETS_CACHE \
  --hub_token $HF_TOKEN \
  --template $template \
  --num_shots $num_shot \
  --retriever $retriever \
  --data_seed $data_seed
