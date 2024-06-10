#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

task_list=mafand_e2t,mafand_t2e
n_shot=0
retriever=zero
data_seed=0
bs=16

for model in bloomz xglm mt0 llama2; do
    output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/
    bash $PROJECT_DIR/scripts/eval_llm.sh $model $task_list $n_shot $retriever $data_seed $output_dir $bs
done
