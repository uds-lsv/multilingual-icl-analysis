#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

task_list=xquad,tydiqa
n_shot=4
template=completion_template
data_seed=0
retriever=flip
bs=4

for model in xglm llama2; do
    output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/${retriever}_${data_seed}_minone
    bash $PROJECT_DIR/scripts/eval_llm.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir $bs
done
