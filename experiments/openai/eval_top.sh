#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh


retriever=topk
template=chat_template
data_seed=0
n_shot=4
task_list=xcopa,afrisenti,xquad,tydiqa

for model in gpt3.5; do
   output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/${retriever}_${data_seed}
   bash $PROJECT_DIR/scripts/eval_openai.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir
done
