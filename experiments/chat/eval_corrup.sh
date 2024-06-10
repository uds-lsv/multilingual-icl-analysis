#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

model=llama2-chat
data_seed=0
n_shot=4
template=chat_template
task_list=xcopa,afrisenti,xquad,tydiqa
retriever=flip

output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/${retriever}_${data_seed}
bash $PROJECT_DIR/scripts/eval_chat.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir
