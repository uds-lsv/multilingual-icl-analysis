#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

model=llama2-chat
template=chat_template # 'chat_template_star' for using optimized template
data_seed=0
retriever=random
n_shot=4
task_list=xcopa,afrisenti,xquad,tydiqa

output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/${retriever}_${data_seed}
bash $PROJECT_DIR/scripts/eval_chat.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir



