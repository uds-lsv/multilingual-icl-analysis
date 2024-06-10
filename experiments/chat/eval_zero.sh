#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

model=llama2-chat
task_list=pawsx,xcopa,afrisenti,xnli,indicxnli,xstorycloze,mafand_e2t,mafand_t2e,xquad,tydiqa
template=chat_template  # 'chat_template_star' -> using optmized templates for the four tasks
n_shot=0
data_seed=0
retriever=zero

output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/
bash $PROJECT_DIR/scripts/eval_chat.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir

