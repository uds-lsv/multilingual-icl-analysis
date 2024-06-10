#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

task_list=pawsx,xcopa,afrisenti,xnli,indicxnli,xstorycloze,mafand_e2t,mafand_t2e,xquad,tydiqa
n_shot=0
retriever=zero
data_seed=0
template=chat_template

for model in gpt3.5 gpt4; do
    output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/
    bash $PROJECT_DIR/scripts/eval_openai.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir
done

