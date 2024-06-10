#!/bin/bash

export PROJECT_DIR=/nethome/mzhang/Project/ICL/project
source $PROJECT_DIR/scripts/setup.sh

model=gpt3.5
template=chat_template
retriever=random
n_shot=4
task_list=pawsx,xcopa,afrisenti,xnli,indicxnli,xstorycloze,mafand_e2t,mafand_t2e,xquad,tydiqa

output_dir=$PROJECT_DIR/results/${model}/${n_shot}_shot/${retriever}_${data_seed}
bash $PROJECT_DIR/scripts/eval_openai.sh $model $task_list $template $n_shot $retriever $data_seed $output_dir
