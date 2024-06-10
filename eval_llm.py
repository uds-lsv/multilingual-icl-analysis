#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a model on a multilingual task for all languages"""

import logging
import os
import sys
import json
import datasets
import numpy as np
from collections import defaultdict
from time import time, ctime

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed
)

from huggingface_hub import login

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever import ZeroRetriever, RandomRetriever, FlipRetriever, TopkRetriever
from openicl.icl_inferencer import PPLInferencer, Seq2seq_PPLInferencer, GenInferencer
from openicl.icl_evaluator import AccEvaluator, SquadEvaluator, ChrfEvaluator

from arguments import DataTrainingArguments, ModelArguments, InContextLearningArguments, SUPPORTED_MODELS
from utils import load_config, get_predictions_from_file, load_multilingual_data

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def compuate_metrics(task_name, all_predictions, all_references) :
    if task_name in ['xquad', 'tydiqa'] :
        evaluator = SquadEvaluator()
    elif 'mafand' in task_name :
        evaluator = ChrfEvaluator()
    else :
        evaluator = AccEvaluator()

    metrics = defaultdict(dict)
    for l in all_predictions.keys() :
        assert len(all_predictions[l]) == len(all_references[l])
        score = evaluator.score(predictions=all_predictions[l], references=all_references[l])
        if isinstance(score, dict) :
            for k, v in score.items() :
                metrics[k][l] = v
        else :
            metrics['accuracy'][l] = score
    return metrics


def run_task(model, model_args, data_args, training_args, icl_args) :
    # create output dir
    os.makedirs(training_args.output_dir, exist_ok=True)

    # load config & templates
    data_config = load_config(data_args.task_config_path, data_args.task_name)
    languages = data_config['languages']
    template_conf = data_config[icl_args.template]
    ice_token = data_config['ice_token']
    sep_token = data_config['sep_token'] if 'sep_token' in data_config.keys() else None
    column_token_map = data_config['column_token_map']

    if 'ice_template' not in template_conf.keys() :
        template_conf['ice_template'] = template_conf['prompt_template']

    ice_template = PromptTemplate(template=template_conf['ice_template'],
                                  column_token_map=column_token_map,
                                  ice_token=ice_token,
                                  sep_token=sep_token)
    prompt_template = PromptTemplate(template=template_conf['prompt_template'],
                                     column_token_map=column_token_map,
                                     ice_token=ice_token,
                                     sep_token=sep_token)

    # load data
    if data_args.task_name in ['mafand_e2t', 'mafand_t2e']:
        data_preprocess_func = "process_" + data_args.task_name
    else:
        data_preprocess_func = template_conf['preprocess_func'] if 'preprocess_func' in template_conf.keys() else None

    eval_datasets = load_multilingual_data(task_name=data_args.task_name,
                                           hf_dataset=data_config['hf'],
                                           languages=data_config['languages'],
                                           cache_dir=data_args.dataset_cache_dir,
                                           process_func=data_preprocess_func)

    # check if the evaluation result already exist
    all_results_file = os.path.join(training_args.output_dir, "results.jsonl")
    if os.path.exists(all_results_file) :
        logger.warning(f"Results file {all_results_file} exists, exiting.")
        return

    # ---------------------- Evaluation --------------------------
    all_references = {}
    all_predictions = {}
    for lang in languages :
        # if 'en' not in lang :
        #    continue
        cur_dataset = eval_datasets[lang]

        # dataset_reader
        dataset_reader = DatasetReader(cur_dataset,
                                       input_columns=data_config['input_columns'],
                                       output_column=data_config['output_column'],
                                       test_split=data_config['test_split'],
                                       ds_size=4 if icl_args.limit else None)
        all_references[lang] = dataset_reader.references

        # check if the predictions already exist
        output_json_filepath = training_args.output_dir
        output_json_filename = f"{lang}_predictions"
        prediction_file = f'{output_json_filepath}/{output_json_filename}.json'

        if os.path.exists(prediction_file) :
            logger.warning(f'Predictions for {lang} already exist. Skipping the evaluation...')
            predictions = get_predictions_from_file(prediction_file)
            all_predictions[lang] = predictions
            continue

        # retriever
        if icl_args.retriever == 'zero' :
            retriever = ZeroRetriever(dataset_reader,
                                      index_split=data_config['index_split'],
                                      test_split=data_config['test_split'])
        elif icl_args.retriever == 'random' :
            retriever = RandomRetriever(dataset_reader,
                                        index_split=data_config['index_split'],
                                        test_split=data_config['test_split'],
                                        ice_num=icl_args.num_shots,
                                        seed=training_args.data_seed)
        elif icl_args.retriever == 'flip' :
            #assert data_args.task_name not in ['tydiqa', 'xquad', 'mafand_e2t', 'mafand_t2e']
            retriever = FlipRetriever(dataset_reader,
                                      index_split=data_config['index_split'],
                                      test_split=data_config['test_split'],
                                      ice_num=icl_args.num_shots,
                                      seed=training_args.data_seed)
        elif icl_args.retriever == 'topk' :
            retriever = TopkRetriever(dataset_reader,
                                      index_split=data_config['index_split'],
                                      test_split=data_config['test_split'],
                                      sentence_transformers_model_name='LaBSE',
                                      tokenizer_name='sentence-transformers/LaBSE',
                                      ice_num=icl_args.num_shots)
        else :
            raise NotImplementedError

        # inferencer
        if data_args.task_name in ['xquad', 'tydiqa', 'mafand_e2t', 'mafand_t2e'] :
            max_new_tokens = 50 if 'mafand' in data_args.task_name else 20

            inferencer = GenInferencer(model,
                                       tokenizer_name=model_args.tokenizer_name,
                                       max_model_token_num=model_args.max_seq_len,
                                       batch_size=training_args.per_device_eval_batch_size,
                                       generation_kwargs={"max_new_tokens" : max_new_tokens}, #, "min_new_tokens": 1},
                                       model_parallel=True,
                                       quantize=True if model_args.int8 else False)
            predictions = inferencer.inference(retriever, ice_template=ice_template,
                                               prompt_template=prompt_template,
                                               output_json_filepath=output_json_filepath,
                                               output_json_filename=output_json_filename)
        else :
            ppl_inferencer = Seq2seq_PPLInferencer if 'mt0' in model_args.model_name_or_path else PPLInferencer
            inferencer = ppl_inferencer(model,
                                        tokenizer_name=model_args.model_name_or_path,
                                        max_model_token_num=model_args.max_seq_len,
                                        batch_size=training_args.per_device_eval_batch_size,
                                        quantize=True if model_args.int8 else False)
            predictions = inferencer.inference(retriever, ice_template=ice_template,
                                               prompt_template=prompt_template,
                                               output_json_filepath=output_json_filepath,
                                               output_json_filename=output_json_filename,  #
                                               normalizing_str='')  # ignore prefix loss by using '' as the normalizing_str

        all_predictions[lang] = predictions

    # ---------------------- Compute metrics --------------------------
    metrics = compuate_metrics(data_args.task_name, all_predictions, all_references)

    # ---------------------- Save results --------------------------
    all_results = defaultdict(dict)

    # get average score across languages
    for metric, val in metrics.items() :
        all_results[metric] = val
        all_results['avg_' + metric] = np.mean([s for _, s in val.items()])

    # ---------------------- Save results --------------------------
    all_results["config"]["model_name_or_path"] = model_args.model_name
    all_results["config"]["task_name"] = data_args.task_name
    all_results["config"]["template"] = icl_args.template
    all_results["config"]["source_lang"] = icl_args.source_lang
    all_results["config"]["num_shots"] = icl_args.num_shots
    all_results["config"]["retriever"] = icl_args.retriever
    all_results["config"]["data_seed"] = training_args.data_seed
    all_results["config"]["log_time"] = ctime(time())

    # save to .jsonl
    with training_args.main_process_first(desc="Write results to file") :
        dumped = json.dumps(all_results, indent=2)
        with open(all_results_file, "w") as f :
            f.write(dumped)


def main() :
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, InContextLearningArguments))
    model_args, data_args, training_args, icl_args = parser.parse_args_into_dataclasses()

    model_args.model_name_or_path = SUPPORTED_MODELS[model_args.model_name]['model_name_or_path']
    model_args.max_seq_len = SUPPORTED_MODELS[model_args.model_name]['max_seq_len']
    model_args.tokenizer_name = model_args.model_name_or_path
    print(f"model arguments: {model_args}")
    print(f"ICL arguments: {icl_args}")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # login to huggingface hub to use llama-2
    login(token=training_args.hub_token)

    # set seed before initializing model.
    set_seed(training_args.seed)

    if icl_args.compute_metric_only :
        logger.warning("Computing metrics only. No evaluation will be performed.")
        model = None
    else :
        if ('t5' in model_args.model_name_or_path) or ('t0' in model_args.model_name_or_path) :
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path,
                                                          cache_dir=model_args.cache_dir,
                                                          load_in_8bit=True if model_args.int8 else False,
                                                          device_map='auto')
        else :
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                         cache_dir=model_args.cache_dir,
                                                         load_in_8bit=True if model_args.int8 else False,
                                                         device_map='auto')

    # ---------------------- RUN --------------------------
    root_path = training_args.output_dir
    for task_name in data_args.task_list.split(",") :
        data_args.task_name = task_name.strip()
        training_args.output_dir = os.path.join(root_path, task_name)
        run_task(model, model_args, data_args, training_args, icl_args)


if __name__ == "__main__" :
    main()
