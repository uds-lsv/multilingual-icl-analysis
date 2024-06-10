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
import re
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

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever import ZeroRetriever, RandomRetriever, FlipRetriever, TopkRetriever
from openicl.icl_inferencer import AzureInferencer
from openicl.icl_evaluator import SquadEvaluator, EMEvaluator, ChrfEvaluator

from arguments import DataTrainingArguments, ModelArguments, InContextLearningArguments, SUPPORTED_MODELS
from utils import load_config, get_predictions_from_file, load_multilingual_data

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def post_process(all_predictions, task_name):
    def _format(response):
        if task_name in ['xnli', 'indicxnli', 'pawsx'] :
            pattern = r'\b(True|False|Neither|true|false|neither)\b'
            match = re.search(pattern, response)
            if match :
                response = match.group(0)

        if task_name in ['xcopa', 'xstorycloze'] :
            pattern = r'Choice number: \s*(\d+)'
            match = re.search(pattern, response)
            if match :
                response = match.group(0)
            if ":" in response:  # avoid code switching, e.g. gpt4 xstorycloze, eu "Aukera zenbakia: 1"  ru "Выбор номер: 1"
                response = response.replace(response.split(":")[0], "Choice number")

        if task_name == 'afrisenti' :
            pattern = r'\b(positive|negative|neutral|Positive|Negative|Neutral)\b'
            match = re.search(pattern, response)
            if match :
                response = match.group(0)

        return response

    post_preds = {}
    for l in all_predictions:
        post_preds[l] = [_format(p) for p in all_predictions[l]]
    return post_preds

def compuate_metrics(task_name, all_predictions, all_references) :
    if task_name in ['xquad', 'tydiqa'] :
        evaluator = SquadEvaluator()
    elif 'mafand' in task_name :
        evaluator = ChrfEvaluator()
    else :
        evaluator = EMEvaluator()  # exact match
        all_predictions = post_process(all_predictions, task_name)

    metrics = defaultdict(dict)
    for l in all_predictions.keys() :
        score = evaluator.score(predictions=all_predictions[l], references=all_references[l])
        if isinstance(score, dict) :
            for k, v in score.items() :
                metrics[k][l] = v
        else :
            metrics['accuracy'][l] = score

    return metrics


def run_task(model_args, data_args, training_args, icl_args) :
    # create output dir
    os.makedirs(training_args.output_dir, exist_ok=True)

    # load config & templates
    data_config = load_config(data_args.task_config_path, data_args.task_name)
    languages = data_config['languages']
    language_names = data_config['language_names']
    template_conf = data_config[icl_args.template]
    column_token_map = data_config['column_token_map']
    output_column = data_config['output_column']

    task_instruction = template_conf['system']
    input_template = PromptTemplate(template=template_conf['input_template'],
                                    column_token_map=column_token_map)
    output_template = PromptTemplate(template=template_conf['output_template'],
                                     column_token_map=column_token_map,
                                     selected_column_name=output_column,
                                     selected_column_map={})

    # ---------------------- Data --------------------------

    # load data
    if data_args.task_name in ['mafand_e2t', 'mafand_t2e'] :
        data_preprocess_func = "process_" + data_args.task_name
    else :
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
    for lang, lang_name in zip(languages, language_names) :
        # if 'zh' not in lang :
        #    continue
        if icl_args.incorrect_language_role :
            incorrect_langs = [l for l in language_names if l != lang_name]
            instruction = task_instruction.replace('EVALUATION_LANGUAGE', np.random.choice(incorrect_langs))
        elif icl_args.no_language_role :
            instruction = task_instruction.replace(' in EVALUATION_LANGUAGE', '')
        else :
            instruction = task_instruction.replace('EVALUATION_LANGUAGE', lang_name)

        # dataset_reader
        cur_dataset = eval_datasets[lang]
        dataset_reader = DatasetReader(cur_dataset,
                                       input_columns=data_config['input_columns'],
                                       output_column=data_config['output_column'],
                                       output_template=output_template,
                                       test_split=data_config['test_split'],
                                       ds_size=4 if icl_args.limit else data_args.num_openai_samples)
        all_references[lang] = dataset_reader.generate_output_field_corpus(
            dataset_reader.dataset[data_config['test_split']])

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
        elif icl_args.retriever == 'flip':
            # flip the labels for classification tasks
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
        max_new_tokens = 100 if 'mafand' in data_args.task_name else 50

        inferencer = AzureInferencer(api_name=model_args.model_name,
                                         max_model_token_num=model_args.max_seq_len,
                                         max_new_tokens=max_new_tokens)

        predictions = inferencer.inference(retriever, system=instruction,
                                           input_template=input_template,
                                           output_template=output_template,
                                           output_json_filepath=output_json_filepath,
                                           output_json_filename=output_json_filename)

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
    print(f"model arguments: {model_args}")
    print(f"ICL arguments: {icl_args}")

    model_args.max_seq_len = SUPPORTED_MODELS[model_args.model_name]['max_seq_len']
    if data_args.task_config_path is None :
        data_args.task_config_path = SUPPORTED_MODELS[model_args.model_name]['task_config_path']
        print(f"Using the config: {data_args.task_config_path}")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # set seed before initializing model.
    set_seed(training_args.seed)

    if icl_args.compute_metric_only :
        logger.warning("Computing metrics only. No evaluation will be performed.")

    # ---------------------- RUN --------------------------
    root_path = training_args.output_dir
    for task_name in data_args.task_list.split(",") :
        data_args.task_name = task_name.strip()
        training_args.output_dir = os.path.join(root_path, task_name)
        run_task(model_args, data_args, training_args, icl_args)


if __name__ == "__main__" :
    main()
