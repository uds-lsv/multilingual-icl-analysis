import json
import os
import yaml
import sys
import glob
from datasets import load_dataset, Dataset, DatasetDict

MAFAND_LANGUAGE_NAMES = {
    "amh": "Amharic",
    "hau": "Hausa",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Luo",
    "nya": "Chichewa",
    "pcm": "Nigerian Pidgin",
    "sna": "Shona",
    "swa": "Swahili",
    "tsn": "Setswana",
    "twi": "Twi",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu"
}


def load_config(config_path, task_name):
    if task_name in ["mafand_e2t", "mafand_t2e"]:
        task_name = "mafand"

    with open(os.path.join(config_path, f'{task_name}.yaml'), 'r') as config_file:
        data_config = yaml.safe_load(config_file)
    return data_config


def get_predictions_from_file(filename):
    dict = json.load(open(filename, 'r'))
    return [sample['prediction'] for sample in dict.values()]


def process_xcopa(dataset):
    """XGLM template"""

    def _process(example):
        # input: (1) remove the comma at the end of the premise (2) add the connecting words
        example['premise'] = example["premise"].strip()[:-1]
        example['choice1'] = example['choice1'][0].lower() + example['choice1'][1:]
        example['choice2'] = example['choice2'][0].lower() + example['choice2'][1:]

        conn = 'because' if example['question'] == 'cause' else 'so'
        example['premise'] = example["premise"] + f" {conn}"
        return example

    return dataset.map(_process)


def process_xcopa2(dataset):
    """PromptSource template (for instruction tuning models)"""

    def _process(example):
        conn = 'This happened because...' if example['question'] == 'cause' else 'As a consequence...'
        example['premise'] = example["premise"] + f" {conn}"
        return example

    return dataset.map(_process)


def process_qa(dataset):
    def _process(example):
        example['answers'] = example["answers"]["text"][0]
        return example

    # get the answer span
    dataset = dataset.map(_process)
    # remove unanswerable questions in IndicQA
    dataset = dataset.filter(lambda example: len(example["answers"]) > 0)  # e.g., hi 1.5k -> 1k
    return dataset


def process_mafand_e2t(dataset):
    splits = list(dataset.keys())
    target_lang = [l for l in list(dataset[splits[0]][0]['translation'].keys()) if l != 'en'][0]

    def _add_column(d):
        num_rows = len(d)
        d = d.add_column(name="source_lang", column=['English'] * num_rows)
        d = d.add_column(name="source_text", column=[a['en'] for a in d['translation']])
        d = d.add_column(name="target_lang", column=[MAFAND_LANGUAGE_NAMES[target_lang]] * num_rows)
        d = d.add_column(name="target_text", column=[a[target_lang] for a in d['translation']])
        return d

    for split in splits:
        dataset[split] = _add_column(dataset[split])

    return dataset


def process_mafand_t2e(dataset):
    splits = list(dataset.keys())
    source_lang = [l for l in list(dataset[splits[0]][0]['translation'].keys()) if l != 'en'][0]

    def _add_column(d):
        num_rows = len(d)
        d = d.add_column(name="target_lang", column=['English'] * num_rows)
        d = d.add_column(name="target_text", column=[a['en'] for a in d['translation']])
        d = d.add_column(name="source_lang", column=[MAFAND_LANGUAGE_NAMES[source_lang]] * num_rows)
        d = d.add_column(name="source_text", column=[a[source_lang] for a in d['translation']])
        return d

    for split in splits:
        dataset[split] = _add_column(dataset[split])

    return dataset


def load_multilingual_data(task_name, hf_dataset, languages, cache_dir, process_func):
    d = {}
    for lang in languages:
        if "mafand" in task_name:
            d[lang] = load_dataset(hf_dataset, f"en-{lang}", cache_dir=cache_dir)
        else:
            d[lang] = load_dataset(hf_dataset, lang, cache_dir=cache_dir)

    if process_func is not None:
        for lang in languages:
            d[lang] = eval(process_func)(d[lang])

    return d


def add_template_name(path, template_name):
    result_files = glob.glob(os.path.join(path, '**/results.jsonl'), recursive=True)
    for file in result_files:
        with open(file, 'r') as fp:
            print(file)
            d = json.load(fp)
        d['config']['template'] = template_name
        with open(file, 'w') as fp:
            json.dump(d, fp, indent=2)


if __name__ == "__main__":
    path = sys.argv[1]
    template_name = sys.argv[2]

    add_template_name(path, template_name)
