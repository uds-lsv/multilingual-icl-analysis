from dataclasses import dataclass, field, fields
from typing import Optional

SUPPORTED_MODELS = {
    "xglm" : {
        'model_name_or_path' : "facebook/xglm-7.5B",
        'max_seq_len' : 2048
    },
    "llama2" : {
        'model_name_or_path' : "meta-llama/Llama-2-13b-hf",
        'max_seq_len' : 4096
    },
    "llama2-chat" : {
        'model_name_or_path' : "meta-llama/Llama-2-13b-chat-hf",
        'max_seq_len' : 4096
    },
    "bloomz" : {
        'model_name_or_path' : "bigscience/bloomz-7b1",
        'max_seq_len' : 2048
    },
    "mt0" : {
        'model_name_or_path' : "bigscience/mt0-xxl",
        'max_seq_len' : 1024
    },
    "gpt3.5":{
        'max_seq_len' : 16384
    },
    "gpt4":{
        'max_seq_len' : 32768,
    }
}


@dataclass
class DataTrainingArguments :
    task_list: Optional[str] = field(
        default=None,
        metadata={
            "help" : "The tasks to evaluate. Seperated by comma",
        },
    )
    task_config_path: Optional[str] = field(
        default='./configs/',
        metadata={
            "help" : "The config directory",
        }
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help" : "Where to save the cached dataset."}
    )
    num_openai_samples: Optional[int] = field(
        default=200, metadata={"help" : "Number of samples (per language) to evaluate OpenAI API"}
    )

@dataclass
class ModelArguments :
    model_name: str = field(
        metadata={
            "help" : "Pretrained model name",
            "choices" : SUPPORTED_MODELS.keys()
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help" : "Pretrained tokenizer name or path if not the same as model_name_or_path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help" : "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    int8: bool = field(
        default=True,
        metadata={"help" : "Use 8-bit precision via bitsandbytes"},
    )


@dataclass
class InContextLearningArguments :
    template: Optional[str] = field(
        default=None, metadata={
            "help": "The template for prompting"
        }
    )
    source_lang: Optional[str] = field(
        default='target', metadata={
            "choices" : [None, 'target'],
            "help" : "The language for getting few-shot demonstrations."}
    )
    num_shots: int = field(
        default=0,
        metadata={"help" : (
            "Total number of demonstrations in the context.")},
    )
    retriever: Optional[str] = field(
        default="zero", metadata={
            "choices" : ['zero', 'random', 'flip', 'topk'],
            "help" : "The demonstration retriever"}
    )
    compute_metric_only: bool = field(
        default=False,
        metadata={"help": (
            "Compute the metrics based on saved prediction files")},
    )
    incorrect_language_role: bool = field(
        default=False,
        metadata={"help": (
            "For OpenAI API, assign a random language in the task instruction")},
    )
    no_language_role: bool = field(
        default=False,
        metadata={"help": (
            "For OpenAI API,remove the language indicator in the task instruction")},
    )
    limit: bool = field(
        default=False,
        metadata={"help" : "Debug with limited data samples"},
    )
