# The Impact of Demonstrations on Multilingual In-Context Learning
This repository contains code for our paper: 
[The Impact of Demonstrations on Multilingual In-Context Learning: A Multidimensional Analysis](https://arxiv.org/abs/2402.12976).
It allows for performing in-context learning with different types of demonstrations and templates 
on diverse models and tasks.  

**Models:**
- [XGLM](https://huggingface.co/facebook/xglm-7.5B)
- [LLama2](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [Llama2-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [mT0](https://huggingface.co/bigscience/mt0-xxl)
- [BLOOMZ](https://huggingface.co/bigscience/bloomz-7b1)
- OpenAI APIs ([via Microsoft Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/))

**Multilingual tasks:**
- [XNLI](https://huggingface.co/datasets/facebook/xnli)
- [IndicXNLI](https://huggingface.co/datasets/Divyanshu/indicxnli)
- [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x)
- [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa)
- [XStoryCloze](https://huggingface.co/datasets/juletxara/xstory_cloze)
- [AfriSenti](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment)
- [XQuAD](https://huggingface.co/datasets/google/xquad)
- [TyDiQA-GoldP](https://huggingface.co/datasets/khalidalt/tydiqa-goldp)
- [MAFAND](https://huggingface.co/datasets/masakhane/mafand)

## Setup
Create your docker image or virtual environment based on the provided docker file (./Dockerfile). Specifically,
```
python: 3.8
torch: 1.14.0
CUDA: 11.8.0
```

## Code
- `configs/`: configuration files for each task  
- `experiments/`: reproduce results in the paper (See [experiments/README.md](https://github.com/uds-lsv/multilingual-icl-analysis/tree/master/experiments) for details.)
- `openicl/`: modified version of [OpenICL](https://github.com/Shark-NLP/OpenICL)
- `scripts/`: bash scripts that used by experiments
- `arguments.py`: all input arguments
- `eval_openai.py`: evaluation pipeline for OpenAI models
- `eval_chat.py`: evaluation pipeline for Llama2-chat
- `eval_llm.py`: evaluation pipeline for other models, e.g., XGLM
- `utils.py`: utility functions, e.g., data processing 

## Acknowledgement
- Thanks to the [OpenICL](https://github.com/Shark-NLP/OpenICL) codebase 
and Microsoft Azure. 




