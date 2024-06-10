# Run experiments
Example scripts are provided here for reproducing results.

## Folders
- `openai`: OpenAI models
- `chat`: LLama2-chat
- `llm`: all other models, e.g., XGLM (please adjust the batch size in scripts according to your hardware)


## Main arguments
- `task_list`: the tasks to evaluate, separated by comma
- `n_shot`: number of demonstrations
- `data_seed`: the random seed used to sample demonstrations
- `retriever`: the type of demonstrations
  - `zero`: no demonstrations (zero-shot)
  - `random`: random selection
  - `flip`: corrupted labels
  - `topk`: the top-k most similar demonstrations
- `template`: prompting template specified in configuration files