"""Direct Generation Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import BaseRetriever
# from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from accelerate import Accelerator

logger = get_logger(__name__)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class LlamaInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for llama-2-chat.

    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class.
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file.
        api_name (:obj:`str`, optional): Name of API service.
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method.
    """

    def __init__(self,
                 model_name: Optional[str] = 'meta-llama/Llama-2-13b-chat-hf',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs

    def format_tokens(self, dialog):
        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]

        dialog_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [self.tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )

        dialog_tokens += self.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )

        return dialog_tokens

    def inference(self, retriever: BaseRetriever,
                  system: Optional[str] = None,
                  input_template: Optional[PromptTemplate] = None,
                  output_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate messages for testing input
        all_messages = []
        for idx, ice_idx in enumerate(ice_idx_list):
            test_sample = retriever.test_ds[idx]
            messages = []
            if len(system) > 0:
                messages.append({"role": "system", "content": system})

            for ice_id in ice_idx:
                demo = retriever.index_ds[ice_id]
                messages.append({"role": "user", "content": input_template.generate_item(demo)})
                messages.append({"role": "assistant", "content": output_template.generate_item(demo)})
            messages.append({"role": "user", "content": input_template.generate_item(test_sample)})

            if self.max_model_token_num is not None:
                ice_num = len(ice_idx)
                prompt_token_num = len(self.format_tokens(messages))
                while ice_num > 0 and prompt_token_num > self.max_model_token_num:
                    ice_num -= 1
                    del messages[-3:-1]
                    prompt_token_num = len(self.format_tokens(messages))
            all_messages.append(messages)

        output_handler.save_orgin_prompts([json.dumps(m, ensure_ascii=False) for m in all_messages])

        # 4. messages -> llama-chat input tokens
        chats = [self.format_tokens(messages) for messages in all_messages]

        # 5. Inference for prompts
        logger.info("Starting inference process...")
        responses = []
        for chat in tqdm(chats, disable=not self.is_main_process):
            prompt_len = len(chat)
            with torch.no_grad():
                input_ids = torch.tensor(chat).long().unsqueeze(0).to(self.device)
                output = self.model.generate(input_ids=input_ids, **self.generation_kwargs)[0]
                gen_text = self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                responses.append(gen_text)

        # 5-3. Save current output
        for response in responses:
            prediction = output = response
            output_handler.save_prediction_and_output(prediction, output, index)
            index = index + 1

        # 6. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)
        return [sample['prediction'] for sample in output_handler.results_dict.values()]
