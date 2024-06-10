"""Direct Generation Inferencer"""
from openicl import PromptTemplate
from openicl.icl_retriever import BaseRetriever
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.azure_service import is_api_available, api_get_tokens
from openicl.utils.logging import get_logger
from typing import List, Optional
import json
import tiktoken

logger = get_logger(__name__)


class AzureInferencer:
    """OpenAI In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
    """

    def __init__(self,
                 api_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 max_new_tokens: Optional[int] = 50):
        self.api_name = api_name
        self.max_model_token_num = max_model_token_num
        self.max_new_tokens = max_new_tokens

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        assert is_api_available(self.api_name), f"API {self.api_name} is not available."

    def num_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def inference(self, retriever: BaseRetriever,
                  system: Optional[str] = None,
                  input_template: Optional[PromptTemplate] = None,
                  output_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num)
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
                prompt_token_num = self.num_tokens_from_messages(messages)
                while ice_num > 0 and prompt_token_num > self.max_model_token_num:
                    ice_num -= 1
                    del messages[-3:-1]
                    prompt_token_num = self.num_tokens_from_messages(messages)
            all_messages.append(messages)

        output_handler.save_orgin_prompts([json.dumps(m, ensure_ascii=False) for m in all_messages])

        # 4. Inference for prompts in each batch
        logger.info("Starting inference process...")
        responses = api_get_tokens(self.api_name, all_messages, self.max_new_tokens)

        # 5-3. Save current output
        for response in responses:
            prediction = output = response
            output_handler.save_prediction_and_output(prediction, output, index)
            index = index + 1

        # 6. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()]
