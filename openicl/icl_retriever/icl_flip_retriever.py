'''Random Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class FlipRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class  -> permute the labels
        Class of Random Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.seed = seed

        # modify the labels for index_ds -> also works for xquad cause self.test_ds is not changed
        self.index_ds = self.flip_labels(self.index_ds, output_column=dataset_reader.output_column)

    def flip_labels(self, dataset, output_column):
        label_set = set(dataset[output_column])

        def _flip(example):
            #select_labels = label_set.copy()
            #select_labels.remove(example[output_column])
            example[output_column] = np.random.choice(list(label_set))
            return example
        return self.index_ds.map(_flip)

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for idx in trange(len(self.test_ds), disable=not self.is_main_process):
            # Added:  deal with index_split == test_split  (for xquad dataset)
            idx_list = np.random.choice(num_idx, self.ice_num + 1, replace=False).tolist()

            if self.index_split != self.test_split:
                idx_list = idx_list[:self.ice_num]
            else:
                idx_list = [n for n in idx_list if n != idx][:self.ice_num]
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
