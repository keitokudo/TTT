from pathlib import Path
import pickle
import json

from tqdm import tqdm
from logzero import logger
from torch.utils.data import IterableDataset, Dataset

import collators
from utils.pickle_file import PickleFileLoader

class PickleDataset(Dataset):
    def __init__(self, dataset_path:Path):
        self.dataset_path = dataset_path
        self.basic_info_path = self.dataset_path / "basic_info.json"
        self.tokenized_data_file_path = self.dataset_path / "tokenized_data.pkl"
        
        with self.basic_info_path.open(mode="r") as f:
            self.basic_info = json.load(f)
        
        self.collator = getattr(collators, self.basic_info["collator"])

        if (not ("save_format" in self.basic_info)) or self.basic_info["save_format"] == "iterable":
            logger.info("Loading iterable format dataset")
            self.tokenized_data = [
                obj for obj in PickleFileLoader(self.tokenized_data_file_path)
            ]
        else:
            logger.info("Loading non-iterable format dataset")
            self.tokenized_data = next(iter(PickleFileLoader(self.tokenized_data_file_path)))
            
    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, item):
        return self.tokenized_data[item]
    
    
class IterablePickleDataset(IterableDataset):
    def __init__(self, dataset_path:Path):
        self.dataset_path = dataset_path
        self.basic_info_path = self.dataset_path / "basic_info.json"
        self.tokenized_data_file_path = self.dataset_path / "tokenized_data.pkl"
        self.collator = getattr(collators, self.basic_info["collator"])
        assert (not ("save_format" in self.basic_info)) or self.basic_info["save_format"] == "iterable"
        
    def __iter__(self):
        yield from PickleFileLoader(self.tokenized_data_file_path)


class PickleMaxTokensDataset(PickleDataset):
    def __init__(self, dataset_path:Path, max_tokens:int):
        super().__init__(dataset_path)
        self.max_tokens = max_tokens
        # Only "input_ids": List of lisi of [List[batch_idxs], max_sentence_length]
        # with "labels": List of List of
        # [List[batch_idxs], max_input_ids_length, max_labels_length]
        self.batch_indices = []
        self.generate_batch_indices()


    def infer_whether_seq2seq(self):
        if "labels" not in self.tokenized_data[0]:
            return False
        
        return not all(
            [
                len(d["input_ids"]) == len(d["labels"])
                for d in self.tokenized_data
            ]
        )
    
    def generate_batch_indices_for_seq2seq(self):
        self.tokenized_data.sort(key=lambda d: len(d["input_ids"]))
        if len(self.tokenized_data[-1]["input_ids"]) > self.max_tokens:
            raise RuntimeError(
                "max_tokens is smaller than the length of the longest input_ids. "
                f"max_tokens: {self.max_tokens}, longest input_ids: {len(self.tokenized_data[0-1]['input_ids'])}"
            )
            
        self.tokenized_data.sort(key=lambda d: len(d["labels"]))
        if len(self.tokenized_data[-1]["labels"]) > self.max_tokens:
            raise RuntimeError(
                "max_tokens is smaller than the length of the longest labels. "
                f"max_tokens: {self.max_tokens}, longest labels: {len(self.tokenized_data[-1]['labels'])}"
            )

        current_batch = []
        current_max_input_ids_length = 0
        current_max_labels_length = 0        
        for i, data in enumerate(tqdm(self.tokenized_data)):
            new_max_input_ids_length = max(
                current_max_input_ids_length,
                len(data['input_ids'])
            )
            new_max_labels_length = max(
                current_max_labels_length,
                len(data['labels'])
            )
            new_total_input_ids_num_tokens = (len(current_batch) + 1) * new_max_input_ids_length
            new_total_labels_num_tokens = (len(current_batch) + 1) * new_max_labels_length
            
            if new_total_input_ids_num_tokens + new_total_labels_num_tokens <= self.max_tokens:
                current_batch.append(i)
                current_max_input_ids_length = new_max_input_ids_length
                current_max_labels_length = new_max_labels_length
            else:
                self.batch_indices.append(current_batch)
                current_batch = [i]
                current_max_input_ids_length = len(data['input_ids'])
                current_max_labels_length = len(data['labels'])
                
        if current_batch:
            self.batch_indices.append(current_batch)


    def generate_batch_indices_for_single_input(self):
        self.tokenized_data.sort(key=lambda d: len(d["input_ids"]))
        if len(self.tokenized_data[0]["input_ids"]) > self.max_tokens:
            raise RuntimeError(
                "max_tokens is smaller than the length of the longest input_ids. "
                f"max_tokens: {self.max_tokens}, longest input_ids: {len(self.tokenized_data[0]['input_ids'])}"
            )

        current_batch = []
        current_max_input_ids_length = 0
        for i, data in tqdm(enumerate(self.tokenized_data)):
            new_max_input_ids_length = max(
                current_max_input_ids_length,
                len(data['input_ids'])
            )
            new_total_input_ids_num_tokens = (len(current_batch) + 1) * new_max_input_ids_length

            if new_total_input_ids_num_tokens <= self.max_tokens:
                current_batch.append(i)
                current_max_input_ids_length = new_max_input_ids_length
            else:
                self.batch_indices.append(current_batch)
                current_batch = [i]
                current_max_input_ids_length = len(data['input_ids'])
                

        if current_batch:
            self.batch_indices.append(current_batch)

    def generate_batch_indices(self):
        if self.infer_whether_seq2seq():
            logger.info("Detected seq2seq task.")
            self.generate_batch_indices_for_seq2seq()
        else:
            self.generate_batch_indices_for_single_input()
    

    def __len__(self):
        return len(self.batch_indices)
    
    def __getitem__(self, item):
        return [self.tokenized_data[i] for i in self.batch_indices[item]]

            
