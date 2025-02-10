from pathlib import Path
import json
from itertools import count

from logzero import logger
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from mmap_datset.indexed_dataset import MMapIndexedDataset
import collators


class MMapDatsetUnit(Dataset):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.basic_info_path = self.dataset_path / "basic_info.json"

        self._dataset = MMapIndexedDataset(str(dataset_path / "data"))
        self.sizes = self._dataset.sizes
        
        with self.basic_info_path.open(mode="r") as f:
            self.basic_info = json.load(f)
            
        # self.collator = getattr(collators, self.basic_info["collator"])
        self.name = self.basic_info["name"]
        self.shape = self.basic_info.get("shape", None)
        
    def __len__(self):
        return self._dataset.__len__()
    
    def __getitem__(self, item):
        if self.shape is not None:
            return self._dataset[item].reshape(self.shape).tolist()
        else:
            return self._dataset[item].tolist()


class MMapDatset(Dataset):
    def __init__(self, dataset_path: Path):
        self.datasets = {}
        self.basic_info_path = dataset_path / "basic_info.json"
        assert self.basic_info_path.exists(), f"There is no basic_info.json in {dataset_path}"
        for i in count():
            subdataset_path = dataset_path / f"subdataset_{i}"
            if subdataset_path.exists():
                assert (subdataset_path / "basic_info.json").exists()
                dataset = MMapDatsetUnit(subdataset_path)
                self.datasets[dataset.name] = dataset
            else:
                break
            
        if len(self.datasets) == 0:
            dataset = MMapDatsetUnit(dataset_path)
            self.datasets[dataset.name] = dataset
            
            
        with self.basic_info_path.open(mode="r") as f:
            self.basic_info = json.load(f)
        for name, dataset in self.datasets.items():
            self.basic_info[name] = dataset.basic_info
            
        self.collator = getattr(collators, self.basic_info["collator"])
        self._validate_datasets()

    def _validate_datasets(self):
        # Check whether all datasets have the same length
        lengths = [len(dataset) for dataset in self.datasets.values()]
        assert all(length == lengths[0] for length in lengths), "All datasets must have the same length."
        
    def __len__(self):
        return len(next(iter(self.datasets.values())))
    

    def __getitem__(self, item):
        instance = {
            dataset.name: dataset[item] for dataset in self.datasets.values()
        }
        instance["id"] = torch.tensor(item, dtype=torch.long)
        return instance
        
    
class MMapMaxTokensDataset(MMapDatset):
    def __init__(self, dataset_path:Path, max_tokens:int):
        super().__init__(dataset_path)
        self.max_tokens = max_tokens
        self.batch_indices = []
        self.max_token_target_name = self.basic_info["max_token_target_name"]
        if type(self.max_token_target_name) == str:
            self.max_token_target_name = [self.max_token_target_name]
        self.generate_batch_indices()
                
    def generate_batch_indices(self):
        sizes = {
            name: self.datasets[name].sizes for name in self.max_token_target_name
        }
        if all(sum(size_tuple) <= self.max_tokens for size_tuple in zip(*sizes.values())):
            logger.warning(
                "max_tokens is smaller than the length of the longest input_ids."
            )
            
        current_batch = []
        current_max_input_ids_lengths = [0] * len(sizes)
        for i, size_tuple in tqdm(enumerate(zip(*sizes.values()))):
            new_max_input_ids_lengths = [
                max(l, s)
                for l, s in zip(current_max_input_ids_lengths, size_tuple)
            ]
            new_max_input_ids_num_tokens = [
                (len(current_batch) + 1) * l
                for  l in new_max_input_ids_lengths
            ]

            if all(num_tokens <= self.max_tokens for num_tokens in new_max_input_ids_num_tokens):
                current_batch.append(i)
                current_max_input_ids_lengths = new_max_input_ids_lengths
            else:
                self.batch_indices.append(current_batch)
                current_batch = [i]
                current_max_input_ids_lengths = list(size_tuple)
                                
        if current_batch:
            self.batch_indices.append(current_batch)

            
    def __len__(self):
        return len(self.batch_indices)
        
    def __getitem__(self, item):
        batch = []
        for idx in self.batch_indices[item]:
            instance = {
                dataset.name: dataset[idx] for dataset in self.datasets.values()
            }
            instance["id"] = torch.tensor(idx, dtype=torch.long)
            batch.append(instance)
        return batch


class DummyIterableDataset(IterableDataset):
    def __init__(self):
        self.basic_info = {}
        self.collator = collators.DummyCollator
        
    def __iter__(self):
        while True:
            yield None
