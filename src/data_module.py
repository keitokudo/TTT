from pathlib import Path
import concurrent.futures
from queue import Queue
import gc
import json

from logzero import logger
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from pickle_dataset import PickleDataset, PickleMaxTokensDataset
from mmap_datset_wrapper import MMapDatset, MMapMaxTokensDataset, DummyIterableDataset

class DataModulePL(pl.LightningDataModule):
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Datasets")
        parser.add_argument(
            "--train_data_file_paths",
            help="Specify train data file path",
            type=Path,
            nargs="*",
            required=True
        )
        parser.add_argument(
            "--valid_data_file_paths",
            help="Specify valid data file path",
            type=Path, nargs="*",
            required=True
        )
        parser.add_argument(
            "--test_data_file_paths",
            help="Specify test data file paths",
            nargs='*',
            type=Path,
            default=[]
        )
        parser.add_argument(
            "--batch_size",
            help="Specify batch size.",
            type=int
        )
        parser.add_argument(
            "--max_tokens",
            help="Specify number of max tokens.",
            type=int
        )
        parser.add_argument(
            "--valid_batch_size",
            help="Specify valid batch size.",
            type=int,
        )
        parser.add_argument(
            "--valid_max_tokens",
            help="Specify number of valid max tokens.",
            type=int
        )
        parser.add_argument(
            "--test_batch_size",
            help="Specify test batch size.",
            type=int,
        )
        parser.add_argument(
            "--test_max_tokens",
            help="Specify number of test max tokens.",
            type=int
        )

        parser.add_argument(
            "--num_workers",
            help="Specify number of workers.",
            type=int,
            default=0
        )
        parser.add_argument(
            "--train_data_shuffle",
            help="Specify whether shuffle train data or not",
            action="store_true"
        )
        return parent_parser

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        self.is_multiple_train_dataset = (len(self.hparams.train_data_file_paths) > 1)
        self.valid_datasets = None
        self.test_datasets = None
        assert (self.hparams.batch_size is not None) ^ (self.hparams.max_tokens is not None), "Specify either batch size or max tokens."

        self.dataset_type = self.detect_dataset_type()

        if self.dataset_type == "pickle":        
            if self.hparams.max_tokens is not None:
                self.batch_size = 1
                self.dataset_cls = lambda dataset_path: PickleMaxTokensDataset(dataset_path, self.hparams.max_tokens)
            else:
                self.batch_size = self.hparams.batch_size
                self.dataset_cls = PickleDataset
        elif self.dataset_type == "mmap":
            if self.hparams.max_tokens is not None:
                self.batch_size = 1
                self.dataset_cls = lambda dataset_path: MMapMaxTokensDataset(dataset_path, self.hparams.max_tokens)
            else:
                self.batch_size = self.hparams.batch_size
                self.dataset_cls = MMapDatset
        else:
            raise NotImplementedError(f"Unknown dataset type: {self.dataset_type}")

        
        if self.hparams.valid_max_tokens is not None:
            self.valid_batch_size = 1
        elif self.hparams.valid_batch_size is not None:
            self.valid_batch_size = self.hparams.valid_batch_size
        else:
            self.valid_batch_size = self.batch_size

        if self.hparams.test_max_tokens is not None:
            self.test_batch_size = 1
        elif self.hparams.test_batch_size is not None:
            self.test_batch_size = self.hparams.test_batch_size
        else:
            self.test_batch_size = self.batch_size


        assert (not self.is_multiple_train_dataset) or (self.hparams.reload_dataloaders_every_n_epochs == 1), "Specify reload_dataloaders_every_n_epochs = 1 when multiple train datasets are used."
   

    def detect_dataset_type(self):
        first_basic_info_path = self.hparams.train_data_file_paths[0] / "basic_info.json"
        assert first_basic_info_path.exists(), f"{first_basic_info_path} does not exist."

        with first_basic_info_path.open("r") as f:
            basic_info = json.load(f)
            first_dataset_type = basic_info.pop("dataset_type", "pickle")
            
        for path in self.hparams.train_data_file_paths[1:] + self.hparams.valid_data_file_paths + self.hparams.test_data_file_paths:
            basic_info_path_ = path / "basic_info.json"
            with basic_info_path_.open("r") as f:
                basic_info_ = json.load(f)
                dataset_type = basic_info_.pop("dataset_type", "pickle")
            assert dataset_type == first_dataset_type, \
                f"Dataset type is different. {dataset_type} != {first_dataset_type}"
        return first_dataset_type
        
            
    def prepare_data(self):
        pass
    
    def setup(self, stage):
        if stage == "fit":
            if self.is_multiple_train_dataset:
                self.train_dataset = None
            else:
                self.train_dataset = self.dataset_cls(
                    self.hparams.train_data_file_paths[0]
                )
                
        elif stage == "validate":
            self.valid_datasets = []
            for path in self.hparams.valid_data_file_paths:
                self.valid_datasets.append(
                    self.dataset_cls(path)
                )
                
        elif stage == "test":
            self.test_datasets =[]
            for path in self.hparams.test_data_file_paths:
                self.test_datasets.append(
                    self.dataset_cls(path)
                )

        elif stage == "predict":
            self.predict_datasets = []
            self.predict_datasets.append(DummyIterableDataset())
                
        else:
            raise NotImplementedError()

    def prepare_next_train_dataset(self):
        dataset_idx = self.trainer.current_epoch % len(self.hparams.train_data_file_paths)
        self.train_dataset = self.dataset_cls(
            self.hparams.train_data_file_paths[dataset_idx]
        )
        
    def train_dataloader(self):
        if self.is_multiple_train_dataset:
            self.prepare_next_train_dataset()
            
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collator(
                self.hparams,
                self.tokenizer,
                self.train_dataset.basic_info,
                split="train",
            ),
            shuffle=self.hparams.train_data_shuffle,
        )
        return dataloader
        
    def val_dataloader(self):
        if self.valid_datasets is None:
            self.setup(stage="validate")
            
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.valid_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=dataset.collator(
                    self.hparams,
                    self.tokenizer,
                    dataset.basic_info,
                    split="valid",
                ),
                shuffle=False,
            ) for dataset in self.valid_datasets
        ]
        return dataloaders
    
    def test_dataloader(self):
        if self.test_datasets is None:
            self.setup(stage="test")

        dataloaders = [
            DataLoader(
                dataset,
                batch_size=self.test_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
                collate_fn=dataset.collator(
                    self.hparams,
                    self.tokenizer,
                    dataset.basic_info,
                    split="test",
                ),
                shuffle=False,
            ) for dataset in self.test_datasets
        ]
        return dataloaders
    
    def predict_dataloader(self):
        if self.predict_datasets is None:
            self.setup(stage="predict")
            
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,
                pin_memory=True,
                collate_fn=dataset.collator(
                    self.hparams,
                    self.tokenizer,
                    dataset.basic_info,
                    split="predict",
                ),
                shuffle=False,
            ) for dataset in self.predict_datasets
        ]
        return dataloaders
