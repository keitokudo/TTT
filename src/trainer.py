import argparse
from pathlib import Path
import os
from datetime import datetime
import json

from logzero import logger
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.plugins.io import AsyncCheckpointIO
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import models
from data_module import DataModulePL
from utils.argparse_helpers import float_or_int
from custom_strategy import get_strategy

class Trainer:
    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode.lower()
        self.current_check_point = None
        seed_everything(self.args.seed, workers=True)
        # torch.backends.cudnn.deterministic = True
        assert os.environ.get("PYTHONHASHSEED") == "0",\
            "Set enviroment variable \"PYTHONHASHSEED\" = \"0\""
        os.environ["TOKENIZERS_PARALLELISM"] = str(args.num_workers == 0).lower()
        
        cls = getattr(models, self.args.pl_model_name)
        logger.info(f"Selected model : {args.pl_model_name}")
        
        self.args.log_model_output_dir.mkdir(parents=True, exist_ok=True)
        self.args.default_root_dir.mkdir(parents=True, exist_ok=True)
        self.args.weights_save_path.mkdir(parents=True, exist_ok=True)
        self.args.log_dir.mkdir(parents=True, exist_ok=True)
        self.args.checkpoint_save_path.mkdir(parents=True, exist_ok=True)

        self.pl_trainer_setting()
        
        with self.pl_trainer.init_module():
            if self.args.load_check_point is not None:
                if str(self.args.load_check_point) == "best":
                    with (self.args.default_root_dir / "best_model_path.text").open(mode="r") as f:
                        self.args.load_check_point = Path(f.read().strip())

                logger.info(f"Load model from \"{args.load_check_point}\"")
                if self.args.load_check_point.is_dir():
                    logger.info("Detect zero checkpoint.")
                    converted_ckpt_path = self.args.load_check_point.parent / "_converted_model.ckpt"
                    if os.environ.get("NODE_RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0":
                        logger.info("Convert zero checkpoint to fp32 checkpoint.")
                        convert_zero_checkpoint_to_fp32_state_dict(
                            self.args.load_check_point,
                            converted_ckpt_path,
                        )
                    self.args.load_check_point = converted_ckpt_path

                self.pl_model = cls.load_from_checkpoint(
                    self.args.load_check_point,
                    config=args,
                    strict=False,
                    map_location="cpu",
                )
                self.current_check_point = Path(self.args.load_check_point)
            else:
                logger.info("Learning from beggning.")
                self.pl_model = cls(self.args)

            if self.args.torch_compile_mode is not None:
                self.pl_model = torch.compile(
                    self.pl_model,
                    # dynamic=True,
                    mode=self.args.torch_compile_mode,
                )


            
        self.data_module = DataModulePL(
            args, self.pl_model.tokenizer
        )
        
    
    def pl_trainer_setting(self):
        start_time = datetime.today().strftime("%Y%m%d%H%M%S")
        
        pl_logger = WandbLogger(
            save_dir=str(self.args.log_dir),
            name=f"{self.args.version}_{start_time}",
            project=self.args.project_name
        )
        
        check_point_filename = "checkpoint-{epoch:02d}-{step:02d}-{" + self.args.monitor + ":.4f}-" + start_time
        
        callbacks = [
            ModelCheckpoint(
                monitor=self.args.monitor,
                mode=self.args.stop_mode,
                verbose=True,
                dirpath=self.args.checkpoint_save_path,
                save_top_k=self.args.save_top_k,
                filename=check_point_filename,
                save_last=True,
            ),
            
            LearningRateMonitor(
                logging_interval="step",
            ),
        ]
        
        
        if self.args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=self.args.monitor,
                    min_delta=0.0,
                    patience=self.args.early_stopping_patience,
                    mode=self.args.stop_mode,
                    stopping_threshold=self.args.stopping_threshold,
                    check_on_train_epoch_end=not self.args.check_on_each_evaluation_step,
                    strict=True,
                )
            )
        else:
            logger.info("Don't use EarlyStopping!")
        

        if self.args.max_epochs == -1:
            self.args.max_epochs = None
            
        # Specify fp16
        if self.args.fp16:
            precision = f"16-{self.args.precision_mode}"
        elif self.args.bf16:
            precision = f"bf16-{self.args.precision_mode}"
        else:
            precision = 32

        if self.args.transformer_engine:
            precision = "transformer-engine"
            if self.args.fp16:
                precision += "-float16"
            else:
                assert self.args.bf16, "Specify bf16 or fp16 when using transformer-engine"
                
        logger.info(f"Precision {precision}")
        logger.info(
            f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == torch.cuda.device_count(), "Number of visible GPU error"            
        

        plugins = []
        if self.args.async_checkpointing:
            plugins.append(AsyncCheckpointIO())

        
        if torch.cuda.device_count() != 0:
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [-1]
        logger.info(f"GPU IDS: {gpu_ids}")
        
        assert (not self.args.accelerator == "cpu") or gpu_ids == [-1], \
            "Specify accelerator to cpu!"
        assert (self.args.max_steps != -1) or (self.args.max_epochs != -1)
        
        # if (self.args.strategy is not None) and self.args.strategy.startswith("deepspeed"):
        #     strategy = DeepSpeedStrategy(
        #         stage=int(self.args.strategy.split("_")[2]),
        #         initial_scale_power=4,
        #         min_loss_scale=0.001,
        #         # offload_optimizer=True,
        #         # offload_parameters=True,
        #     )
        # else:

        if self.args.strategy.startswith("custom_"):
            strategy = get_strategy(self.args.strategy)
        else:
            strategy = self.args.strategy
        
        self.pl_trainer = pl.Trainer(
            check_val_every_n_epoch=self.args.check_val_every_n_epoch,
            val_check_interval=self.args.val_check_interval,
            deterministic=False,
            callbacks=callbacks,
            devices=gpu_ids,
            accelerator=self.args.accelerator,
            strategy=strategy,
            fast_dev_run=self.args.fast_dev_run,
            log_every_n_steps=self.args.log_every_n_steps,
            max_epochs=self.args.max_epochs,
            max_steps=self.args.max_steps,
            profiler=None,
            default_root_dir=self.args.default_root_dir,
            logger=pl_logger,
            precision=precision,
            gradient_clip_val=self.args.gradient_clip_val,
            reload_dataloaders_every_n_epochs=self.args.reload_dataloaders_every_n_epochs,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            plugins=plugins,
        )

        if isinstance(self.pl_trainer.strategy, DeepSpeedStrategy) and self.args.load_check_point is not None:
            self.pl_trainer.strategy.load_full_weights = True
            
    def test(self):
        logger.info("Test start!!")
        test_result_dict = {
            "tag": self.args.tag,
            "results": []
        }
        
        if self.args.load_only_weight:
            ckpt_path = None
        else:
            ckpt_path = self.args.load_check_point

        test_results = self.pl_trainer.test(
            self.pl_model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path,
        )


        for idx, (path, test_result) in enumerate(
                zip(
                    self.args.test_data_file_paths,
                    test_results,
                )
        ):
            test_result["idx"] = idx
            test_result["dataset_path"] = str(path)
            test_result["check_point_path"] = str(self.current_check_point)
            test_result_dict["results"].append(test_result)
        
        now = datetime.today().strftime("%Y%m%d%H%M%S")
        test_result_file_name = f"test_result_{now}.json"
        with (self.args.default_root_dir / test_result_file_name).open(mode="w") as f:
            json.dump(test_result_dict, f, ensure_ascii=False, indent=4)
        
        logger.info("Test finish!!")
        return test_result_dict
    

    def train(self):
        assert self.mode == "train", "Set seelf.mode = \"train\" or \"resume\"!"
        
        # Start train!
        logger.info("Train start!!")

        if self.args.load_only_weight:
            ckpt_path = None
        else:
            ckpt_path = self.args.load_check_point
        
        self.pl_trainer.fit(
            self.pl_model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path,
        )        
        
        logger.info("Train finish!!")
        
        if self.pl_trainer.is_global_zero:
            best_model_path = str(self.pl_trainer.checkpoint_callback.best_model_path)
            logger.info(f"Best model path: {best_model_path}")
            if isinstance(self.pl_trainer.strategy, DeepSpeedStrategy):
                assert Path(best_model_path).is_dir()
                merged_ckpt_path = str(
                    self.args.checkpoint_save_path / "pl_best_model.ckpt"
                )
                convert_zero_checkpoint_to_fp32_state_dict(
                    best_model_path,
                    merged_ckpt_path,
                )
                self.pl_trainer.checkpoint_callback.best_model_path = merged_ckpt_path
                best_model_path = merged_ckpt_path
                cls = self.pl_model.__class__
                # Release memory in advance
                del self.pl_model
                self.pl_model = cls.load_from_checkpoint(
                    merged_ckpt_path,
                    strict=False,
                    map_location="cpu",

                )
            else:
                self.pl_model = self.pl_model.__class__.load_from_checkpoint(
                    best_model_path,
                    strict=False,
                    map_location="cpu",
                )
                
            with (self.args.default_root_dir / "best_model_path.text").open(mode="w") as f:
                if best_model_path == "":
                    logger.warning("No best_model_path exists...")
                else:  
                    f.write(best_model_path)
                    # os.link(best_model_path, self.args.checkpoint_save_path / "best.ckpt")
                    logger.info("Save transformers model")

            # Save transformers model
            logger.info("Save transformers model")
            model_weight_save_path = self.args.weights_save_path / self.args.pl_model_name
            model_weight_save_path.mkdir(parents=True, exist_ok=True)
            self.pl_model.model.save_pretrained(model_weight_save_path)
            logger.info(f"Save model weight to {model_weight_save_path}")

    def predict(self):
        assert self.mode == "predict", "Set seelf.mode = \"predict\"!"
        logger.info("Prediction start!!")

        if self.args.load_only_weight:
            ckpt_path = None
        else:
            ckpt_path = self.args.load_check_point
        
        self.pl_trainer.predict(
            self.pl_model,
            datamodule=self.data_module,
            ckpt_path=ckpt_path,
            # strict=False,
        )
        logger.info("Prediction finish!!")
        
    def __call__(self, train_only=False):
        # train
        self.train()
        
        if not train_only:
            if self.args.strategy is not None:
                torch.distributed.destroy_process_group()
                if self.pl_trainer.is_global_zero:
                    self.pl_trainer.training_type_plugin.num_nodes = 1
                    self.pl_trainer.training_type_plugin.num_processes = 1
                    self.pl_trainer.accelerator_connector.replace_sampler_ddp = False
                else:
                    return
            
            try:
                self.pl_model = self.pl_model.__class__.load_from_checkpoint(
                    self.pl_trainer.checkpoint_callback.best_model_path,
                    strict=False,
                )
            except Exception as e:
                print(f"({e})")
                logger.error("No checkpoint were saved...")
                return 
            
            self.current_check_point = Path(
                self.pl_trainer.checkpoint_callback.best_model_path
            )
            logger.info(f"Load best model from {self.pl_trainer.checkpoint_callback.best_model_path}")
            return self.test()
        
    
    
    @staticmethod
    def add_args(parent_parser):
        initial_parser = argparse.ArgumentParser()
        initial_parser.add_argument("--pl_model_name", help="Select pl_model_name", type=str, required=True)
        initial_args, unrecognized = initial_parser.parse_known_args()
        cls = getattr(models, initial_args.pl_model_name)
        cls.add_model_specific_args(parent_parser)
        
        parent_parser = Trainer.add_global_setting_args(parent_parser)
        parent_parser = Trainer.add_trainer_setting_args(parent_parser)
        parent_parser = Trainer.add_logger_setting_args(parent_parser)
        parent_parser = Trainer.add_callbacks_args(parent_parser)
        parent_parser = DataModulePL.add_args(parent_parser)
        return parent_parser

    
    @staticmethod
    def add_global_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("global_setting")
        parser.add_argument("--pl_model_name", help="Specify pl_model_name", type=str, required=True)
        parser.add_argument("--seed", help="Specify random seed", type=int, required=True)
        parser.add_argument("--tag", help="Specify tag.", type=str, required=True)
        parser.add_argument("--log_model_output_dir", help="Specify log_model_output_dir.", type=Path, required=True)
        parser.add_argument("--load_check_point", help="Specify checkpoint", type=Path)
        parser.add_argument("--torch_compile_mode", help="Specify torch compile mode", type=str)
        parser.add_argument("--load_only_weight", help="Specify whether to load only weight", action="store_true")
        return parent_parser


    @staticmethod
    def add_trainer_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument("--max_epochs", type=int, default=-1)
        parser.add_argument("--max_steps", type=int, default=-1)    
        parser.add_argument("--check_val_every_n_epoch", help="Specify frequency of validation steps.", type=int)
        parser.add_argument("--val_check_interval", help="Specify frequency of validation steps.", type=float_or_int, required=True)
        parser.add_argument("--default_root_dir", help="Specify default root dir.", type=Path, required=True)
        parser.add_argument("--weights_save_path", help="Specify weights save path.", type=Path, required=True)
        parser.add_argument("--fp16", help="Specify whether to use fp16", action="store_true")
        parser.add_argument("--bf16", help="Specify whether to use bf16", action="store_true")
        parser.add_argument("--precision_mode", help="Specify precision mode", type=str, choices=["mixed", "true"], default="mixed")
        parser.add_argument("--transformer_engine", help="Specify whether to use transformer-engine", action="store_true")
        parser.add_argument("--fast_dev_run", help="Specify fast_dev_run step", type=int, default=0)
        parser.add_argument("--accelerator", help="Specify accelerator", type=str, default="gpu")
        parser.add_argument("--strategy", help="Specify strategy", type=str, default="auto")
        parser.add_argument("--gradient_clip_val", help="Specify gradient_clip_val", type=float)
        parser.add_argument("--reload_dataloaders_every_n_epochs", help="Specify whether to reload dataloaders for every epoch", type=int, default=0)
        parser.add_argument("--accumulate_grad_batches", help="Specify accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--log_every_n_steps", help="Specify log_every_n_steps", type=int, default=50)
        return parent_parser
    
    @staticmethod
    def add_logger_setting_args(parent_parser):
        parser = parent_parser.add_argument_group("Logger")
        parser.add_argument("--project_name", help="Specify wandb project name.", type=str, required=True)
        parser.add_argument("--log_dir", help="Specify log dir.", type=Path, required=True)
        parser.add_argument("--version", help="Specify log version.", type=str, default="")
        return parent_parser

    @staticmethod
    def add_callbacks_args(parent_parser):
        parser = parent_parser.add_argument_group("Callbacks")
        parser.add_argument("--checkpoint_save_path", help="Specify checkpoint_save_path.", type=Path, required=True)
        parser.add_argument("--save_top_k", help="Specify checkpoint_save_path.", type=int, required=True)
        parser.add_argument("--early_stopping_patience", help="Specify early_stopping_patience.", type=int, required=True)
        parser.add_argument("--monitor", help="Specify monitor.", type=str, default="valid_loss")
        parser.add_argument("--stop_mode", help="Specify stop mode.", type=str, default="min")
        parser.add_argument("--stopping_threshold", help="Specify stopping_threshold.", type=float)
        parser.add_argument("--check_on_each_evaluation_step", help="Specify check_on_each_evaluation_step mode", action="store_true")
        parser.add_argument("--async_checkpointing", help="Specify async checkpointing mode", action="store_true")
        return parent_parser
    




def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
