from pathlib import Path
import json
from datetime import datetime, timedelta
import os

import ujson
from logzero import logger
import torch
import lightning.pytorch as pl
from tqdm import trange
from transformers import (
    get_scheduler,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    PromptTuningInit,
    PromptTuningConfig,
    PromptEncoderConfig,
    IA3Config,
    get_peft_model,
    PeftModel,
    PeftConfig,
)

from .utils import pid_to_port

class LanguageModelPL(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("pl_module_setting")

        # lr scheduler configurations
        parser.add_argument("--lr", type=float, required=True)
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="cosine_schedule_with_warmup",
        )
        parser.add_argument(
            "--lr_scheduler_interval",
            type=str,
            default="step",
            choices=["step", "epoch"],
        )
        parser.add_argument("--num_warmup_steps", type=int)
        
        # Options for ReduceLROnPlateau
        parser.add_argument("--lr_reduce_mode", type=str, choices=["min", "max"])
        parser.add_argument("--lr_reduce_factor", type=float, default=0.1)
        parser.add_argument("--lr_reduce_patience", type=int, default=1)
        parser.add_argument("--lr_reduce_threshold", type=float, default=1e-4)
        parser.add_argument(
            "--lr_reduce_threshold_mode",
            type=str,
            default="rel",
            choices=["rel", "abs"]
        )
        parser.add_argument("--min_lr", type=float, default=0.0)
        parser.add_argument("--lr_reduce_eps", type=float, default=0.0)
        parser.add_argument("--lr_reduce_monitor", type=str, default="valid_loss")
        

        parser.add_argument(
            "--use_8bit_adam",
            action="store_true",
            help="Whether or not to use 8-bit Adam from bitsandbytes."
        )

        # AdamW configurations
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--eps", type=float, default=1e-08)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        
        # Model, Tokenizer initialization configurations
        parser.add_argument("--model_name_or_path", help="Select model name or path", type=str, required=True)
        parser.add_argument("--tokenizer_name_or_path", help="Select model name or path", type=str, required=True)
        parser.add_argument("--from_scratch", help="Select whether to use pretrained model", action="store_true")
        parser.add_argument("--label_smoothing", type=float, default=0.0)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument(
            "--attn_implementation",
            type=str,
            default="eager",
            choices=["eager", "sdpa", "flash_attention_2"],
        )
        parser.add_argument("--freeze_all_embeddings", action="store_true")

        parser.add_argument("--dropout", type=float)
        
        # Configurations for PEFT
        parser.add_argument("--peft_model_name_or_path", type=str)
        parser.add_argument("--peft_type", type=str, default=None, choices=["lora", "prefix_tuning", "prompt_tuning", "ia3", "p_tuning"])
        
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_target_modules", type=str, nargs="*", default=[])
        parser.add_argument("--lora_dropout", type=float, default=0.0)
        parser.add_argument("--lora_alpha", type=int, default=8)
        parser.add_argument("--lora_fan_in_fan_out", action="store_true")
        parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
        parser.add_argument("--lora_modules_to_save", type=str, default=[], nargs="*")
        parser.add_argument("--relora_freq", type=int)
        parser.add_argument("--relora_none_reinitilization", action="store_true")
        parser.add_argument("--reset_relora_step_count", action="store_true")
        parser.add_argument("--lora_strat_merge", action="store_true")

        
        parser.add_argument("--prefix_tuning_num_virtual_tokens", type=int)

        parser.add_argument("--prompt_tuning_num_virtual_tokens", type=int)
        parser.add_argument("--prompt_tuning_prompt_tuning_init_text", type=str)
        
        parser.add_argument("--ia3_target_modules", type=str, nargs="*", default=[])
        parser.add_argument("--ia3_feedforward_modules", type=str, nargs="*", default=[])
        parser.add_argument("--ia3_fan_in_fan_out", action="store_true")

        parser.add_argument("--p_tuning_num_virtual_tokens", type=int)
        
        
        # Evaluation configurations
        parser.add_argument("--num_beams", type=int, default=1)
        parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
        parser.add_argument("--penalty_alpha", type=float, default=0.0)
        parser.add_argument("--top_k", type=int, default=0)
        parser.add_argument("--length_penalty", type=float, default=1.0)
        parser.add_argument("--model_max_length", type=int)
        parser.add_argument("--max_new_tokens", type=int)
        parser.add_argument("--eos_token", type=str)
        parser.add_argument("--eos_token_id", type=int)
        parser.add_argument("--save_hidden_states", action="store_true")
        parser.add_argument("--save_attention", action="store_true")
        parser.add_argument("--hidden_states_save_dir", type=Path)
        parser.add_argument("--layer_save_step", type=int)
        parser.add_argument("--save_layer_idxs", type=int, nargs="*")
        parser.add_argument("--save_hideen_states_bf16", action="store_true")
        parser.add_argument("--pad_token_id", type=int)
        parser.add_argument("--shift_size", type=int)
        parser.add_argument("--source_key", type=str, default="source")
        parser.add_argument("--gold_key", type=str)
        parser.add_argument("--suffix_match_eval", action="store_true")
        parser.add_argument("--gold_char_index", type=str)
        return parser
    
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = True
        self.save_hyperparameters(config)
        self._post_init()
        
    def _post_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        assert not (self.hparams.freeze_all_embeddings and self.model_config.tie_word_embeddings), "freeze_all_embeddings is not available with tie_word_embeddings"
        assert not (self.hparams.suffix_match_eval and self.hparams.gold_char_index is not None), "suffix_match_eval and gold_char_index cannot be specified at the same time"
        
    def configure_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            trust_remote_code=True,
        )
        self.overwrite_model_config()
        
        if self.hparams.from_scratch:            
            self.model = AutoModelForCausalLM.from_config(
                config=self.model_config,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            logger.info("Learning from scrach!")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hparams.model_name_or_path,
                config=self.model_config,
                trust_remote_code=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            logger.info(f"Load pretrained model from \"{self.hparams.model_name_or_path}\"")
            print(f"Global rank: {self.global_rank}, device: {self.model.device}")

        if self.hparams.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.hparams.peft_type is not None:
            self.setup_peft()
        
        
    def overwrite_model_config(self):
        self.model_config.label_smoothing = self.hparams.label_smoothing
        if self.hparams.dropout is not None:
            self.model_config.dropout = self.hparams.dropout
            
    def setup_peft(self):
        if self.hparams.peft_model_name_or_path is not None:
            self.peft_config = PeftConfig.from_pretrained(
                self.hparams.peft_model_name_or_path
            )
            if self.peft_config.base_model_name_or_path != self.hparams.model_name_or_path:
                logger.warning(f"Model path ({self.peft_config.base_model_name_or_path}) in peft config is different from the model path in self.hparams.model_name_or_path ({self.hparams.model_name_or_path})")
            self.model = PeftModel.from_pretrained(
                self.model, self.hpams.peft_model_name_or_path
            )
        else:
            if self.hparams.peft_type == "lora":
                self.peft_config = LoraConfig(
                    r=self.hparams.lora_r,
                    target_modules=self.hparams.lora_target_modules,
                    lora_dropout=self.hparams.lora_dropout,
                    fan_in_fan_out=self.hparams.lora_fan_in_fan_out,
                    lora_alpha=self.hparams.lora_alpha,
                    bias=self.hparams.lora_bias,
                    modules_to_save=self.hparams.lora_modules_to_save,
                    inference_mode=False,
                    task_type=TaskType.CAUSAL_LM,
                )
            elif self.hparams.peft_type == "prefix_tuning":
                self.peft_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    num_virtual_tokens=self.hparams.prefix_tuning_num_virtual_tokens,
                )
            elif self.hparams.peft_type == "prompt_tuning":
                self.peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.hparams.prompt_tuning_num_virtual_tokens,
                    prompt_tuning_init_text=self.hparams.prompt_tuning_init_text,
                    tokenizer_name_or_path=self.hparams.tokenizer_name_or_path,
                    inference_mode=False,
                )
            elif self.hparams.peft_type == "ia3":
                self.peft_config = IA3Config(
                    target_modules=self.hparams.ia3_target_modules,
                    feedforward_modules=self.hparams.ia3_feedforward_modules,
                    inference_mode=False,
                    fan_in_fan_out=self.hparams.ia3_fan_in_fan_out,
                )
            elif self.hparams.peft_type == "p_tuning":
                self.peft_config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    num_virtual_tokens=self.hparams.p_tuning_num_virtual_tokens,
                )
            else:
                raise ValueError(f"Invalid peft_type: {self.peft_type}")
            self.model = get_peft_model(self.model, self.peft_config)

        if self.hparams.relora_freq is not None:
            assert self.hparams.peft_type == "lora", \
                "relora_freq is only available with lora"
            self.prevoous_relora_update_step = -1
            
        assert self.hparams.relora_freq is None or self.hparams.peft_type == "lora", "relora_freq is only available with lora"
        self.model.print_trainable_parameters()
        
    def configure_optimizers(self):
        if self.hparams.use_8bit_adam:
            # assert not self.hparams.fp16
            # optimizer_cls = bnb.optim.AdamW8bit
            raise NotImplementedError("Not implemented yet for 8-bit Adam")
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
            raise NotImplementedError("Not implemented yet for DeepSpeed")
        else:
            optimizer_cls = torch.optim.AdamW
            
        optimizer = optimizer_cls(
            self.trainer.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )


        if self.hparams.lr_scheduler == "ReduceLROnPlateau":
            lr_sheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.hparams.lr_reduce_mode,
                factor=self.hparams.lr_reduce_factor,
                patience=self.hparams.lr_reduce_patience,
                threshold=self.hparams.lr_reduce_threshold,
                threshold_mode=self.hparams.lr_reduce_threshold_mode,
                min_lr=self.hparams.min_lr,
                eps=self.hparams.lr_reduce_eps,
            )
            lr_sheduler_config = {
                "scheduler": lr_sheduler,
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1,
                "monitor": self.hparams.lr_reduce_monitor,
                "strict": True,
            }
        else:
            lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            lr_sheduler_config = {
                "scheduler": lr_scheduler,
                "interval": self.hparams.lr_scheduler_interval,
                "frequency": 1,
            }            
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_sheduler_config,
        }        
        

    def on_fit_start(self):
        if self.hparams.lora_strat_merge:
           raise NotImplementedError("Not Supported LoRa Strat Merge")
       
        if self.hparams.freeze_all_embeddings:
            embedding = self.model.get_input_embeddings()
            embedding.weight.requires_grad = False
    
    def training_step(self, batch, batch_idx=None):
        batch.pop("id", None)            
        output = self.model(**batch)
        
        self.log(
            "train_loss",
            output.loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": output.loss}        

    
    def validation_step(self, batch, batch_idx=None, dataloader_idx=0):
        batch.pop("id", None)
        batch.pop("is_number_token", None)
        
        output = self.model(**batch)
        self.log(
            "valid_loss",
            output.loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        
    # Convert from (generate_seq_len, num_layer, batch_size, seq_len, hidden_size) to (num_layers, batch_size, seq_len, hidden_size)
    def reorder_hidden_states(self, generate_hidden_states):
        reordered_states = [torch.stack(states) for states in generate_hidden_states]
        # (num_layer, batch_size, seq_len, hidden_size)
        reordered_states = torch.cat(reordered_states, dim=2)
        # (batch_size, num_layer, seq_len, hidden_size)
        return reordered_states.permute(1, 0, 2, 3)

    def on_test_epoch_start(self):
        self.test_memory = []
        if self.hparams.save_hidden_states:
            if self.hparams.hidden_states_save_dir is None:
                self.hidden_state_save_dir = self.hparams.log_dir / "hidden_states"
            else:
                self.hidden_state_save_dir = self.hparams.hidden_states_save_dir
                
            if self.trainer.is_global_zero:
                self.hidden_state_save_dir.mkdir(exist_ok=True, parents=True)

        self.prompt_ids = None
        self.prompt_attention_mask = None
        self.prompt_key_values = None

        assert self.hparams.layer_save_step is None or self.hparams.save_layer_idxs is None, "layer_save_step and save_layer_idxs cannot be specified at the same time"

        assert self.hparams.eos_token is None or self.hparams.eos_token_id is None, "eos_token and eos_token_id cannot be specified at the same time"
        if self.hparams.eos_token_id is not None:
            self.eos_token_id = self.hparams.eos_token_id
        else:
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.hparams.eos_token) if self.hparams.eos_token is not None else self.tokenizer.eos_token_id
            assert self.eos_token_id != self.tokenizer.unk_token_id, "eos_token_id is unk_token_id"

        try:
            self.tcp_port = int(os.environ["MASTER_PORT"]) + 1
        except KeyError:
            assert self.trainer.strategy.world_size == 1, "MASTER_PORT is not set"
            self.tcp_port = pid_to_port(os.getpid())
            
            
    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        max_length = self.hparams.model_max_length if self.hparams.model_max_length is not None else self.tokenizer.model_max_length

        prompt_ids = batch.pop("prompt_ids", None)
        if prompt_ids is not None:                
            if self.prompt_ids is None or (not torch.equal(prompt_ids, self.prompt_ids)):
                self.prompt_ids = prompt_ids
                self.prompt_attention_mask = batch.pop("prompt_attention_mask", None)
                output = self.model(
                    input_ids=prompt_ids,
                    attention_mask=self.prompt_attention_mask,
                    use_cache=True,
                )
                self.prompt_key_values = output.past_key_values
                
            assert self.prompt_key_values is not None
            
            attention_mask = batch.pop("attention_mask", None)
            if attention_mask is None and self.prompt_attention_mask is None:
                attention_mask = torch.ones_like(batch["input_ids"])
            
            if self.prompt_attention_mask is None:
                attention_mask = torch.cat(
                    [torch.ones_like(self.prompt_ids), attention_mask], dim=-1
                )
            else:
                attention_mask = torch.cat(
                    [self.prompt_attention_mask, attention_mask], dim=-1
                )
     
            
        else:
            self.prompt_ids = None
            self.prompt_attention_mask = None
            self.prompt_key_values = None
            attention_mask = batch.pop("attention_mask", None)
            
        if self.hparams.save_hidden_states:
            bsz = batch["input_ids"].size(0)
            if self.prompt_ids is None:
                input_ids = batch["input_ids"]
            else:
                input_ids = torch.cat([self.prompt_ids, batch["input_ids"]], dim=-1)
                
            assert attention_mask.size(-1) == input_ids.size(-1)
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.prompt_key_values,
                num_beams=self.hparams.num_beams,
                do_sample=False,
                min_length=0,
                max_length=max_length,
                max_new_tokens=self.hparams.max_new_tokens,
                num_beam_groups=1,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=0,
                length_penalty=self.hparams.length_penalty,
                eos_token_id=self.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=self.hparams.save_attention,
                use_cache=True,
            )
            
            if self.hparams.save_attention:
                raise NotImplementedError("Not implemented yet")
            
            
            hidden_states = self.reorder_hidden_states(output.hidden_states).cpu()
            decoded_ids = output.sequences

            if attention_mask is None:
                attention_mask = [None] * bsz
            else:
                attention_mask = attention_mask.cpu()
                
            for text_id, state, decoded_id, attention_mask in zip(
                    batch["id"].tolist(),
                    hidden_states,
                    decoded_ids.cpu(),
                    attention_mask,
            ):
                save_dir = self.hidden_state_save_dir / f"{text_id}"
                save_dir.mkdir(exist_ok=True, parents=True)
                if self.prompt_ids is None:
                    save_decoded_id = decoded_id
                else:
                    save_decoded_id = decoded_id[self.prompt_ids.size(-1):]
                torch.save(
                    {
                        "id": text_id,
                        "decoded_id": save_decoded_id,
                        "attention_mask": attention_mask,
                        "input_length": batch["input_ids"].size(-1),
                    },
                    save_dir / "meta_data.pt",
                )

                states_dir = save_dir / "states"
                states_dir.mkdir(exist_ok=True, parents=True)

                if self.hparams.layer_save_step is not None:
                    self.save_layer_idxs = list(range(0, state.size(0), self.hparams.layer_save_step))
                elif self.hparams.save_layer_idxs is not None:
                    self.save_layer_idxs = self.hparams.save_layer_idxs
                else:
                    self.save_layer_idxs = list(range(state.size(0)))
                    
                    
                for layer_idx, layer_state in zip(
                        self.save_layer_idxs, state[self.save_layer_idxs]
                ):
                    if self.hparams.save_hideen_states_bf16:
                        layer_state = layer_state.to(torch.bfloat16)
                    torch.save(
                        layer_state,
                        states_dir / f"{layer_idx}.pt",
                    )
                    
        else:
            if self.prompt_ids is None:
                input_ids = batch["input_ids"]
            else:
                input_ids = torch.cat([self.prompt_ids, batch["input_ids"]], dim=-1)
                
            assert attention_mask.size(-1) == input_ids.size(-1)
            decoded_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self.prompt_key_values,
                num_beams=self.hparams.num_beams,
                do_sample=False,
                min_length=0,
                max_length=max_length,
                max_new_tokens=self.hparams.max_new_tokens,
                num_beam_groups=1,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=0,
                length_penalty=self.hparams.length_penalty,
                eos_token_id=self.eos_token_id,
                use_cache=True,
            )
            
        hyp_texts = self.tokenizer.batch_decode(
            decoded_ids[:, input_ids.size(-1):].cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        for text_id, hyp_text in zip(batch["id"].tolist(), hyp_texts):
            self.test_memory.append(
                {
                    "id": text_id,
                    "hyp_text": hyp_text.strip(),
                    "data_loader_idx": dataloader_idx,
                }
            )
            
    def on_test_epoch_end(self):
        # Start Server
        self.trainer.strategy.barrier()        
        logger.info(f"Rank{self.global_rank} is waiting at port {self.tcp_port}")
        self.store = torch.distributed.TCPStore(
            host_name="localhost",
            port=self.tcp_port,
            is_master=self.trainer.is_global_zero,
            world_size=self.trainer.strategy.world_size,
            timeout=timedelta(seconds=600),
        )
        self.store.set(
            f"result_{self.global_rank}",
            ujson.dumps(self.test_memory, ensure_ascii=False, escape_forward_slashes=False),
        )
        
        # Merge
        if self.trainer.is_global_zero:
            logger.info("Rank0 is merging..")
            all_data = []
            for rank in trange(self.trainer.strategy.world_size):
                shared_data = ujson.loads(self.store.get(f"result_{rank}"))
                all_data.extend(shared_data)
                
            # Remove duplicated data
            dataloader_idx = all_data[0]["data_loader_idx"]
            all_data = list({data["id"]: data for data in all_data}.values())
            all_data = sorted(all_data, key=lambda x: x["id"])
            assert len(all_data) == len(self.trainer.datamodule.test_datasets[dataloader_idx]), \
                f"len(all_data)={len(all_data)} != len(self.trainer.datamodule.test_datasets[dataloader_idx])={len(self.trainer.datamodule.test_datasets[dataloader_idx])}"
            
            basic_info = self.trainer.datamodule.test_datasets[dataloader_idx].basic_info
            basic_info_path = self.trainer.datamodule.test_datasets[dataloader_idx].basic_info_path
            golds = []
            sources = []
            corpus_path = Path(basic_info["corpus_path"])
            with corpus_path.open(mode="r") as f:
                for line in f:
                    data = json.loads(line)
                    sources.append(data[self.hparams.source_key])

                    if self.hparams.gold_key is not None:
                        golds.append(data[self.hparams.gold_key])
                    elif "scratchpad" in data:
                        golds.append(data["scratchpad"].strip())
                    else:
                        golds.append(data["original_answer"].strip())
                        

            # Save results
            now = datetime.today().strftime("%Y%m%d%H%M%S")
            output_json = {
                "date": now,
                "corpus_path": basic_info["corpus_path"],
                "basic_info": str(basic_info_path),
                "model_name_or_path": self.hparams.model_name_or_path,
                "tokenizer_name_or_path": self.hparams.tokenizer_name_or_path,
                "result": {},
            }
            if self.hparams.save_hidden_states:
                output_json["hidden_states_dir"] = str(self.hidden_state_save_dir)
                output_json["save_hideen_states_bf16"] = self.hparams.save_hideen_states_bf16
                output_json["eos_token_id"] = self.eos_token_id
                output_json["save_layer_idxs"] = self.save_layer_idxs
                
                
            correct_count = 0
            assert len(golds) == len(all_data) == len(sources), \
                f"len(golds)={len(golds)} != len(all_data)={len(all_data)} != len(sources)={len(sources)}"
            
            for data, gold, source in zip(all_data, golds, sources):
                output_json["result"][data["id"]] = {}
                output_json["result"][data["id"]]["hyp_text"] = data["hyp_text"]
                output_json["result"][data["id"]]["gold"] = gold
                output_json["result"][data["id"]]["source"] = source
                
                if self.hparams.suffix_match_eval:
                    if gold.replace(" ", "").endswith(data["hyp_text"].replace(" ", "")):
                        correct_count += 1
                        output_json["result"][data["id"]]["correct"] = True
                    else:
                        output_json["result"][data["id"]]["correct"] = False
                elif self.hparams.gold_char_index is not None:
                    try:
                        sl_or_index = int(self.hparams.gold_char_index)
                    except ValueError:
                        start, stop = map(int, self.hparams.gold_char_index.split(":"))
                        sl_or_index = slice(start, stop)
                    
                    if gold[sl_or_index].replace(" ", "") == data["hyp_text"].replace(" ", ""):
                        correct_count += 1
                        output_json["result"][data["id"]]["correct"] = True
                    else:
                        output_json["result"][data["id"]]["correct"] = False
                else:
                    if data["hyp_text"].replace(" ", "") == gold.replace(" ", ""):
                        correct_count += 1
                        output_json["result"][data["id"]]["correct"] = True
                    else:
                        output_json["result"][data["id"]]["correct"] = False
                    
            output_json["accuracy"] = correct_count / len(all_data)
            self.log("test_accuracy", output_json["accuracy"])
            
            # Save result
            versions = self.hparams.version.replace("/", "_")
            result_file_path = self.hparams.log_dir / f"result_{versions}_{now}.json"
            with result_file_path.open(mode="w") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)
            logger.info(f"Save result to {result_file_path}")
            # Make symlink as result_latest.json
            result_latest_path = self.hparams.log_dir / "result_latest.json"
            if result_latest_path.is_symlink():
                result_latest_path.unlink()
            result_latest_path.symlink_to(result_file_path.name)
            
            self.logger.experiment.save(str(result_file_path))
            self.logger.experiment.save(output_json["corpus_path"])
                        
        self.test_memory.clear()
        

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        print(flush=True)
        max_length = self.hparams.model_max_length if self.hparams.model_max_length is not None else self.tokenizer.model_max_length
        self.tokenizer.model_max_length = max_length
        eos_token_id = self.tokenizer.convert_tokens_to_ids(self.hparams.eos_token) if self.hparams.eos_token is not None else self.tokenizer.eos_token_id
        
        while True:
            try:
                user_input = input(">> ")
            except UnicodeDecodeError:
                print("Input Unicode Error.")
                continue
            
            if user_input in ["q", "quit"]:
                break
            elif user_input == "exit":
                raise RuntimeError("exit")
            
            input_text = f"{user_input}{self.tokenizer.sep_token}"
            batch = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            decoded_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.hparams.num_beams,
                do_sample=False,
                min_length=0,
                max_length=max_length,
                max_new_tokens=self.hparams.max_new_tokens,
                num_beam_groups=1,
                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                length_penalty=self.hparams.length_penalty,
                eos_token_id=eos_token_id,
                penalty_alpha=self.hparams.penalty_alpha,
                top_k=self.hparams.top_k,
            )
            response = self.tokenizer.decode(
                decoded_ids[0, input_ids.size(-1):].cpu(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            print(f"++ {response}", flush=True)
            
