import os
from pathlib import Path
from datetime import datetime, timedelta

from logzero import logger
import torch
from tqdm import trange
import ujson

from .lm_pl import LanguageModelPL
from .utils import pid_to_port

class ActivationPatcher:    
    def __init__(self, hidden_states_dict):
        self.hidden_states_dict = hidden_states_dict # {position_idx: hidden_states}
        self._enable = True
        
    def __call__(self, model, input, output):
        # hidden_states: (batch_size, seq_len, dim)
        if not self._enable:
            return output
        
        if type(output) is tuple:
            hidden_states = output[0]
        else:
            hidden_states = output
            
        bsz = hidden_states.size(0)
        for pos, state in self.hidden_states_dict.items():
            hidden_states[:, pos] = state.expand(
                bsz, -1
            ).to(hidden_states.dtype).to(hidden_states.device)
            
        if type(output) is tuple:
            pathed_output = (hidden_states, *output[1:])
            return pathed_output
        else:
            return hidden_states

    def enable(self):
        self._enable = True
        
    def disable(self):
        self._enable = False
        


class LanguageModelDynamicInterventionPL(LanguageModelPL):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LanguageModelPL.add_model_specific_args(parent_parser)
        parser.add_argument("--patch_start_position", type=int)
        parser.add_argument("--patch_end_position", type=int)
        parser.add_argument("--patch_start_layer", type=int)
        parser.add_argument("--patch_end_layer", type=int)
        return parser

    def included(self, rng, value):
        return rng[0] <= value <= rng[1]
    
    def set_activation_patch(self, hideen_states):
        # hideen_states: (num_layers, seq_len, batch_size, dim)
        activation_patch = {}
        
        for layer_idx, layer_states in enumerate(hideen_states):
            for position_idx, state in enumerate(layer_states):
                if not self.included(self.patch_position_range, position_idx):
                    continue
                if not self.included(self.patch_layer_range, layer_idx):
                    continue
                
                if layer_idx not in activation_patch:
                    activation_patch[layer_idx] = {}
                activation_patch[layer_idx][position_idx] = state
        
        for layer_idx, hidden_states_dict in activation_patch.items():
            self.model.add_activation_patch(
                layer_idx, ActivationPatcher(hidden_states_dict)
            )

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

        # Load activation patch config
        self.patch_position_range = (
            self.hparams.patch_start_position if self.hparams.patch_start_position is not None else 0,
            self.hparams.patch_end_position if self.hparams.patch_end_position is not None else float("inf"),
        )
        self.patch_layer_range = (
            self.hparams.patch_start_layer if self.hparams.patch_start_layer is not None else 0,
            self.hparams.patch_end_layer if self.hparams.patch_end_layer is not None else float("inf"),
        )


        try:
            self.tcp_port = int(os.environ["MASTER_PORT"]) + 1
        except KeyError:
            assert self.trainer.strategy.world_size == 1, "MASTER_PORT is not set"
            self.tcp_port = pid_to_port(os.getpid())

        self.trainer.strategy.barrier()

        
    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        max_length = self.hparams.model_max_length if self.hparams.model_max_length is not None else self.tokenizer.model_max_length
        
        
        # Inference for prompts part
        prompt_ids = batch.pop("prompt_ids", None)
        if prompt_ids is not None:                
            if self.prompt_ids is None or (not torch.equal(prompt_ids, self.prompt_ids)):
                self.prompt_ids = prompt_ids
                self.prompt_attention_mask = batch.pop("prompt_attention_mask", None)
                self.model.disable_activation_patch()
                self.prompt_key_values = self.model(
                    input_ids=prompt_ids,
                    attention_mask=self.prompt_attention_mask,
                    use_cache=True,
                ).past_key_values
                
            assert self.prompt_key_values is not None            
        else:
            self.prompt_ids = None
            self.prompt_attention_mask = None
            self.prompt_key_values = None


        # Base inference (without patching)
        self.model.disable_activation_patch()
        base_attention_mask = batch.pop("base_attention_mask", None)
        if self.prompt_attention_mask is None:
            base_attention_mask = torch.cat(
                [torch.ones_like(self.prompt_ids), base_attention_mask], dim=-1
            )
        else:
            base_attention_mask = torch.cat(
                [self.prompt_attention_mask, base_attention_mask], dim=-1
            )
            
        output = self.model(
            input_ids=batch["base_input_ids"],
            attention_mask=base_attention_mask,
            past_key_values=self.prompt_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=self.hparams.save_attention,
        )
        past_key_values = output.past_key_values
        # input_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        
        new_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat(
            [
                self.prompt_ids,
                batch["base_input_ids"],
                new_ids
            ],
            dim=-1
        )
        attention_mask = torch.cat(
            [
                base_attention_mask,
                torch.ones_like(new_ids)
            ],
            dim=-1
        )
        input_part_hidden_states = torch.stack(
            output.hidden_states
        ).transpose(0, 1).cpu()
        
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
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
        
        # hidden_states: (num_layers, batch_size, seq_len, dim)
        hidden_states = self.reorder_hidden_states(output.hidden_states).cpu()
        hidden_states = torch.cat((input_part_hidden_states, hidden_states), dim=2)
        # base_decoded_ids = torch.cat((batch["base_input_ids"], output.sequences), dim=-1)
        base_decoded_ids = output.sequences[:, self.prompt_ids.size(-1):]
        
        # Set activation patch
        self.set_activation_patch(hidden_states.permute(1, 2, 0, 3))
        
        attention_mask = batch.pop("attention_mask", None)
        if self.prompt_attention_mask is None:
            attention_mask = torch.cat(
                [torch.ones_like(self.prompt_ids), attention_mask], dim=-1
            )
        else:
            attention_mask = torch.cat(
                [self.prompt_attention_mask, attention_mask], dim=-1
            )
            
        # Intervented inference (with patching)
        if self.hparams.save_hidden_states:
            self.model.enable_activation_patch()
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                past_key_values=self.prompt_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=self.hparams.save_attention,
            )
            past_key_values = output.past_key_values
            new_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat(
                [
                    self.prompt_ids,
                    batch["input_ids"],
                    new_ids
                ],
                dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(new_ids)], dim=-1
            )
            input_part_hidden_states = torch.stack(
                output.hidden_states
            ).transpose(0, 1).cpu()
            
            self.model.disable_activation_patch()
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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
                attentions = output.attentions
                raise NotImplementedError("Not implemented yet")
            
            hidden_states = self.reorder_hidden_states(output.hidden_states).cpu()
            
            hidden_states = torch.cat((input_part_hidden_states, hidden_states), dim=2)
            decoded_ids = output.sequences[:, self.prompt_ids.size(-1):]
            
            if attention_mask is None:
                attention_mask = [None] * len(batch["input_ids"])
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
                torch.save(
                    {
                        "id": text_id,
                        "decoded_id": decoded_id,
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
            self.model.enable_activation_patch()            
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=attention_mask,
                past_key_values=self.prompt_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=self.hparams.save_attention,
            )
            
            past_key_values = output.past_key_values
            new_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat(
                [
                    self.prompt_ids,
                    batch["input_ids"],
                    new_ids
                ],
                dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(new_ids)], dim=-1
            )
            
            self.model.disable_activation_patch()
            decoded_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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
            decoded_ids = decoded_ids[:, self.prompt_ids.size(-1):]


        if self.hparams.shift_size is None:
            input_ids = batch["input_ids"]
            base_input_ids = batch["base_input_ids"]
        else:
            input_ids = batch["input_ids"][:, :-self.hparams.shift_size]
            base_input_ids = batch["base_input_ids"][:, :-self.hparams.shift_size]
            
        hyp_texts = self.tokenizer.batch_decode(
            decoded_ids[:, input_ids.size(-1):].cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        base_hyp_texts = self.tokenizer.batch_decode(
            base_decoded_ids[:, base_input_ids.size(-1):].cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        detok_source_texts = self.tokenizer.batch_decode(
            input_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        detok_base_source_texts = self.tokenizer.batch_decode(
            base_input_ids.cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        
        for text_id, detok_source, detok_base_source, hyp_text, base_hyp_text in zip(
                batch["id"].tolist(), detok_source_texts, detok_base_source_texts, hyp_texts, base_hyp_texts
        ):
            self.test_memory.append(
                {
                    "id": text_id,
                    "detok_source_text": detok_source,
                    "detok_base_source_text": detok_base_source,
                    "hyp_text": hyp_text.strip(),
                    "base_hyp_text": base_hyp_text.strip(),
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
            base_golds = []
            sources = []
            base_sources = []
            corpus_path = Path(basic_info["corpus_path"])
            with corpus_path.open(mode="r") as f:
                for line in f:
                    data = ujson.loads(line)
                    sources.append(data["source"])
                    base_sources.append(data["base_source"])
                    if "scratchpad" in data:
                        golds.append(data["scratchpad"].strip())
                    else:
                        golds.append(data["original_answer"].strip())

                    if "base_scratchpad" in data:
                        base_golds.append(data["base_scratchpad"].strip())
                    else:
                        base_golds.append(data["base_original_answer"].strip())

                        
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
                

            base_correct_count = 0
            correct_count = 0
            intervention_correct_count = 0
            assert len(golds) == len(all_data) == len(sources) == len(base_sources), \
                f"len(golds)={len(golds)} != len(all_data)={len(all_data)} != len(sources)={len(sources)} != len(base_sources)={len(base_sources)}"
            
            for data, gold, base_gold, source, base_source in zip(
                    all_data, golds, base_golds, sources, base_sources
            ):
                output_json["result"][data["id"]] = {}
                output_json["result"][data["id"]]["detok_source_text"] = data["detok_source_text"]
                output_json["result"][data["id"]]["detok_base_source_text"] = data["detok_base_source_text"]
                output_json["result"][data["id"]]["hyp_text"] = data["hyp_text"]
                output_json["result"][data["id"]]["base_hyp_text"] = data["base_hyp_text"]
                output_json["result"][data["id"]]["gold"] = gold
                output_json["result"][data["id"]]["base_gold"] = base_gold
                output_json["result"][data["id"]]["source"] = source
                output_json["result"][data["id"]]["base_source"] = base_source
                
                if data["base_hyp_text"].replace(" ", "") == base_gold.replace(" ", ""):
                    base_correct_count += 1
                    output_json["result"][data["id"]]["base_correct"] = True
                else:
                    output_json["result"][data["id"]]["base_correct"] = False

                if data["hyp_text"].replace(" ", "") == gold.replace(" ", ""):
                    correct_count += 1
                    output_json["result"][data["id"]]["correct"] = True
                else:
                    output_json["result"][data["id"]]["correct"] = False
                    
                if data["hyp_text"].replace(" ", "") == base_gold.replace(" ", ""):
                    intervention_correct_count += 1
                    output_json["result"][data["id"]]["intervention_correct"] = True
                else:
                    output_json["result"][data["id"]]["intervention_correct"] = False
            
                    
            output_json["base_accuracy"] = base_correct_count / len(all_data)
            self.log("test_base_accuracy", output_json["base_accuracy"])
            output_json["accuracy"] = correct_count / len(all_data)
            self.log("test_accuracy", output_json["accuracy"])
            output_json["intervention_accuracy"] = intervention_correct_count / len(all_data)
            self.log("test_intervention_accuracy", output_json["intervention_accuracy"])
            
            # Save result
            versions = self.hparams.version.replace("/", "_")
            result_file_path = self.hparams.log_dir / f"result_{versions}_{now}.json"
            with result_file_path.open(mode="w") as f:
                ujson.dump(
                    output_json,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    escape_forward_slashes=False
                )
            # Make symlink as result_latest.json
            result_latest_path = self.hparams.log_dir / "result_latest.json"
            if result_latest_path.is_symlink():
                result_latest_path.unlink()
            result_latest_path.symlink_to(result_file_path.name)
            
            self.logger.experiment.save(str(result_file_path))
            self.logger.experiment.save(output_json["corpus_path"])
                        
        self.test_memory.clear()
        
