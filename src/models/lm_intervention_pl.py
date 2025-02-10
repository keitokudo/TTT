from pathlib import Path
import json
from datetime import datetime

from logzero import logger
import torch
from tqdm import trange

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
        


class LanguageModelInterventionPL(LanguageModelPL):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LanguageModelPL.add_model_specific_args(parent_parser)
        parser.add_argument("--activation_patch_path", type=Path)
        return parser


    def set_activation_patch(self, activation_patch_path):
        # {layer_idx: {position_idx: hidden_states}}
        activation_patch = torch.load(activation_patch_path / "activation_patch.pt")
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
            
        # Load activation patch
        if self.hparams.activation_patch_path is not None:
            # {layer_idx: {position_idx: hidden_states}}
            self.set_activation_patch(
                self.hparams.activation_patch_path
            )
        self.trainer.strategy.barrier()

        
    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        max_length = self.hparams.model_max_length if self.hparams.model_max_length is not None else self.tokenizer.model_max_length
        

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
            input_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(input_ids)], dim=-1
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
            decoded_ids = torch.cat((batch["input_ids"], output.sequences), dim=-1)
            
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
            input_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(input_ids)], dim=-1
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
            decoded_ids = torch.cat((batch["input_ids"], decoded_ids), dim=-1)
            
        hyp_texts = self.tokenizer.batch_decode(
            decoded_ids[:, batch["input_ids"].size(-1):].cpu(),
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
        share_file_path = self.hparams.log_dir / f"share_{self.trainer.global_rank}.jsonl"
        with share_file_path.open(mode="w") as f:
            for data in self.test_memory:
                print(json.dumps(data), file=f)
                
        # sync process
        logger.info(f"Rank{self.global_rank} is waiting..")
        self.trainer.strategy.barrier()
        if self.trainer.is_global_zero:
            logger.info("Rank0 is merging..")
            all_data = []
            for rank in trange(self.trainer.strategy.world_size):
                share_file_path = self.hparams.log_dir / f"share_{rank}.jsonl"
                with share_file_path.open(mode="r") as f_share:
                    for line in f_share:
                        all_data.append(json.loads(line))
            
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
                    sources.append(data["source"])
                    if "scratchpad" in data:
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
            if self.hparams.activation_patch_path is not None:
                output_json["activation_patch_path"] = str(self.hparams.activation_patch_path)
                
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
                
                if data["hyp_text"].replace(" ", "") == gold.replace(" ", ""):
                    correct_count += 1
                    output_json["result"][data["id"]]["correct"] = True
                else:
                    output_json["result"][data["id"]]["correct"] = False
                    
            output_json["accuracy"] = correct_count / len(all_data)
            self.log("test_accuracy", output_json["accuracy"])
            
            # Save result
            result_file_path = self.hparams.log_dir / f"result_{now}.json"
            with result_file_path.open(mode="w") as f:
                json.dump(output_json, f, indent=4, ensure_ascii=False)
            # Make symlink as result_latest.json
            result_latest_path = self.hparams.log_dir / "result_latest.json"
            if result_latest_path.is_symlink():
                result_latest_path.unlink()
            result_latest_path.symlink_to(result_file_path.name)
            
            self.logger.experiment.save(str(result_file_path))
            self.logger.experiment.save(output_json["corpus_path"])
                        
        self.test_memory.clear()
        
