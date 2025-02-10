import torch

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False

    
class CollatorBase:
    def __init__(self, args, tokenizer, basic_info, split="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.basic_info = basic_info
        self.split = split
        
    @torch.no_grad()
    def __call__(self, batch_source):
        raise NotImplementedError

class LMPretrainCollator(CollatorBase):        
    @torch.no_grad()
    def __call__(self, batch_source):
        if self.args.max_tokens is not None:
            batch_source = batch_source[0]
            
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)
                
        result_batch = {}
        result_batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
        result_batch["attention_mask"] = torch.ones_like(result_batch["input_ids"])
        result_batch["labels"] = result_batch["input_ids"].clone()
        
        if "id" in batch:
            result_batch["id"] = torch.stack(batch["id"])
        return dict(result_batch)
    
class LMInstructTuningCollator(CollatorBase):
    def __init__(self, args, tokenizer, basic_info, split="train"):
        super().__init__(args, tokenizer, basic_info, split)
        if self.tokenizer.pad_token_id is None:
            if self.args.pad_token_id is not None:
                self.tokenizer.pad_token_id = self.args.pad_token_id
            else:
                assert self.tokenizer.unk_token is not None
                self.tokenizer.pad_token = self.tokenizer.unk_token
            
    @torch.no_grad()
    def __call__(self, batch_source):
        if self.args.max_tokens is not None:
            batch_source = batch_source[0]
            
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)
                
        # Concat source ids and target ids
        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        batch["input_ids"] = []
        
        for source_id, target_id in zip(source_ids, target_ids):
            if self.split == "test":
                batch["input_ids"].append(source_id)
            else:
                batch["input_ids"].append(source_id + target_id)
                
        del batch["source_ids"]
        del batch["target_ids"]

        if "prompt_ids" in batch:
            assert all(len(batch["input_ids"][0]) == len(ids) for ids in batch["input_ids"]), "All input_ids should have the same length when prompt_ids is given"
            prompt_batch = self.tokenizer.pad(
                {"input_ids": batch["prompt_ids"]},
                return_attention_mask=True,
                return_tensors="pt"
            )
        else:
            prompt_batch = None
            
        
        result_batch = self.tokenizer.pad(
            {"input_ids": batch["input_ids"]},
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        if self.split != "test":
            result_batch["labels"] = result_batch["input_ids"].clone()
            result_batch["labels"][
                result_batch["attention_mask"] == 0
            ] = -100
            for i, ids in enumerate(target_ids):
                result_batch["labels"][i, :-len(ids)] = -100

        if prompt_batch is not None:
            result_batch["prompt_ids"] = prompt_batch["input_ids"]
            result_batch["prompt_attention_mask"] = prompt_batch["attention_mask"]
            
            
        if "id" in batch:
            result_batch["id"] = torch.stack(batch["id"])
        return dict(result_batch)



class LMDynamicInterventionCollator(LMInstructTuningCollator):
    @torch.no_grad()
    def __call__(self, batch_source):
        if self.args.max_tokens is not None:
            batch_source = batch_source[0]
            
        batch = {k:[] for k in batch_source[0].keys()}
        for instance in batch_source:
            for k, v in instance.items():
                batch[k].append(v)
                
        # Concat source ids and target ids
        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]
        batch["input_ids"] = []
        
        for source_id, target_id in zip(source_ids, target_ids):
            if self.split == "test":
                batch["input_ids"].append(source_id)
            else:
                batch["input_ids"].append(source_id + target_id)
                
        del batch["source_ids"]
        del batch["target_ids"]

        if "prompt_ids" in batch:
            assert all(len(batch["input_ids"][0]) == len(ids) for ids in batch["input_ids"]), "All input_ids should have the same length when prompt_ids is given"
            prompt_batch = self.tokenizer.pad(
                {"input_ids": batch["prompt_ids"]},
                return_attention_mask=True,
                return_tensors="pt"
            )
        else:
            prompt_batch = None
            
        
        result_batch = self.tokenizer.pad(
            {"input_ids": batch["input_ids"]},
            return_attention_mask=True,
            return_tensors="pt"
        )
        if self.split != "test":
            result_batch["labels"] = result_batch["input_ids"].clone()
            result_batch["labels"][
                result_batch["attention_mask"] == 0
            ] = -100
            for i, ids in enumerate(target_ids):
                result_batch["labels"][i, :-len(ids)] = -100



        base_source_ids = batch["base_source_ids"]
        base_target_ids = batch["base_target_ids"]
        batch["base_input_ids"] = []
        
        for base_source_id, base_target_id in zip(base_source_ids, base_target_ids):
            if self.split == "test":
                batch["base_input_ids"].append(base_source_id)
            else:
                batch["base_input_ids"].append(base_source_id + base_target_id)
                
        del batch["base_source_ids"]
        del batch["base_target_ids"]
        base_result_batch = self.tokenizer.pad(
            {"input_ids": batch["base_input_ids"]},
            return_attention_mask=True,
            return_tensors="pt"
        )
        if self.split != "test":
            base_result_batch["labels"] = base_result_batch["input_ids"].clone()
            base_result_batch["labels"][
                base_result_batch["attention_mask"] == 0
            ] = -100
            for i, ids in enumerate(base_target_ids):
                base_result_batch["labels"][i, :-len(ids)] = -100

        result_batch["base_input_ids"] = base_result_batch["input_ids"]
        result_batch["base_attention_mask"] = base_result_batch["attention_mask"]
        
        if prompt_batch is not None:
            result_batch["prompt_ids"] = prompt_batch["input_ids"]
            result_batch["prompt_attention_mask"] = prompt_batch["attention_mask"]
            
        if "id" in batch:
            result_batch["id"] = torch.stack(batch["id"])
        return dict(result_batch)
    
    



    
    
class DummyCollator(CollatorBase):
    @torch.no_grad()
    def __call__(self, batch_source):
        return {
            "indexes": torch.arange(len(batch_source)),
        }
    
