from lightning.pytorch.strategies import DeepSpeedStrategy


# Default bucket sizes for DeepSpeed = 200000000 (2e8)
STRATEGY_REGISTRY = {
    "deepspeed_stage_3": {
        "cls": DeepSpeedStrategy,
        "args": {
            "stage": 3,
            "allgather_bucket_size": 2e5, 
            "reduce_bucket_size": 2e5,
            # "contiguous_gradients": True,
            "zero_allow_untested_optimizer": True,
        }
    },
    
    "deepspeed_stage_3_offload": {
        "cls": DeepSpeedStrategy,
        "args": {
            "stage": 3,
            "offload_optimizer": True,
            "offload_parameters": True,
            "allgather_bucket_size": 2e5,
            "reduce_bucket_size": 2e5,
            "zero_allow_untested_optimizer": True,
        }
    },
}

def get_strategy(name: str):
    assert name.startswith("custom_"), f"strategy name must start with 'custom_', got {name}"
    name = name[len("custom_"):]
    strategy_config = STRATEGY_REGISTRY[name]
    return strategy_config["cls"](**strategy_config["args"])



