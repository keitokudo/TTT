import json
import torch

def args_to_json_serializable(args):
    args_dict = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            args_dict[k] = v
        except TypeError:
            args_dict[k] = str(v)
    return args_dict


def get_torch_dtype(dtype_text:str):
    if dtype_text == "float32":
        return torch.float32
    elif dtype_text == "float16" or dtype_text == "fp16":
        return torch.float16
    elif dtype_text == "bfloat16" or dtype_text == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_text}")
