from dataclasses import dataclass

import torch

torch_dtype_float16 = {"torch_dtype": torch.float16}


@dataclass
class ModelConfig:
    model_name: str
    model_kwargs: dict = {}
    tokenizer_kwargs: dict = {}
