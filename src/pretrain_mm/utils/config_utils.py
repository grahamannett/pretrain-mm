from dataclasses import dataclass, field

import torch

torch_dtype_float16 = {"torch_dtype": torch.float16}


@dataclass
class ModelConfig:
    model_name: str
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)

    ModelCls: callable = None
    ProcessorCls: callable = None
