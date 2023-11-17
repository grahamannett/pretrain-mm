from dataclasses import dataclass, field

import torch

torch_dtype_float16 = {"torch_dtype": torch.float16}


"""
Note: this is called ModelInfo and not ModelConfig so that it doesn't conflict with
the ModelConfig class from transformers.  It might make sense to rename this to
ModelInfo or something
"""


@dataclass
class ModelInitInfo:
    model_name: str
    model_kwargs: dict = field(default_factory=dict)
    tokenizer_kwargs: dict = field(default_factory=dict)

    ModelCls: callable = None
    ProcessorCls: callable = None
