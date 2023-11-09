from dataclasses import dataclass

from pretrain_mm.utils.config_utils import ModelConfig, torch_dtype_float16


@dataclass
class FuyuConfig(ModelConfig):
    model_name = "adept/fuyu-8b"
    model_kwargs = {**torch_dtype_float16}
    tokenizer_kwargs = {}
