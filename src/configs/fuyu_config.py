from dataclasses import dataclass, field

import torch
import transformers

from pretrain_mm.utils.config_utils import ModelConfig, torch_dtype_float16

FuyuConfig = ModelConfig(
    model_name="adept/fuyu-8b",
    model_kwargs={"torch_dtype": torch.float16},
    ModelCls=transformers.models.fuyu.FuyuForCausalLM,
    ProcessorCls=transformers.models.fuyu.FuyuProcessor,
)
