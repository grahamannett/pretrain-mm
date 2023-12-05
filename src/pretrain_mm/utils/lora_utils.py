from dataclasses import dataclass, field

import torch
from peft import LoraConfig, TaskType

from pretrain_mm.utils.config_utils import BaseConfig


@dataclass
class BaseLoraConfig(BaseConfig):
    enabled: bool = True
    bias: str = "none"  # "None"
    dropout: float = 0.05
    alpha: int = 32
    r: int = 8
    target_modules: list[str] = field(default_factory=list)


def setup_lora(model: torch.nn.Module, lora_config: BaseLoraConfig):
    lora_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
    )

    model.add_adapter(lora_peft_config)
    return model, lora_peft_config
