from dataclasses import dataclass

from transformers import AutoModel, AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from torch import nn


@dataclass
class ModelConfig:
    base_model_name: str = "adept/fuyu-8b"


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
