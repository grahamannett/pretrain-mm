from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn as nn
import transformers

from transformers.modeling_outputs import CausalLMOutputWithPast


from pretrain_mm.model.fuyu import CombineEmbeddings

"""
this model is to test the various stages of training while not using the full model/multiple GPU's

to get fuyu type processor working with another model, we either need to resize the model embedding or add tokens to the tokenizer
"""

from transformers import FuyuConfig

_mistral_target_modules = ["q_proj", "k_proj", "v_proj"]
_fuyu_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]


class MockModel(transformers.PreTrainedModel):
    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        num_key_value_heads: int = 2,
        patch_size: int = 30,
        num_channels: int = 3,
        *args,
        **kwargs,
    ):
        # self.config = transformers.models.mistral.configuration_mistral.MistralConfig(
        self.config = transformers.models.fuyu.configuration_fuyu.FuyuConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        super().__init__(config=self.config, *args, **kwargs)
        self.model = transformers.models.fuyu.modeling_fuyu.FuyuForCausalLM(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
