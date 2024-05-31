from functools import partial
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel


class TokenRouter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight_predictor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = self.weight_predictor(x).squeeze(-1)  # [batch_size, seq_len]
        return weights


class MixtureOfDepth(nn.Module):
    """
    https://github.com/astramind-ai/Mixture-of-depths/tree/main

    much clearer implementation here:
        from https://github.com/Mixture-AI/Mixture-of-Depths/blob/master/MoD/MoD.py
    but not sure if it is correct and does not handle attention_mask,position_ids,past_key_value,cache,etc

    things i changed:
    - casting to device for position_ids and cache_position
        - this could be avoided if you use this within the models __init__ since the reason it is necessary
            is because the hooks related to device_map='auto' on transformers are already set
    - allowing skip_position_ids to be passed in
        - this avoids errors with models that dont have the recent version of `apply_rotary_pos_emb` where the
            position_ids used might correspond to indexes that are not in the selected_tokens
        - i am not sure if this will impact the implementation as the recent llama model does it this way but models
            that are from before 2024 it seems to be common
    """

    def __init__(
        self, block: nn.Module, capacity: float = 0.125, skip_position_ids: bool = False, ensure_devices: bool = True
    ):
        super().__init__()
        self.block = block
        self.capacity = capacity
        self.router = TokenRouter(block.hidden_size)

        self.training_step = 0
        # for some models, the implementation hasnt been updated in awhile and the position_ids being used will cause
        # an error. Specifically for Fuyu but seems like any model that uses old `apply_rotary_pos_emb` will have this
        self._skip_position_ids = skip_position_ids
        # to be taken out if there is a way to fix the hooks on the models when using device_map='auto'
        self._ensure_devices = ensure_devices

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
        use_cache: bool,
        cache_position: torch.Tensor = None,
        **kwargs: Any,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
        b, s, d = x.shape
        weights = self.router(x)

        if self.router.training:
            self.training_step += 1 if self.training_step < 1000 else 999
            self.capacity = 0.125 + ((1 - 0.125) * (1.0 / self.training_step))

        k = int(self.capacity * s)
        top_k_values, _ = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, -1]
        selected_mask = weights > threshold.unsqueeze(-1)
        cache = None

        processed_tokens = torch.zeros_like(x)

        for i in range(b):
            current_selected_mask = selected_mask[i]
            selected_tokens = x[i][current_selected_mask]
            if self._ensure_devices:
                position_ids = position_ids.to(x.device)  # CAST-PATCH if using apply_mod_to_hf after init
            selected_position_ids = position_ids[i][current_selected_mask].unsqueeze(0)
            if attention_mask is not None:
                current_causal_mask = attention_mask[i, 0]
                current_causal_mask = (
                    current_causal_mask[current_selected_mask][:, current_selected_mask].unsqueeze(0).unsqueeze(0)
                )  # first if for the one second is for the bs
            else:
                current_causal_mask = None

            if selected_tokens.size(0) > 0:
                # Dynamic cache management
                if cache_position is not None:
                    if self._ensure_devices:
                        cache_position = cache_position.to(x.device)  # CAST-PATCH if using apply_mod_to_hf after init
                    selected_cache_position = cache_position[selected_mask[i]]
                    block_output = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,
                        position_ids=selected_position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=selected_cache_position,
                        **kwargs,
                    )
                else:
                    # warning: this wont work as is for many models since
                    # selected_position_ids shape is less than selected_tokens
                    # not clear what the impact of this is but it allows model to work
                    block_output = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,
                        position_ids=None if self._skip_position_ids else selected_position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )

                if len(block_output) == 2:
                    processed_tokens[i][selected_mask[i]], cache = block_output
                else:
                    processed_tokens[i][selected_mask[i]] = block_output[0]

                processed_tokens[i][selected_mask[i]] = processed_tokens[i][selected_mask[i]] * weights[i][
                    selected_mask[i]
                ].unsqueeze(-1)

        output = processed_tokens + (x * (~selected_mask).unsqueeze(-1).to(x.dtype))
        return (output, cache) if cache is not None else (output,)


def _mod_func(mod: nn.Module):
    # _archs = "BloomForCausalLM", "FalconMoDForCausalLM"
    if hasattr(mod, "language_model"):  # for multimodal like fuyu models in general
        ret_mod, ret_settr = mod.language_model.model.layers, partial(setattr, mod.language_model.model, "layers")
    elif hasattr(mod, "model"):  # for GPT models
        ret_mod, ret_settr = mod.model.layers, partial(setattr, mod.model, "layers")
    elif hasattr(mod, "transformers"):  # for ??? models
        ret_mod, ret_settr = mod.transformers.h, partial(setattr, mod.transformers, "h")
    else:
        raise ValueError("Model not supported")
    return ret_mod, ret_settr


def apply_mod_to_hf(
    model: PreTrainedModel,
    enabled: bool = True,
    skip_position_ids: bool = False,
    mod_helper: callable = _mod_func,
    update_name: bool = True,
    device: bool | str = True,
) -> PreTrainedModel:
    if not enabled:
        return model

    new_layers = nn.ModuleList()
    mod_layers, mod_settr_func = mod_helper(model)

    for i, layer in enumerate(mod_layers):
        if i % 2 != 0:
            new_layer = MixtureOfDepth(layer, 0.125, skip_position_ids=skip_position_ids)

            if device is True:
                new_layer.to(layer.parameters().__next__().device)
            elif isinstance(device, str):
                new_layer.to(device)

        else:
            new_layer = layer

        new_layers.append(new_layer)
    mod_settr_func(new_layers)

    if update_name:  # idk why they do this?
        class_name = model.__class__.__name__
        # Insert MoD before the For
        if "For" in class_name:
            parts = class_name.split("For", 1)
            modified_class_name = parts[0] + "MoDFor" + parts[1]
        else:
            modified_class_name = "MoD" + class_name  # If it doesn't find any i prepends MoD

        model.__class__.__name__ = modified_class_name

    return model
