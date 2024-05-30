from typing import Any
import torch
import torch.nn as nn
from functools import partial
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
    """

    def __init__(self, capacity: float, block: nn.Module):
        super().__init__()
        self.block = block
        self.capacity = capacity
        self.router = TokenRouter(block.hidden_size)

        self.training_step = 0

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

                    if len(block_output) == 2:
                        processed_tokens[i][selected_mask[i]], cache = block_output
                    else:
                        processed_tokens[i][selected_mask[i]] = block_output[0]

                    processed_tokens[i][selected_mask[i]] = processed_tokens[i][selected_mask[i]] * weights[i][
                        selected_mask[i]
                    ].unsqueeze(-1)
                else:
                    processed_tokens[i][selected_mask[i]] = self.block(
                        selected_tokens.unsqueeze(0),
                        attention_mask=current_causal_mask,
                        position_ids=selected_position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs,
                    )[0] * weights[i][selected_mask[i]].unsqueeze(-1)

        output = processed_tokens + (x * (~selected_mask).unsqueeze(-1).to(x.dtype))
        return (output, cache) if cache is not None else (output,)


def _mod_func(mod: nn.Module):
    # _archs = "BloomForCausalLM", "FalconMoDForCausalLM"
    def _settr(mod_obj, layers_field, layers):
        setattr(mod_obj, layers_field, layers)

    if hasattr(mod, "language_model"):  # for multimodal like fuyu models in general
        ret_mod, ret_settr = mod.language_model.layers, partial(_settr, mod.language_model, "layers")
    elif hasattr(mod, "model"):  # for GPT models
        ret_mod, ret_settr = mod.model.layers, partial(_settr, mod.model, "layers")
    elif hasattr(mod, "transformers"):  # for ??? models
        ret_mod, ret_settr = mod.transformers.h, partial(_settr, mod.transformers, "h")
    else:
        raise ValueError("Model not supported")
    return ret_mod, ret_settr


def apply_mod_to_hf(
    model: PreTrainedModel,
    enabled: bool = True,
    mod_helper: callable = _mod_func,
    set_device: bool = True,
) -> PreTrainedModel:
    if not enabled:
        return model

    new_layers = nn.ModuleList()
    mod_layers, mod_settr_func = mod_helper(model)

    for i, layer in enumerate(mod_layers):
        if i % 2 != 0:
            new_layer = MixtureOfDepth(0.125, layer)
        else:
            new_layer = layer
            if set_device:
                new_layer.to(layer.parameters().__next__().device)

        new_layers.append(new_layer)
    mod_settr_func(new_layers)

    class_name = model.__class__.__name__
    # Insert MoD before the For
    if "For" in class_name:
        parts = class_name.split("For", 1)
        modified_class_name = parts[0] + "MoDFor" + parts[1]
    else:
        modified_class_name = "MoD" + class_name  # If it doesn't find any i prepends MoD

    model.__class__.__name__ = modified_class_name

    return model
