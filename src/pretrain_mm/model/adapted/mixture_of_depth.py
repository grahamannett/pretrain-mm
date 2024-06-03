from functools import partial
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def _verify_selected_causal_mask(attention_mask, selected_mask):
    causal_masks = []
    for i in range(selected_mask.shape[0]):
        current_causal_mask = attention_mask[i, 0]
        current_causal_mask = current_causal_mask[selected_mask[i]][:, selected_mask[i]].unsqueeze(0).unsqueeze(0)
        causal_masks.append(current_causal_mask)
    return torch.cat(causal_masks)


def _ensure_devices(should, device, *args):
    if should:
        return [arg.to(device) if arg is not None else None for arg in args]
    return args


class MixtureOfDepth(nn.Module):
    """ """

    def __init__(
        self,
        block: nn.Module,
        capacity: float = 0.125,
        smooth_interval: int = 1000,
        skip_position_ids: bool = False,
        ensure_devices: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.block = block

        # idk
        self._capacity = capacity
        self._smooth_interval = smooth_interval
        self._train_step = 0

        self.router = nn.Linear(block.hidden_size, 1, bias=bias)

        # to be taken out if there is a way to fix the hooks on the models when using device_map='auto' after init
        self._ensure_devices = ensure_devices
        # for some models, the implementation hasnt been updated in awhile and the position_ids being used will cause
        # an error. Specifically for Fuyu but seems like any model that uses old `apply_rotary_pos_emb` will have this
        self._skip_position_ids = skip_position_ids

    def _check_device(self, device, *args):
        # CAST-PATCH if using apply_mod_to_hf after init
        if self._ensure_devices:
            return [arg.to(device) if arg is not None else None for arg in args]
        return args

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool,
        use_cache: bool,
        cache_position: torch.Tensor = None,
        **kwargs: Any,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
        if self.router.training and self._train_step < self._smooth_interval:
            # this "smooths" the capacity from 1.0 to self.capacity_end over the first 1000 steps
            capa = self.capacity + ((1 - self.capacity) * (1.0 / self.training_step))
            self._train_step += 1

        b, s, d = hidden_states.shape
        position_ids, cache_position = self._check_device(hidden_states.device, position_ids, cache_position)

        weights = self.router(hidden_states).squeeze(-1)

        k = int(capa * s)
        top_k_values, k_idxs = torch.topk(weights, k, dim=1, sorted=True)
        threshold = top_k_values[:, -1]
        selected_mask = weights > threshold.unsqueeze(-1)

        cache = None

        output = torch.zeros_like(hidden_states)
        # make sure that the number of tokens selected is the same for all the batches
        if selected_mask.sum(1).unique(return_counts=True)[1].shape[0] != 1:
            raise ValueError("The number of tokens selected is not the same for all the batches")

        # make sure that the position ids are the same for all the sequences in the batch
        if (position_ids.ndim == 2) and (position_ids.shape[0] != 1):
            raise ValueError("The position_ids should have a batch dimension of 1?")

        selected_tokens = hidden_states[selected_mask].reshape(b, -1, d)

        # you want to use masked_select for position_ids/cache_position as they are likely 1D but we need
        # the batch dimension to be the same as the selected_tokens
        selected_position_ids = torch.masked_select(position_ids, selected_mask).view(b, -1)

        if attention_mask is not None:
            if (attention_mask.ndim == 4) and (attention_mask.shape[1] != 1):
                raise ValueError("I only tested this when the attention mask 4dim has 1 as the second dim")

            # has taken me an insanely long time to figure this out..., the attention mask is likely to be
            #       [batch_size, 1, seq_len, seq_len]
            # the first mask we take, we must reshape with -1 in 1st dim as that corresponds to the new seq len
            # from there the last dim will be original seq len
            current_causal_mask = attention_mask[selected_mask[:, None]].view(b, -1, s)
            # the mask_s is the new seq len, the original seq len is the last dim, it is the same as the new seq length
            # of selected tensors above but it is easier to understand the masking if i keep it shown here
            mask_s = current_causal_mask.shape[1]
            # from here we need to select the causal mask along the last dim (the s from above) but this requires us to
            # Tensor.expand along the dim we already masked along.
            current_causal_mask = current_causal_mask[selected_mask[:, None].expand(-1, mask_s, -1)]
            # then lastly we need to reshape it since taking the mask along multiple dims causes you to lose
            # dims with boolean masking
            current_causal_mask = current_causal_mask.view(b, 1, mask_s, mask_s)
            # I believe i verified all of this but spent way too long figuring this bit out.  very frustrating
            # to make sure in future, im leaving the `_verify_selected_causal_mask(attention_mask, selected_mask)` which
            # i am more sure is correct and that should equal current_causal_mask
            if mask_s != selected_tokens.shape[1]:
                raise ValueError("The attention mask should have the same number of tokens as the selected tokens")
        else:
            current_causal_mask = None

        if cache_position is not None:
            if not self._skip_position_ids:
                kwargs["position_ids"] = selected_position_ids
            kwargs["cache_position_position"] = torch.masked_select(cache_position, selected_mask).view(b, -1)

        # this should correspond to the sequence length? wtf does it even mean to have size > 0?
        if not (selected_tokens.size(1) > 0):
            # when would this even happen?? but its in the other implementation
            # only thing i can think of is if there are no selected tokens?
            # hidden_states = torch.zeros_like(selected_tokens)
            raise ValueError("No selected tokens")
        else:
            hidden_states = self.block(
                selected_tokens,
                attention_mask=current_causal_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            if len(hidden_states) == 2:
                hidden_states, cache = hidden_states

        # feel like the better way to do this is use scatter_add but then you need indexes as int, and I am frankly
        # kind of confused by how the indexing of this stuff works when you get to 4D tensors.
        # this should be broadcastable as is though
        # confused how exactly to do that operation and seems maybe not ideal if i need to repeat
        # the indexes along the embed dim
        output[selected_mask] = (hidden_states * weights[selected_mask].view(b, -1, 1)).view(-1, d)
        output = output + (hidden_states * (~selected_mask[..., None]).to(hidden_states.dtype))

        output = (output,)

        if cache:
            output += (cache,)

        return output


def _mod_func(mod: nn.Module):
    if hasattr(mod, "language_model"):  # for multimodal like fuyu models in general
        ret_mod, ret_settr = mod.language_model.model.layers, partial(setattr, mod.language_model.model, "layers")
    elif hasattr(mod, "model"):  # for GPT models
        ret_mod, ret_settr = mod.model.layers, partial(setattr, mod.model, "layers")
    elif hasattr(mod, "transformers"):  # for ??? models
        ret_mod, ret_settr = mod.transformers.h, partial(setattr, mod.transformers, "h")
    else:
        raise ValueError("Model not supported")
    return ret_mod, ret_settr


def convert_hf_model(
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

    if update_name:
        class_name = model.__class__.__name__
        # Insert MoD before the For
        if "For" in class_name:
            parts = class_name.split("For", 1)
            modified_class_name = parts[0] + "MoDFor" + parts[1]
        else:
            modified_class_name = "MoD" + class_name  # If it doesn't find any i prepends MoD

        model.__class__.__name__ = modified_class_name

    return model
