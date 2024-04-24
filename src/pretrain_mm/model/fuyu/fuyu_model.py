from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.fuyu.modeling_fuyu import FuyuForCausalLM as BaseFuyuForCausalLM

from pretrain_mm.model.fuyu.fuyu_config import FuyuConfig


def _chop_model(config: FuyuConfig, num_hidden_layers: int):
    config.text_config.num_hidden_layers = num_hidden_layers
    config.num_hidden_layers = num_hidden_layers
    return config


class LossKey:
    IMAGE_PATCH_LOSS = "image_patch_loss"
    CLM = "clm"

    # other
    LOSS_KW = "loss_kwargs"


class ImagePatchOut(nn.Module):
    def __init__(self, hidden_size: int, patch_out: int):
        super().__init__()

        self.image_patch_layers = nn.Sequential(
            nn.Linear(hidden_size, patch_out),
            nn.Tanh(),
        )

    def forward(self, logits, image_patches: torch.Tensor = None, image_patch_idx: int = None):
        patch_loss = 0

        # Patch language modeling loss
        if (image_patches is not None) and (image_patch_idx < image_patches.shape[1]):
            patch_logits = self.image_patch_layers(logits[:, image_patch_idx])
            loss_func = nn.MSELoss()
            # patch_loss = loss_func(patch_logits, image_patches[:, image_patch_idx].to(patch_logits.device))
            patch_loss = loss_func(patch_logits.to(image_patches.device), image_patches[:, image_patch_idx])

        return patch_loss


class FuyuForCausalLM(BaseFuyuForCausalLM):
    def __init__(self, config: FuyuConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._do_patch_loss = False
        if config.patch_image_out:
            self.image_patch_out = ImagePatchOut(config.hidden_size, config.num_channels * (config.patch_size**2))
            self._do_patch_loss = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra={},
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        if self.training and self._do_patch_loss:
            output_hidden_states = True

        outputs = super().forward(
            input_ids=input_ids,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.training and self._do_patch_loss and (extra_loss := extra.get("extra_loss")):
            patch_loss = self.image_patch_out(
                logits=outputs.hidden_states[-1],
                image_patches=image_patches,
                image_patch_idx=extra_loss["patch_idx"],
            )

            outputs.loss += patch_loss
        return outputs
