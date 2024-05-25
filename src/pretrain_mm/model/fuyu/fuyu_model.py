from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.fuyu.modeling_fuyu import FuyuForCausalLM as HFFuyuForCausalLM

from pretrain_mm import logger
from pretrain_mm.model.adapted.loss_adapter import CLMLossAdapter
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
    def __init__(self, hidden_size: int, patch_out: int, use_patch_idx_latent: int = False):
        super().__init__()

        self.use_patch_idx_latent = use_patch_idx_latent
        if use_patch_idx_latent:
            self.patch_idx_latent = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.ReLU(),
            )

        self.image_patch_layers = nn.Sequential(
            nn.Linear(hidden_size, patch_out, bias=False),
            nn.Tanh(),
        )

    def forward(self, logits, image_patches: torch.Tensor = None, image_patch_idx: torch.Tensor = None):
        patch_loss = 0

        # Patch language modeling loss
        if (image_patches is not None) and (image_patch_idx < image_patches.shape[1]):
            if self.use_patch_idx_latent:
                patch_idx_latent = self.patch_idx_latent(image_patch_idx.to(logits.device).to(logits.dtype)[None, ...])
                logits += patch_idx_latent

            # should use idk of logits or sum across time and then pass through image_patch_layers?
            # e.g. alternative is patch_logits = self.image_patch_layers(logits[:, image_patch_idx])
            logits = self.image_patch_layers(logits).sum(1)

            loss_func = nn.MSELoss()
            patch_loss = loss_func(logits.to(image_patches.device), image_patches[:, image_patch_idx])

        return patch_loss


class FuyuForCausalLM(HFFuyuForCausalLM):
    def __init__(self, config: FuyuConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self._do_patch_loss = False
        if config.patch_image_out:
            self.image_patch_out = ImagePatchOut(
                config.hidden_size,
                config.num_channels * (config.patch_size**2),
                use_patch_idx_latent=config.patch_idx_latent,
            )
            self._do_patch_loss = True

        self._forward = self.forward

        if hasattr(config, "causal_lm_loss"):
            self._forward = CLMLossAdapter(self._forward, config)

        self.forward = self.patched_forward

        # TODO: refactor above so can just use something like the following
        # if hasattr(config, "causal_lm_loss"):
        #     CLMLossAdapter.use_and_patch_forward(self)

        # make this optional to allow for easier testing
        if getattr(config, "patch_gather_continuous_embeddings", True):
            logger.warn("Patching gather_continuous_embeddings for model as HF one may be broken")
            self.gather_continuous_embeddings = self._gather_continuous_embeddings
        else:
            logger.warn("Not patching gather_continuous_embeddings for model. Likely will not work on 4+ GPU shard")

    def _gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        image_patch_input_indices = image_patch_input_indices.to(word_embeddings.device)
        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                src_indices = src_indices[: continuous_embeddings[batch_idx].shape[0]]
                dst_indices = dst_indices[: len(src_indices)]
                raise ValueError(f"{continuous_embeddings[batch_idx].shape=} does not match ")

            word_embeddings[batch_idx][dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[
                src_indices
            ]
        return word_embeddings

    def patched_forward(
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

        outputs = self._forward(
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
