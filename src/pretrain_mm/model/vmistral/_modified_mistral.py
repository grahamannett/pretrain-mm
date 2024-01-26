from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from pretrain_mm.model.combine_embeddings import CombineEmbeddings


class ModifiedMistralModel(transformers.models.mistral.MistralForCausalLM):
    """similar to transformers.models.mistral.MistralForCausalLM but with vision embeddings
    with the vision embedding structure is similar to transformers.models.fuyu.FuyuForCausalLM

    Args:
        transformers (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config.patch_size = 30
        self.config.num_channels = 3
        # self.config.hidden_size

        self.vision_embed_tokens = nn.Linear(
            self.config.patch_size * self.config.patch_size * self.config.num_channels, self.config.hidden_size
        )

    def modality_merger(
        self,
        input_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        image_placeholder_idxs: list[list[tuple[int, int]]],  # (batch_idx, start, end)
        **kwargs,
    ):
        """merges the image and text embeddings

        Args:
            input_embeds (torch.Tensor): [batch_size, seq_len, hidden_size]
            image_embeds (torch.Tensor): [batch_size, num_total_patches, hidden_size]
            image_placeholder_idxs (list[list[tuple[int, int]]]): [batch_size, num_total_patches, 2]
        """

        for batch_idx, start, end in image_placeholder_idxs:
            input_embeds[batch_idx, start:end] = image_embeds[batch_idx]
        return input_embeds


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        patch_embeddings = [
            self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0)
            for patch in image_patches
        ]
        word_embeddings = self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.combine_embeddings(
            word_embeddings=word_embeddings,
            patch_embeddings=patch_embeddings,
            image_patches_indices=image_patches_indices,
        )

        outputs = self.language_model(
            input_ids=input_ids,
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

        return outputs
