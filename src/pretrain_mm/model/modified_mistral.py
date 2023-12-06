from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModifiedMistralModel(transformers.PreTrainedModel):
    """similar to transformers.models.mistral.MistralForCausalLM but with vision embeddings
    with the vision embedding structure is similar to transformers.models.fuyu.FuyuForCausalLM

    Args:
        transformers (_type_): _description_
    """

    def __init__(self):
        self.language_model = transformers.AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.vision_embed_tokens = nn.Linear(self.language_model.config.hidden_size)


    def combine_embeddings(word_embeddings: torch.Tensor, patch_embeddings: torch.Tensor, image_patchees_indices: torch.Tensor):
        """
        Combine word and patch embeddings

        modified version of `gather_continuous_embeddings` in `transformers.models.fuyu.FuyuForCausalLM`
        """

        for batch_idx in range(word_embeddings.shape[0]):
            dst_indices = torch.nonzero(image_patchees_indices[batch_idx] >= 0, as_tuple=True)[0]
            src_indices = image_patchees_indices[batch_idx][dst_indices]
            if src_indices.shape[0] > patch_embeddings[batch_idx].shape[0]:
                src_indices = src_indices[:patch_embeddings[batch_idx].shape[0]]
                dst_indices = dst_indices[:len(src_indices)]
            word_embeddings[batch_idx][dst_indices] = patch_embeddings[batch_idx][src_indices]
        return word_embeddings

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
            image_patch_input_indices=image_patches_indices,
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
