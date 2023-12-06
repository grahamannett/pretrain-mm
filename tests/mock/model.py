from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers

from transformers.modeling_outputs import CausalLMOutputWithPast


from pretrain_mm.model.combine_embed import CombineEmbeddings

"""
this model is to test the various stages of training while not using the full model/multiple GPU's

to get fuyu type processor working with another model, we either need to resize the model embedding or add tokens to the tokenizer
"""


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
        self.config = transformers.models.mistral.configuration_mistral.MistralConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        super().__init__(config=self.config, *args, **kwargs)
        self.language_model = transformers.models.mistral.modeling_mistral.MistralForCausalLM(self.config)
        self.vision_embed_tokens = nn.Linear(patch_size * patch_size * num_channels, hidden_size)
        self.combine_embeddings = CombineEmbeddings()

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
        word_embeddings = self.language_model.model.embed_tokens(input_ids)

        inputs_embeds = self.combine_embeddings(
            word_embeddings=word_embeddings,
            patch_embeddings=patch_embeddings,
            image_patches_indices=image_patches_indices,
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs
