from typing import List, Optional
import torch
from pretrain_mm import logger

from transformers import FuyuForCausalLM, PreTrainedModel, modeling_outputs


def gather_continuous_embeddings(
    self,
    word_embeddings: torch.Tensor,
    continuous_embeddings: list[torch.Tensor],
    image_patch_input_indices: torch.Tensor,
) -> torch.Tensor:
    """This function places the continuous_embeddings into the word_embeddings at the locations
    indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
    embeddings.

    Args:
        word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
        continuous_embeddings:
            Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
        shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
        indices in image_patch_input_indices for that batch element.
        image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
    """
    if word_embeddings is not None:
        logger.warn("Need to know where this is called from if we use it!")
        raise NotImplementedError("Verify that we are using this from ")

    if not (word_embeddings.shape[0] == len(continuous_embeddings)):
        continuous_embeddings = [continuous_embeddings[0] for _ in range(word_embeddings.shape[0])]
    output_embeddings = word_embeddings.clone()

    for batch_idx in range(word_embeddings.shape[0]):
        # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
        # positions in word_embeddings that we want to replace with content from continuous_embeddings.
        dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
        # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
        # want to use to replace the values in word_embeddings.
        src_indices = image_patch_input_indices[batch_idx][dst_indices]

        if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
            src_indices = src_indices[: continuous_embeddings[batch_idx].shape[0]]
            dst_indices = dst_indices[: len(src_indices)]

        output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx].to(src_indices.device)[src_indices]
    return output_embeddings


# FuyuForCausalLM.gather_continuous_embeddings = gather_continuous_embeddings


class PatchedFuyu(PreTrainedModel):
    """
    need way to use fuyu with incontext images+text intersperced
    """

    def __init__(self):
        self.main_model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

    # def forward(self, input_ids: torch.Tensor = None):
    #     inputs_embeds = self.language_model.embed_tokens(input_ids)
    def _make_position_ids(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        position_ids: torch.LongTensor,
        past_key_values_length,
        seq_length,
    ):
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        return position_ids

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: list[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> modeling_outputs.BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0)
                    for patch in image_patches
                ]
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            labels=labels,
        )
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
        return outputs
        # image patches will come in as either list[torch.Tensor] or torch.Tensor
        # if list[torch.Tensor] then they should come in as batch x [num_patches x patch_dim]
        # or
        # if torch.Tensor then [batch x image_in_seq x num_patches x patch_dim]
