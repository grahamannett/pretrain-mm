import torch
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast
from transformers.utils import TensorType

from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstantsClass
from pretrain_mm.processor.processor import ProcessorMixin
from pretrain_mm.processor.tokenizer_constants import SetConstants


"""
note, for patching this model atm need to fix casting in PaliGemmaProcessor._merge_input_ids_with_image_features

I am not sure exactly which devices/where the issue arises from so ended up just casting multiple things as otherwise
debugging/fixing on borah is a pain

if edited in place on borah, location @
`/bsuhome/gannett/mambaforge/envs/pt/lib/python3.11/site-packages/transformers/models/paligemma/modeling_paligemma.py`
then remove the cast
"""


class PaliGemmaConstantsClass(FuyuConstantsClass):
    pass


PaliGemmaConstants = PaliGemmaConstantsClass()


class PaliGemmaForConditionalGeneration(HFPaliGemmaForConditionalGeneration):
    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, labels, token_type_ids, cache_position
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min

        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # expand masks to match embedding dimension
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim).to(final_embedding.device)  # CATCHPATCH
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim).to(final_embedding.device)  # CASTPATCH
        # insert padding and text token embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        # insert image embeddings - the image mask is always less or equal to the sentence in length
        image_mask = image_mask.to(final_embedding.device)  # CASTPATCH
        scaled_image_features = scaled_image_features.to(final_embedding.device)  # CASTPATCH
        final_embedding = final_embedding.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(final_embedding), scaled_image_features
        )
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
        if attention_mask is not None:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1)
        else:
            position_ids = None

        if token_type_ids is not None and labels is not None:
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            target_length = cache_position[-1] + 1
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                causal_mask = causal_mask.to(attention_mask.device)  # CASTPATCH
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                # unmask the prefill
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :] == 0, 0
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

            final_labels = torch.full(
                (batch_size, sequence_length), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
            final_labels = torch.where(input_ids != self.pad_token_id, labels, final_labels)
        else:
            causal_mask = attention_mask.unsqueeze(1).unsqueeze(2) * attention_mask.unsqueeze(1).unsqueeze(-1)
            causal_mask = causal_mask.to(dtype).expand(-1, self.config.text_config.num_key_value_heads, -1, -1)
            final_labels = None
        return final_embedding, causal_mask, final_labels, position_ids

    def _loss_func(self, logits, labels, input_attention_mask: torch.Tensor = None):
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if input_attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                shift_attention_mask = input_attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens

            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = torch.nn.functional.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=-100,
                reduction="mean",
                label_smoothing=0.0,
            )
        return loss

    def ideal_forward(self, *args, **kwargs):
        """
        not possible to use this given AlignDevicesHook with device_map='auto' as it puts the
        logits and past_key_values on the model_outputs rather than the loss
        """
        labels = kwargs.pop("labels", None)
        model_output = super().forward(*args, **kwargs)
        if labels is not None:
            model_output.loss = self._loss_func(
                model_output.logits, labels, input_attention_mask=kwargs.get("attention_mask", None)
            )
        return model_output

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        token_type_ids: torch.LongTensor = None,
        cache_position: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # the attention mask is turned 4d after, we keep track of the original one
        input_attention_mask = attention_mask

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
                selected_image_feature = image_outputs.last_hidden_state
                image_features = self.multi_modal_projector(selected_image_feature)

                if cache_position is None:
                    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels, token_type_ids, cache_position
                )

            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    # TODO @molbap this will only work for dynamic cache.
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = cache_position[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses PaliGemma+ Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        attention_mask = attention_mask.to(inputs_embeds.dtype)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = outputs.logits
        logits = logits.float()
        loss = self._loss_func(logits, labels, input_attention_mask=input_attention_mask)
        # loss = None
        # if labels is not None:
        #     shift_logits = logits[..., :-1, :]
        #     shift_labels = labels[..., 1:]
        #     if input_attention_mask is not None:
        #         # we use the input attention mask to shift the logits and labels, because it is 2D.
        #         shift_attention_mask = input_attention_mask[..., 1:]
        #         shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
        #         shift_labels = shift_labels[shift_attention_mask.to(logits.device) != 0].contiguous()
        #     else:
        #         shift_logits = shift_logits.contiguous()
        #         shift_labels = shift_labels.contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()

        #     flat_logits = shift_logits.view(-1, self.config.vocab_size)
        #     flat_labels = shift_labels.view(-1).to(shift_logits.device)
        #     loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@SetConstants(PaliGemmaConstants)
class PaliGemmaProcessor(HFPaliGemmaProcessor, ProcessorMixin):
    def __call__(
        self,
        text=None,
        images=None,
        tokenize_newline_separately=True,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=TensorType.PYTORCH,
        do_resize=None,
        do_normalize=None,
        image_mean=None,
        image_std=None,
        data_format="channels_first",
        input_data_format=None,
        resample: "PILImageResampling" = None,  # noqa: F821 # type: ignore
        do_convert_rgb: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_rescale: bool = None,
        suffix=None,
        **kwargs,
    ):
        suffix = suffix or kwargs.get("label", None)
        return super().__call__(
            text=text,
            images=images,
            tokenize_newline_separately=tokenize_newline_separately,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            do_resize=do_resize,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
            input_data_format=input_data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
            do_thumbnail=do_thumbnail,
            do_align_long_axis=do_align_long_axis,
            do_rescale=do_rescale,
            suffix=suffix,
        )


MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
