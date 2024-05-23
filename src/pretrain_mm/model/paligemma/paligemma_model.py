import torch
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor
from transformers.utils import TensorType

from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstantsClass
from pretrain_mm.processor.processor import ProcessorMixin


"""
note, for patching this model atm need to fix casting in PaliGemmaProcessor._merge_input_ids_with_image_features

I am not sure exactly which devices/where the issue arises from so ended up just casting multiple things as otherwise
debugging/fixing on borah is a pain

if edited in place on borah, location @
`/bsuhome/gannett/mambaforge/envs/pt/lib/python3.11/site-packages/transformers/models/paligemma/modeling_paligemma.py`
then remove the cast
"""


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
            shift_labels = labels[..., 1:].to(logits.device)
            if input_attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                shift_attention_mask = input_attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(logits.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens

            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = torch.nn.functional.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction="mean", label_smoothing=0.0
            )
            # loss = loss_fct(flat_logits, flat_labels)
        return loss

    def forward(self, *args, **kwargs):
        labels = kwargs.pop("labels", None)
        model_output = super().forward(*args, **kwargs)
        if labels is not None:
            model_output.loss = self._loss_func(
                model_output.logits, labels, input_attention_mask=kwargs.get("attention_mask", None)
            )
            breakpoint()
        return model_output


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


class PaliGemmaConstantsClass(FuyuConstantsClass):
    pass


PaliGemmaConstants = PaliGemmaConstantsClass()

MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
