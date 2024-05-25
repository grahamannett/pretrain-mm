import torch
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor
from transformers.models.paligemma.modeling_paligemma import PaliGemmaCausalLMOutputWithPast
from transformers.utils import TensorType

from pretrain_mm.processor.processor import ProcessorMixin
from pretrain_mm.processor.tokenizer_constants import SetConstants, TokenizerConstants


"""
note, for patching this model atm need to fix casting in PaliGemmaProcessor._merge_input_ids_with_image_features

I am not sure exactly which devices/where the issue arises from so ended up just casting multiple things as otherwise
debugging/fixing on borah is a pain

if edited in place on borah, location @
`/bsuhome/gannett/mambaforge/envs/pt/lib/python3.11/site-packages/transformers/models/paligemma/modeling_paligemma.py`
then remove the cast

spaces with actual working implementation of seg/detect/etc


- https://huggingface.co/spaces/big-vision/paligemma-hf
    - this one has the VAE for decoding to mask
- https://huggingface.co/spaces/big-vision/paligemma
    - this one uses the big-vision stuff which is not ideal


models:
https://huggingface.co/google/paligemma-3b-ft-docvqa-896
https://huggingface.co/google/paligemma-3b-ft-ocrvqa-896
"""


class PaliGemmaConstantsClass(TokenizerConstants):
    # from the processor.tokenizer
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    image_placeholder_token: str = "<image>"

    repr_bbox_open_text: str = "<box>"
    repr_bbox_close_text: str = "<box>"  # not </box>
    repr_point_open_text: str = "<point>"
    repr_point_close_text: str = "<point>"  # not </point>


PaliGemmaConstants = PaliGemmaConstantsClass()


class PaliGemmaForConditionalGeneration(HFPaliGemmaForConditionalGeneration):
    pass


def coords_raw_to_scaled(coords: list[str], scale_factor: float = 1.0) -> list[str]:
    """
    takes a list of string ints and scales them by a factor then returns a list of string (that are ints) to be tokenized

    goes from full size (e.g. 1920) to 1/2 size (e.g. 960)
    """

    def _scale_fn(val):
        return str(round((float(val) / 2) * scale_factor))

    return [_scale_fn(val) for val in coords]


class OCRMixin:
    def convert_text_to_model_input(self, text: str):
        pass


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
