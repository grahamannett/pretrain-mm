import re
from functools import cache

import torch
from PIL.Image import Image
from transformers import PaliGemmaConfig as HFPaliGemmaConfig
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemmaForConditionalGeneration
from transformers import PaliGemmaProcessor as HFPaliGemmaProcessor
from transformers.utils import TensorType

from pretrain_mm.processor.processor import ProcessorMixin, TextProcessorMixin
from pretrain_mm.processor.tokenizer_constants import SetConstants, TokenizerConstants
from pretrain_mm.utils.token_tag_utils import TagType, box_pattern, point_pattern, segment_str


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

PROCESSOR_MAX_SIZE: int = 1024  # loc/seg must be scaled to this max

_r_loc = r"<loc(\d{4})>"
re_loc = re.compile(_r_loc)
re_loc_point = re.compile(_r_loc * 2)
re_loc_box = re.compile(_r_loc * 4)

re_seg = re.compile(r"<seg(\d{3})>")


def _scale_val(val: int, dim_scale_factor: int, max_size: int = PROCESSOR_MAX_SIZE):
    return min(int(val * dim_scale_factor), max_size)


def _make_seg_text(val: int, tag: str = "loc", digits: int = 4):
    return f"<{tag}{val:0>{digits}}>"


@cache
def _make_scale_dim_func(image_dim: int):
    # image_dim is either height or width
    def func(*vals: int):
        return [round((int(val) / PROCESSOR_MAX_SIZE) * image_dim) for val in vals]

    return func


class PaliGemmaConfig(HFPaliGemmaConfig):
    pass


class PaliGemmaConstantsClass(TokenizerConstants):
    # from the processor.tokenizer
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    image_placeholder_token: str = "<image>"

    repr_bbox_open_text: str = "<box>"
    repr_bbox_close_text: str = "</box>"
    repr_point_open_text: str = "<point>"
    repr_point_close_text: str = "</point>"


PaliGemmaConstants = PaliGemmaConstantsClass()


class PaliGemmaForConditionalGeneration(HFPaliGemmaForConditionalGeneration):
    pass


@SetConstants(PaliGemmaConstants)
class PaliGemmaProcessor(HFPaliGemmaProcessor, ProcessorMixin, TextProcessorMixin):
    constants: PaliGemmaConstantsClass

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
        extra: dict | bool = False,
        **kwargs,
    ):
        suffix = suffix or kwargs.get("label", None)
        if text:
            text = self.preprocess_text(text, images)

        if suffix:
            suffix = self.preprocess_text(suffix, images)

        batch = super().__call__(
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

        batch = self.create_attachable(batch, extra)(text=text, images=images, label=suffix)

        return batch

    def decode(self, outputs: torch.Tensor, do_post: bool = True, **kwargs) -> str:
        """this is specific to PaliGemma"""
        # converts the tokens to text
        outputs = self.tokenizer.decode(outputs, **kwargs)
        return outputs

    def preprocess_text(
        self,
        text: str,
        images: list[torch.Tensor | Image] | Image = None,
        max_size: int = PROCESSOR_MAX_SIZE,
    ) -> str:
        # not sure what to do for multiple images if need to scale

        if isinstance(images, list):
            images = images[0]

        if images is not None:
            image_width, image_height = images.size
            height_scale = max_size / image_height
            width_scale = max_size / image_width

        segments = segment_str(text, box_pattern=box_pattern, point_pattern=point_pattern)

        out_text = ""
        for seg, seg_type in segments:
            if seg_type:
                if seg_type == TagType.POINT:
                    x, y = map(int, seg)
                    # Scale the coordinates
                    scaled_x = _make_seg_text(_scale_val(x, width_scale, max_size))
                    scaled_y = _make_seg_text(_scale_val(y, height_scale, max_size))
                    # model uses y, x in examples
                    scaled_toks = f"{scaled_y}{scaled_x} point"
                elif seg_type == TagType.BOX:
                    x1, y1, x2, y2 = map(int, seg)
                    # Scale the coordinates
                    scaled_x1 = _make_seg_text(_scale_val(x1, width_scale, max_size))
                    scaled_y1 = _make_seg_text(_scale_val(y1, height_scale, max_size))
                    scaled_x2 = _make_seg_text(_scale_val(x2, width_scale, max_size))
                    scaled_y2 = _make_seg_text(_scale_val(y2, height_scale, max_size))
                    # they do y1, x1, y2, x2 in examples
                    scaled_toks = f"{scaled_y1}{scaled_x1}{scaled_y2}{scaled_x2} box"
                out_text += scaled_toks
            else:
                out_text += seg
        return out_text

    def handle_token_loc_seg(self, text: str, image_height: int, image_width: int):
        _scale_height = _make_scale_dim_func(image_height)
        _scale_width = _make_scale_dim_func(image_width)
        box_tags = ("<box>", "</box>")
        point_tags = ("<point>", "</point>")

        def _make_text(tag_open: str, tag_close: str, *vals):
            return (
                text[: tag_open[1]] + f"{tag_open[0]}{', '.join(map(str, vals))}{tag_close[0]}" + text[tag_close[1] :]
            )

        def _make_yx(points: list[int]):
            return _scale_height(*points[0::2]), _scale_width(*points[1::2])

        while loc_match := re_loc_box.match(text):
            start_idx, end_idx = zip(box_tags, loc_match.span())
            (y1, y2), (x1, x2) = _make_yx(list(loc_match.groups()))
            text = _make_text(start_idx, end_idx, y1, x1, y2, x2)

        while loc_match := re_loc_point.match(text):
            tag_open, tag_close = zip(point_tags, loc_match.span())
            (y1,), (x1,) = _make_yx(list(loc_match.groups()))
            text = _make_text(tag_open, tag_close, y1, x1)

        return text


MODEL_ID: str = "google/paligemma-3b-ft-docvqa-896"
