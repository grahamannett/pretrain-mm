"""
wip

"""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from transformers import ProcessorMixin, TensorType
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature

from PIL import Image

import torch


TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>


class FuyuConstants:
    text_repr_bbox_open = TEXT_REPR_BBOX_OPEN
    text_repr_bbox_close = TEXT_REPR_BBOX_CLOSE
    text_repr_point_open = TEXT_REPR_POINT_OPEN
    text_repr_point_close = TEXT_REPR_POINT_CLOSE

    token_bbox_open_string = TOKEN_BBOX_OPEN_STRING
    token_bbox_close_string = TOKEN_BBOX_CLOSE_STRING
    token_point_open_string = TOKEN_POINT_OPEN_STRING
    token_point_close_string = TOKEN_POINT_CLOSE_STRING

    boa_string: str = BEGINNING_OF_ANSWER_STRING

    eos_string: str = "|ENDOFTEXT|"
    image_newline_string: str = "|NEWLINE|"
    image_placeholder_string: str = "|SPEAKER|"

class _ModifiedFuyuProcessor(ProcessorMixin):
    # the original FuyuProcessor is not good
    # need to test against https://github.com/huggingface/transformers/blob/main/tests/models/fuyu/test_processing_fuyu.py
    # interleaved should be like sample = ["here is the image", image1, "here is another image", image2]

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"
    constants = FuyuConstants

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1

    def _encode_images(self, images: list[torch.Tensor | Image.Image] | list[list[torch.Tensor | Image.Image]]) -> list[torch.Tensor]:

        if isinstance(images[0], list):
            return [self._encode_images(imgs) for imgs in images]
        return [self.image_processor.preprocess(image, return_tensors="pt") for image in images]

    def __call__(
        self,
        text=None,
        images=None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        is_interleaved: bool = False,
        **kwargs,
    ) -> "FuyuBatchFeature":

        # help when its single of each
        if text is not None and isinstance(text, str):
            text = [text]
        if images is not None and isinstance(images, Image.Image):
            images = [images]

        if isinstance(images[0], list):
            # images are interleaved
            is_interleaved = True

        image_encoding = self.image_processor.preprocess(images, return_tensors="pt")

        # allow interleaved images
        # if isinstance(images, list) and :
