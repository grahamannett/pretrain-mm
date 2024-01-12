"""
wip

"""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import pad, to_channel_dimension_format
from transformers.image_utils import ChannelDimension, get_image_size
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

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


class FuyuImageProcessor(BaseImageProcessor):
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_pad = True
        self.do_normalize = True
        self.do_rescale = True
        self.do_resize = False
        self.rescale_factor: float = 1 / 255

        self.image_mean: float = 0.5
        self.image_std: float = 0.5

        self.num_channels = 3
        self.patch_size = 30

        # default size from fuyu

        self.target_size = {"height": 1080, "width": 1920}

    def resize(self, image, size):
        pass

    def pad_image(
        self,
        image: np.ndarray,
        # image_size: tuple[int, int, int]
        size: Dict[str, int],
        original_image_size: Dict[str, int] = None,
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # image_height, image_width = get_image_size(image, input_data_format)
        image_height, image_width = original_image_size["height"], original_image_size["width"]
        target_height, target_width = size["height"], size["width"]
        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width

        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def _get_image_size(self, image_size: tuple[int, int, int]) -> dict[str, int]:
        return {
            "height": image_size[0],
            "width": image_size[1],
            "channels": image_size[2],
        }

    def _calc_target_size(self, val: int, patch_size: int) -> int:
        if val % patch_size:
            return (val + patch_size) - (val % patch_size)
        return val
        # return (val + patch_size) - (val % patch_size)

    def preprocess(
        self,
        image: List[Image.Image | torch.Tensor],
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    ) -> Tuple[torch.Tensor, tuple[int, int, int]]:
        # the base normalize/rescale/etc rely on numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if isinstance(image, Image.Image):
            image = np.array(image)

        original_image_size = self._get_image_size(image.shape)

        if self.do_resize:
            image = self.resize(image, self.image_size)

        if self.do_pad:
            target_size = {
                "height": self._calc_target_size(original_image_size["height"], self.patch_size),
                "width": self._calc_target_size(original_image_size["width"], self.patch_size),
            }

            image = self.pad_image(image, target_size, original_image_size)

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor)

        if self.do_normalize:
            image = self.normalize(image, mean=self.image_mean, std=self.image_std)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format)

        image = torch.from_numpy(image).unsqueeze(0)
        return image, original_image_size

    def patchify(self, image, patch_height: int = None, patch_width: int = None):
        patch_height = patch_height or self.patch_size
        patch_width = patch_width or self.patch_size

        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
        return patches

    # def patchify_image(self, image: torch.Tensor, patch_size: int = None, )


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

    def _encode_images(
        self, images: list[torch.Tensor | Image.Image] | list[list[torch.Tensor | Image.Image]]
    ) -> list[torch.Tensor]:
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
        # return_tensors: Optional[Union[str, TensorType]] = None,
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


import unittest


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.image = Image.open("tmp/test-ss1.png")

    def test_preprocess(self):
        image_processor = FuyuImageProcessor()
        image, original_image_size = image_processor.preprocess(self.image)

        self.assertEqual(image.shape, (3, 1080, 1280))
        self.assertEqual(image.dtype, torch.float32)

    def test_patchify(self):
        image_processor = FuyuImageProcessor()

        patch_size = 4
        test_patch = torch.ones((3, patch_size, patch_size))

        # make image with patch sizes and then stack so that i can assert the patches are in the right place
        image = torch.cat([test_patch, 2 * test_patch, 3 * test_patch], dim=2)
        image = torch.cat([image, 3 + image], dim=1)

        image = image.unsqueeze(0)
        patches = image_processor.patchify(image, 4, 4)

        self.assertTrue((patches[0, 0] == 1).all())
        self.assertTrue((patches[0, -1] == 6).all())

        image, original_image_size = image_processor.preprocess(self.image)

        patchified_image = image_processor.patchify(image)
        self.assertEqual(patchified_image.shape, (1, 1548, 2700))
        self.assertEqual(patchified_image.dtype, torch.float32)
