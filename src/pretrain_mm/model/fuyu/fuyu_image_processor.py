from typing import TypedDict

import torch
from PIL import Image
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor as TFuyuImageProcessor

from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.processor.image_processor_helpers import patchify_image
from pretrain_mm.processor.image_processor_mixin import (
    ChannelDimension,
    ImageProcessorMixin,
    to_channel_dimension_format,
)


class PatchSize(TypedDict):
    height: int
    width: int


class FuyuImageProcessor(TFuyuImageProcessor, ImageProcessorMixin):
    """
    Note: Use this FuyuImageProcessor as BaseFuyuImageProcessor had bugs with padding/normalizing.
    maybe it will be fixed in future but also was slow
    """

    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] = None,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
        do_pad: bool = True,
        padding_value: float = 1.0,
        padding_mode: str = "constant",
        do_normalize: bool = True,
        image_mean: float | list[float] = 0.5,
        image_std: float | list[float] = 0.5,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        # use height and width instead of singular size since its what original model does
        patch_size: PatchSize = {"height": 30, "width": 30},
        # think below are my own defined params
        num_channels: int = 3,
        target_size: dict[str, int] = VIEWPORT_SIZE_DICT,
        *args,
        **kwargs,
    ):
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_pad=do_pad,
            padding_value=padding_value,
            padding_mode=padding_mode,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            patch_size=patch_size,
            *args,
            **kwargs,
        )

        self.num_channels = num_channels
        # default size from fuyu
        # default width from original fuyu processor is 1920 but makes context length longer by a fair amount for wide image
        self.target_size = target_size

        # dont want these hardcoded but leaving for reference
        self._image_placeholder_id = 71011
        self._image_newline_id = 71019

    def get_patch_idx_from_midpoint(self, midpoint: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
        patch_col = midpoint[0] // self.patch_size["height"]
        patch_row = midpoint[1] // self.patch_size["width"]

        patches_per_row = image_size[0] // self.patch_size["width"]
        patch_idx = patch_row * patches_per_row + patch_col
        return patch_idx

    def prepare_image(
        self,
        image: list[Image.Image | torch.Tensor],
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """equivalent to preprocess on FuyuImageProcessor
        Checks image and then does resize/padding/scaling/normalization

        Args:
            image (List[Image.Image  |  torch.Tensor]): _description_
            data_format (Optional[Union[str, ChannelDimension]], optional): _description_. Defaults to ChannelDimension.FIRST.

        Returns:
            Tuple[torch.Tensor, tuple[int, int, int]]: _description_
        """
        # the base normalize/rescale/etc rely on numpy
        image, image_info = self._check_image(image, data_format=data_format)

        if self.do_resize:
            # NOTE: resize is a method on FuyuImageProcessor
            image = self._resize(image, target_size=self.target_size, image_size=image_info)

        if self.do_pad:
            target_size = {
                "height": self._calc_target_size(image_info["height"], self.patch_size["height"]),
                "width": self._calc_target_size(image_info["width"], self.patch_size["width"]),
            }

            # NOTE: pad_image is a method on FuyuImageProcessor
            image = self._pad_image(image, target_size=target_size, image_size=image_info)

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor)

        if self.do_normalize:
            # using normalize from transformers but normalize from torch seems noticeably faster
            image = self.normalize(image, mean=self.image_mean, std=self.image_std)

        # WARN: if i need to enable this remove from _check_image

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format)
            image_info["channel_format"] = data_format

        image = torch.from_numpy(image)  # [None, ...]

        return image, image_info

    def encode_image(
        self,
        image: Image.Image | torch.Tensor | str,
        image_placeholder_id: int,
        image_newline_id: int,
        attach_sizes: bool = False,
        patch_size: PatchSize = None,  # allow override
        extra: dict = {},
        **kwargs,
    ) -> FuyuBatchFeature:
        """encode is what prepares (i.e. scales/resize/normalize) then transform to patches with image ids/pos ids
        the ids to be used with tokenizer

        Args:
            image (Image.Image | torch.Tensor): _description_
            return_tensors (str, optional): _description_. Defaults to "pt".

        Returns:
            torch.Tensor: _description_
        """

        if not patch_size:
            patch_size = self.patch_size

        patch_size_height, patch_size_width = self.patch_size["height"], self.patch_size["width"]

        image, image_info = self.prepare_image(image)  # converts image and
        n_patch_cols = image.shape[-1] // patch_size_height
        n_patch_rows = image.shape[-2] // patch_size_width

        # [batch_size, num_patches, patch_dim_h, patch_dim_w, channels]
        image_patches = patchify_image(image, patch_dim_h=patch_size_height, patch_dim_w=patch_size_width)

        # since we only are dealing with 1 image at a time for time being, take out batch dim since we add extra dim
        # later to all
        bs, n, p_h, p_w, c = image_patches.shape
        if bs > 1:
            raise NotImplementedError("Batched image encoding not implemented yet")

        # not sure if its quicker to reshape or flatten(-3).squeeze(0)
        # [batch_size,num_patches, img_input_dim]
        image_patches = image_patches.reshape(n, p_h * p_w * c)

        image_ids, image_pos_ids = self._make_image_tokens(
            image_placeholder_id=image_placeholder_id or self._image_placeholder_id,
            image_newline_id=image_newline_id or self._image_newline_id,
            patch_cols=n_patch_cols,
            patch_rows=n_patch_rows,
            device=image_patches.device,
        )

        if attach_sizes:
            extra = {
                **extra,
                "image_info": image_info,
                "patch_sizes": (n_patch_cols, n_patch_rows),
            }

        return FuyuBatchFeature(
            data={
                "image_patches": image_patches,
                "input_ids": image_ids,
                "image_patches_indices": image_pos_ids,
                **extra,
            }
        )
