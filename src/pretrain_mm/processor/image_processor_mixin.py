from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.image_transforms import pad, resize, to_channel_dimension_format
from transformers.image_utils import ChannelDimension, PILImageResampling


class ImageInfo(TypedDict):
    height: int
    width: int
    channels: int
    channel_format: ChannelDimension


class ImageProcessorMixin(ProcessorMixin):
    """
    in general

    method with _ before name are from me
    """

    """general processor for images.  not specific to any model."""

    def _setup_image_tokens(self, *args, **kwargs):
        """use this to set ids on class
        e.g. self._image_newline_id = tokenizer.convert_tokens_to_ids(image_newline_token)
        """
        raise NotImplementedError

    def _check_image(
        self,
        image: torch.Tensor | Image.Image | np.ndarray | str,
        data_format: ChannelDimension = ChannelDimension.FIRST,
    ) -> tuple[np.ndarray, ImageInfo]:
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        image_info: ImageInfo = self._get_image_size_dict(image.shape)

        return image, image_info

    def _end_check(self, image: np.ndarray, channel_dim: ChannelDimension = ChannelDimension.FIRST) -> np.ndarray:
        return to_channel_dimension_format(image, channel_dim)

    def _get_image_size_dict(self, image_size: tuple[int, int, int]) -> ImageInfo:
        height, width, channel = image_size

        if ((channel != 3) or (channel != 1)) and ((height == 3) or (height == 1)):
            channel, height, width = image_size

        return {
            "height": height,
            "width": width,
            "channels": channel,
        }

    def _calc_target_size(self, val: int, patch_size: int) -> int:
        """helper to calculate target size for patchify_image

        Args:
            val (int): _description_
            patch_size (int): _description_

        Returns:
            int: _description_
        """
        if val % patch_size:
            return (val + patch_size) - (val % patch_size)
        return val

    def _make_image_tokens(
        self,
        image_placeholder_id: int,
        image_newline_id: int,
        patch_rows: int,
        patch_cols: int,
        device: str = None,
        non_index_token: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create image tokens (e.g. signifying if an index of the image_patches is a patch or a newline) and position IDs for each patch.

        Args:
            image_placeholder_id (int): Placeholder token for each patch.
            image_newline_id (int): Newline token for each row.
            patch_rows (int): Number of rows of patches.
            patch_cols (int): Number of columns of patches.
            image_patches (torch.Tensor, optional): Image patches. Defaults to None.
            device (str, optional): Device to use. Defaults to None.
            non_index_token (int, optional): Non-index token. Defaults to -1.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image tokens and position IDs.
        """

        # Create image IDs with placeholder tokens
        image_ids = torch.full(
            (patch_rows, patch_cols),
            image_placeholder_id,
            dtype=torch.int32,
            device=device,
        )

        # Add newline tokens at the end of each row
        # image ids is placeholder that is used to signify the image patch before combination of text and image
        # for each patch row signify a newline, e.g. newline ids at the end of each row
        image_ids = torch.cat(
            [
                image_ids,
                torch.full((patch_rows, 1), image_newline_id, dtype=torch.int32, device=device),
            ],
            dim=1,
        )

        # Create position IDs corresponding to the patch index in the flattened patches

        # position corresponds to the patch index in the flattened patches
        # e.g. for image with patches such that patches are 3x2: [[1, 2, 3], [4,5,6]]
        image_pos_ids = torch.cat(
            [
                torch.arange(0, patch_cols * patch_rows).view(patch_rows, patch_cols),
                torch.full((patch_rows, 1), non_index_token, dtype=torch.int32, device=device),
            ],
            dim=1,
        )

        return image_ids.view(-1), image_pos_ids.view(-1)

    def _pad_image(
        self,
        image: np.ndarray,
        # image_size: tuple[int, int, int]
        target_size: dict[str, int],
        image_size: dict[str, int] = None,
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: str | ChannelDimension = None,
        input_data_format: str | ChannelDimension = None,
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
        # image_h, image_w = get_image_size(image, input_data_format)
        image_h, image_w = image_size["height"], image_size["width"]
        target_h, target_w = target_size["height"], target_size["width"]
        padding_top, padding_left = 0, 0
        padding_bottom, padding_right = target_h - image_h, target_w - image_w

        return pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

    def _resize(
        self,
        image: np.ndarray,
        target_size: dict[str, int],
        image_size: dict[str, int] = None,
        resample: Image.Resampling = PILImageResampling.BILINEAR,  # type: ignore
        **kwargs,
    ) -> np.ndarray:
        image_size = image_size or self._get_image_size_dict(image.shape)

        if (image_size["height"] < target_size["height"]) and (image_size["width"] < target_size["width"]):
            return image

        # height/width scale is the ratio of the target size to the image size
        h_scale = target_size["height"] / image_size["height"]
        w_scale = target_size["width"] / image_size["width"]
        scale_factor = min(h_scale, w_scale)

        new_h, new_w = int(image_size["height"] * scale_factor), int(image_size["width"] * scale_factor)

        return resize(
            image=image,
            size=(new_h, new_w),
            resample=resample,
            **kwargs,
        )

    def inverse_prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """inverse of prepare_image"""
        image = (image + 1) * 255 / 2
        return image

    def inverse_patchify(self, patches: torch.Tensor, original_height: int, original_width: int) -> torch.Tensor:
        """
        want a way to inverse patchify_image, this seems to work.  ideally woudl use in place like
        torch.nn.functional.fold(patches.T, output_size=(original_height, originaL_width),
        kernel_size=(patch_size, patch_size), stride=(patch_size,patch_size))

        but that seems to have some weird issue with channels (where image is kinda there but its not right)
        """
        batch_size, num_patches, patch_height, patch_width, channels = patches.shape
        patches = patches.permute(
            0, 1, 4, 2, 3
        )  # Change to (batch_size, num_patches, channels, patch_height, patch_width)

        # Calculate the number of patches along height and width
        num_patches_height = original_height // patch_height
        num_patches_width = original_width // patch_width

        # Initialize the output tensor
        reconstructed = torch.zeros((batch_size, channels, original_height, original_width), device=patches.device)

        # Loop over the patches and place them in the correct position
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                patch_idx = i * num_patches_width + j
                reconstructed[
                    :, :, i * patch_height : (i + 1) * patch_height, j * patch_width : (j + 1) * patch_width
                ] = patches[:, patch_idx]

        return reconstructed
