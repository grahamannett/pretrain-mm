from enum import StrEnum, auto
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from einops import rearrange


class ChannelDimension(StrEnum):
    FIRST = auto()
    LAST = auto()


def resize_to_patch_divisable(image: torch.Tensor, x: int, y: int):
    b, c, h, w = image.shape
    new_h = ((h - 1) // x + 1) * x
    new_w = ((w - 1) // y + 1) * y
    return F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)


def image_to_patches(image: torch.Tensor, p1: int, p2: int) -> torch.Tensor:
    return rearrange(image, "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=p1, p2=p2)


def make_patch_indices(patches: torch.Tensor, original_w: int, p2: int) -> torch.Tensor:
    return [[i // (original_w // p2), i % (original_w // p2)] for i in range(patches.shape[1])]


def is_scaled_image(image: torch.Tensor, min_scale: float = 0.0, max_scale: float = 1.0) -> bool:
    """similar to transformers.image_utils.is_scaled_image but for torch.Tensor

    Args:
        image (torch.Tensor): _description_
    """

    if image.dtype == torch.uint8:
        return False

    return torch.min(image) >= min_scale and torch.max(image) <= max_scale


def ensure_channels_first(image: torch.Tensor) -> torch.Tensor:
    """
    Ensure that the channels are the first dimension of the image tensor.

    Args:
        image: Image tensor. Shape: [batch, height, width, channels] or [batch, channels, height, width]

    Returns:
        Image tensor with channels as the first dimension. Shape: [batch, channels, height, width]
    """
    if image.ndim != 4:
        image = image.unsqueeze(0)

    if image.shape[1] != 3:
        image = image.permute(0, 3, 1, 2)
    return image


def patchify_image(image: torch.Tensor, patch_dim_h: int, patch_dim_w: int) -> "torch.Tensor":
    """
    Convert an image into a tensor of patches.

    Args:
        image: Image to convert. Shape: [batch, channels, height, width]
        patch_dim_h: Height of each patch.
        patch_dim_w: Width of each patch.
    """

    batch_size, channels, height, width = image.shape
    patches = image.unfold(2, patch_dim_h, patch_dim_h)  # unfolded_along_height
    patches = patches.unfold(3, patch_dim_w, patch_dim_w)  # unfolded_along_width

    # patches reshaped
    patches = patches.contiguous().view(batch_size, channels, -1, patch_dim_h, patch_dim_w)

    # [batch_size, channels, num_patches, patch_dim_h, patch_dim_w]
    patches_final = patches.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, channels * patch_dim_h * patch_dim_w)

    return patches_final


def infer_channel_dimension_format(
    image: torch.Tensor, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`torch.Tensor`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels or (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    shape_len = len(image.shape)
    if shape_len not in [3, 4]:
        raise ValueError(f"Unsupported number of image dimensions: {shape_len}")

    first_dim, last_dim = (0, 2) if shape_len == 3 else (1, 3)

    if image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST

    raise ValueError("Unable to infer channel dimension format")


def normalize(
    image: torch.Tensor,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> torch.Tensor:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`torch.Tensor`):
            The image to normalize.
        mean (`float` or `Iterable[float]`):
            The mean to use for normalization.
        std (`float` or `Iterable[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not torch.is_tensor(image):
        raise ValueError("image must be a torch.Tensor")

    input_data_format = input_data_format or infer_channel_dimension_format(image)
    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    mean = [mean] * num_channels if not isinstance(mean, Iterable) else mean
    if len(mean) != num_channels:
        raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")

    std = [std] * num_channels if not isinstance(std, Iterable) else std
    if len(std) != num_channels:
        raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")

    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        mean = torch.tensor(mean, dtype=image.dtype)
        std = torch.tensor(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.permute(1, 2, 0) - mean) / std).permute(2, 0, 1)

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format else image
    return image


def get_channel_dimension_axis(image: torch.Tensor, input_data_format: Optional[Union[str, ChannelDimension]] = None):
    """
    Returns the channel dimension axis of `image`.

    Args:
        image (`torch.Tensor`):
            The image to get the channel dimension axis of.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        The channel dimension axis of the image.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    if input_data_format == ChannelDimension.LAST:
        return image.ndim - 1

    raise ValueError("Unsupported data format")


def to_channel_dimension_format(
    image: torch.Tensor,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> torch.Tensor:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`torch.Tensor`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `torch.Tensor`: The image with the channel dimension set to `channel_dim`.
    """
    if not torch.is_tensor(image):
        raise ValueError(f"Input image must be of type torch.Tensor, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.permute((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.permute((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


def resize_image_below_max(image: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
    *_, image_height, image_width = image.shape

    if image_width <= target_width and image_height <= target_height:
        return image

    height_scale_factor = target_height / image_height
    width_scale_factor = target_width / image_width
    optimal_scale_factor = min(height_scale_factor, width_scale_factor)

    new_height = int(image_height * optimal_scale_factor)
    new_width = int(image_width * optimal_scale_factor)

    scaled_image = resize(img=image, size=(new_height, new_width))
    return scaled_image
