from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import ProcessorMixin
from torchvision.transforms.functional import normalize

from pretrain_mm.processesor.image_utils import (
    is_scaled_image,
    make_patch_indices,
    ensure_channels_first,
    patchify_image,
    resize_image_below_max,
    resize_to_patch_divisable,
)


@dataclass
class ImageProcessingConfig:
    rgb_max: float = 255.0

    # from fuyu config - called target_width and target_height
    max_width: int = 1920
    max_height: int = 1080

    norm_mean: float = 0.5
    norm_std: float = 0.5

    scale_image: bool = True
    pad_image: bool = True
    padding_value: float = 1.0
    padding_mode: str = "constant"

    # patch related
    patch_dim_h: int = 30
    patch_dim_w: int = 30


class ImageProcessor(ProcessorMixin):
    def __init__(self):
        self.config = ImageProcessingConfig()

    def pad_image(self, image: torch.Tensor) -> torch.Tensor:
        image_height, image_width = image.shape[2], image.shape[3]

        padding_top = 0
        padding_left = 0
        padding_bottom = self.config.max_height - image_height
        padding_right = self.config.max_width - image_width

        padded_image = torch.nn.functional.pad(
            image,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode=self.config.padding_mode,
            value=self.config.padding_value,
        )
        return padded_image

    def pre_process_image(self, image: torch.Tensor) -> torch.Tensor:
        """equivalent to apply_transformation in fuyu
        - skipping pad_image/rescale until I understand why they are needed

        Args:
            image (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # original_height, original_width = image.shape[2], image.shape[3]

        if not is_scaled_image(image):
            image = image / self.config.rgb_max

        # using normalize from torchvision.transforms.functional as seems more standardized
        image = normalize(image, self.config.norm_mean, self.config.norm_std)
        return image

    def __call__(
        self, images: List[torch.Tensor] = None, image: torch.Tensor = None, **kwargs
    ) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """
        This method processes the input images and returns the patches and their indices.

        Args:
            images (List[torch.Tensor], optional): List of images to be processed. Defaults to None.
            image (torch.Tensor, optional): Single image to be processed. Defaults to None.
            **kwargs: Additional parameters.

        Returns:
            Tuple[List[torch.Tensor], List[Tuple[int, int]]]: Returns a tuple where the first element is a list of patches and the second element is a list of patch indices.
        """
        images = [image] if images is None else images

        images = [ensure_channels_first(img) for img in images]
        images = [self.pre_process_image(img) for img in images]

        images = [resize_to_patch_divisable(img, self.config.patch_dim_h, self.config.patch_dim_w) for img in images]
        im_sz = [img.shape for img in images]
        patches = [patchify_image(img, self.config.patch_dim_h, self.config.patch_dim_w) for img in images]
        patch_idxs = [
            make_patch_indices(patch, im_sz[i][3], self.config.patch_dim_w) for i, patch in enumerate(patches)
        ]

        return patches, patch_idxs
