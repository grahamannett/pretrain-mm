from einops import rearrange
import torch
import torch.nn.functional as F


def resize_to_patch_divisable(image: torch.Tensor, x: int, y: int):
    b, c, h, w = image.shape
    new_h = ((h - 1) // x + 1) * x
    new_w = ((w - 1) // y + 1) * y
    return F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)


def image_to_patches(image: torch.Tensor, p1: int, p2: int) -> torch.Tensor:
    return rearrange(image, "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=p1, p2=p2)


def make_patch_indices(patches: torch.Tensor, w: int, p2: int) -> torch.Tensor:
    return [[i // (w // p2), i % (w // p2)] for i in range(patches.shape[1])]


def patchify(pixel_values):
    """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
            Patchified pixel values.
    """
    patch_size, num_channels = 8, 3
    # sanity checks
    if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
        raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
    if pixel_values.shape[1] != num_channels:
        raise ValueError(
            "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
        )

    # patchify
    batch_size = pixel_values.shape[0]
    num_patches_one_direction = pixel_values.shape[2] // patch_size
    patchified_pixel_values = pixel_values.reshape(
        batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
    )
    patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
    )
    return patchified_pixel_values


if __name__ == "__main__":
    image_input = patchify()
