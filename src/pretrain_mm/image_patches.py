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


