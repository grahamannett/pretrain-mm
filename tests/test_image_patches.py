import torch

import unittest

class TestImagePatches(unittest.TestCase):
    def test_image_patches(self):

        from clippymm.pretrain.datasets import resize_to_nearest_multiple
        from einops import rearrange
        from PIL import Image
        import numpy as np

        image = Image.open("/Users/graham/code/clippymm_datasets/pretrain_data/common-screen

p1, p2 = 64, 64
resize_image = resize_to_nearest_multiple(image.unsqueeze(0), p1, p2)
h, w = resize_image.shape[2:]

# patches = rearrange(resize_image, "b c (h p1) (w p2) -> b h w(p1 p2 c)", p1=8, p2=8)

# rearrange(image, "b (h1 h2) (w1 w2) c -> (h1 w2) (b w1 h2) c", h2=8, w2=8)

# patches = rearrange(ims, 'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=8, p2=8)

first_image = str(sorted(Path("data/tasks/current").glob("*.png"))[0])
image = torchvision.io.read_image(first_image, mode=torchvision.io.image.ImageReadMode.RGB)
def plot_patches(patches: torch.Tensor, patch_idxs: torch.Tensor, num_rows: int = 4, num_cols: int = 20):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 20, figsize=(20, 4))
    fig.patch.set_facecolor("xkcd:mint green")
    p_idx = 0
    for i in range(4):
        for j in range(20):
            axes[i, j].imshow(patches[0, i * 20 + j].permute(1, 2, 0))
            axes[i, j].axis("off")
            # axes[i, j].set_title(f"({i}, {j})")
            axes[i, j].set_title(patch_indices[p_idx])
            p_idx += 1

    # plt.tight_layout()
    # plt.savefig("patches.png")
