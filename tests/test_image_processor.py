import unittest

import pytest
import torch
from torchvision import transforms

from pretrain_mm.processor import image_processor_helpers, image_processor_mixin
from tests.mock.image_info import mac_screenshot

ChannelDimension = image_processor_mixin.ChannelDimension
infer_channel_dimension_format = image_processor_helpers.infer_channel_dimension_format
normalize = image_processor_helpers.normalize
patchify_image = image_processor_helpers.patchify_image
make_patch_indices = image_processor_helpers.make_patch_indices

screenshot_image = torch.randint(0, 255, (1, mac_screenshot.c, mac_screenshot.h, mac_screenshot.w), dtype=torch.uint8)
other_image = torch.randint(0, 255, (1, 3, 1000, 1000), dtype=torch.uint8)


class TestImageUtils(unittest.TestCase):
    def test_channel_dimension_format(self):
        # Test when image has 3 dimensions and channel is the first dimension
        image = torch.randn(3, 100, 100)
        self.assertEqual(
            image_processor_helpers.infer_channel_dimension_format(image),
            image_processor_helpers.ChannelDimension.FIRST,
        )

        # Test when image has 3 dimensions and channel is the last dimension
        image = torch.randn(100, 100, 3)
        self.assertEqual(
            image_processor_helpers.infer_channel_dimension_format(image), image_processor_helpers.ChannelDimension.LAST
        )

        # Test when image has 4 dimensions and channel is the first dimension
        image = torch.randn(10, 3, 100, 100)
        self.assertEqual(
            image_processor_helpers.infer_channel_dimension_format(image),
            image_processor_helpers.ChannelDimension.FIRST,
        )

        # Test when image has 4 dimensions and channel is the last dimension
        image = torch.randn(10, 100, 100, 3)
        self.assertEqual(
            image_processor_helpers.infer_channel_dimension_format(image), image_processor_helpers.ChannelDimension.LAST
        )

    def test_infer_channel_dimension(self):
        # Test we fail with invalid input
        with pytest.raises(ValueError):
            infer_channel_dimension_format(torch.randint(0, 256, (10, 10)))

        with pytest.raises(ValueError):
            infer_channel_dimension_format(torch.randint(0, 256, (10, 10, 10, 10, 10)))

        # Test we fail if neither first not last dimension is of size 3 or 1
        with pytest.raises(ValueError):
            infer_channel_dimension_format(torch.randint(0, 256, (10, 1, 50)))

        # But if we explicitly set one of the number of channels to 50 it works
        inferred_dim = infer_channel_dimension_format(torch.randint(0, 256, (10, 1, 50)), num_channels=50)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        # Test we correctly identify the channel dimension
        image = torch.randint(0, 256, (3, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

        image = torch.randint(0, 256, (1, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

        image = torch.randint(0, 256, (4, 5, 3))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        image = torch.randint(0, 256, (4, 5, 1))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        # We can take a batched array of images and find the dimension
        image = torch.randint(0, 256, (1, 3, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

    def test_normalize(self):
        image = torch.randint(0, 256, (224, 224, 3)) / 255

        # Test that exception is raised if inputs are incorrect
        # Not a numpy array image
        with self.assertRaises(ValueError):
            normalize(5, 5, 5)

        # Number of mean values != number of channels
        with self.assertRaises(ValueError):
            normalize(image, mean=(0.5, 0.6), std=1)

        # Number of std values != number of channels
        with self.assertRaises(ValueError):
            normalize(image, mean=1, std=(0.5, 0.6))

        # Test result is correct - output data format is channels_first and normalization
        # correctly computed
        mean = (0.5, 0.6, 0.7)
        std = (0.1, 0.2, 0.3)
        mean = torch.tensor(mean, dtype=image.dtype)
        std = torch.tensor(std, dtype=image.dtype)
        expected_image = (image - mean) / std

        normalized_image = normalize(image, mean=mean, std=std, data_format=ChannelDimension.LAST)
        self.assertIsInstance(normalized_image, torch.Tensor)
        self.assertEqual(normalized_image.shape, (224, 224, 3))
        self.assertTrue(torch.allclose(normalized_image, expected_image))

        # Test that its similar to torchvision.transforms.Normalize
        tv_Normalize = transforms.Normalize(mean=mean, std=std)
        tv_normalized_image = tv_Normalize(image.permute(2, 0, 1)).permute(1, 2, 0)
        self.assertTrue(torch.allclose(normalized_image, tv_normalized_image))

        # Test image with 4 channels is normalized correctly
        image = torch.randint(0, 256, (224, 224, 4)) / 255
        mean = (0.5, 0.6, 0.7, 0.8)
        std = (0.1, 0.2, 0.3, 0.4)
        mean = torch.tensor(mean, dtype=image.dtype)
        std = torch.tensor(std, dtype=image.dtype)
        expected_image = (image - mean) / std

        self.assertTrue(
            torch.allclose(normalize(image, mean, std, input_data_format=ChannelDimension.LAST), expected_image)
        )


class TestImagePatches(unittest.TestCase):
    def test_image_patches(self):
        for p1, p2 in [(16, 16), (24, 24), (30, 30), (54, 54)]:
            for image in [screenshot_image, other_image]:
                patches = patchify_image(image, p1, p2)
                self.assertEqual(patches.shape[-1], p1 * p2 * 3)

    def test_image_patch_indices(self):
        for p1, p2 in [(16, 16), (24, 24), (30, 30), (54, 54)]:
            for image in [screenshot_image, other_image]:
                image_height, image_width = image.shape[2], image.shape[3]
                patches = patchify_image(image, p1, p2)
                indices = make_patch_indices(image, image_width, p2)


class TestImageProcessor(unittest.TestCase):
    def test_image_processor(self):
        processor = image_processor_mixin.ImageProcessor()
        patches, patch_idxs = processor(image=screenshot_image)
        self.assertEqual(patches[0].shape[-2], len(patch_idxs[0]))


if __name__ == "__main__":
    unittest.main()
