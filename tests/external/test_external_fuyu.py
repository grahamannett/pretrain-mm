import unittest

import torch
from PIL import Image

MODEL_ID = "adept/fuyu-8b"  # https://huggingface.co/adept/fuyu-8b
from transformers import FuyuForCausalLM, FuyuProcessor


class TestFuyuProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    def test_image_patches(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        image_path = "tests/fixtures/bus.png"
        text = "image of a bus"
        image = Image.open(image_path)
        data = processor(text=text, images=[image, image])

        # if these change we will need to update our code
        self.assertEquals(data.input_ids.ndim, 2)
        self.assertIsInstance(data.image_patches, list)
        self.assertIsInstance(data.image_patches[0], torch.Tensor)
        self.assertEquals(data.image_patches[0].shape[0], 1)

    def test_processor_fail(self):
        # keys above 1000 fail but only with images
        text = "Task: Test that 1000 keys fail \nNext Action @ <box>1000, 1010, 2000, 4233</box>"
        image = torch.rand(3, 1280, 1080)

        processor = FuyuProcessor.from_pretrained(MODEL_ID)

        with self.assertRaises(KeyError):
            processor(text=text, images=image)


class TestFuyuModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FuyuForCausalLM.from_pretrained(MODEL_ID)
        return super().setUpClass()

    def test_model(self):
        pass
