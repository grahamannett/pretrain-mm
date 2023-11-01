import unittest

from PIL import Image
from transformers import FuyuProcessor, FuyuForCausalLM

MODEL_ID = "adept/fuyu-8b"  # https://huggingface.co/adept/fuyu-8b


class TestFuyuProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = FuyuProcessor.from_pretrained(MODEL_ID)

        return super().setUpClass()

    def test_image_patches(self):
        image_path = "fixtures/bus.png"
        image = Image.open(image_path)

        


class TestFuyuModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FuyuForCausalLM.from_pretrained(MODEL_ID)
        return super().setUpClass()

    def test_model(self):
        pass
