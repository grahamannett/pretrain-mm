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
        processor = self.processor
        image_path = "tests/fixtures/bus.png"
        text = "image of a bus"
        image = Image.open(image_path)

        data = processor(text=text, images=image)


class TestFuyuModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FuyuForCausalLM.from_pretrained(MODEL_ID)
        return super().setUpClass()

    def test_model(self):
        pass
