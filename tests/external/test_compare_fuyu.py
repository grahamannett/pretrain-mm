import unittest

import torch
from PIL import Image
from transformers import FuyuForCausalLM

from config.fuyu import FuyuInfo
from pretrain_mm.processor.fuyu.fuyu_processing import FuyuProcessor

# from


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

        data = processor(text=text, images=[image, image])


class TestProcessor(unittest.TestCase):
    def test_fuyu_processor(self):
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

        image = torch.rand(3, 1280, 1080)
        text = "Task: Find JetBlue career openings in New York Previous Actions Next Action:CLICK  @ <box>172, 4215, 234, 4233</box>"

        data = processor(text=text, images=image)


class TestFuyuModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FuyuForCausalLM.from_pretrained(MODEL_ID)
        return super().setUpClass()

    def test_model(self):
        pass
