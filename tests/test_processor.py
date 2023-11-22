import unittest

import torch

from config.fuyu import FuyuInfo
from pretrain_mm.processor.fuyu.fuyu_processing import FuyuProcessor


class TestFuyuProcessor(unittest.TestCase):
    def test_processor(self):
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

        image = torch.rand(3, 1280, 1080)
        text = "Task: Find JetBlue career openings in New York Previous Actions Next Action:CLICK  @ <box>172, 4215, 234, 4233</box>"

        data = processor(text=text, images=image)

        self.assertEquals(data.input_ids.ndim, 2)
        self.assertIsInstance(data.image_patches, list)
        self.assertIsInstance(data.image_patches[0], torch.Tensor)
        self.assertEquals(data.image_patches[0].shape[0], 1)