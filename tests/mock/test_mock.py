import unittest

import torch
import transformers
from tests.mock.model import MockModel


class TestMockModel(unittest.TestCase):
    def test_mock_model(self):
        device = "cuda"

        text, image = "this is the input sentence", torch.rand(3, 1000, 1000)

        processor = transformers.AutoProcessor.from_pretrained("adept/fuyu-8b")
        # processor.tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        model = MockModel()
        model.language_model.model.resize_token_embeddings(len(processor.tokenizer))

        batch = processor(text=text, images=image)
        model.to(device)
        outputs = model(**batch.to(device))

        self.assertEqual(outputs.logits.ndim, 3)
