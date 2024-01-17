import unittest

import torch
import transformers

from pretrain_mm.model.fuyu.processing import FuyuConstants, FuyuProcessor
from pretrain_mm.utils.generate_utils import generate_helper


class TestGenerateHelper(unittest.TestCase):
    def setUp(self):
        image_size = (1000, 1000)
        self.model_id = "adept/fuyu-8b"
        self.inputs = {"text": "Hello", "images": torch.rand(3, *image_size)}
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.processor = FuyuProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.max_new_tokens = 10
        self.stop_tokens = [1, 2, 3]
        self.temperature = 0.8
        self.top_k = 5
        self.indices_placeholder = torch.tensor([[-1]])
        self.mask_placeholder = torch.tensor([[1]])
        self.drop_last_of_input = True

    def test_generate_helper(self):
        output = generate_helper(
            model=self.model,
            processor=self.processor,
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            stop_tokens=self.stop_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            indices_placeholder=self.indices_placeholder,
            mask_placeholder=self.mask_placeholder,
            drop_last_of_input=self.drop_last_of_input,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([1, self.max_new_tokens]))

    def test_generate_helper_with_stop_tokens(self):
        additional_tokens = ["black", " The", "The", "image", "the image", "The Image", "a"]

        stop_tokens = FuyuConstants.get_stop_tokens(
            self.processor,
            additional_tokens=additional_tokens,
        )

        output = generate_helper(
            model=self.model,
            processor=self.processor,
            inputs=self.inputs,
            max_new_tokens=self.max_new_tokens,
            stop_tokens=stop_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            indices_placeholder=self.indices_placeholder,
            mask_placeholder=self.mask_placeholder,
            drop_last_of_input=self.drop_last_of_input,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([1, self.max_new_tokens]))


if __name__ == "__main__":
    unittest.main()
