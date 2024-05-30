import unittest

import torch
from transformers import AutoModel, AutoTokenizer

from pretrain_mm.model.adapted.mixture_of_depth import MixtureOfDepth, apply_mod_to_hf


class TestMixtureOfDepth(unittest.TestCase):
    def setUp(self):
        self.model = AutoModel.from_pretrained("gpt2")
        self.processor = AutoTokenizer.from_pretrained("gpt2")
        self.mod = MixtureOfDepth(0.125, self.model.transformer.h[0])

    def test_forward(self):
        x = torch.rand(1, 10, self.model.config.n_embd)
        attention_mask = torch.ones(1, 10)
        position_ids = torch.arange(10).unsqueeze(0)
        past_key_value = (
            torch.rand(1, 2, 10, self.model.config.n_embd),
            torch.rand(1, 2, 10, self.model.config.n_embd),
        )
        output_attentions = False
        use_cache = False
        cache_position = None

        output = self.mod(x, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[0].shape, x.shape)


class TestApplyModToHf(unittest.TestCase):
    def setUp(self):
        self.model = AutoModel.from_pretrained("gpt2")
        self.processor = AutoTokenizer.from_pretrained("gpt2")

    def test_apply_mod_to_hf(self):
        modified_model = apply_mod_to_hf(self.model, enabled=True)
        self.assertEqual(modified_model.__class__.__name__, "MoDGPT2Model")


if __name__ == "__main__":
    unittest.main()
