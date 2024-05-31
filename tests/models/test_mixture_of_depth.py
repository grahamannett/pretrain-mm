import unittest

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from pretrain_mm.model import fuyu, paligemma
from pretrain_mm.model.adapted.mixture_of_depth import MixtureOfDepth, apply_mod_to_hf
from tests.fixtures.fuyu_fixtures import image, input_string


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


class TestModMM(unittest.TestCase):
    def setUp(self):
        self.device_map = "auto"

    def test_apply_mod_to_llama(self):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer(text=input_string, return_tensors="pt")

        model = apply_mod_to_hf(model)
        inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))

    def test_apply_mod_to_persimmon(self):
        # for testing locally, we need to use this config
        model_id = "adept/persimmon-8b-base"
        model_config = AutoConfig.from_pretrained(
            model_id,
            hidden_size=512,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
        )

        model = AutoModelForCausalLM.from_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer(text=input_string, return_tensors="pt")
        inputs.to(model.device)

        model = apply_mod_to_hf(model, skip_position_ids=True)

        with torch.no_grad():
            outputs = model(**inputs)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(model.__class__.__name__, "PersimmonMoDForCausalLM")

    def test_apply_mod_to_fuyu(self):
        # need to drop layers for testing locally, i think we need 2 at least for MoD to work
        model_config = fuyu.FuyuConfig.from_pretrained(fuyu.MODEL_ID).patch(num_hidden_layers=2)

        model = fuyu.FuyuForCausalLM.from_pretrained(fuyu.MODEL_ID, device_map=self.device_map, config=model_config)

        model = apply_mod_to_hf(model)
        self.assertEqual(model.__class__.__name__, "FuyuMoDForCausalLM")
        processor = fuyu.FuyuProcessor.from_pretrained(fuyu.MODEL_ID)

        inputs = processor(text=input_string, images=image)
        inputs.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))

    def test_apply_mod_to_gemma(self):
        model = paligemma.PaliGemmaForConditionalGeneration.from_pretrained(
            paligemma.MODEL_ID, device_map=self.device_map
        )

        modified_model = apply_mod_to_hf(model)
        self.assertEqual(modified_model.__class__.__name__, "PaliGemmaMoDForConditionalGeneration")

        processor = paligemma.PaliGemmaProcessor.from_pretrained(paligemma.MODEL_ID)

        inputs = processor(text=input_string, images=image)
        inputs.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertTrue(hasattr(outputs, "logits"))


if __name__ == "__main__":
    unittest.main()
