import json
import time
import unittest

import torch
import transformers
from pretrain_mm.utils.testing_utils import TimerMixin


def load_torch(
    folder: str,
    device: str,
    ModelArgs: type,
    ModelCls: type,
    TokenizerCls: type,
    dtype: torch.dtype = torch.float16,
):
    torch.set_default_dtype(dtype)

    tokenizer = TokenizerCls(f"{folder}/tokenizer.model")

    with open(f"{folder}/config.json", "r") as f:
        model_args = ModelArgs(**json.load(f))

    with torch.device(device):
        model_state_dict = torch.load(folder + "/pytorch_model.bin", map_location="cpu", mmap=True)
        model = ModelCls(model_args)
        model.load_state_dict(model_state_dict)

        model.to(device)

    return model, tokenizer


def load_hf(dtype=torch.bfloat16):
    model_name = "mistralai/Mistral-7B-v0.1"
    # model_name = "adept/fuyu-8b"
    torch.set_default_dtype(dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map="auto")
    return model, tokenizer


class TestLoadTorch(TimerMixin, unittest.TestCase):
    def test_load_torch(self):
        folder = "~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773/"
        load_torch(
            folder=folder,
            device="cuda:0",
            dtype=torch.bfloat16,
            ModelArgs=transformers.models.mistral.MistralConfig,
        )

    def test_load_hf(self):
        load_hf()
        self.check_timer("load_hf")
