from collections import UserDict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from .base import Sample, IGNORE_INDEX


@dataclass
class Batch(UserDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    images: torch.Tensor


@dataclass
class DataCollator:
    def __post_init__(self):
        self.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, samples: Sequence[Sample]):
        


@dataclass
class DataCollatorWithProcessor:
    text_processor: PreTrainedTokenizer

    text_tokenizer_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": False,
    }

    def __call__(self, samples: Sequence[Sample]):
        input_ids = pad_sequence([s.input_ids for s in samples], batch_first=True, padding_value=self.pad_token_id)
        image_patches = pad_sequence([s.image_patches for s in samples], batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.pad_token_id)

        # labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        labels = input_ids.clone()
        images = [sample.image for sample in samples]

        return Batch(input_ids=input_ids, labels=labels, attention_mask=attention_mask, images=images)

    def text_tokenize_func(self, sample, sample_key):
        """TODO: need to do ignore_index and padding stuff somewhere"""
        tokenized_vals = self.tokenizer(
            getattr(sample, sample_key), max_length=self.tokenizer.model_max_length, **self.tokenizer_kwargs
        )
        input_ids, attention_mask = tokenized_vals.input_ids, tokenized_vals.attention_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # sample.input_ids = input_ids[0]
        # sample.label = sample.input_ids.clone()
        # sample.attention_mask = attention_mask[0]
        # return sample

    def _from_raw_sample(self, samples: Sequence[Sample]):
        samples = [self.tokenize_func(sample, sample_key="text") for sample in samples]
