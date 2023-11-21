from typing import Any
from transformers import ProcessorMixin
from tokenizers import Tokenizer

import torch
from PIL import Image

from pretrain_mm.processor.image_processor import ImageProcessor


class Processor(ProcessorMixin):
    def __init__(self, text_tokenizer: Tokenizer, image_processor: ImageProcessor):
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor

    def __call__(self, text, images: torch.Tensor | Image.Image, **kwds: Any) -> Any:
        image_patches, image_patch_idxs = self.image_processor(images)

        text_encodings = self.process_text(text, return_tensors="pt", padding=True)
        # attention_mask =

    def process_text(self, text: str, **kwargs) -> torch.Tensor:
        transformed_text = self.text_tokenizer(text)

    # def make_attention_mask(self, text_encodings, image_patches, left_pad: int):
    #     return torch.ones((text_encodings.input_ids.shape[0], image_patches.shape[1]), dtype=torch.long
