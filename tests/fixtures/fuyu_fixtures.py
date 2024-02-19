import io
from typing import Any

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.model.fuyu import MODEL_ID, FuyuPatches, FuyuConstants
from tests.fixtures.fixture_tools import DataFixture
from tests.fixtures.common import input_label, input_string, screenshot

default_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
default_processor = AutoProcessor.from_pretrained(MODEL_ID)


def _get_image_from_url(url: str) -> Image.Image:
    return Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")


hf_image_urls = {
    "box": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg",
    "jobs": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/jobs.png",
    "chart": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/chart.png",
    "skateboard": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/skateboard.png",
    "vacations": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/vacation_days_hr.png",
    "fish": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/fish_carrots.png",
}

hf_prompts = {
    "box": "Answer the following DocVQA question based on the image. \n Which is the metro in California that has a good job Outlook?",
    "jobs": "Answer the following DocVQA question based on the image. \n What if the maximum male life expectancy?",
    "chart": "Answer the following DocVQA question based on the image. \n What sport is that?",
    "skateboard": "Answer the following DocVQA question based on the image. \n What was the fair amount of paid vacation days in the United Kingdom?",
    "vacations": "Answer the following VQAv2 question based on the image: What type of foods are in the image?",
    "fish": "Answer the following VQAv2 question based on the image: What type of foods are in the image?",
}

FuyuFixture = DataFixture(
    image_urls=hf_image_urls,
)


def get_hf_examples():
    # Define image URLs and prompts

    # Retrieve images from URLs
    images = {key: _get_image_from_url(url) for key, url in image_urls.items()}

    # Create dictionary of examples
    examples = {key: {"image": images[key], "prompt": prompts[key]} for key in image_urls.keys()}

    return examples


def fuyu_model_kwargs() -> dict:
    logger.warn('Using device_map="auto" and torch_dtype=torch.float16 for model as 24gb GPU wont work otherwise')
    return {
        "device_map": "auto",
        **FuyuInfo.model_kwargs,
    }


def get_kwargs_for_preprocess_with_tokenizer_info(images, processor=default_processor):
    image_encoding = processor.image_processor.preprocess(images, return_tensors="pt")
    batch_images = image_encoding["images"]
    image_unpadded_heights = image_encoding["image_unpadded_heights"]
    image_unpadded_widths = image_encoding["image_unpadded_widths"]
    scale_factors = image_encoding["image_scale_factors"]

    image_placeholder_id = processor.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
    image_newline_id = processor.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
    tensor_batch_images = torch.stack([img[0] for img in batch_images]).unsqueeze(1)

    return {
        # "scale_factors": [scale_factors],
        "image_present": torch.ones(1, 1, 1),
        "image_unpadded_h": torch.tensor([image_unpadded_heights[0]]),
        "image_unpadded_w": torch.tensor([image_unpadded_widths[0]]),
        "image_placeholder_id": image_placeholder_id,
        "image_newline_id": image_newline_id,
        "image_input": tensor_batch_images[0].unsqueeze(0),
        "variable_sized": True,
    }


def get_model_and_patch(device_map: str = "auto", trust_remote_code=True, torch_dtype=torch.bfloat16, **kwargs):
    model = FuyuPatches.patch(
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    )

    # model = FuyuPatches.patch_gather_embeddings(model)
    return model


def get_fuyu_example_inputs() -> dict:
    # extra tokens that should be added by processor
    input_string_special_tokens = f"{FuyuConstants.bos_string} " + input_string + f"{FuyuConstants.boa_string}"
    input_label_special_tokens = input_label + f"{FuyuConstants.eos_string}"

    return {
        # first 3 are named images/text/label to match processor kwargs
        "images": screenshot,
        "text": input_string,
        "label": input_label,
        # the rest are stubs for testing
        "input_string_with_label": input_string + input_label,
        "input_string_special_tokens": input_string_special_tokens,
        "input_label_special_tokens": input_label_special_tokens,
        "input_string_and_label_special_tokens": input_string_special_tokens + input_label_special_tokens,
    }


# text stubs for testing
example_inputs = get_fuyu_example_inputs()

image = example_inputs["images"]
input_string = example_inputs["text"]
input_label = example_inputs["label"]
input_string_with_label = example_inputs["input_string_with_label"]
input_string_special_tokens = example_inputs["input_string_special_tokens"]
input_label_special_tokens = example_inputs["input_label_special_tokens"]
input_string_and_label_special_tokens = example_inputs["input_string_and_label_special_tokens"]
