import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor

from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.model.fuyu import FuyuConstants

MODEL_ID = "adept/fuyu-8b"

default_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
default_processor = AutoProcessor.from_pretrained(MODEL_ID)


def fuyu_model_kwargs() -> dict:
    logger.warn('Using device_map="auto" and torch_dtype=torch.float16 for model as 24gb GPU wont work otherwise')
    return {
        "device_map": "auto",
        **FuyuInfo.model_kwargs,
    }


def get_fuyu_example_inputs() -> dict:
    image = Image.open("tests/fixtures/screenshot0.png")
    input_string = 'Given the following HTML provide the bounding box\\n <button backend_node_id="661"></button>'
    input_label = "<box>54, 1066, 102, 1200</box>"
    # extra tokens that should be added by processor
    input_string_special_tokens = f"{FuyuConstants.bos_string} " + input_string + f"{FuyuConstants.boa_string}"
    input_label_special_tokens = input_label + f"{FuyuConstants.eos_string}"

    return {
        "images": image,
        "input_string": input_string,
        "label": input_label,
        "input_string_with_label": input_string + input_label,
        "input_string_special_tokens": input_string_special_tokens,
        "input_label_special_tokens": input_label_special_tokens,
        "input_string_and_label_special_tokens": input_string_special_tokens + input_label_special_tokens,
    }


example_inputs = get_fuyu_example_inputs()

image = example_inputs["images"]
input_string = example_inputs["input_string"]
input_label = example_inputs["label"]
input_string_with_label = example_inputs["input_string_with_label"]
input_string_special_tokens = example_inputs["input_string_special_tokens"]
input_label_special_tokens = example_inputs["input_label_special_tokens"]
input_string_and_label_special_tokens = example_inputs["input_string_and_label_special_tokens"]


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
