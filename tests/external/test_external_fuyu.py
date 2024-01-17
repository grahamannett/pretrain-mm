import io
import os
import unittest

import requests
import torch
from PIL import Image, ImageDraw
from transformers import FuyuForCausalLM, FuyuProcessor

from pretrain_mm import logger

# from pretrain_mm.model.fuyu.fuyu_processing import FuyuProcessor as PatchedFuyuProcessor
from pretrain_mm.model.fuyu import FuyuProcessor as PatchedFuyuProcessor
from pretrain_mm.utils.eval_utils import box_pattern

# MODEL_ID = "adept/fuyu-8b"  # https://huggingface.co/adept/fuyu-8b
from tests.fixtures.fuyu_fixtures import MODEL_ID, fuyu_model_kwargs

# NEED TO PATCH GATHER CONTINOUS EMBEDDINGS
from pretrain_mm.model.fuyu.fuyu_embed import CombineEmbeddings

image_path = "tests/fixtures/bus.png"
image_text = "image of a bus"


class TestFuyuProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    def test_image_patches(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID)

        image = Image.open(image_path)
        data = processor(text=image_text, images=[image, image])

        # if these change we will need to update our code
        self.assertEquals(data.input_ids.ndim, 2)
        self.assertIsInstance(data.image_patches, list)
        self.assertIsInstance(data.image_patches[0], torch.Tensor)
        self.assertEquals(data.image_patches[0].shape[0], 1)

    def test_processor_fail(self):
        # keys above 1000 fail but only with images, if this passes we can remove our patched processor
        text = "Task: Test that 1000 keys fail \nNext Action @ <box>1000, 1010, 2000, 4233</box>"
        image = torch.rand(3, 1280, 1080)

        processor = FuyuProcessor.from_pretrained(MODEL_ID)

        with self.assertRaises(KeyError):
            processor(text=text, images=image)


class TestFuyuModel(unittest.TestCase):
    """
    https://huggingface.co/adept/fuyu-8b/discussions/44#6544c5e6ee7bbb5952bdebfb
    """

    def test_ocr_box(self):
        # Given:
        # Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
        # y1, x1, y2, x2

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        matched = box_pattern.search(bbox_prompt)

        y1, x1, y2, x2 = map(int, matched.groups())

        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        draw = ImageDraw.Draw(bbox_image_pil)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        if save_path := os.environ.get("test_image_out", None):  # probably want like "test_box.png"
            logger.info(f"saving path to: {save_path}")
            bbox_image_pil.save(save_path)

    def test_text_extract(self):
        processor = PatchedFuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())
        model.gather_continuous_embeddings = CombineEmbeddings.gather_continuous_embeddings

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        breakpoint()

        model_outputs = processor.batch_decode(generated_tokens[:, -10:], skip_special_tokens=True)[0]
        prediction = model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        self.assertTrue("\x04" in model_outputs)

    def test_box_generate(self):
        # bbox localisation from text
        processor = PatchedFuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())
        # model = FuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n Williams"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        outputs = model.generate(**model_inputs, max_new_tokens=10)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
        matched = box_pattern.search(decoded_outputs)
        self.assertEquals(len(matched.groups()), 4)
