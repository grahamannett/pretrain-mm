import io
import os
import unittest

import requests
import torch
from PIL import Image, ImageDraw
from transformers import FuyuForCausalLM as HFFuyuForCausalLM
from transformers import FuyuProcessor as HFFuyuProcessor

from pretrain_mm import logger
from pretrain_mm.model.fuyu import CombineEmbeddings
from pretrain_mm.model.fuyu import FuyuProcessor as PatchedFuyuProcessor

# NEED TO PATCH GATHER CONTINOUS EMBEDDINGS
from pretrain_mm.utils.token_tag_utils import token_box_pattern

# MODEL_ID = "adept/fuyu-8b"  # https://huggingface.co/adept/fuyu-8b
from tests.fixtures.fuyu_fixtures import MODEL_ID, fuyu_model_kwargs, FuyuFixture

image_path = "tests/fixtures/bus.png"
image_text = "image of a bus"


breakpoint()


class TestFuyuProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()

    def test_image_patches(self):
        processor = HFFuyuProcessor.from_pretrained(MODEL_ID)

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

        processor = HFFuyuProcessor.from_pretrained(MODEL_ID)

        with self.assertRaises(KeyError):
            processor(text=text, images=image)


class TestFuyuModel(unittest.TestCase):
    """
    https://huggingface.co/adept/fuyu-8b/discussions/44#6544c5e6ee7bbb5952bdebfb
    """

    def setUp(self) -> None:
        self.vacations_image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/vacation_days_hr.png"
        )
        self.vacations_image_pil = Image.open(io.BytesIO(requests.get(self.vacations_image_url).content)).convert("RGB")

        self.skateboard_image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/skateboard.png"
        )
        self.skateboard_image_pil = Image.open(io.BytesIO(requests.get(self.skateboard_image_url).content)).convert(
            "RGB"
        )
        return super().setUp()

    def test_other_tokens(self):
        max_new_tokens = 20
        num_gens = 10
        skip_special_tokens = False

        prompt = FuyuFixture.box.prompt

        # <0x05> maps to
        prompt = "Complete <0x07>"
        # if you want to add boa token: prompt = f"<0x04>{prompt}"
        image = self.skateboard_image_pil

        processor = HFFuyuProcessor.from_pretrained(MODEL_ID)

        model_inputs = processor(text=prompt, images=image)

        # processor adds boa token <0x04> no matter what to end of prompt
        model_inputs.input_ids = model_inputs.input_ids[:, :-1]

        show_from = model_inputs.input_ids.shape[-1] - 6

        model = HFFuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
        model = CombineEmbeddings.patch_gather_embeddings(model)

        model_inputs.to(model.device)
        gen_kwargs = {
            "pad_token_id": processor.tokenizer.unk_token_id,  # or processor.tokenizer.eos_token_id
        }
        logger.info(f"Generating {num_gens} inputs.  0 is no sampling, 1-4 is sampling with temp 0.1-0.4.")
        logger.info(f"Base prompt: {processor.tokenizer.convert_ids_to_tokens(model_inputs.input_ids[0, show_from:])}")

        for idx in range(num_gens):
            # model_inputs.input_ids[idx, -1] = processor.tokenizer.eos_token_id
            if idx != 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = 0.1 * idx

            with torch.no_grad():
                model_outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens, **gen_kwargs)

            decoded_text = processor.decode(model_outputs[0, show_from:], skip_special_tokens=skip_special_tokens)
            logger.info(f"{idx}: {decoded_text}")

    def test_vqa(self):
        processor = HFFuyuProcessor.from_pretrained(MODEL_ID)

        model = HFFuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
        model = CombineEmbeddings.patch_gather_embeddings(model)

        fourth_text_prompt = "Answer the following DocVQA question based on the image. \n What was the fair amount of paid vacation days in the United Kingdom?"

        model_inputs = processor(text=fourth_text_prompt, images=self.vacations_image_pil).to(model.device)
        model_outputs = model.generate(**model_inputs, max_new_tokens=10)
        decoded_model_outputs = processor.batch_decode(model_outputs[:, -10:], skip_special_tokens=True)

    def test_vqa2(self):
        fifth_text_prompt = (
            "Answer the following VQAv2 question based on the image: What type of foods are in the image?"
        )
        fish_image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/fish_carrots.png"
        )
        fish_image_pil = Image.open(io.BytesIO(requests.get(fish_image_url).content))

        # get both processors
        processor = HFFuyuProcessor.from_pretrained(MODEL_ID)

        model = HFFuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
        model = CombineEmbeddings.patch_gather_embeddings(model)

        model_inputs = processor(text=fifth_text_prompt, images=fish_image_pil)
        model_outputs = model.generate(**model_inputs, max_new_tokens=10)

        decoded_model_outputs = processor.batch_decode(model_outputs[:, -10:], skip_special_tokens=True)

    def test_ocr_box(self):
        # Given:
        # Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
        # y1, x1, y2, x2

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n <box>388, 428, 404, 488</box>"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        matched = token_box_pattern.search(bbox_prompt)

        y1, x1, y2, x2 = map(int, matched.groups())

        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        draw = ImageDraw.Draw(bbox_image_pil)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        if save_path := os.environ.get("test_image_out", None):  # probably want like "test_box.png"
            logger.info(f"saving path to: {save_path}")
            bbox_image_pil.save(save_path)

    def test_text_extract(self):
        processor = PatchedFuyuProcessor.from_pretrained(MODEL_ID)
        model = HFFuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())
        model.gather_continuous_embeddings = CombineEmbeddings.gather_continuous_embeddings

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        generated_tokens = model.generate(**model_inputs, max_new_tokens=10)

        model_outputs = processor.batch_decode(generated_tokens[:, -10:], skip_special_tokens=True)[0]
        prediction = model_outputs.split("\x04", 1)[1] if "\x04" in model_outputs else ""
        self.assertTrue("\x04" in model_outputs)

    def test_box_generate(self):
        # bbox localisation from text
        processor = PatchedFuyuProcessor.from_pretrained(MODEL_ID)
        model = HFFuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())
        # model = FuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float16)

        bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n Williams"
        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")

        outputs = model.generate(**model_inputs, max_new_tokens=10)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
        matched = token_box_pattern.search(decoded_outputs)
        self.assertEquals(len(matched.groups()), 4)
