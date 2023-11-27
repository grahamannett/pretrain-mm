import io
import unittest


import requests
import torch
from PIL import Image
from transformers import FuyuForCausalLM

from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.model.fuyu import FuyuProcessor
from pretrain_mm.utils.eval_utils import box_pattern, bbox_metric, bbox_metric_from_str
from tests.fixtures.fuyu_fixtures import MODEL_ID, fuyu_model_kwargs

bbox_image_url = (
    "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
)
bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n Williams"
bbox_prompt_with_box = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n<box>388, 428, 404, 488</box>"
bbox_prompt_fail = "This should not respond with a bounding box.\\nTesting"

bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))


class TestBoxEval(unittest.TestCase):
    def test_eval(self):
        input_str = ""

        processor = FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name)

        text = (
            "SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER||SPEAKER"
            + "||NEWLINE|<s> When presented with a box, perform OCR to extract text contained within it. If provided"
            + " with text, generate the corresponding bounding box.\\n Williams\x04<box>388, 428, 22, 111</box>"
        )

        image = Image.open("tests/fixtures/bus.png")

        encoded_text = processor(text=text, images=image).input_ids
        post_processsed_bbox_tokens = processor.post_process_bbox_tokens(encoded_text)

    def test_box_metric(self):
        # similar to tests.external.test_external_fuyu.TestFuyuModel
        # which in turn comes from:
        # "https://huggingface.co/adept/fuyu-8b/discussions/44#6544d4d14200ae379eaafd33"
        # bbox localisation from text
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())

        # this one is known to be good from

        model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")
        outputs = model.generate(**model_inputs, max_new_tokens=10)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)

        target = torch.tensor(list(map(int, box_pattern.search(bbox_prompt_with_box).groups())), dtype=float)
        pred = torch.tensor(list(map(int, box_pattern.search(decoded_outputs).groups())), dtype=float)

        metric = bbox_metric(target, pred)
        self.assertLessEqual(metric, 1.0)

    def test_box_metric_none(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID)
        model = FuyuForCausalLM.from_pretrained(MODEL_ID, **fuyu_model_kwargs())

        model_inputs = processor(text=bbox_prompt_fail, images=bbox_image_pil).to("cuda")

        outputs = model.generate(**model_inputs, max_new_tokens=10)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)

        metric = bbox_metric_from_str(target_str=bbox_prompt, pred_str=decoded_outputs)
        self.assertEquals(metric, 1.0)  # 1.0 means failure
