import unittest

from PIL import Image
from config.fuyu import FuyuInfo


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
