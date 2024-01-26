import unittest

import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    SiglipModel,
    SiglipTextModel,
    SiglipVisionModel,
)
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from tests.fixtures.common import screenshot, input_label, input_string


class TestVisionModel(unittest.TestCase):
    def test_model(self):
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        texts = [["a photo of a cat", "a photo of a dog"]]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        breakpoint()

    def test_pix2struct(self):
        processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(images=image, return_tensors="pt")

        # autoregressive generation
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

        # conditional generation
        text = "A picture of"
        inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

        inputs = processor(
            text="What location is chosen?", images=screenshot, return_tensors="pt", add_special_tokens=False
        )

        generated_ids = model.generate(**inputs, max_new_tokens=50)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

    def test_pix2struct_ai2d(self):
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)

        model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base")
        processor = Pix2StructProcessor.from_pretrained("google/pix2struct-ai2d-base")

        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"

        inputs = processor(images=image, text=question, return_tensors="pt")

        predictions = model.generate(**inputs)
        print(processor.decode(predictions[0], skip_special_tokens=True))

        inputs = processor(text="What location is chosen?", images=screenshot, return_tensors="pt", add_special_tokens=False)

        generated_ids = model.generate(**inputs, max_new_tokens=50)

        breakpoint()


@unittest.skip("SiglipTextModel not yet implemented")
class TestSiglipTextModel(unittest.TestCase):
    def test_model(self):
        model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        # important: make sure to set padding="max_length" as that's how the model was trained
        inputs = tokenizer(text=texts, padding="max_length", return_tensors="pt")
        # outputs = model(**inputs)
        # last_hidden_state = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output  # pooled (EOS token) states

        m_inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            m_outputs = model(**m_inputs)

        logits_per_image = m_outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image)  # these are the probabilities
        print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
