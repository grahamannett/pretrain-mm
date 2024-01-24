import json
import os
import random
import unittest

import torch
import transformers
from PIL import Image, ImageDraw

from config.dev import get_dev_config
from pretrain_mm import constants
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, Mind2WebTaskProcessor, TaskAdapter
from pretrain_mm.model.fuyu import MODEL_ID, CombineEmbeddings, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.utils.generate_utils import generate_helper
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


class TestContextLength(unittest.TestCase):
    def test_backpass(self):
        num_backwards = 5000
        min_initial_context_length = int(os.environ.get("MIN_INPUT_IDS", 2000))
        dataset_name = "mind2web"

        m2w_info = get_dev_config(dataset_name)

        dataset_config = Mind2WebConfig(
            task_dir=m2w_info["task_dir"],
            **m2w_info["train"],
        )
        dataset = Mind2Web(dataset_config)
        # model = transformers.AutoModelForCausalLM.from_pretrained(

        model = FuyuForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        processor = FuyuProcessor.from_pretrained(MODEL_ID)

        pretrain_task_processor = Mind2WebPretrainProcessor()
        pretrain_task_processor.cands_range = (90, 100)

        task_processor = Mind2WebTaskProcessor(
            processor=processor,
            ignore_index=constants.IGNORE_INDEX,
        )

        transforms = {
            "pretrain_task": pretrain_task_processor.pretrain_func_generate_possible_actions,
            "processor": task_processor.encode_data,
        }

        task_dataset = TaskAdapter(dataset, transforms=transforms)
        sample = task_dataset[0]

        all_ids = list(processor.vocab.values())

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        image_patches = sample["image_patches"]
        image_patches_indices = sample["image_patches_indices"]

        extra_attention_mask_val = torch.tensor([[1]])
        extra_image_patches_indices_val = torch.tensor([[-1]])

        optimizer = torch.optim.SGD(model.parameters())

        def add_extra_fn(input_ids, attention_mask, image_patches_indices):
            extra_input_id_val = torch.tensor(
                [[random.choice(all_ids)]], dtype=input_ids.dtype, device=input_ids.device
            )
            input_ids = torch.cat([input_ids, extra_input_id_val], dim=-1)
            attention_mask = torch.cat([attention_mask, extra_attention_mask_val], dim=-1)
            image_patches_indices = torch.cat([image_patches_indices, extra_image_patches_indices_val], dim=-1)

            return input_ids, attention_mask, image_patches_indices

        while input_ids.shape[-1] < min_initial_context_length:
            input_ids, attention_mask, image_patches_indices = add_extra_fn(
                input_ids, attention_mask, image_patches_indices
            )

        for _ in range(num_backwards):
            torch.cuda.empty_cache()

            input_ids, attention_mask, image_patches_indices = add_extra_fn(
                input_ids, attention_mask, image_patches_indices
            )

            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_patches=image_patches,
                image_patches_indices=image_patches_indices,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            print("context length", input_ids.shape[-1])


class TestModel(unittest.TestCase):
    def setUp(self):
        MODEL_ID = "adept/fuyu-8b"
        self.image_size = (1000, 1000)

    def test_decode(self):
        processor = FuyuProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        text = "text, generate the corresponding bounding box.\n Williams<box>388, 428, 404, 900</box>"
        label = "Williams<box>388, 428, 404, 900</box>"

        image = torch.rand(3, 1280, 1280)
        outputs = processor(text=text, images=image).input_ids

        post_processed_bbox_tokens = processor.post_process_box_coordinates(
            outputs, target_sizes=torch.tensor([image.shape[-2:]])
        )[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)

        metric_val = loc_metric_from_str(target_str=label, pred_str=decoded_outputs)

    def test_generate_helper(self):
        text = "Caption the following image\n"
        additional_tokens = ["black", " The", "The", "image", "the image", "The Image", "a"]
        image = torch.rand(3, *self.image_size)

        max_new_tokens = 100
        temperature = 0.7

        processor = FuyuProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        stop_tokens = FuyuConstants.get_stop_tokens(
            processor,
            additional_tokens=additional_tokens,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        model.gather_continuous_embeddings = CombineEmbeddings.gather_continuous_embeddings

        outputs_helper = generate_helper(
            model,
            processor=processor,
            inputs={"text": text, "images": image},
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
        )

        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            eos_token_id=stop_tokens,
            pad_token_id=processor.pad_token_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        processor.decode(outputs[0])
        processor.decode(outputs_helper[0])

    def test_ocr(self):
        # import io, requests

        # bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
        # image = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
        # ignore above

        # 1087, 63, 1180, 85 (x1, y1, x2, y2)
        # box = [1087, 63, 1180, 85]
        # box = [55, 772, 431, 799]
        box = (
            [21, 59, 122, 90],
            [1087, 63, 1180, 85],
            [167, 67, 344, 84],
            [201, 623, 320, 641],
            [384, 623, 452, 642],
            [601, 623, 642, 642],
            [775, 623, 818, 642],
            [940, 623, 1011, 643],
            [604, 641, 697, 660],
            [202, 643, 264, 660],
            [777, 643, 820, 660],
            [385, 643, 505, 661],
            [941, 644, 1014, 661],
            [55, 772, 431, 799],
        )
        # box = [777, 643, 820, 660], [385, 643, 505, 661], [941, 644, 1014, 661], [55, 772, 431, 799]
        box = box[-3]
        # box = [428, 388, 488, 404]  # from hf
        x1, y1, x2, y2 = box
        # text = f"When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box. \n <box>{y1}, {x1}, {y2}, {x2}</box>"
        text = (
            f"Given the HTML Perform OCR to extract text contained within the box.\n<box>{y1}, {x1}, {y2}, {x2}</box>"
        )
        image = Image.open("tests/fixtures/screenshot0.png")

        # image = image.crop((0, 0, 1280, 1080))
        # image = image.crop((0, 0, 1400, 1080))

        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        model = CombineEmbeddings.patch_gather_embeddings(model)

        processor = FuyuProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        inputs = processor(text=text, images=image, add_boa_token=True, add_bos_token=True, return_tensors="pt")

        # processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        # inputs = processor(text=text, images=image)

        inputs = inputs.to(model.device)

        print(f"len of image patches: {inputs['image_patches'].size(1)}")

        max_new_tokens = 30
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)
        decoded_outputs = processor.tokenizer.convert_ids_to_tokens(post_processed_bbox_tokens[-50:])
        # decoded_outputs = processor.decode(post_processed_bbox_tokens[-50:], skip_special_tokens=True)

        print("output is:", decoded_outputs)

        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        image.save("xstmp/fuyu-ss0.png")
