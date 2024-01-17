import json
import random
import unittest

import torch
import transformers

from pretrain_mm.model.fuyu import CombineEmbeddings
from pretrain_mm.model.fuyu import FuyuProcessor, FuyuConstants
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


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model_id = "adept/fuyu-8b"
        self.image_size = (1000, 1000)

    def test_decode(self):
        processor = FuyuProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        text = "text, generate the corresponding bounding box.\n Williams<box>388, 428, 404, 900</box>"
        label = "Williams<box>388, 428, 404, 900</box>"

        image = torch.rand(3, 1280, 1280)
        outputs = processor(text=text, images=image).input_ids

        post_processed_bbox_tokens = processor.post_process_box_coordinates(
            outputs, target_sizes=torch.tensor([image.shape[-2:]])
        )[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)

        metric_val = loc_metric_from_str(target_str=label, pred_str=decoded_outputs)
        breakpoint()

    def test_context_length(self):
        text = "1 2 3 4 5 7 8 9 " * 100
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # model.gather_continuous_embeddings = CombineEmbeddings.gather_continuous_embeddings
        model = CombineEmbeddings.patch_gather_embeddings(model)
        processor = FuyuProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        image = torch.rand(3, *(self.image_size))
        inputs = processor(text=text, images=image, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_patches = inputs["image_patches"]
        image_patches_indices = inputs["image_patches_indices"]

        extra_attention_mask_val = torch.tensor([[1]])
        extra_image_patches_indices_val = torch.tensor([[-1]])

        while True:
            extra_input_id_val = torch.tensor([[random.randint(100, 1000)]])

            input_ids = torch.cat([input_ids, extra_input_id_val], dim=-1)
            attention_mask = torch.cat([attention_mask, extra_attention_mask_val], dim=-1)
            image_patches_indices = torch.cat([image_patches_indices, extra_image_patches_indices_val], dim=-1)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_patches=image_patches,
                    image_patches_indices=image_patches_indices,
                )

            latest_shape = input_ids.shape[-1]
            print("context length", latest_shape)

    def test_generate_helper(self):
        text = "Caption the following image\n"
        additional_tokens = ["black", " The", "The", "image", "the image", "The Image", "a"]
        image = torch.rand(3, *self.image_size)

        max_new_tokens = 100
        temperature = 0.7

        processor = FuyuProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        stop_tokens = FuyuConstants.get_stop_tokens(
            processor,
            additional_tokens=additional_tokens,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
