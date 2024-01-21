from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser

from transformers import AutoModelForCausalLM
from pretrain_mm.model.fuyu import FuyuProcessor, CombineEmbeddings, MODEL_ID

from pretrain_mm.datasets import pretrain_instructions


@dataclass
class Config:
    model_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain"
    processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    device_map: str = "auto"

    # input related
    # prompt: str = "Given the following page, generate a list of bounding boxes for possible actions. If the bounding box contains text, include the text after the bounding box. \n"
    # prompt = "Generate the bounding box of 3 potential actions for the screenshot. Give the action text if relevant. \n"
    prompt = pretrain_instructions.GenerateNumPotentialActions(num_candidates=3)
    input_img: str = "tests/fixtures/screenshot0.png"

    # generate related
    max_new_tokens: int = 100


def examine(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map=config.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = CombineEmbeddings.patch_gather_embeddings(model)

    processor = FuyuProcessor.from_pretrained(
        MODEL_ID,
    )

    inputs = processor(text=config.prompt, images=config.input_img, add_boa_token=True, add_bos_token=True)
    inputs = inputs.to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)

    decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=False)

    breakpoint()
    decoded_outputs_tokens = processor.tokenizer.convert_ids_to_tokens(
        post_processed_bbox_tokens[-config.max_new_tokens :]
    )

    tokens = processor.convert_ids_to_tokens(outputs)
    breakpoint()

    # ModelInitInfo.model_kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config
    examine(config)
