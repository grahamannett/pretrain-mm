from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM
from PIL import Image

from pretrain_mm.datasets import pretrain_instructions
from pretrain_mm.model.fuyu import MODEL_ID, CombineEmbeddings, FuyuConstants, FuyuProcessor

from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import box_pattern


@dataclass
class Config:
    model_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain"
    processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    device_map: str = "auto"

    # input related
    instruction = pretrain_instructions.GenerateNumPotentialActions(num_candidates=1)
    input_img: str = "tests/fixtures/screenshot0.png"

    # generate related
    max_new_tokens: int = 10
    # temperature
    temperature: float = 1.0

    def __post_init__(self):
        self.input_img = Image.open(self.input_img)


def examine(config):
    image = config.input_img
    text = f"{config.instruction(num_candidates=3)}{FuyuConstants.boa_string}  \n{FuyuConstants.token_bbox_open_string}"

    processor = FuyuProcessor.from_pretrained(MODEL_ID)

    # tok_id_dict = FuyuConstants.get_all_ids(processor)
    # breakpoint()

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        device_map=config.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = CombineEmbeddings.patch_gather_embeddings(model)

    inputs = processor(text=text, images=image, add_boa_token=False, add_bos_token=True)
    inputs = inputs.to(model.device)

    input_ids = inputs.input_ids[0]

    # outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens)
    stop_tokens = FuyuConstants.get_stop_tokens(processor)
    output = generate_helper(
        model,
        model_inputs=inputs,
        max_new_tokens=config.max_new_tokens,
        stop_tokens=stop_tokens,
        temperature=config.temperature,
    )

    bos_idx = (output[0] == processor.vocab[FuyuConstants.bos_string]).nonzero().view(-1)[0].item()

    output = output[0, bos_idx:]

    generated_text = processor.full_decode(output)
    decoded_tokens = processor.convert_ids_to_tokens(output)

    bounding_box = box_pattern.search(generated_text).groups()

    if len(bounding_box) != 4:
        raise ValueError(f"Could not find bounding box in generated text: {generated_text}")

    bounding_box = list(map(int, bounding_box))
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config
    examine(config)
