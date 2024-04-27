"""
SCRIPT TO EXAMINE A MODEL AND ITS GENERATED OUTPUT

"""

from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM

from config.dev import get_dev_config
from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, pretrain_instructions
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.model.fuyu import MODEL_ID, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import box_pattern


@dataclass
class Config:
    model_path: str = "/data/graham/models/pretrain-mm/fuyu/latest"
    # processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

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


def generate_and_draw_box(model, inputs, image):
    pass


def examine(config):
    image = config.input_img
    text = f"{config.instruction(num_candidates=1)}{FuyuConstants.boa_token} \n {FuyuConstants.bbox_open_string}"

    m2w_info = get_dev_config("mind2web")
    ds_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        # subset=config.data_subset,
        **m2w_info["train"],
    )
    # config = Mind2WebConfig()
    dataset = Mind2Web(config=ds_config)

    sample = dataset[0]
    image = sample.image

    image_width, image_height = image.size
    image = image.crop((0, 0, VIEWPORT_SIZE_DICT["width"], VIEWPORT_SIZE_DICT["height"]))
    # image = image.crop((0, 0, image_width, VIEWPORT_SIZE_DICT["height"]))

    # train_dataset = Mind2Web(train_data_config)

    processor = FuyuProcessor.from_pretrained(f"{config.model_path}/processor")
    stop_ids = FuyuConstants.get_stop_ids(processor)

    model = FuyuForCausalLM.from_pretrained(
        config.model_path,
        device_map=config.device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    breakpoint()

    inputs = processor(text=text, images=image, add_boa_token=False, add_bos_token=True)
    inputs = inputs.to(model.device)

    input_ids = inputs.input_ids[0]

    num_gens = 5
    draw = ImageDraw.Draw(image)

    output = model.generate(**inputs, max_new_tokens=50)
    gen_text = processor.full_decode(output)

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(sample.cleaned_html, "html.parser")

    for gen_i in range(num_gens):
        output = generate_helper(
            model,
            model_inputs=inputs,
            max_new_tokens=config.max_new_tokens,
            stop_ids=stop_ids,
            temperature=config.temperature,
        )

        bos_idx = (output[0] == processor.vocab[FuyuConstants.bos_token]).nonzero().view(-1)[0].item()

        output = output[0, bos_idx:]

        decoded_tokens = processor.convert_ids_to_tokens(output)
        generated_text = processor.full_decode(output)
        generated_text_ = generated_text.split("\n")[1]

        print(f"{gen_i + 1}. Got decoded tokens: {generated_text_}")

        if box_match := box_pattern.search(generated_text):
            if len(box_match.groups()) != 4:
                raise ValueError(f"Could not find bounding box in generated text: {generated_text}")

            box_vals = list(map(int, box_match.groups()))
            # they come in as y1, x1, y2, x2
            box_vals = [box_vals[1], box_vals[0], box_vals[3], box_vals[2]]
            x1, y1, x2, y2 = box_vals

            if (y1 > y2) or (x1 > x2):
                # raise ValueError(f"Invalid bounding box: {box_match.groups()}")
                logger.warn(f"Invalid bounding box: {box_match.groups()}")
            else:
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                draw.text((x1, y1), f"{gen_i}", fill="red", font_size=30)
                image.save("tmp/examine.png")
    breakpoint()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config
    examine(config)
