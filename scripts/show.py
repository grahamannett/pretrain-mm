from dataclasses import dataclass
import torch

from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser
import transformers
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebTaskProcessor, TaskAdapter

from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.model.fuyu.processing_fuyu import FuyuProcessor
from pretrain_mm import logger

from pretrain_mm.utils.config_utils import BaseConfig
from config.dev import get_dev_config
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.eval_utils import box_pattern
from pretrain_mm.model.combine_embed import CombineEmbeddings
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate


@dataclass
class Config(BaseConfig):
    model_id: str = "adept/fuyu-8b"
    model_output_dir: str = "/data/graham/models/pretrain-mm/fuyu/masked_output_checkpoint_0"  # another possible model: "/data/graham/models/pretrain-mm/fuyu/masked_output"
    processor_output_dir: str = "/data/graham/models/pretrain-mm/fuyu/processor"

    dataset_name: str = "mind2web"

    max_new_tokens: int = 25
    temperature: float = 0.5


# if __name__ == "__main__":
parser = ArgumentParser()
parser.add_arguments(Config, dest="config")
args = parser.parse_args()
config: Config = args.config


processor = FuyuProcessor.from_pretrained(config.model_id, trust_remote_code=True)

model = transformers.AutoModelForCausalLM.from_pretrained(
    # config.model_id,
    config.model_output_dir,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
# patch combine embeddings
model.gather_continuous_embeddings = CombineEmbeddings.gather_continuous_embeddings


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


m2w_info = get_dev_config(config.dataset_name)

train_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    **m2w_info["train"],
)


train_dataset = Mind2Web(train_data_config)
task_processor = Mind2WebTaskProcessor(processor=processor)
print("original width: ", processor.image_processor.size["width"])
# task_processor.add_stop_token("</bb")

sample = train_dataset[2550]


# processor.image_processor.size["width"] = 1290
# sample.image = sample.image.crop((0, 0, 1290, 1080))
# sample.image = sample.image.resize((1920, 1080))
# sample.image = sample.image.crop((0, 0, 1920, 1080))
# sample.image = add_margin(sample.image, 0, 640, 0, 0, (0, 0, 0))
# task_sample = Mind2WebTaskProcessor.preprocessor(sample)
task_sample = task_processor.task_mind2web(sample)
image = task_sample["image"]

target_sizes = torch.tensor([task_sample["image"].size])
# target_sizes = None


image.save("x-cropped.png")
draw = ImageDraw.Draw(image)


draw.rectangle(task_sample["box"], outline="blue", width=3)

# draw.rectangle(
#     (
#         981,
#         731,
#         981 + 235,
#         731 + 22,
#     ),
#     outline="purple",
#     width=5,
# )
image.save("x-target.png")
# breakpoint()

model_inputs = processor(
    text="When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n Refine Your Search",
    images=task_sample["image"],
).to("cuda")
outputs = model.generate(**model_inputs, max_new_tokens=10)
post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs, target_sizes=target_sizes)[0]
decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
y1, x1, y2, x2 = list(map(int, box_pattern.search(decoded_outputs).groups()))

draw.rectangle((x1, y1, x2, y2), outline="green", width=3)
# draw.rectangle((y1, x1, y2, x2), outline="green", width=6)

inputs = {"text": task_sample["text"], "images": task_sample["image"]}

# this generates
# for idx in range(2):
#     logger.info(f"doing i: {idx}")
#     outputs = generate_helper(
#         model,
#         processor,
#         inputs,
#         config.max_new_tokens,
#         task_processor.extra_stop_tokens,
#         0.5,
#         disable_progress_bar=False,
#     )
#     logger.info(f"got outputs: {outputs[0, -30:]}")

#     post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs, target_sizes=target_sizes)[0]
#     logger.info(f"post_processed: {post_processed_bbox_tokens[-30:]}")
#     decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=False)
#     logger.info(f"fully generated: {decoded_outputs[-30:]}")

#     # model_outputs = model(**batch)
#     try:
#         y1, x1, y2, x2 = list(map(int, box_pattern.search(decoded_outputs).groups()))
#         draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
#     except Exception as err:
#         logger.warn(f"Error: {err}")

neg_candidates_to_draw = sample.neg_candidates[500:510]

for cand in neg_candidates_to_draw:
    attrs = parse_candidate(cand, parse_bounding_box=True)["attributes"]
    coords = list(map(int, attrs["bounding_box_rect"]))
    draw.rectangle(coords, outline="purple", width=3)

image.save("x-pred.png")
breakpoint()

# output_embeddings = get_embeddings(model, **batch)

# breakpoint()

# compute loss based on box.  0 is perfect 1 means not even bbox.


# metric_val = loc_metric_from_str(target_str=combined_text, pred_str=decoded_outputs)


# TESTING
# ----

# import io
# import requests

# bbox_prompt = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\n Williams"
# bbox_image_url = (
#     "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.jpeg"
# )
# bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))
# original_size = bbox_image_pil.size
# bbox_image_pil = bbox_image_pil.crop((0, 0, 1280, 1080))
# bbox_image_pil = bbox_image_pil.resize(original_size)
# model_inputs = processor(text=bbox_prompt, images=bbox_image_pil).to("cuda")
# outputs = model.generate(**model_inputs, max_new_tokens=10)
# post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
# decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
# y1, x1, y2, x2 = list(map(int, box_pattern.search(decoded_outputs).groups()))
# draw = ImageDraw.Draw(bbox_image_pil)
# draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

# bbox_image_pil.save("x-will.png")
# breakpoint()
# # ####
