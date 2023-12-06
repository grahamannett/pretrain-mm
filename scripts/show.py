from dataclasses import dataclass

from simple_parsing import ArgumentParser
import transformers
from pretrain_mm.datasets import (
    Mind2Web,
    Mind2WebConfig,
    Mind2WebTaskProcessor,
    TaskAdapter,
    task_mind2web,
)

from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.model.fuyu.processing_fuyu import FuyuProcessor
from pretrain_mm import logger

from pretrain_mm.utils.config_utils import BaseConfig
from config.dev import get_dev_config


@dataclass
class Config(BaseConfig):
    model_name: str = "adept/fuyu-8b"
    model_output_dir: str = "output/model_output"  # where model was saved

    dataset_name: str = "mind2web"

    max_new_tokens: int = 30


# if __name__ == "__main__":
parser = ArgumentParser()
parser.add_arguments(Config, dest="config")
args = parser.parse_args()
config: Config = args.config


processor = FuyuProcessor.from_pretrained(f"{config.model_output_dir}/processor", trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(config.model_output_dir, device_map="auto")


m2w_info = get_dev_config(config.dataset_name)

train_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    **m2w_info["train"],
)

train_dataset = Mind2Web(train_data_config)

task_processor = Mind2WebTaskProcessor(processor=processor)


sample = train_dataset[3]
# task_sample = Mind2WebTaskProcessor.preprocessor(sample)
task_sample = task_mind2web(sample)
task_sample_preprocessed = Mind2WebTaskProcessor.preprocessor(task_sample)
# breakpoint()

batch = processor(text=task_sample["text"], images=task_sample["image"])


batch["input_ids"] = batch["input_ids"].to(model.device)
batch["attention_mask"] = batch["attention_mask"].to(model.device)
batch["image_patches_indices"] = batch["image_patches_indices"].to(model.device)
batch["image_patches"] = [im.to(model.device) for im in batch["image_patches"]]


# top_k=0,
outputs = model.generate(**batch, max_new_tokens=config.max_new_tokens, do_sample=True, temperature=0.6)
post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)


# model_outputs = model(**batch)


# output_embeddings = get_embeddings(model, **batch)

breakpoint()

# compute loss based on box.  0 is perfect 1 means not even bbox.


# metric_val = loc_metric_from_str(target_str=combined_text, pred_str=decoded_outputs)
