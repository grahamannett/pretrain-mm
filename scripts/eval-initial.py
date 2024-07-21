import os
import random
from dataclasses import dataclass
from functools import partial

import torch
import torchmetrics
import tyro
from bs4 import BeautifulSoup

import wandb

# from simple_parsing import ArgumentParser, choice
from config.dev import get_dev_config
from config.model_configs import ExperimentConfigModelInfo, ModelInitInfo

# from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.model.fuyu import FuyuConstants
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler
from pretrain_mm.utils.config_utils import BaseTrainConfig, FromConfig, WandBConfig
from pretrain_mm.utils.generate_utils import StopOnToken
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.utils.json_utils import read_json
from pretrain_mm.utils.eval_utils import remove_label


@dataclass
class WandBConfigExp(WandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "eval"
    tags: tuple[str, ...] = ("eval", "mind2web")


@dataclass
class EvalConfig(BaseTrainConfig):
    wandb: WandBConfig = FromConfig[WandBConfigExp]

    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    # model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    # model_config = FuyuInfo
    model_path: ExperimentConfigModelInfo = ExperimentConfigModelInfo.Fuyu

    output_dir: str = None  # "output/model_output"

    # dataset
    dataset_name: str = "mind2web"
    get_text_from: str = "ocr"

    data_subset: int = None
    batch_size: int = 1

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    # generate kwargs
    max_new_tokens: int = 20
    temperature: float = 1.0
    do_sample: bool = True

    max_length: int = 2700
    add_cand_outline: bool = False
    skip_include_text: bool = False
    # MARK: mask related
    label_mask_text_ids: bool = True
    label_mask_image_patches: bool = True

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"

    @property
    def model_info(self) -> ModelInitInfo:
        return self.model_path.resolve()


def eval_model(
    config: EvalConfig,
    model,
    dataloader,
    generate_kwargs: dict = {
        "max_new_tokens": EvalConfig.max_new_tokens,
        "temperature": EvalConfig.temperature,
        "do_sample": EvalConfig.do_sample,
    },
    eval_func: callable = None,
):
    # train_config.masked_values = [71019, 71011]
    # masked_values = torch.tensor(train_config.masked_values) if train_config.masked_values else None
    stop_ids = FuyuConstants.get_stop_ids(processor)

    logger.info("starting eval")

    metrics = []
    losses = []
    progress = logger.progress()
    for batch in dataloader:
        batch.to(model.device)
        outputs = model.generate(**batch, **generate_kwargs)

        metric = eval_func(outputs)

        # EVAL RELATED
        eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_ids)

        eval_acc_metric = eval_metrics["eval/acc_metric"]
        logger.log(f"E[{epoch}][L:{epoch_loss:.2f}][LR:{scheduler.get_last_lr()[0]:.4f}][Eval:{eval_acc_metric:.4f}]")
        wandb.log({"train/epoch_loss": epoch_loss, **eval_metrics})


def make_eval_func(processor):
    def eval_func(outputs, label: str):
        decoded_output = processor.full_decode(outputs)
        if "</box>" in decoded_output:
            decoded_output = decoded_output.split("</box>")[1]

        decoded_output.rstrip("|ENDOFTEXT|")
        pred_str = " ".join(decoded_output.split())

        if pred_str in label:
            return 1
        return 0

    # having the model output the bounding box is much more convoluted because i dont know what the text may be.
    # eval based on box accuracy
    def eval_func(outputs, label: list[int]):
        decoded_output = processor.full_decode(outputs)
        if box_match := box_pattern.search(decoded_output):
            box_match = box_match.group()
            box_vals = list(map(int, box_match.groups()))
            metric = [(l1 - l2) ** 2 for l1, l2 in zip(label, box_vals)]


# go through data and make
def make_dataset_map_fn(task_dir: str, screenshot_file: str):
    def dataset_map_fn(data: dict):
        output = {
            "div_text": [],
            "bounding_box": [],
            "annotation_id": [],
            "json_filepath": [],
            "action_idx": [],
            "action_repr": [],
            "node_text": [],
        }

        for idx, (ann_id, actions, action_reprs) in enumerate(
            zip(data["annotation_id"], data["actions"], data["action_reprs"])
        ):
            json_filepath = f"{task_dir}/task/{ann_id}/{screenshot_file}"

            # might want to remove later
            json_data = read_json(json_filepath, use_cache=True)

            for act_idx, (action, action_repr) in enumerate(zip(actions, action_reprs)):
                # not clear if i should go through neg_candidates as well?

                image = read_image_from_b64(json_data[act_idx]["before"]["screenshot"])
                action_repr_loc, action_rep_target = action_repr.split("->", 1)

                div_text = action_repr_loc.split("]", 1)[1].strip()

                if div_text == "":
                    continue

                pos_candidate = action["pos_candidates"][0]
                pos_cand = m2w_utils.parse_candidate(action["pos_candidates"][0], parse_bounding_box=True, to_int=True)
                bounding_box = pos_cand["attributes"]["bounding_box_rect"]  # x1, y1, x2, y2

                # dsoup = BeautifulSoup(action["raw_html"], "html.parser")
                soup = BeautifulSoup(action["cleaned_html"], "html.parser")
                node = soup.find(backend_node_id=pos_candidate["backend_node_id"])

                # NOTE: node text may contain sub elements so its kind of a shitty match but finding the
                node_text = node.text

                output["div_text"].append(div_text)
                output["bounding_box"].append(bounding_box)
                output["annotation_id"].append(ann_id)
                output["json_filepath"].append(json_filepath)
                output["action_idx"].append(act_idx)
                output["action_repr"].append(action_repr)
                output["node_text"].append(node_text)

        data["bounding_box"] = bounding_boxes
        return data

    return dataset_map_fn


config = tyro.cli(EvalConfig)

# not entirely necessary to make these vars but was previously using simple-parsing
wandb_config: WandBConfig = config.wandb
model_info = config.model_info

ModelInfo = config.model_info
ModelConstants = ModelInfo.ModelConstants

ModelConfigCls = ModelInfo.ModelConfigCls
ModelCls = ModelInfo.ModelCls
ModelProcessorCls = ModelInfo.ProcessorCls


logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
logger.tools.check_exp_config(config=config, exp_type="initial-eval")

m2w_info = get_dev_config(config.dataset_name)

train_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    subset=config.data_subset,
    **m2w_info["train"],
)

test_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    subset=config.data_subset,
    **m2w_info["test"],
)

train_dataset = Mind2Web(train_data_config)
test_dataset = Mind2Web(test_data_config)


# model_chop = {"num_hidden_layers": 1, "text_config": {"model_type": "persimmon", "num_hidden_layers": 1}}
# model_config = ModelConfigCls.from_pretrained(model_info.model_name) if ModelConfigCls else None
# model = ModelCls.from_pretrained(model_info.model_name, config=model_config, device_map=config.device)
processor = ModelProcessorCls.from_pretrained(model_info.model_name, **model_info.tokenizer_kwargs)
model = ModelCls.from_pretrained(model_info.model_name, torch_dtype=torch.bfloat16, device_map=config.device)
# this goes from raw sample -> sample in task format
task_processor: Mind2WebPretrainProcessor = Mind2WebPretrainProcessor(
    get_text_from=config.get_text_from,
    tokenizer_constants=ModelConstants,
)


encode_func = partial(
    processor.encode_sample,
    # label_mask_image_patches=config.label_mask_image_patches,
    # label_mask_text_ids=config.label_mask_text_ids,
    max_length=config.max_length,
    truncation=True,
)

ocr_bounding_box_completion = partial(task_processor.ocr_eval)

transforms = {
    "task": ocr_bounding_box_completion,  # or agent_train_func
    "encode": encode_func,
}
train_dataset_adapter = TaskAdapter(
    train_dataset,
    transforms=transforms,
)

sample = train_dataset_adapter[0]


def eval_ocr_bounding_box(
    model,
    dataset,
    transforms,
    generate_kwargs: dict = {
        "max_new_tokens": EvalConfig.max_new_tokens,
        "temperature": EvalConfig.temperature,
        "do_sample": EvalConfig.do_sample,
    },
    max_new_tokens: int = 7,
    do_sample: bool = True,
    temperature: float = 0.1,
):
    stopping_criteria: list[callable] = [StopOnToken(ModelConstants.get_stop_ids(tokenizer=processor.tokenizer))]

    wer = torchmetrics.text.WordErrorRate()

    def _batch_transform(batch):
        for transform in transforms:
            batch = transform(batch)
            if batch is None:
                return None
        return batch

    for b_idx, batch in enumerate(dataset):
        if (batch := _batch_transform(batch)) is None:
            logger.info(f"Skipping batch: {b_idx}")
            continue

        try:
            boa_idx = processor.get_inputs_start_idx(batch.input_ids, labels=batch.labels, offset=-1)
            batch, (rem_ids, rem_lab) = remove_label(batch, to_idx=boa_idx)
        except:
            breakpoint()

        batch = batch.to(model.device)

        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
        # decoded_output = processor.full_decode(output)

        generated_label = output[:, boa_idx:]
        decoded_generated_label = processor.full_decode(generated_label)
        if "|ENDOFTEXT|" in decoded_generated_label:
            decoded_generated_label = decoded_generated_label.split("|ENDOFTEXT|")[0]

        text_target = batch.extra["label"]

        metric_val = torchmetrics.functional.text.word_error_rate(decoded_generated_label, text_target)
        logger.info(f"Got metric val: {metric_val}")


eval_ocr_bounding_box(model, train_dataset, transforms=[ocr_bounding_box_completion, encode_func])
