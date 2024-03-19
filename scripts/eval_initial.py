import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import wandb
from bs4 import BeautifulSoup
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebEncoder, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.model.fuyu import FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig  # , check_train_config, setup_wandb
from pretrain_mm.utils.dev_utils import make_profiler
from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.utils.json_utils import read_json
from pretrain_mm.utils.image_utils import read_image_from_b64


@dataclass
class WandBConfig(BaseWandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "eval"


@dataclass
class EvalConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    output_dir: str = None  # "output/model_output"

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    IGNORE_INDEX: int = constants.IGNORE_INDEX

    data_subset: int = None
    batch_size: int = 1

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    # generate kwargs
    max_new_tokens: int = 20
    temperature: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"


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
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    logger.info("starting eval")

    metrics = []
    losses = []
    progress = logger.progress()
    for batch in dataloader:
        batch.to(model.device)
        outputs = model.generate(**batch, **generate_kwargs)

        metric = eval_func(outputs)

        # EVAL RELATED
        eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_tokens)

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

                if "]" not in action_repr_loc:
                    breakpoint()

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EvalConfig, dest="eval_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")

    args = parser.parse_args()

    config: EvalConfig = args.eval_config
    wandb_config: WandBConfig = args.wandb_config
    model_config = config.model_config

    # setup wandb + check config such that yaml printed config is in wandb console logs
    logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
    logger.tools.check_train_config(config)

    m2w_info = get_dev_config(config.dataset_name)

    m2w_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        subset=config.data_subset,
        **m2w_info["train"],
    )

    eval_dataset = Mind2Web(m2w_data_config)

    map_fn = make_dataset_map_fn(m2w_data_config.task_dir, m2w_data_config.screenshot_file)
    _dataset = eval_dataset.dataset.map(
        map_fn,
        batched=True,
        # change all below after dev
        batch_size=4,
        num_proc=1,
        load_from_cache_file=False,
    )
    processor = FuyuProcessor.from_pretrained(config.model_id)

    model = FuyuForCausalLM.from_pretrained(config.model_id, device_map=config.device, torch_dtype=torch.bfloat16)

    task_encoder = Mind2WebEncoder(
        processor=processor,
        ignore_index=config.IGNORE_INDEX,
        loc_before_action_repr=config.loc_before_action_repr,
        max_length=config.max_length,
    )

    # generate possible actions pretrain task
    transforms = {
        "pretrain_task": pretrain_task_processor.pretrain_func_generate_possible_actions,
        "encode": task_encoder.encode_data,
    }

    task_eval_dataset = TaskAdapter(train_dataset, transforms=transforms)
    # sample = task_train_dataset[0]
    # task_eval_dataset = TaskAdapter(test_dataset, transforms=pretrain_task_processor.pretrain_func)

    # sample = task_train_dataset[1000]
    collate_fn = DataCollator(processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=True)
    eval_dataloader = torch.utils.data.DataLoader(
        task_eval_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.dl_num_workers,
        pin_memory=config.dl_pin_memory,
        shuffle=True,
    )
    # test_dl = torch.utils.data.DataLoader(
    #     task_train_dataset,
    #     collate_fn=collate_fn,
    #     batch_size=config.batch_size,
    #     num_workers=config.dl_num_workers,
    #     pin_memory=config.dl_pin_memory,
    # )

    # pretrain(
    #     config,
    #     model,
    #     train_dl,
    #     eval_dataset=test_dataset,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     task_processor=task_processor,
    # )
