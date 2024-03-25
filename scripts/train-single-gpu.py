import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import torchmetrics
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebEncoder, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer import Trainer
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseTrainConfig, LocalDataConfig, WandBConfig


wandb_config = WandBConfig(group="testing/pretrain-fuyu", job_type="pretrain")

infolm_metric = torchmetrics.text.infolm.InfoLM(
    "google/bert_uncased_L-2_H-128_A-2",
    idf=False,
    verbose=False,
    information_measure="l2_distance",
)


@dataclass
class TrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    do_eval: bool = True
    do_eval_pre: bool = False
    do_batch_eval_every: int = None
    eval_num_samples: int = 2
    eval_num_generations: int = 2
    eval_use_past_key_values: bool = False
    output_dir: str = None  # "output/model_output"
    save_every: Optional[str] = choice("epoch", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    loc_type: str = "box"
    IGNORE_INDEX: int = constants.IGNORE_INDEX
    loc_before_action_repr: bool = False
    max_length: int = 2700
    get_text_from: str = choice("html", "ocr", default="ocr")
    ocr_use_gpu: bool = False

    data_subset: int = None
    epochs: int = 10
    batch_size: int = 1
    grad_accum_steps: int = 4

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 0
    dl_pin_memory: bool = True
    dl_prefetch_factor: int = None
    dl_persistent_workers: bool = False
    dl_worker_init: bool = False
    dl_timeout: float = 5

    optimizer_type: str = "AdamW"  # allow for
    use_groups: bool = True
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    learning_rate: float = 1e-04
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    gamma: float = 0.85
    eps: float = 1e-8
    momentum: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)

    gradient_checkpointing: bool = False

    # tokenzier related
    extra_tokenizer_toks: bool = True

    # task related
    instruction: str = "AssistantResponse"
    task_function: str = "agent_training"
    skip_include_text: bool = False

    use_profiler: bool = False
    test_dataloader: bool = False

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"

        if (self.dl_num_workers == 0) and (self.dl_timeout != 0):
            logger.warn("timeout must be 0 if num_workers is 0.  Setting to 0")
            self.dl_timeout = 0

        if (self.dl_num_workers == 0) and (self.dl_prefetch_factor != None):
            logger.warn("prefetch factor must be None if num_workers is 0.  Setting to None")
            self.dl_prefetch_factor = None


@torch.no_grad()
def eval_with_metric(
    config: TrainConfig,
    data_iter: Iterable[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    metric_fn: Callable = infolm_metric,
    max_new_tokens: int = 20,
    do_sample: bool = True,
    temperature: float = 0.1,
):
    metric_vals = []
    generated_strs = []

    model.eval()
    for n in range(config.eval_num_samples):
        batch = next(data_iter)

        if not batch.is_valid:
            # might need to skip during eval
            logger.log(f"skipping: {n} eval batch due to invalid batch")
            continue

        label = batch.extra["label"]
        gen_start_idx = batch.input_ids.shape[-1]
        batch.to(model.device)

        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.pad_token_id,
        )

        generated_output = processor.full_decode(output[:, gen_start_idx:])
        generated_strs.append(generated_output)

        if FuyuConstants.eos_string in generated_output:
            generated_output = generated_output.rstrip(FuyuConstants.eos_string)

        metric_val = metric_fn(generated_output, label)
        metric_vals.append(metric_val.item())

    avg_metric_vals = sum(metric_vals) / len(metric_vals)
    eval_results = {"log/eval/batch_metric": avg_metric_vals, "eval/generated_strs": generated_strs}

    return eval_results


# -----------------------------------
#  __  __    _    ___ _   _         |
# |  \/  |  / \  |_ _| \ | |        |
# | |\/| | / _ \  | ||  \| |        |
# | |  | |/ ___ \ | || |\  |        |
# |_|  |_/_/   \_\___|_| \_|        |
#
# NOTE:


parser = ArgumentParser()
parser.add_arguments(TrainConfig, dest="pretrain_config")
parser.add_arguments(wandb_config, dest="wandb_config", prefix="wandb.")
parser.add_arguments(LocalDataConfig, dest="local_data_config", prefix="local_data.")

args = parser.parse_args()

config: TrainConfig = args.pretrain_config

# setup wandb + check config such that yaml printed config is in wandb console logs
logger.tools.setup_wandb(wandb_config=args.wandb_config, config=config)
logger.tools.setup_local_data(local_data_config=args.local_data_config, config=config)
logger.tools.check_train_config(train_config=config)

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

processor = FuyuProcessor.from_pretrained(config.model_id)
model = FuyuForCausalLM.from_pretrained(config.model_id, device_map=config.device, torch_dtype=torch.bfloat16)

train_task_processor = Mind2WebPretrainProcessor(
    instruction=config.instruction,
    task_function=config.task_function,
    get_text_from=config.get_text_from,
    # ocr_preprocessed=torch.load("output/processed/train_ds_raw_output.pt"),
)

task_processor = Mind2WebEncoder(
    processor=processor,
    ignore_index=config.IGNORE_INDEX,
    max_length=config.max_length,
    encode_kwargs={"label_mask_text_ids": True},
)

# generate possible actions pretrain task
transforms = {
    "pretrain_task": train_task_processor,
    "encode": task_processor.encode_data,
}


# to eval, we want the label but not as part of the encoded data
def encode_for_eval(sample, encode):
    label = sample.pop("label")
    encoded_sample = encode(sample)
    encoded_sample.extra["label"] = label
    return encoded_sample


train_dataset_adapter = TaskAdapter(train_dataset, transforms=transforms)
test_dataset_adapter = TaskAdapter(
    test_dataset,
    transforms={
        "pretrain_task": train_task_processor,
        "encode": partial(encode_for_eval, encode=task_processor.encode_data),
    },
)

collate_fn = DataCollator(processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=True)

train_dl = torch.utils.data.DataLoader(
    train_dataset_adapter,
    collate_fn=collate_fn,
    batch_size=config.batch_size,
    num_workers=config.dl_num_workers,
    pin_memory=config.dl_pin_memory,
    prefetch_factor=config.dl_prefetch_factor,
    persistent_workers=config.dl_persistent_workers,
    timeout=config.dl_timeout,
    # worker_init_fn=pretrain_task_processor._worker_init_func if config.dl_worker_init else None,
    shuffle=True,
)

test_dl = torch.utils.data.DataLoader(
    test_dataset_adapter,
    collate_fn=DataCollator(processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=False),
    batch_size=config.batch_size,
    num_workers=config.dl_num_workers,
    pin_memory=config.dl_pin_memory,
    prefetch_factor=config.dl_prefetch_factor,
    timeout=config.dl_timeout,
    persistent_workers=config.dl_persistent_workers,
    shuffle=True,  # shuffle since we may create new iter each eval
)

num_training_steps = len(train_dl) * config.epochs
optimizer = get_optimizer(
    model,
    optimizer_type=config.optimizer_type,
    # general optimizer kwargs
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    use_groups=config.use_groups,
    #  adam related
    betas=config.betas,
    eps=config.eps,
    # sgd related
    momentum=config.momentum,
)
scheduler = get_scheduler(
    config.scheduler_type,
    optimizer,
    num_training_steps=num_training_steps,
    warmup_ratio=config.warmup_ratio,
)

show_optim_info(optimizer, scheduler, num_training_steps, warmup_ratio=config.warmup_ratio)


def save_model_callback(model, epoch, trainer, **kwargs):
    if trainer.config.output_dir is None:
        return

    output_path = f"{trainer.config.output_dir}"
    if trainer.config.save_every == "epoch":
        output_path += f"/checkpoint_{epoch}"
    model.save_pretrained(output_path)
    logger.log(f"model for epoch: {epoch} saved to: {output_path}")


def clean_for_log(data: dict):
    return {k.lstrip("log/"): v for k, v in data.items() if k.startswith("log/")}


def _show_train_pre():
    logger.log(f"show that we started training with `{len(train_dl)}` batches")


def _show_train_post_needs_args(val1: str, optional_val: int = 10):
    logger.log(f"showing how you would need to do this one! {val1} and {optional_val}")


test_iter = iter(test_dl)


def _do_eval_every_batch(batch_idx: int, batch_loss: float = None):
    if config.do_batch_eval_every and ((batch_idx % config.do_batch_eval_every) == 0):
        eval_results = eval_with_metric(
            config,
            data_iter=test_iter,
            model=model,
            metric_fn=infolm_metric,
            max_new_tokens=15,
        )
        # log data to wandb
        logger.log_data(clean_for_log(eval_results))

        # create str to log to terminal
        log_str = f"[B-IDX:{batch_idx}]"
        if batch_loss:
            log_str += f"[L:{batch_loss:.3f}]"

        logger.log(log_str)

    model.train()


callbacks = Trainer.CallbackHandler(
    {
        Trainer.Events.train_pre: (_show_train_pre, _show_train_post_needs_args),
        # Trainer.Events.batch_pre: (_do_eval_every_batch),
        Trainer.Events.batch_post: (_do_eval_every_batch),
    }
)

trainer = Trainer(config=config, callbacks=callbacks)

trainer.setup_helpers(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dl,
    processor=processor,
)

# save here
if config.output_dir:
    processor.save_pretrained(f"{config.output_dir}/processor")


# do_eval_callback(config, test_dl, model, metric=infolm_metric)
trainer.train()
