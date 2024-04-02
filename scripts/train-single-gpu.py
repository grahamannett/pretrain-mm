from functools import partial
import os
from dataclasses import dataclass
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
from pretrain_mm.utils.generate_utils import StopOnToken


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
    train_type: str = choice("epoch", "iter", default="iter")

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    do_eval: bool = True
    do_eval_pre: bool = False
    do_batch_eval_every: int = None
    eval_num_samples: int = 2
    eval_num_generations: int = 2
    eval_use_past_key_values: bool = False
    output_dir: str = None  # "output/model_output"
    save_every_n_batch: int = 200
    save_every: Optional[str] = choice("epoch", "iter", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    loc_type: str = "box"
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
        if self.train_type == "iter" and not (self.num_iters > 0):
            raise ValueError("num_iters must be greater than 0 if train_type is iter")

        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"

        if (self.dl_num_workers == 0) and (self.dl_timeout != 0):
            logger.warn("timeout must be 0 if num_workers is 0.  Setting to 0")
            self.dl_timeout = 0

        if (self.dl_num_workers == 0) and (self.dl_prefetch_factor is not None):
            logger.warn("prefetch factor must be None if num_workers is 0.  Setting to None")
            self.dl_prefetch_factor = None


parser = ArgumentParser()
parser.add_arguments(TrainConfig, dest="pretrain_config")
parser.add_arguments(wandb_config, dest="wandb_config", prefix="wandb.")
parser.add_arguments(LocalDataConfig, dest="local_data_config", prefix="local_data.")

args = parser.parse_args()

config: TrainConfig = args.pretrain_config
# initialize trainer here because it can be useful in the functions below
trainer = Trainer(config=config)


@torch.no_grad()
def eval_with_metric(
    config: TrainConfig,
    data_iter: Iterable[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    metric_fn: Callable = infolm_metric,
    max_new_tokens: int = 15,
    do_sample: bool = True,
    temperature: float = 0.1,
    # do this so that it is initialized every call
    stopping_criteria: list[Callable] = [StopOnToken(FuyuConstants.get_stop_tokens())],
):
    metric_vals = []
    generated_strs = []
    losses = []
    # eval_num_samples = config.eval_num_samples
    model.eval()

    def _remove_label(batch, to_idx):
        batch.labels = None
        batch.input_ids = batch.input_ids[:, :to_idx]
        batch.attention_mask = batch.attention_mask[:, :to_idx]
        batch.image_patches_indices = batch.image_patches_indices[:, :to_idx]
        return batch

    while len(generated_strs) < config.eval_num_samples:
        if not (batch := next(data_iter)).is_valid:
            continue
        batch.to(model.device)
        target_label_str: str = batch.extra["label"]
        boa_idx = processor.get_inputs_start_idx(batch.input_ids, offset=-1)

        # first we just get the loss
        output = model(**batch)
        losses.append(output.loss.item())

        # remove all label from related tensors
        batch = _remove_label(batch, to_idx=boa_idx)

        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.pad_token_id,
            # stop_tokens=stop_tokens,
            stopping_criteria=stopping_criteria,
        )

        if FuyuConstants.eos_string in (generated_output := processor.full_decode(output[:, boa_idx:])):
            generated_output = generated_output.rstrip(FuyuConstants.eos_string)

        generated_strs.append(generated_output)

        metric_val = metric_fn(generated_output, target_label_str)
        metric_vals.append(metric_val.item())

    avg_metric_vals = sum(metric_vals) / len(metric_vals)
    avg_loss = sum(losses) / len(losses)
    return {
        # log/ allows filter logging
        "log/eval/batch_metric": avg_metric_vals,
        "log/eval/batch_loss": avg_loss,
        # unimportant vals
        "generated_strs": generated_strs,
        "losses": losses,
    }


#            _ _ _                _
#   ___ __ _| | | |__   __ _  ___| | _____
#  / __/ _` | | | '_ \ / _` |/ __| |/ / __|
# | (_| (_| | | | |_) | (_| | (__|   <\__ \
#  \___\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
#
# NOTE: Callbacks are used exclusively for trainer


def _do_train_pre():
    show_optim_info(optimizer, scheduler, num_training_steps, warmup_ratio=config.warmup_ratio)
    if config.output_dir:
        logger.info("Using callback to setup train related... saving processor.")
        processor.save_pretrained(f"{config.output_dir}/processor")
    else:
        logger.info("Not saving processor")


def _do_grad_accum_post(batch_idx: int, batch_loss: float):
    logger.log(f"[B-IDX:{batch_idx}][L:{batch_loss:.3f}][LR:{trainer.last_lr:.2e}]")
    logger.log_data({"train/batch_loss": batch_loss, "learning_rate": trainer.last_lr})


def _do_batch_eval(batch_idx: int):
    if config.do_batch_eval_every and (batch_idx > 0) and ((batch_idx % config.do_batch_eval_every) == 0):
        eval_results = eval_with_metric(
            config,
            data_iter=trainer.test_iter,
            model=model,
            metric_fn=infolm_metric,
        )

        _data_logged = logger.log_data_filter(filter_by="log/")(data=eval_results)
        logger.log(f"[Eval|{_data_logged}]")

    if (batch_idx > 0) and (batch_idx % config.save_every_n_batch == 0):
        if config.output_dir:
            # if config.save_every in ["iter", "best"]:
            #     save_dir = f"{config.output_dir}/checkpoint_{batch_idx}"
            #     model.save_pretrained(save_dir)
            # else:
            #     save_dir = f"{config.output_dir}/latest"
            save_dir = f"{config.output_dir}/checkpoint_{batch_idx}"
            model.save_pretrained(save_dir)
            logger.log(f"saving model at batch_idx: {batch_idx} to {save_dir}")

    model.train()


# -----------------------------------
#  __  __    _    ___ _   _         |
# |  \/  |  / \  |_ _| \ | |        |
# | |\/| | / _ \  | ||  \| |        |
# | |  | |/ ___ \ | || |\  |        |
# |_|  |_/_/   \_\___|_| \_|        |
#
# NOTE:


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
    ignore_index=constants.IGNORE_INDEX,
    max_length=config.max_length,
    encode_kwargs={"label_mask_text_ids": True},
)

# generate possible actions pretrain task
transforms = {
    "pretrain_task": train_task_processor,
    "encode": task_processor.encode_data,
}


train_dataset_adapter = TaskAdapter(train_dataset, transforms=transforms)
test_dataset_adapter = TaskAdapter(test_dataset, transforms=transforms)

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
    # DataCollator(processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=False),
    collate_fn=collate_fn,
    batch_size=config.batch_size,
    num_workers=config.dl_num_workers,
    pin_memory=config.dl_pin_memory,
    prefetch_factor=config.dl_prefetch_factor,
    timeout=config.dl_timeout,
    persistent_workers=config.dl_persistent_workers,
    shuffle=True,  # shuffle since we may create new iter each eval
)

if config.train_type == "epoch":
    num_training_steps = len(train_dl) * config.epochs
    run_func = trainer.train
elif config.train_type == "iter":
    num_training_steps = config.num_iters
    run_func = trainer.train_num_iters

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

callbacks = Trainer.CallbackHandler(
    {
        Trainer.Events.train_pre: [_do_train_pre],
        Trainer.Events.grad_accum_post: [_do_grad_accum_post],
        Trainer.Events.batch_post: [_do_batch_eval],
    }
)

logger.info(f"[green]CALLBACKS:{callbacks}[/green]")

trainer.setup_helpers(
    callbacks=callbacks,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dl,
    test_iter=iter(test_dl),
    processor=processor,
)

# trainer.train()
# trainer.train_num_iters()
run_func()
