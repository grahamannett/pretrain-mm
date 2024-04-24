import os
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import torchmetrics
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.pretrain_instructions import PretrainTask
from pretrain_mm.model.fuyu import FuyuConfig, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer import Trainer
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseConfig, BaseTrainConfig, LocalDataConfig, WandBConfig
from pretrain_mm.utils.functional_utils import wpartial
from pretrain_mm.utils.generate_utils import StopOnToken


# helper to only log data that starts with "log/" for dict keys
logger._filtered_log = logger.log_data_filter(filter_by="log/")

wandb_config = WandBConfig(group="testing/pretrain-fuyu", job_type="pretrain")


@dataclass
class ExtraDatasets(BaseConfig):
    # unless specified use 'train'
    names: list[str] = field(default_factory=lambda: ["mosaicml/instruct-v3"])

    @property
    def datasets(self):
        from datasets import load_dataset

        datasets = [load_dataset(name, split="train") for name in self.names]
        return datasets


@dataclass
class ModelConfig(BaseConfig):
    chop_model: bool = False
    patch_image_out: bool = False


# MARK: CONFIG
@dataclass
class TrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset
    train_type: str = choice("epoch", "iter", default="iter")

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_patch_forward: bool = False
    model_image_patch_loss: bool = False
    model_chop: bool = False  # for making the model have only 1 decoder block, e.g. local dev

    do_eval: bool = True
    do_eval_pre: bool = False
    do_batch_eval_every: int = None
    eval_num_samples: int = 2
    eval_num_generations: int = 2
    eval_use_past_key_values: bool = False
    output_dir: str = None  # "output/model_output"
    clean_output_dir: str = False
    save_every_n_batch: int = 200
    save_every: Optional[str] = choice("epoch", "iter", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    use_extra_datasets: bool = False
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
    warmup_ratio: float = 0.05
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
    add_cand_outline: bool = False
    skip_include_text: bool = False
    # MARK: mask related
    label_mask_text_ids: bool = True
    label_mask_image_patches: bool = True

    test_dataloader: bool = False

    metric_prefix: str = "log/eval/"

    stop_ids = FuyuConstants.get_stop_ids()

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

        if self.model_image_patch_loss and not self.model_patch_forward:
            logger.warn("must set patch_forward to True if using patch loss.  Setting to True")
            self.model_patch_forward = True

        # setup instruction func
        self.instruction_func = PretrainTask[self.instruction]()

    @property
    def model_info(self):
        # TODO: fix this, right now its not json serializeable
        return FuyuInfo


parser = ArgumentParser()
parser.add_arguments(TrainConfig, dest="pretrain_config")
parser.add_arguments(ExtraDatasets, dest="extra_datasets", prefix="extra_datasets.")
parser.add_arguments(wandb_config, dest="wandb_config", prefix="wandb.")
parser.add_arguments(LocalDataConfig, dest="local_data_config", prefix="local_data.")

args = parser.parse_args()

config: TrainConfig = args.pretrain_config
extra_datasets: ExtraDatasets = args.extra_datasets
# initialize trainer here to be able to use it in the functions below
trainer = Trainer(config=config)


def get_num_training_steps():
    if config.train_type == "epoch":
        return len(trainer.train_dataloader) * config.epochs
    if config.train_type == "iter":
        return config.num_iters


def remove_label(batch, to_idx):
    batch.attention_mask = batch.attention_mask[:, :to_idx]
    batch.input_ids, removed_input_ids = batch.input_ids[:, :to_idx], batch.input_ids[:, to_idx:]
    batch.labels, removed_labels = None, batch.labels
    batch.image_patches_indices = batch.image_patches_indices[:, :to_idx]
    return batch, (removed_input_ids, removed_labels)


def rstrip_eos(s: str):
    return s.rstrip(FuyuConstants.eos_string)


# MARK: METRICS
infolm = torchmetrics.text.infolm.InfoLM(
    "google/bert_uncased_L-2_H-128_A-2",
    idf=False,
    verbose=False,
    information_measure="l2_distance",
)

edit_distance = torchmetrics.text.ExtendedEditDistance()
match_error_rate = torchmetrics.text.MatchErrorRate()

# tensor based
perplexity = torchmetrics.text.Perplexity(ignore_index=constants.IGNORE_INDEX)

collection_str = torchmetrics.MetricCollection([infolm, edit_distance, match_error_rate], prefix=config.metric_prefix)
collection_int = torchmetrics.MetricCollection([perplexity], prefix=config.metric_prefix)


# MARK: EVAL
@torch.no_grad()
def eval_with_metric(
    config: TrainConfig,
    data_iter: Iterable[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    metric_fn: torchmetrics.MetricCollection = collection_str,
    tensor_metric_fn: torchmetrics.MetricCollection = collection_int,
    max_new_tokens: int = 15,
    do_sample: bool = True,
    temperature: float = 0.1,
    # do this so that it is initialized every call
    stopping_criteria: list[Callable] = [StopOnToken(config.stop_ids)],
):
    gen_strs = []
    gen_losses = []
    # eval_num_samples = config.eval_num_samples
    model.eval()

    tensor_metric_fn.to(model.device)

    while len(gen_strs) < config.eval_num_samples:
        if not (batch := next(data_iter)).is_valid:
            continue
        batch.to(model.device)

        # decode the target label as that is closest to what model is trained on.
        # the actual label before encoding sometimes is missing extra tokens/additional spaces
        target_label_str: str = processor.full_decode(batch.labels)
        boa_idx = processor.get_inputs_start_idx(batch.input_ids, offset=-1)

        # first we just get the loss of the input/labels
        output = model(**batch)
        gen_losses.append(output.loss.item())

        _ = tensor_metric_fn(output.logits, batch.labels)

        # remove all label from related tensors (otherwise (_inp_ids, labels))
        batch, (rem_ids, rem_lab) = remove_label(batch, to_idx=boa_idx)

        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

        generated_str = processor.full_decode(output[:, boa_idx:])

        generated_str = rstrip_eos(generated_str)
        target_label_str = rstrip_eos(target_label_str)

        gen_strs.append(generated_str)

        # use .update to not return any value
        _ = metric_fn(generated_str, target_label_str)

    # avg_metric_vals = sum(metric_vals) / len(metric_vals)
    # metric_vals: dict = metric_fn.compute()
    metric_vals: dict[str, torch.Tensor] = {
        **metric_fn.compute(),
        **tensor_metric_fn.cpu().compute(),
    }
    metric_vals = {k: v.item() for k, v in metric_vals.items()}

    avg_loss = sum(gen_losses) / len(gen_losses)

    return {
        **metric_vals,
        # log/ allows filter logging
        # "log/eval/batch_metric": avg_metric_vals,
        "log/eval/batch_loss": avg_loss,
        # unimportant vals
        "generated_strs": gen_strs,
        "losses": gen_losses,
    }


# MARK: CALLBACKS
#            _ _ _                _
#   ___ __ _| | | |__   __ _  ___| | _____
#  / __/ _` | | | '_ \ / _` |/ __| |/ / __|
# | (_| (_| | | | |_) | (_| | (__|   <\__ \
#  \___\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
#
# NOTE: Callbacks are used exclusively for trainer


def format_key(key: str):
    return key.lstrip("eval/")[:12]


def pretty_data_dict(data, digits=3):
    return {format_key(k): round(v, digits) if isinstance(v, float) else v for k, v in data.items()}


def _eval_helper(eval_str: str = "Eval"):
    eval_res = eval_with_metric(
        config,
        data_iter=trainer.test_iter,
        model=model,
        metric_fn=collection_str,
    )

    eval_data = logger._filtered_log(eval_res)
    eval_data = pretty_data_dict(eval_data)
    logger.log(f"[[bold magenta]{eval_str}[/bold magenta]|{eval_data}]")


def _do_train_pre(metric_fn: Callable = None):
    show_optim_info(optimizer, scheduler, num_training_steps, warmup_ratio=config.warmup_ratio)

    logger.log("Instruction")
    logger.log(config.instruction_func)

    if isinstance(metric_fn, torchmetrics.MetricCollection):
        table = logger.use_table(title="Metric Collection Info", box=logger.get_box_type("rounded"))
        table.add_column("metric", justify="left", style="cyan")
        table.add_column("key", justify="left", style="cyan")
        table.add_column("higher is better", justify="left", style="cyan")

        for _metric_key, _fn in metric_fn.items():
            table.add_row(_fn._get_name(), _metric_key, str(_fn.higher_is_better).lower())

        logger.log(table)

    if config.output_dir:
        logger.info("Using callback to setup train related... saving processor.")
        model.config.save_pretrained(f"{config.output_dir}/model_config")
        processor.save_pretrained(f"{config.output_dir}/processor")
    else:
        logger.info("Not saving processor")

    if config.do_eval_pre:
        logger.info("Doing eval PRE")
        _eval_helper("PreEval")


def _do_grad_accum_post(batch_idx: int, batch_loss: float):
    logger.log(f"[B-IDX:{batch_idx}][L:{batch_loss:.3f}][LR:{trainer.last_lr:.2e}]")
    logger.log_data({"train/batch_loss": batch_loss, "learning_rate": trainer.last_lr})


def _do_batch_eval(batch_idx: int):
    if config.do_batch_eval_every and (batch_idx > 0) and ((batch_idx % config.do_batch_eval_every) == 0):
        _eval_helper(f"Eval@{batch_idx}")

    if (batch_idx > 0) and (batch_idx % config.save_every_n_batch == 0):
        if config.output_dir:
            save_dir = f"{config.output_dir}/checkpoint_{batch_idx}"
            model.save_pretrained(save_dir)
            logger.log(f"saving model at batch_idx: {batch_idx} to {save_dir}")

    model.train()


def _do_post_train():
    if config.output_dir:
        save_dir = f"{config.output_dir}/latest"
        model.save_pretrained(save_dir)
        logger.log(f"Saving model to {save_dir}")


# def wrapped_model_forward(self, image_patches: torch.Tensor=None, extra: dict =None, **kwargs):


# MARK: SETUP
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


# model_config = FuyuConfig.from_pretrained(config.model_id, patch_image_out=True)
# # why doesnt passing these into from_pretrained even work? so many bugs/issues with hf stuff
# model_config.num_hidden_layers = 1
# model_config.text_config.num_hidden_layers = 1
model_config = FuyuConfig.from_pretrained(config.model_id, patch_image_out=True)
# model_config.patch(num_hidden_layers=1)

model = FuyuForCausalLM.from_pretrained(
    config.model_id,
    device_map=config.device,
    # torch_dtype=torch.bfloat16, # wtf is this giving errors now
    config=model_config,
)

# this goes from raw sample -> sample in task format
task_processor = Mind2WebPretrainProcessor(
    instruction_func=config.instruction_func,
    task_function=config.task_function,
    get_text_from=config.get_text_from,
    add_cand_outline=config.add_cand_outline,
)


encode_func = partial(
    processor.encode_sample,
    label_mask_image_patches=config.label_mask_image_patches,
    label_mask_text_ids=config.label_mask_text_ids,
    max_length=config.max_length,
)


train_dataset_adapter = TaskAdapter(
    train_dataset,
    transforms={
        "pretrain_task": partial(
            task_processor.agent_training,
            include_patch_idx=config.model_image_patch_loss,
            image_processor=processor.image_processor,
        ),
        "encode": encode_func,
    },
)
test_dataset_adapter = TaskAdapter(
    test_dataset,
    transforms={"pretrain_task": task_processor, "encode": encode_func},
)


collate_fn = DataCollator(
    processor.pad_token_id,
    squeeze=(config.batch_size != 1),
    include_labels=True,
    include_extra_loss_kwargs=True,  # or should be based on config.model_image_patch_loss
)

train_dl = torch.utils.data.DataLoader(
    train_dataset_adapter,
    collate_fn=collate_fn,
    batch_size=config.batch_size,
    num_workers=config.dl_num_workers,
    pin_memory=config.dl_pin_memory,
    prefetch_factor=config.dl_prefetch_factor,
    persistent_workers=config.dl_persistent_workers,
    timeout=config.dl_timeout,
    # worker_init_fn=pretask_processor._worker_init_func if config.dl_worker_init else None,
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

num_training_steps = get_num_training_steps()
scheduler = get_scheduler(
    config.scheduler_type,
    optimizer,
    num_training_steps=num_training_steps,
    warmup_ratio=config.warmup_ratio,
)


callbacks = Trainer.CallbackHandler(
    {
        # saving processor and showing optimizer info
        Trainer.Events.train_pre: [wpartial(_do_train_pre, metric_fn=collection_str)],
        Trainer.Events.train_post: [_do_post_train],  # saving model
        Trainer.Events.grad_accum_post: [_do_grad_accum_post],  # logging batch loss
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


# MARK: RUN
if config.train_type == "epoch":
    trainer.train_epochs()
elif config.train_type == "iter":
    trainer.train_num_iters()
