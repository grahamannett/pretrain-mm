import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Literal, Optional, Literal

import torch
import torchmetrics

# from simple_parsing import ArgumentParser, choice
from config.dev import get_dev_config

# from config.fuyu import FuyuInfo
from config.model_configs import ExperimentConfigModelInfo, ExperimentModelConfigMixin
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.adapted.loss_adapter import CLMLossKwargs
from pretrain_mm.trainer import Trainer
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseConfig, BaseTrainConfig, FromConfig, LocalDataConfig, WandBConfig
from pretrain_mm.utils.eval_utils import remove_label
from pretrain_mm.utils.functional_utils import wpartial
from pretrain_mm.utils.generate_utils import StopOnToken


# helper to only log data that starts with "log/" for dict keys
logger._filtered_log = logger.log_data_filter(filter_by="log/")


# wandb_config = WandBConfig(group="testing/pretrain-fuyu", job_type="pretrain")
@dataclass
class WandBConfig(WandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "pretrain"
    tags: tuple[str, ...] = ("pretrain", "fuyu")


@dataclass
class ExtraDatasets(BaseConfig):
    # unless specified use 'train'
    names: tuple[str, ...] = ("mosaicml/instruct-v3",)

    @property
    def datasets(self):
        from datasets import load_dataset

        datasets = [load_dataset(name, split="train") for name in self.names]
        return datasets


# MARK: CONFIG
@dataclass
class TrainConfig(BaseTrainConfig, ExperimentModelConfigMixin):
    wandb: WandBConfig = FromConfig[WandBConfig]
    extra_datasets: ExtraDatasets = FromConfig[ExtraDatasets]
    local_data_config: LocalDataConfig = FromConfig[LocalDataConfig]

    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    train_type: Literal["epoch", "iter"] = "iter"
    num_iters: int = False  # num iters if not going through full dataset
    epochs: int = 10

    model_info: ExperimentConfigModelInfo = ExperimentConfigModelInfo.Fuyu
    model_patch_forward: bool = False
    model_image_patch_loss: bool = False
    model_patch_idx_latent: bool = False
    model_patch_gather_continuous_embeddings: bool = True

    causal_lm_loss: CLMLossKwargs.CLMLossKwargsType = CLMLossKwargs.DC_FIELD

    do_eval: bool = True
    do_eval_pre: bool = False
    do_batch_eval_every: int | None = None
    eval_num_samples: int = 2
    eval_num_generations: int = 2
    eval_use_past_key_values: bool = False
    output_dir: str | None = None  # "output/model_output"
    clean_output_dir: str = False
    save_every_n_batch: int = 200
    save_every: Optional[Literal["epoch", "iter", "best"]] = None

    # dataset
    dataset_name: str = "mind2web"
    use_extra_datasets: bool = False
    loc_type: str = "box"
    loc_before_action_repr: bool = False
    max_length: int = 2700
    get_text_from: Literal["html", "ocr"] = "ocr"
    # get_text_from: str = choice("html", "ocr", default="ocr")
    ocr_use_gpu: bool = False

    data_subset: int | None = None

    batch_size: int = 1
    grad_accum_steps: int = 8

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 0
    dl_pin_memory: bool = True
    dl_prefetch_factor: int | None = None
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
    infolm_measure: Literal[
        "kl_divergence",
        "alpha_divergence",
        "beta_divergence",
        "ab_divergence",
        "renyi_divergence",
        "l1_distance",
        "l2_distance",
        "l_infinity_distance",
        "fisher_rao_distance",
    ] = "kl_divergence"  # was using "l2_distance",
    metric_prefix: str = "log/eval/"

    def __post_init__(self):
        if self.train_type == "iter" and not (self.num_iters > 0):
            logger.warn("num_iters must be greater than 0 if train_type is iter.  Setting to 1")
            self.num_iters = 1

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

    @property
    def model_config_kwargs(self):
        model_info = getattr(self.model_info, "resolve", lambda: self.model_info)()
        if callable(model_info.get_model_config_kwargs):
            config_kwargs = model_info.get_model_config_kwargs(self)
            if self.causal_lm_loss.pop("use"):
                config_kwargs["causal_lm_loss"] = self.causal_lm_loss
            return config_kwargs
        return {}


config: TrainConfig = TrainConfig.cli()

# not entirely necessary to make these vars but was previously using simple-parsing
extra_datasets: ExtraDatasets = config.extra_datasets
local_data_config: LocalDataConfig = config.local_data_config
wandb_config: WandBConfig = config.wandb
model_info = config.model_info.resolve()

trainer = Trainer(config=config)

ModelConstants = model_info.ModelConstants
ModelConfigCls = model_info.ModelConfigCls
ModelCls = model_info.ModelCls
ModelProcessorCls = model_info.ProcessorCls


def rstrip_eos(s: str):
    return s.rstrip(ModelConstants.eos_token)


# MARK: EVAL
@torch.no_grad()
def eval_with_metric(
    config: TrainConfig,
    data_iter: Iterable[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    processor: ModelProcessorCls,
    metric_fn: torchmetrics.MetricCollection,
    tensor_metric_fn: torchmetrics.MetricCollection,
    max_new_tokens: int = 15,
    do_sample: bool = True,
    temperature: float = 0.1,
    # do this so that it is initialized every call
):
    gen_strs = []
    gen_losses = []

    stopping_criteria: list[Callable] = [StopOnToken(ModelConstants.get_stop_ids(tokenizer=processor.tokenizer))]

    # eval_num_samples = config.eval_num_samples
    model.eval()

    tensor_metric_fn.to(model.device)

    while len(gen_strs) < config.eval_num_samples:
        batch = next(data_iter)
        if not batch.okay:
            continue

        batch.to(model.device)

        # decode the target label as that is closest to what model is trained on.
        # the actual label before encoding sometimes is missing extra tokens/additional spaces
        target_label_str: str = processor.full_decode(batch.labels)

        # image_width, image_height = batch.extra["images"].size

        boa_idx = processor.get_inputs_start_idx(batch.input_ids, labels=batch.labels, offset=-1)

        # first we just get the loss of the input/labels
        output = model(**batch)
        gen_losses.append(output.loss.item())

        # should perplexity be offset?
        _ = tensor_metric_fn(output.logits[..., :-1, :], batch.labels[..., 1:])

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

    metric_fn.reset()
    tensor_metric_fn.reset()
    model.train()

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
        processor=processor,
        metric_fn=met_collect_str,
        tensor_metric_fn=met_collect_int,
    )

    eval_data = logger._filtered_log(eval_res)
    eval_data = pretty_data_dict(eval_data)
    logger.log(f"[[bold magenta]{eval_str}[/bold magenta]|{eval_data}]")


def _do_train_pre(metric_fn: Callable = None):
    show_optim_info(optimizer, scheduler, num_training_steps, warmup_ratio=config.warmup_ratio)

    logger.log(f"Instruction/Task Function: {config.task_function}")

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


def _do_post_train():
    if config.output_dir:
        save_dir = f"{config.output_dir}/latest"
        model.save_pretrained(save_dir)
        logger.log(f"Saving model to {save_dir}")


# MARK: MAIN
#
# -----------------------------------
#  __  __    _    ___ _   _         |
# |  \/  |  / \  |_ _| \ | |        |
# | |\/| | / _ \  | ||  \| |        |
# | |  | |/ ___ \ | || |\  |        |
# |_|  |_/_/   \_\___|_| \_|        |
# -----------------------------------


# setup wandb + check config such that yaml printed config is in wandb console logs
logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
logger.tools.setup_local_data(local_data_config=local_data_config, config=config)
logger.tools.check_exp_config(config=config, exp_type="train")

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


model_config = (
    ModelConfigCls.from_pretrained(model_info.model_name, **config.model_config_kwargs) if ModelConfigCls else None
)

# necessary as setting the model_config_kwargs_ext is extremely case specific per model so easier to just figure out
# where to set num_layers for local dev and model_chop it seems
if config.model_modify_config and callable(conf_cb := model_info.modify_model_config_callback):
    model_config = conf_cb(model_config, exp_config=config)

processor = ModelProcessorCls.from_pretrained(model_info.model_name, **model_info.tokenizer_kwargs)
model = ModelCls.from_pretrained(model_info.model_name, config=model_config, **config.model_init_kwargs)

# this goes from raw sample -> sample in task format
task_processor: Mind2WebPretrainProcessor = Mind2WebPretrainProcessor(
    get_text_from=config.get_text_from,
    tokenizer_constants=ModelConstants,
)


encode_func = partial(
    processor.encode_sample,
    label_mask_image_patches=config.label_mask_image_patches,
    label_mask_text_ids=config.label_mask_text_ids,
    max_length=config.max_length,
    truncation=True,
)

agent_train_func = partial(
    task_processor.agent_training,
    include_patch_idx=config.model_image_patch_loss,
    add_cand_outline=config.add_cand_outline,
    image_processor=processor.image_processor,
)

pretrain_func_generate_possible_actions = partial(
    task_processor.pretrain_func_generate_possible_actions,
    cands_range=(3, 10),
    skip_include_text=config.skip_include_text,
)


multiple_tasks = {
    "agent_training": agent_train_func,
    "pretrain_func_generate_possible_actions": pretrain_func_generate_possible_actions,
}


_task_keys = list(multiple_tasks.keys())


def sample_from_tasks(sample: dict):
    task_key = random.choice(_task_keys)
    transformed_sample = multiple_tasks[task_key](sample)
    return transformed_sample


train_dataset_adapter = TaskAdapter(
    train_dataset,
    transforms={
        "task": sample_from_tasks,  # or agent_train_func
        "encode": encode_func,
    },
)

test_dataset_adapter = TaskAdapter(
    test_dataset,
    transforms={"pretrain_task": agent_train_func, "encode": encode_func},
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

num_training_steps = trainer.get_num_training_steps(config=config, dataloader=train_dl)
scheduler = get_scheduler(
    config.scheduler_type,
    optimizer,
    num_training_steps=num_training_steps,
    warmup_ratio=config.warmup_ratio,
)

# MARK: METRICS
# maybe move to eval_utils
infolm = torchmetrics.text.infolm.InfoLM(
    "google/bert_uncased_L-2_H-128_A-2",
    idf=False,
    verbose=False,
    information_measure=config.infolm_measure,  # "l2_distance",
)

edit_distance = torchmetrics.text.ExtendedEditDistance()
match_error_rate = torchmetrics.text.MatchErrorRate()

# tensor based
perplexity = torchmetrics.text.Perplexity(ignore_index=constants.IGNORE_INDEX)

met_collect_str = torchmetrics.MetricCollection([infolm, edit_distance, match_error_rate], prefix=config.metric_prefix)
met_collect_int = torchmetrics.MetricCollection([perplexity], prefix=config.metric_prefix)


callbacks = Trainer.CallbackHandler(
    {
        # saving processor and showing optimizer info
        Trainer.Events.train_pre: [wpartial(_do_train_pre, metric_fn=met_collect_str)],
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
