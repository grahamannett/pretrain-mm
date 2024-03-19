import os
from dataclasses import dataclass
from typing import Optional

import torch
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebEncoder, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer import CallbackHandler, Trainer
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseTrainConfig, WandBConfig, LocalDataConfig


@dataclass
class WandBConfig(WandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "pretrain"


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
    task_function: str = "AssistantResponse"
    skip_include_text: bool = False

    use_profiler: bool = False
    test_dataloader: bool = False

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"

        if (self.dl_num_workers == 0) and (self.dl_timeout != 0):
            logger.warn(f"timeout must be 0 if num_workers is 0.  Setting to 0")
            self.dl_timeout = 0

        if (self.dl_num_workers == 0) and (self.dl_prefetch_factor != None):
            logger.warn(f"prefetch factor must be None if num_workers is 0.  Setting to None")
            self.dl_prefetch_factor = None


def pretrain_dataloader_test(config, model, dataloader):
    for epoch in range(config.epochs):
        logger.log(f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            batch.to(model.device)
            logger.log(f"Batch: {batch_idx}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="pretrain_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")
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
    train_dataset.setup_pretrain().use_num_iters(config.num_iters)
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(config.model_id)
    model = FuyuForCausalLM.from_pretrained(config.model_id, device_map=config.device, torch_dtype=torch.bfloat16)

    train_task_processor = Mind2WebPretrainProcessor(
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
        "pretrain_task": train_task_processor.pretrain_func_generate_possible_actions,
        "encode": task_processor.encode_data,
    }

    task_train_dataset = TaskAdapter(train_dataset, transforms=transforms)

    collate_fn = DataCollator(processor.pad_token_id, squeeze=(config.batch_size != 1), include_labels=True)
    train_dl = torch.utils.data.DataLoader(
        task_train_dataset,
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
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=config.batch_size,
        num_workers=config.dl_num_workers,
        pin_memory=config.dl_pin_memory,
        prefetch_factor=config.dl_prefetch_factor,
        timeout=config.dl_timeout,
        persistent_workers=config.dl_persistent_workers,
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

    if config.output_dir:
        processor.save_pretrained(f"{config.output_dir}/processor")

    def save_model_callback(model, epoch, trainer, **kwargs):
        if trainer.config.output_dir is None:
            return

        output_path = f"{trainer.config.output_dir}"
        if trainer.config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"
        model.save_pretrained(output_path)
        logger.log(f"model for epoch: {epoch} saved to: {output_path}")

    def log_batch_step(batch_idx, trainer, **kwargs):
        if trainer.do_grad_accum_step(batch_idx):
            logger.log(f"[B-IDX:{batch_idx}][L:{trainer.batch_loss:.3f}]")
            logger.log_data({"train/batch_loss": trainer.batch_loss, "learning_rate": trainer.last_lr})

    callbacks = CallbackHandler({})

    trainer = Trainer(config=config)
    trainer.setup_helpers(model=model, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dl)
    breakpoint()
    trainer.train()
