import os
from dataclasses import dataclass
from typing import Optional

import torch
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import (
    Mind2Web,
    Mind2WebConfig,
    Mind2WebPretrainProcessor,
    TaskAdapter,
)
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import (
    BaseTrainConfig,
    LocalDataConfig,
    WandBConfig,
)
from pretrain_mm.utils.eval_utils import eval_by_completion


wandb_config = WandBConfig(
    group="testing/pretrain-fuyu",
    job_type="pretrain",
)


@dataclass
class PreTrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"

    do_eval: bool = True
    do_eval_pre: bool = False
    do_batch_eval_every: int = -1  # >= 1 for how often to do

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

    # pretrain task related
    task_function: str = "GenerateNumPotentialActions"
    cands_range: tuple[int, int] = (1, 5)
    skip_include_text: bool = False

    use_profiler: bool = False
    test_dataloader: bool = False

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"

        if (self.dl_num_workers == 0) and (self.dl_timeout != 0):
            logger.warn("timeout must be 0 if num_workers is 0.  Setting to 0")
            self.dl_timeout = 0

        if (self.dl_num_workers == 0) and (self.dl_prefetch_factor is not None):
            logger.warn("prefetch factor must be None if num_workers is 0.  Setting to None")
            self.dl_prefetch_factor = None


def pretrain_dataloader_test(config, model, dataloader):
    for epoch in range(config.epochs):
        logger.log(f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            batch.to(model.device)
            logger.log(f"Batch: {batch_idx}")


def pretrain(
    config: PreTrainConfig,
    model,
    train_dataloader,
    eval_dataset,
    optimizer,
    scheduler,
    task_processor,
):
    stop_ids = FuyuConstants.get_stop_ids(processor)

    def clip_grad():
        if config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

    def reset_epoch():
        model.train()
        # explicit list these
        epoch_loss, batch_loss, eval_acc = 0, 0, 0
        return epoch_loss, batch_loss, eval_acc

    def do_grad_accum_step(batch_idx: int):
        if batch_idx == 0:  # dont do it for batch 0
            return False
        if (batch_idx % config.grad_accum_steps) == 0:
            return True
        if batch_idx == len(train_dataloader):
            return True
        return False

    def save_helper(epoch: int):
        if config.output_dir is None:
            return

        output_path = f"{config.output_dir}"
        if config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"

        model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    def _do_eval(batch_idx: int = None):
        """here is where you set whatever eval you want to be completed for
        either the do_eval_pre or do_eval
        """

        return eval_by_completion(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            task_func=pretrain_task_processor.acc_func_complete_box,
            encode_data_func=task_processor.encode_data,
            num_samples=config.eval_num_samples,
            num_generations=config.eval_num_generations,
            get_decode_start_idx_fn=processor.get_inputs_start_idx,
            prepend_str="eval/",
            prepend_str_extra="extra/",
            generate_kwargs={
                "stop_tokens": stop_ids,
                "return_extra": True,
                "max_new_tokens": 10,
                "use_past_key_values": config.eval_use_past_key_values,
                "forward_kwargs": {
                    "output_hidden_states": False,
                },
            },
        )

    def _do_eval_batch_idx(batch_idx: int):
        if (config.do_batch_eval_every >= 1) and (config.do_batch_eval_every % batch_idx) == 0:
            logger.info(f"Doing batch eval for batch: {batch_idx}")
            return _do_eval()
        return

    logger.info("Starting train")

    if config.do_eval_pre:
        eval_info = _do_eval()

    for epoch in range(config.epochs):
        epoch_loss, batch_loss, eval_acc = reset_epoch()

        for batch_idx, batch in enumerate(train_dataloader):
            breakpoint()

            # if you need to check something about batch do here
            batch.to(model.device)
            outputs = model(**batch)

            loss = outputs.loss / config.grad_accum_steps
            loss.backward()
            batch_loss += loss.item()

            clip_grad()

            if do_grad_accum_step(batch_idx):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                logger.log(f"[E/B-IDX:{epoch}/{batch_idx}][L:{batch_loss:.3f}]")
                logger.log_data(
                    {
                        "train/batch_loss": batch_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

                epoch_loss += batch_loss
                batch_loss = 0

            if batch_eval_info := _do_eval_batch_idx(batch_idx=batch_idx):
                logger.log_data(batch_eval_info)

        # save before eval as eval may be error prone/crash
        save_helper(epoch)

        # EVAL RELATED
        if config.do_eval:
            eval_info = _do_eval()

            logger.log_data(
                {
                    "train/epoch_loss": epoch_loss,
                    **{k: v for k, v in eval_info.items() if k.startswith("eval/")},
                }
            )

        logger.log(
            f"E[{epoch}][L:{epoch_loss:.2f}][LR:{scheduler.get_last_lr()[0]:.4f}][Eval:{eval_info['eval/metric_avg']:.2f}]"
        )

    logger.log("Training Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreTrainConfig, dest="pretrain_config")
    parser.add_arguments(wandb_config, dest="wandb_config", prefix="wandb.")
    parser.add_arguments(LocalDataConfig, dest="local_data_config", prefix="local_data.")

    args = parser.parse_args()

    config: PreTrainConfig = args.pretrain_config
    model_config = config.model_config

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

    pretrain_task_processor = Mind2WebPretrainProcessor(
        task_function=config.task_function,
        cands_range=config.cands_range,
        skip_include_text=config.skip_include_text,
        get_text_from=config.get_text_from,
        # ocr_preprocessed=torch.load("output/processed/train_ds_raw_output.pt"),
    )

    # generate possible actions pretrain task
    transforms = {
        "pretrain_task": pretrain_task_processor.pretrain_func_generate_possible_actions,
        "encode": processor.encode_data,
    }

    task_train_dataset = TaskAdapter(train_dataset, transforms=transforms)
    # sample = task_train_dataset[0]
    # sample = task_train_dataset[1000]
    # task_eval_dataset = TaskAdapter(test_dataset, transforms=pretrain_task_processor.pretrain_func)

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

    # num_training_steps = (config.num_iters or len(train_dl)) * config.epochs
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
            logger.log_data(
                {
                    "train/batch_loss": trainer.batch_loss,
                    "learning_rate": trainer.last_lr,
                }
            )

    pretrain(
        config,
        model,
        train_dl,
        eval_dataset=test_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        task_processor=task_processor,
    )
