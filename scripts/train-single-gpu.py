import os
import random
from functools import partial
from dataclasses import dataclass
from typing import Optional

import torch
import transformers
from simple_parsing import ArgumentParser, choice

import wandb
from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.datasets import (
    Mind2Web,
    Mind2WebConfig,
    Mind2WebTaskProcessor,
    TaskAdapter,
    task_mind2web,
)
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu.processing_fuyu import FuyuProcessor
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig, check_train_config, setup_wandb
from pretrain_mm.utils.eval_utils import loc_metric_from_str


@dataclass
class WandBConfig(BaseWandBConfig):
    group: str = "testing/finetune-fuyu"
    job_type: str = "finetune"


@dataclass
class TrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_name: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    output_dir: str = None  # "output/model_output"
    save_every: Optional[str] = choice("epoch", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    task_func: str = "TitleWebsiteTask"
    loc_type: str = "point"

    data_subset: int = None
    epochs: int = 10
    batch_size: int = 1
    grad_accum_steps: int = 4

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    learning_rate: float = 1e-03
    scheduler_type: str = "cosine"
    gamma: float = 0.85

    def get_task_func(self, dataset_info):
        if self.task_func:
            task_func = getattr(dataset_info.tasks, self.task_func)
            return task_func()
        else:
            return dataset_info.task

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"


@torch.no_grad()
def eval_with_generate(model, gen_dataset, processor, max_new_tokens: int = 30, num_choices: int = 5) -> float:
    """
    30 is chosen as seems like that is approximately number of tokens for something like

    Click @ <box> int, int, int, int </box>

    lower is better
    """
    logger.info("DOING EVAL WITH GENERATE")

    choices = list(range(0, len(gen_dataset)))
    random.shuffle(choices)
    choices = choices[:num_choices]

    metrics = []
    model.eval()
    for sample_id in choices:
        sample = gen_dataset[sample_id]
        combined_text = sample["text"] + sample["label"]
        model_inputs = processor(text=sample["text"], images=sample["image"])
        # generate the answer
        model_inputs.to(model.device)
        outputs = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        post_processed_bbox_tokens = processor.post_process_box_coordinates(outputs)[0]
        decoded_outputs = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
        # compute loss based on box.  0 is perfect 1 means not even bbox.
        metric_val = loc_metric_from_str(target_str=combined_text, pred_str=decoded_outputs)
        metrics.append(metric_val)

    return sum(metrics) / len(metrics)


def eval(model, eval_dataloader, get_loss):
    losses = 0
    model.eval()

    progress = logger.progress()
    batch_task = progress.add_task(f"[cyan]Eval Step: ", total=len(eval_dataloader))

    for idx, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch.to(model.device)

            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                image_patches=batch.image_patches,
                image_patches_indices=batch.image_patches_indices,
            )

            loss = get_loss(outputs.logits, batch.input_ids)
            losses += loss.item()

        progress.update(batch_task, advance=1)

    logger.log(f"eval/Loss: {losses}")
    wandb.log(
        {
            "eval/loss": losses,
        }
    )


def train(
    train_config: TrainConfig,
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    eval_with_generate_kwargs: dict = None,
):
    def get_loss(logits, labels):
        # b, l needed when fsdp
        vocab_size = logits.shape[-1]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)
        return loss

    # progress = logger.progress(ensure_exit=True, start=True, disable=train_config.dl_disable_progress)

    def _save_helper(epoch):
        if train_config.output_dir is None:
            return

        output_path = f"{train_config.output_dir}"
        if train_config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"

        model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    logger.info("starting train loop")
    for epoch in range(train_config.epochs):
        # resets
        epoch_loss, batch_loss = 0, 0

        # progress bar info
        # ptask = progress.add_task(f"[cyan]Training Step: ", total=train_config.num_iters or len(train_dataloader))

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # progress.update(ptask, advance=1)
            batch.to(model.device)

            outputs = model(**batch)

            loss = get_loss(outputs.logits, batch["input_ids"])
            loss = loss / train_config.grad_accum_steps
            loss.backward()

            if train_config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

            # for grad accum if batch size has to be 1
            if ((batch_idx % train_config.grad_accum_steps == 0) or (batch_idx == len(train_dataloader))) and (
                batch_idx != 0
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                logger.log(f"[B-IDX:{batch_idx}][L:{batch_loss:.2f}]")
                wandb.log({"train/batch_loss": batch_loss, "learning_rate": scheduler.get_last_lr()[0]})

                batch_loss = 0

            batch_loss += loss.item()
            epoch_loss += loss.item()

            if train_config.num_iters and (train_config.num_iters < batch_idx):
                break

        # stop the batch_task progress so new one can start on next epoch
        # progress.stop()

        # EVAL RELATED SHOULD BE USED HERE
        # eval(model, test_dataloader, get_loss=get_loss)
        # eval_acc_metric = eval_with_generate(model, **eval_with_generate_kwargs)
        eval_acc_metric = 0

        logger.info(f"Epoch[{epoch}] loss: {epoch_loss:.2f} | eval_metric: {eval_acc_metric}")
        wandb.log({"train/epoch_loss": epoch_loss, "eval/bbox_metric": eval_acc_metric})

        _save_helper(epoch)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")
    args = parser.parse_args()

    train_config: TrainConfig = args.train_config
    wandb_config: WandBConfig = args.wandb_config
    model_config = train_config.model_config

    # setup wandb + check config such that yaml printed config is in wandb console logs
    setup_wandb(wandb_config=wandb_config, config=train_config)
    check_train_config(train_config)

    m2w_info = get_dev_config(train_config.dataset_name)

    train_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        subset=train_config.data_subset,
        **m2w_info["train"],
    )
    test_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        subset=train_config.data_subset,
        **m2w_info["test"],
    )

    train_dataset = Mind2Web(train_data_config)
    test_dataset = Mind2Web(test_data_config)

    processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained(
        FuyuInfo.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    task_transforms = {
        "task": partial(task_mind2web, next_action_loc_str=train_config.loc_type),
        "preprocessor": Mind2WebTaskProcessor.preprocessor,
        "processor": processor,
        "postprocessor": Mind2WebTaskProcessor.postprocessor,
    }

    task_train_dataset = TaskAdapter(train_dataset, transforms=task_transforms)
    task_test_dataset = TaskAdapter(test_dataset, transforms=task_transforms)
    gen_test_dataset = TaskAdapter(test_dataset, transforms=[task_mind2web])

    collate_fn = DataCollator(processor.pad_token_id, squeeze=(train_config.batch_size != 1))
    train_dl = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.dl_num_workers,
        pin_memory=train_config.dl_pin_memory,
        shuffle=True,
    )
    test_dl = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.dl_num_workers,
        pin_memory=train_config.dl_pin_memory,
    )

    optimizer = get_optimizer(model, learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
    iters_per_epoch = train_config.num_iters or len(train_dl)
    scheduler = get_scheduler(
        train_config.scheduler_type, optimizer, num_training_steps=(iters_per_epoch * train_config.epochs)
    )

    if train_config.output_dir:
        processor.save_pretrained(f"{train_config.output_dir}/processor")

    train(
        train_config,
        model,
        train_dl,
        test_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_with_generate_kwargs={"gen_dataset": gen_test_dataset, "processor": processor},
    )
