import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import transformers
import wandb
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, Mind2WebTaskProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuConstants, FuyuProcessor, FuyuForCausalLM
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig, check_train_config, setup_wandb
from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.utils.generate_utils import generate_helper


@dataclass
class WandBConfig(BaseWandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "pretrain"


@dataclass
class PreTrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    do_eval: bool = True
    do_eval_pre: bool = False
    output_dir: str = None  # "output/model_output"
    save_every: Optional[str] = choice("epoch", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    loc_type: str = "box"
    IGNORE_INDEX: int = constants.IGNORE_INDEX
    loc_before_action_repr: bool = False

    data_subset: int = None
    epochs: int = 10
    batch_size: int = 1
    grad_accum_steps: int = 4

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    learning_rate: float = 1e-04
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    gamma: float = 0.85

    gradient_checkpointing: bool = False

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"


def eval_with_generate(
    model,
    eval_dataset,
    task_processor,
    max_new_tokens: int = 150,
    num_choices: int = 5,
    pattern_str: str = "box",
    temperature: float = 1.0,
    stop_tokens: list[int] = [],
    drop_last_of_input: bool = False,
    include_loss: bool = True,
) -> float:
    """
    30 is chosen as seems like that is approximately number of tokens for something like

    Click @ <box> int, int, int, int </box>

    lower is better
    """
    logger.info("DOING EVAL WITH GENERATE")
    processor = task_processor.processor

    choices = list(range(0, len(eval_dataset)))
    random.shuffle(choices)
    choices = choices[:num_choices]

    acc_metric, loss_metric = [], []
    # '\x04' + '__' + '\n' + '\x00' => boa + space + newline + box_open
    after_boa = 4

    model.eval()
    for sample_id in choices:
        sample = eval_dataset[sample_id]

        input_for_loss = task_train_dataset.call_transforms(sample).to(model.device)

        boa_idx = input_for_loss.input_ids[0] == processor.vocab[FuyuConstants.boa_string]

        # include the boa token
        boa_idx = boa_idx.nonzero().view(-1)[0].item() + after_boa

        bos_idx = input_for_loss.input_ids[0] == processor.vocab[FuyuConstants.bos_string]
        bos_idx = bos_idx.nonzero().view(-1)[0].item()

        input_for_gen = {
            "input_ids": input_for_loss.input_ids[:, :boa_idx],
            "image_patches": input_for_loss.image_patches,
            "image_patches_indices": input_for_loss.image_patches_indices[:, :boa_idx],
            "attention_mask": input_for_loss.attention_mask[:, :boa_idx],
        }

        with torch.no_grad():
            loss = model(**input_for_loss).loss
            loss_metric.append(loss.item())

            gen_output = generate_helper(
                model,
                model_inputs=input_for_gen,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=temperature,
                drop_last_of_input=drop_last_of_input,
            )

        decoded_output = processor.full_decode(gen_output[0, bos_idx:])
        label_decoded = processor.full_decode(input_for_loss.input_ids[0, bos_idx:])

        logger.info(f"\nOutput generated: {decoded_output}")

        acc_val = 1.0
        try:
            acc_val = loc_metric_from_str(
                target_str=label_decoded,
                pred_str=decoded_output,
                pattern_str=pattern_str,
            )
        except TypeError as err:
            logger.warn(f"Generate string incompatible")
        except ValueError as err:
            logger.warn(f"ValueError for eval_with_generate: {err}")

        acc_metric.append(acc_val)

    return {"eval/acc_metric": sum(acc_metric) / len(acc_metric), "eval/loss": sum(loss_metric)}


def train(
    train_config: PreTrainConfig,
    model,
    train_dataloader,
    eval_dataset,
    optimizer,
    scheduler,
    task_processor,
):
    # train_config.masked_values = [71019, 71011]
    # masked_values = torch.tensor(train_config.masked_values) if train_config.masked_values else None
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    def do_grad_accum_step(batch_idx: int):
        if batch_idx == 0:  # dont do it for batch 0
            return False
        if batch_idx % train_config.grad_accum_steps == 0:
            return True
        if batch_idx == train_config.num_iters:
            return True
        if batch_idx == len(train_dataloader):
            return True
        return False

    def save_helper(epoch: int):
        if train_config.output_dir is None:
            return

        output_path = f"{train_config.output_dir}"
        if train_config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"

        model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    logger.info("starting train loop")

    if train_config.do_eval_pre:
        eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_tokens)

    for epoch in range(train_config.epochs):
        # resets
        epoch_loss, batch_loss = 0, 0

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # if you need to check something about batch do here
            batch.to(model.device)
            outputs = model(**batch)

            loss = outputs.loss / train_config.grad_accum_steps
            loss.backward()

            if train_config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

            batch_loss += loss.item()

            if do_grad_accum_step(batch_idx):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                logger.log(f"[B-IDX:{batch_idx}][L:{batch_loss:.3f}]")
                wandb.log({"train/batch_loss": batch_loss, "learning_rate": scheduler.get_last_lr()[0]})

                epoch_loss += batch_loss
                batch_loss = 0

            if train_config.num_iters and (train_config.num_iters < batch_idx):
                break

        # save before eval as hanging during eval at present
        save_helper(epoch)

        # EVAL RELATED
        eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_tokens)

        eval_acc_metric = eval_metrics["eval/acc_metric"]
        logger.log(f"E[{epoch}][L:{epoch_loss:.2f}][LR:{scheduler.get_last_lr()[0]:.4f}][Eval:{eval_acc_metric:.4f}]")
        wandb.log({"train/epoch_loss": epoch_loss, **eval_metrics})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreTrainConfig, dest="pretrain_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")

    args = parser.parse_args()

    train_config: PreTrainConfig = args.pretrain_config
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
    train_dataset.setup_pretrain()
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(train_config.model_id, trust_remote_code=True)

    model = FuyuForCausalLM.from_pretrained(
        train_config.model_id,
        device_map=train_config.device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # model.language_model.model.layers = model.language_model.model.layers[:1]

    pretrain_task_processor = Mind2WebPretrainProcessor()

    task_processor = Mind2WebTaskProcessor(
        processor=processor,
        ignore_index=train_config.IGNORE_INDEX,
        loc_before_action_repr=train_config.loc_before_action_repr,
    )

    # basic pretrain task
    # transforms = {
    #     "pretrain_task": pretrain_task_processor.pretrain_func,
    #     "processor": task_processor.encode_data,
    #     # "postprocessor": Mind2WebTaskProcessor.postprocessor,
    # }

    # generate possible actions pretrain task
    transforms = {
        "pretrain_task": pretrain_task_processor.pretrain_func_generate_possible_actions,
        "processor": task_processor.encode_data,
    }

    task_train_dataset = TaskAdapter(train_dataset, transforms=transforms)
    # task_eval_dataset = TaskAdapter(test_dataset, transforms=pretrain_task_processor.pretrain_func)

    # sample = task_train_dataset[1000]
    collate_fn = DataCollator(processor.pad_token_id, squeeze=(train_config.batch_size != 1), include_labels=True)
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

    iters_per_epoch = train_config.num_iters or len(train_dl)

    optimizer = get_optimizer(model, learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = get_scheduler(
        train_config.scheduler_type,
        optimizer,
        num_training_steps=(iters_per_epoch * train_config.epochs),
        warmup_ratio=train_config.warmup_ratio,
    )

    if train_config.output_dir:
        processor.save_pretrained(f"{train_config.output_dir}/processor")

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
            wandb.log({"train/batch_loss": trainer.batch_loss, "learning_rate": trainer.last_lr})

    train(
        train_config,
        model,
        train_dl,
        eval_dataset=test_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        task_processor=task_processor,
    )
