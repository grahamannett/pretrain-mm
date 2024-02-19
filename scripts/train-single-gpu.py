import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from simple_parsing import ArgumentParser, choice

import wandb
from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebTaskProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuPatches, FuyuProcessor
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig
from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.lora_utils import BaseLoraConfig, setup_lora


@dataclass
class WandBConfig(BaseWandBConfig):
    group: str = "testing/finetune-fuyu"
    job_type: str = "finetune"


@dataclass
class LoraConfig(BaseLoraConfig):
    target_modules: list[str] = field(default_factory=lambda: FuyuInfo.model_extra_info["lora_target_modules"])
    dropout: float = 1.99


@dataclass
class TrainConfig(BaseTrainConfig):
    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    do_eval: bool = True
    output_dir: str = None  # "output/model_output"
    save_every: Optional[str] = choice("epoch", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    task_func: str = "TitleWebsiteTask"
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
    gen_dataset,
    task_processor,
    max_new_tokens: int = 20,
    num_choices: int = 5,
    pattern_str: str = "box",
    temperature: float = 1.0,
    stop_tokens: list[int] = [],
    include_loss: bool = True,
) -> float:
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
    eval_loss = 0
    model.eval()
    for sample_id in choices:
        sample = gen_dataset[sample_id]
        text = sample["text"] + task_processor.boa_string
        combined_text = sample["text"] + task_processor.boa_string + sample["label"]

        if include_loss:
            with torch.no_grad():
                inputs = task_processor.processor(text=combined_text, images=sample["image"])
                outputs = model(**inputs, labels=inputs["input_ids"])
                eval_loss += outputs.loss.item()

        # generate the answer
        outputs = generate_helper(
            model,
            processor=task_processor.processor,
            inputs={"text": text, "images": sample["image"]},
            max_new_tokens=max_new_tokens,
            stop_tokens=stop_tokens,
            temperature=temperature,
        )
        try:
            post_processed_bbox_tokens = task_processor.processor.post_process_box_coordinates(
                outputs, target_sizes=torch.tensor([sample["image"].size])
            )[0]
            decoded_outputs = task_processor.processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
            # compute loss based on box.  0 is perfect 1 means not even bbox.
            metric_val = loc_metric_from_str(
                target_str=combined_text, pred_str=decoded_outputs, pattern_str=pattern_str
            )
        except ValueError as err:
            # logger.warn(f"Error for outputs: {task_processor.processor.decode(outputs[0][-15:])}")
            logger.warn(f"Error for outputs for eval_with_generate: {err}")
            metric_val = 1.0

        metrics.append(metric_val)

    return {"eval/acc_metric": sum(metrics) / num_choices, "eval/loss": eval_loss / num_choices}


def train(
    train_config: TrainConfig,
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    eval_with_generate_kwargs: dict = None,
):
    # train_config.masked_values = [71019, 71011]
    # masked_values = torch.tensor(train_config.masked_values) if train_config.masked_values else None

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

    # progress = logger.progress(ensure_exit=True, start=True, disable=train_config.dl_disable_progress)

    logger.info("starting train loop")

    for epoch in range(train_config.epochs):
        # resets
        epoch_loss, batch_loss = 0, 0

        # progress bar info - commented out as training is so slow RN it doesnt matter
        # ptask = progress.add_task(f"[cyan]Training Step: ", total=train_config.num_iters or len(train_dataloader))

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            # progress.update(ptask, advance=1)

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
        # progress.stop()  # stop the batch_task progress so new one can start on next epoch

        # EVAL RELATED
        eval_metrics = eval_with_generate(model, **eval_with_generate_kwargs) if train_config.do_eval else 0
        eval_acc_metric = eval_metrics["eval/acc_metric"]
        logger.log(f"E[{epoch}][L:{epoch_loss:.2f}][LR:{scheduler.get_last_lr()[0]:.4f}][Eval:{eval_acc_metric:.4f}]")
        wandb.log({"train/epoch_loss": epoch_loss, **eval_metrics})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")
    parser.add_arguments(LoraConfig, dest="lora_config", prefix="lora.")
    args = parser.parse_args()

    train_config: TrainConfig = args.train_config
    wandb_config: WandBConfig = args.wandb_config
    lora_config: LoraConfig = args.lora_config
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
    processor = FuyuProcessor.from_pretrained(train_config.model_id, trust_remote_code=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        train_config.model_id,
        device_map=train_config.device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.gather_continuous_embeddings = FuyuPatches.gather_continuous_embeddings

    # model.language_model.model.layers = model.language_model.model.layers[:1]

    if lora_config.enabled:
        model, _ = setup_lora(model, lora_config=lora_config)

    task_processor = Mind2WebTaskProcessor(
        processor=processor,
        ignore_index=train_config.IGNORE_INDEX,
        loc_before_action_repr=train_config.loc_before_action_repr,
    )

    task_transforms = {
        "task_func": task_processor.task_mind2web,
        "encode_data": task_processor.encode_data,
        "postprocessor": Mind2WebTaskProcessor.postprocessor,
    }

    task_train_dataset = TaskAdapter(train_dataset, transforms=task_transforms)

    # draw sample as potential errors from samples quickest to find here
    sample = task_train_dataset[1000]

    task_test_dataset = TaskAdapter(test_dataset, transforms=task_transforms)
    gen_test_dataset = TaskAdapter(test_dataset, transforms=[task_processor.task_mind2web])

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

    num_training_steps = (train_config.num_iters or len(train_dl)) * train_config.epochs

    optimizer = get_optimizer(model, learning_rate=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = get_scheduler(
        train_config.scheduler_type,
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=train_config.warmup_ratio,
    )
    show_optim_info(optimizer, scheduler, num_training_steps, warmup_ratio=train_config.warmup_ratio)

    if train_config.output_dir:
        processor.save_pretrained(f"{train_config.output_dir}/processor")

    def eval_with_generate_callback(model, epoch, trainer, **kwargs):
        metrics = eval_with_generate(
            model=model,
            gen_dataset=gen_test_dataset,
            task_processor=task_processor,
            pattern_str=train_config.loc_type,
            stop_tokens=task_processor.extra_stop_tokens,
        )

        eval_str = f"[Eval|Acc:{metrics['eval/acc_metric']:.4f}|Loss:{metrics['eval/loss']:.4f}]"
        logger.log(f"E[{epoch}][L:{trainer.epoch_loss:.2f}][LR:{trainer.last_lr:.4f}]{eval_str}")
        wandb.log({"train/epoch_loss": trainer.epoch_loss, **metrics})

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

    # trainer = Trainer(
    #     config=train_config,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     callbacks={
    #         "train_step": [log_batch_step],
    #         "train_epoch": [save_model_callback, eval_with_generate_callback],
    #     },
    # )

    # logger.log("starting train loop")
    # trainer.setup

    train(
        train_config,
        model,
        train_dl,
        test_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_with_generate_kwargs={
            "gen_dataset": gen_test_dataset,
            "task_processor": task_processor,
            "pattern_str": train_config.loc_type,
            # stop tokens for generate are |SPEAKER| |NEWLINE| |ENDOFTEXT|
            "stop_tokens": task_processor.generate_extra_stop_tokens,
        },
    )
