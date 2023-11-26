from dataclasses import dataclass
import math

import torch
import transformers
from simple_parsing import ArgumentParser, Serializable

from config.fuyu import FuyuInfo
from pretrain_mm.model.fuyu.fuyu_processing import FuyuProcessor
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebTaskProcessor, TaskAdapterProcessor, task_mind2web
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm import logger
from config.dev import get_dev_config

import wandb

from tqdm.auto import tqdm
from rich.progress import track


@dataclass
class TrainConfig(Serializable):
    # logging
    wandb_mode: str = "disabled"
    wandb_project: str = "pretrain-mm"
    wandb_group: str = "testing/fuyu-finetune"
    wandb_job_type: str = "finetune"

    # since slurm seems to fuck up
    batch_log_every: int = False

    model_name: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    output_dir: str = "output/model_output"

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    task_func: str = "TitleWebsiteTask"

    data_subset: int = None
    epochs: int = 2
    batch_size: int = 1
    grad_accum_steps: int = 4

    num_workers_dataloader: int = 4
    pin_memory: bool = True

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    lr: float = 2e-05
    clip_gradients: bool = True
    scheduler_type: str = "cosine"
    gamma: float = 0.85

    def get_task_func(self, dataset_info):
        if self.task_func:
            task_func = getattr(dataset_info.tasks, self.task_func)
            return task_func()
        else:
            return dataset_info.task


def setup_wandb(train_config: TrainConfig):
    wandb.init(
        config=train_config,
        project=train_config.wandb_project,
        group=train_config.wandb_group,
        job_type=train_config.wandb_job_type,
        mode=train_config.wandb_mode,
    )

    wandb.run.save()


def eval(model, eval_dataloader, get_loss):
    losses = 0
    model.eval()

    progress = logger.progress()
    batch_task = progress.add_task(f"[cyan]Eval Step: ", total=len(eval_dataloader))

    for idx, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch.to(model.device)
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            image_patches = batch.image_patches
            image_patches_indices = batch.image_patches_indices

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_patches=image_patches,
                image_patches_indices=image_patches_indices,
            )

            loss = get_loss(outputs.logits, input_ids)
            losses += loss.item()

        progress.update(batch_task, advance=1)

    logger.log(f"eval/Loss: {losses}")
    wandb.log(
        {
            "eval/loss": losses,
        }
    )


def train(train_config, model, train_dataloader, test_dataloader):
    max_steps = len(train_dataloader) * train_config.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_scheduler(train_config.scheduler_type, optimizer, max_steps)

    # loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.functional.cross_entropy(
    def get_loss(logits, labels):
        # b, l needed when fsdp
        b, l, c = logits.shape

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, c)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # loss = loss_func(shift_logits.float(), shift_labels)
        loss = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)
        return loss

    logger.info("starting train loop")
    for epoch in range(train_config.epochs):
        # resets
        losses = 0
        grad_steps_loss = []

        # progress bar info
        # progress = logger.progress()
        # batch_task = progress.add_task(f"[cyan]Training Step: ", total=len(train_dataloader))
        # progress.start()

        model.train()
        # for batch_idx, batch in enumerate(train_dataloader):
        # for batch_idx, batch in track(enumerate(train_dataloader), description="Batch Step..."):
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            batch.to(model.device)
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            image_patches = batch.image_patches
            image_patches_indices = batch.image_patches_indices

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_patches=image_patches,
                image_patches_indices=image_patches_indices,
            )

            loss = get_loss(outputs.logits, input_ids)
            loss.backward()

            if train_config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)

            # for grad accum if batch size has to be 1
            if ((batch_idx + 1) % train_config.grad_accum_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                wandb.log(
                    {
                        "train/batch_loss": sum(grad_steps_loss),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )

                grad_steps_loss = []

            grad_steps_loss.append(loss.item())
            losses += grad_steps_loss[-1]

            if batch_idx > 5:
                break

            # progress.update(batch_task, advance=1)

        # stop the batch_task progress so new one can start on next epoch
        # progress.stop()
        logger.info(f"Epoch[{epoch}] loss: {losses}")
        wandb.log({"train/epoch_loss": losses})

        if train_config.output_dir:
            output_path = f"{train_config.output_dir}/checkpoint_{epoch}"
            model.save_pretrained(output_path)

        logger.info(f"Train loss for epoch: {epoch}: {losses:.2f}")
        eval(model, test_dataloader, get_loss=get_loss)


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)


def get_scheduler(scheduler_type: str, optimizer, max_steps: int):
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    warmup_steps = get_warmup_steps(max_steps)

    logger.info(f"[WARMUP STEPS]: {warmup_steps}")
    logger.info(f"[MAX STEPS]: {max_steps}")
    logger.info(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()

    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config
    setup_wandb(train_config)
    m2w_info = get_dev_config(train_config.dataset_name)

    train_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"], subset=train_config.data_subset, **m2w_info["train"]
    )
    test_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"], subset=train_config.data_subset, **m2w_info["test"]
    )

    train_dataset = Mind2Web(train_data_config)
    test_dataset = Mind2Web(test_data_config)

    processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="auto")

    # check that task adapter with processor is working
    task_train_dataset = TaskAdapterProcessor(
        train_dataset,
        task_func=task_mind2web,
        processor=processor,
        preprocessor=Mind2WebTaskProcessor.preprocessor,  # this converts to just text and images, probably should be done in task_func
        postprocessor=Mind2WebTaskProcessor.postprocessor,  # this is needed as Fuyu processor returns tensors with batch dim already so messes up dataloader
    )

    task_test_dataset = TaskAdapterProcessor(
        test_dataset,
        task_func=task_mind2web,
        processor=processor,
        preprocessor=Mind2WebTaskProcessor.preprocessor,
        postprocessor=Mind2WebTaskProcessor.postprocessor,
    )

    collate_fn = DataCollator(processor.pad_token_id, squeeze=(train_config.batch_size != 1))
    train_dataloader = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=train_config.pin_memory,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=train_config.pin_memory,
    )

    logger.info(f"Running Train. Config:\n{train_config.dumps_yaml()}")
    train(train_config, model, train_dataloader, test_dataloader)
