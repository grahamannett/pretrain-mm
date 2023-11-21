from dataclasses import dataclass

import torch
import transformers
from simple_parsing import ArgumentParser

from config.fuyu import FuyuInfo
from pretrain_mm.datasets import get_dataset, Mind2Web, Mind2WebConfig, TaskAdapterProcessor, task_mind2web
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm import logger
from config.dev import get_dev_config


@dataclass
class TrainConfig:
    model_name: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    model_config = FuyuInfo

    auto_wrap_policy: bool = True

    chop_model: int = -1

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    task_func: str = "TitleWebsiteTask"

    data_subset: int = 10
    epochs: int = 2
    batch_size: int = 1
    num_workers_dataloader: int = 4

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


def dev_load_model(model_name, model_kwargs, ModelCls, train_config):
    if train_config.chop_model > 0:
        model = ModelCls.from_pretrained(model_name, **model_kwargs)
        model.language_model.model.layers = model.language_model.model.layers[: train_config.chop_model]
        model.to("cuda")
    else:
        model = ModelCls.from_pretrained(model_name, device_map="auto", **model_kwargs)
    return model


def eval(model, eval_dataloader, compute_loss):
    losses = 0
    model.eval()

    for idx, batch in enumerate(eval_dataloader):
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

        loss = compute_loss(outputs.logits, input_ids)
        losses += loss.item()


def train(train_config, model, train_dataloader, test_dataloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    loss_func = torch.nn.CrossEntropyLoss()

    def get_loss(logits, labels):
        b, l, c = logits.shape

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, c)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_func(shift_logits.float(), shift_labels)
        return loss

    for epoch in range(train_config.epochs):
        lossses = 0
        progress = logger.progress()
        batch_task = progress.add_task("[cyan]Training Step", total=len(train_dataloader))

        for idx, batch in enumerate(train_dataloader):
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

            # for grad accum if batch size has to be 1
            # if ((idx + 1) % 2) == 0:
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            losses += loss.item()

            progress.update(batch_task, advance=1)

            # logger.print(f"idx: {idx}, loss: {loss.item()}")

        eval(model, test_dataloader, calculate_loss=get_loss)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()

    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config
    m2w_info = get_dev_config(train_config.dataset_name)

    train_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], subset=train_config.data_subset, **m2w_info["train"])
    test_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], subset=train_config.data_subset, **m2w_info["test"])

    train_dataset = Mind2Web(train_config)
    test_dataset = Mind2Web(test_config)

    processor = FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name)
    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b", device_map="auto")

    # check that task adapter with processor is working
    task_train_dataset = TaskAdapterProcessor(
        train_dataset,
        task_func=task_mind2web,
        processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name),
        preprocessor=Mind2Web.task_preprocessor,  # this converts to just text and images, probably should be done in task_func
        postprocessor=Mind2Web.task_postprocessor,  # this is needed as Fuyu processor returns tensors with batch dim already so messes up dataloader
    )

    task_test_dataset = TaskAdapterProcessor(
        test_dataset,
        task_func=task_mind2web,
        processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name),
        preprocessor=Mind2Web.task_preprocessor,
        postprocessor=Mind2Web.task_postprocessor,
    )

    collate_fn = DataCollator(processor.pad_token_id, device=model.device)
    train_dataloader = torch.utils.data.DataLoader(task_train_dataset, batch_size=1, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(task_train_dataset, batch_size=1, collate_fn=collate_fn)

    train(train_config, model, train_dataloader, test_dataloader)
