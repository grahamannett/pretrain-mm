from dataclasses import dataclass
import math
import os

import torch
import transformers
from simple_parsing import ArgumentParser, Serializable

from config.fuyu import FuyuInfo
from pretrain_mm.model.fuyu.processing_fuyu import FuyuProcessor
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebTaskProcessor, TaskAdapter, TaskAdapterProcessor, task_mind2web
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm import logger
from pretrain_mm.utils.eval_utils import box_pattern, bbox_metric
from config.dev import get_dev_config


import wandb

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

    output_dir: str = None # "output/model_output"

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
    task_func: str = "TitleWebsiteTask"

    data_subset: int = None
    epochs: int = 2
    batch_size: int = 1
    grad_accum_steps: int = 4

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    lr: float = 2e-05
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


def setup_wandb(train_config: TrainConfig):
    wandb.init(
        config=train_config,
        project=train_config.wandb_project,
        group=train_config.wandb_group,
        job_type=train_config.wandb_job_type,
        mode=train_config.wandb_mode,
    )

    wandb.run.save()


def check_train_config(train_config: TrainConfig) -> None:
    logger.info(f"Running Train. Config:\n{train_config.dumps_yaml()}")

    if train_config.output_dir is None:
        output_dir_warn = "train_config.output_dir is None"
        output_dir_warn += "\nthis will not save model and if you are doing real train you should exit now"
        logger.warn(output_dir_warn)


def eval_with_generate(model, gen_dataset, processor, max_new_tokens: int = 30):
    """
    30 is chosen as seems like that is approximately number of tokens for something like

    Click @ <box> int, int, int, int </box>
    """
    sample = gen_dataset[0]

    combined_text = sample['text'] + sample['label']
    model_inputs = processor(text=sample['text'], images=sample['image'])
    target_tokens = processor(text=combined_text, images=sample['image'], return_tensors="pt").input_ids[:, model_inputs.input_ids.size(1):]


    generated_output = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    post_processed_tokens = processor.post_process_box_coordinates(generated_output[:, model_inputs.input_ids.shape[1]:])[0]
    model_outputs = processor.decode(post_processed_tokens, skip_special_tokens=True)

    # bbox_metric(target_pos=sample['label'], sequence=model_outputs, tokenizer)
    try:

        target_strs = box_pattern.match(sample['label'])
        target_values = map(int, target_strs.groups())


        if matched := box_pattern.search(decoded_str):
            preds = map(int, matched.groups())
            # matched = torch.tensor([int(m) for m in matched])
            max_value = max(torch.max(target_values, preds))

            # normalize
            metric_value = torch.nn.functional.l1_loss(target_values, preds) / max_value


    except Exception as err:
        logger.warn(f"Error for {target_pos}, computing bbox metric: {err}")
        # return 1.0

    breakpoint()


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

def train_step(model, batch, loss_func, gradient_clipping: float = None):
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

    loss = loss_func(outputs.logits, input_ids)

    return loss



def train(train_config, model, train_dataloader, test_dataloader, eval_with_generate_kwargs: dict = None):
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
        progress = logger.progress(disable=train_config.dl_disable_progress)
        batch_task = progress.add_task(f"[cyan]Training Step: ", total=len(train_dataloader))
        progress.start()

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
        # for batch_idx, batch in track(enumerate(train_dataloader), description="Batch Step..."):
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

            progress.update(batch_task, advance=1)

            if batch_idx > train_config.grad_accum_steps * 10:
                break


        # stop the batch_task progress so new one can start on next epoch
        progress.stop()
        # TODO: Move this to eval
        logger.info("DOING EVAL WITH GENERATE")
        eval_with_generate(model, **eval_with_generate_kwargs)

        logger.info(f"Epoch[{epoch}] loss: {losses}")
        wandb.log({"train/epoch_loss": losses})

        if train_config.output_dir:
            output_path = f"{train_config.output_dir}/checkpoint_{epoch}"
            model.save_pretrained(output_path)

        logger.info(f"Train loss for epoch: {epoch}: {losses:.2f}")

        # EVAL RELATED SHOULD BE USED HERE
        # eval(model, test_dataloader, get_loss=get_loss)


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

    # setup wandb and then check config so printed config goes into logs
    setup_wandb(train_config)
    check_train_config(train_config)

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

    gen_test_dataset = TaskAdapter(
        test_dataset,
        task_func=task_mind2web,
    )

    collate_fn = DataCollator(processor.pad_token_id, squeeze=(train_config.batch_size != 1))
    train_dataloader = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.dl_num_workers,
        pin_memory=train_config.dl_pin_memory,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        task_train_dataset,
        collate_fn=collate_fn,
        batch_size=train_config.batch_size,
        num_workers=train_config.dl_num_workers,
        pin_memory=train_config.dl_pin_memory,
    )

    # eval_with_generate(model, gen_dataset=gen_test_dataset, processor=processor)
    train(train_config, model, train_dataloader, test_dataloader, eval_with_generate_kwargs={"gen_dataset": gen_test_dataset, "processor": processor})
