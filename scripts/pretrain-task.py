import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import constants, logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, Mind2WebTaskProcessor, TaskAdapter
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.model.fuyu import FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.trainer.optim import get_optimizer, get_scheduler, show_optim_info
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig, LocalDataConfig
from pretrain_mm.utils.eval_utils import loc_metric_from_str
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import box_pattern


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
    eval_num_samples: int = 2
    output_dir: str = None  # "output/model_output"
    save_every: Optional[str] = choice("epoch", "best", default=None)

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"
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
    dl_num_workers: int = 4
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

    # pretrain related
    pretrain_task_name: str = "GenerateNumPotentialActions"
    cands_range: tuple[int, int] = (2, 10)
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
    """ """
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


def pretrain_dataloader_test(config, model, dataloader):
    for epoch in range(config.epochs):
        logger.log(f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(dataloader):
            batch.to(model.device)
            logger.log(f"Batch: {batch_idx}")


def add_extra_tokens(config, model, processor):
    # i dont think this is a good idea to modify the model output
    if extra_tokens := FuyuConstants.get_extra_tokenizer_tokens(config.extra_tokenizer_toks):
        num_added = processor.add_extra_tokens(extra_tokens)
        model.resize_token_embeddings(len(processor.tokenizer))
        model.increase_output_size(model.language_model.lm_head, increase_by=num_added, patch_vocab=False)
        model.config.vocab_size = len(processor.tokenizer)
        model.language_model.config.vocab_size = len(processor.tokenizer)


def pretrain(
    config: PreTrainConfig,
    model,
    train_dataloader,
    eval_dataset,
    optimizer,
    scheduler,
    task_processor,
):
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    def clip_grad():
        if config.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)

    def _epoch_reset():
        model.train()
        epoch_loss, batch_loss, eval_acc = 0, 0, 0
        return epoch_loss, batch_loss, eval_acc

    def do_grad_accum_step(batch_idx: int):
        if batch_idx == 0:  # dont do it for batch 0
            return False
        if (batch_idx % config.grad_accum_steps) == 0:
            return True
        if batch_idx == config.num_iters:
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

    def _should_break(batch_idx):
        if config.num_iters and (config.num_iters < batch_idx):
            return True
        return False

    logger.info("Starting train")

    if config.do_eval_pre:
        eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_tokens)

    for epoch in range(config.epochs):

        epoch_loss, batch_loss, eval_acc = _epoch_reset()

        for batch_idx, batch in enumerate(train_dataloader):

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
                logger.log_data({"train/batch_loss": batch_loss, "learning_rate": scheduler.get_last_lr()[0]})

                epoch_loss += batch_loss
                batch_loss = 0

            if _should_break(batch_idx):
                break

        # save before eval as hanging during eval at present
        save_helper(epoch)

        # EVAL RELATED
        if config.do_eval:
            # eval_metrics = eval_with_generate(model, eval_dataset, task_processor, stop_tokens=stop_tokens)
            # eval_val = eval_metrics["eval/acc_metric"]
            eval_metrics = eval_by_completion(model, eval_dataset, num_samples=config.eval_num_samples)
            eval_val = eval_metrics["eval/dist_metric"]

            logger.log_data({"train/epoch_loss": epoch_loss, **eval_metrics})

        logger.log(f"E[{epoch}][L:{epoch_loss:.2f}][LR:{scheduler.get_last_lr()[0]:.4f}][Eval:{eval_val:.4f}]")

    logger.log(f"Training Done")


def eval_by_completion(model, dataset, num_samples: int = 1):

    def measure_func(outputs, label: list[int] | str):
        try:
            if isinstance(label, str):
                label = list(map(int, box_pattern.search(label).groups()))

            decoded_output = processor.full_decode(outputs)
            if box_match := box_pattern.search(decoded_output):
                box_vals = list(map(int, box_match.groups()))
                # metric = [(l1 - l2) ** 2 for l1, l2 in zip(label, box_vals)]
                metric = math.dist(label, box_vals)
                return metric
        except:
            pass

        return False

    metrics = []
    errs = 0
    for n in range(num_samples):
        while True:
            sample = dataset.get_with_transform(pretrain_task_processor.acc_func_complete_box)

            if isinstance(sample, dict):
                break

        sample_enc = task_processor.encode_data(
            sample,
            add_bos_token=False,
            add_boa_token=False,
            label_add_eos_token=False,
            include_label=False,
        )

        gen_output = generate_helper(model, model_inputs=sample_enc.to(model.device), max_new_tokens=5)

        if metric := measure_func(gen_output, sample["label"]):
            metrics.append(metric)
        else:
            errs += 1

    return {
        "eval/dist_metric": sum(metrics) / len(metrics),
        "eval/errs": errs,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreTrainConfig, dest="pretrain_config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")
    parser.add_arguments(LocalDataConfig, dest="local_data_config", prefix="local_data.")

    args = parser.parse_args()

    config: PreTrainConfig = args.pretrain_config
    wandb_config: WandBConfig = args.wandb_config
    local_data_config: LocalDataConfig = args.local_data_config
    model_config = config.model_config

    # setup wandb + check config such that yaml printed config is in wandb console logs
    logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
    logger.tools.setup_local_data(local_data_config=local_data_config, config=config)
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
    train_dataset.setup_pretrain()
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(config.model_id)
    model = FuyuForCausalLM.from_pretrained(config.model_id, device_map=config.device, torch_dtype=torch.bfloat16)

    pretrain_task_processor = Mind2WebPretrainProcessor(
        pretrain_task_name=config.pretrain_task_name,
        cands_range=config.cands_range,
        skip_include_text=config.skip_include_text,
        get_text_from=config.get_text_from,
        # ocr_preprocessed=torch.load("output/processed/train_ds_raw_output.pt"),
    )

    task_processor = Mind2WebTaskProcessor(
        processor=processor,
        ignore_index=config.IGNORE_INDEX,
        loc_before_action_repr=config.loc_before_action_repr,
        max_length=config.max_length,
        encode_kwargs={"label_mask_text_ids": True},
    )

    # generate possible actions pretrain task
    transforms = {
        "pretrain_task": pretrain_task_processor.pretrain_func_generate_possible_actions,
        "encode": task_processor.encode_data,
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

    num_training_steps = (config.num_iters or len(train_dl)) * config.epochs
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

    pretrain(
        config,
        model,
        train_dl,
        eval_dataset=test_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        task_processor=task_processor,
    )
