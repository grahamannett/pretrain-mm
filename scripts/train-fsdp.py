import functools
from dataclasses import dataclass, asdict
import math

import transformers
import torch
from simple_parsing import ArgumentParser
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch import distributed as dist
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from torch.utils.data import DistributedSampler

from pretrain_mm import logger
from pretrain_mm.distributed.distributed_utils import get_dist_info
from pretrain_mm.distributed.policies import mixed_precision_policy
from pretrain_mm.model.model_utils import setup_model
from pretrain_mm.datasets import get_dataset, Mind2Web, Mind2WebConfig, TaskAdapter, task_mind2web
from config.fuyu import FuyuInfo


@dataclass
class TrainConfig:
    model_config = FuyuInfo
    auto_wrap_policy: bool = True
    decoder_layer: torch.nn.Module = model_config.model_extra_info["decoder_layer"]

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"

    # fsdp
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # CPUOffload(offload_params=True) CPUOffload bad https://github.com/pytorch/pytorch/issues/91165
    cpu_offload: CPUOffload = None

    epochs: int = 2
    batch_size: int = 1
    num_workers_dataloader: int = 8

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    lr: float = 2e-05
    clip_gradients: bool = True
    scheduler_type: str = "cosine"


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def get_scheduler(local_rank, scheduler_type: str, max_steps: int, optimizer: torch.optim.Optimizer):
    if scheduler_type.lower() in ["steplr", "step_lr"]:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # warmup_steps = get_warmup_steps(max_steps)
    warmup_steps = math.ceil(max_steps * 0.05)

    if local_rank == 0:
        logger.log(f"[WARMUP STEPS]: {warmup_steps}")
        logger.info(f"[MAX STEPS]: {max_steps}")
        logger.info(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def train(epochs, model, dataloader, optimizer, scheduler, train_config, local_rank):
    model.train()
    dist.barrier()

    loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(logits, labels):
        b, l, c = logits.shape

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, c)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits.float(), shift_labels)
        return loss

    for epoch in range(0, epochs):
        current_epoch = epoch + 1

        progress = logger.progress(disable=local_rank != 0)
        traj_task = progress.add_task("[cyan]Training Step", total=len(dataloader))

        for step, batch in enumerate(dataloader):
            current_step = step + 1

            batch["input_ids"] = batch["input_ids"].squeeze(0)
            batch["attention_mask"] = batch["attention_mask"].squeeze(0)
            batch["image_patches"] = [b.squeeze(0) for b in batch["image_patches"]]
            batch["image_patches_indices"] = batch["image_patches_indices"].squeeze(0)

            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                # "image_patches": [b.to(model.device) for b in batch["image_patches"]],
                # "image_patches_indices": batch["image_patches_indices"].to(model.device),
            }

            # inputs["labels"] = torch.clone(inputs["input_ids"]).to(model.device)

            # forward
            model_output = model(**inputs)
            logits = model_output.logits

            loss = compute_loss(logits, inputs["input_ids"])

            # backward
            loss.backward()

            # clipping
            if train_config.clip_gradients:
                model.clip_grad_norm_(train_config.gradient_clipping).item()
                # grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            progress.update(traj_task, advance=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config

    rank, local_rank, world_size = get_dist_info()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    config = Mind2WebConfig(task_dir=train_config.dataset_dir, subset=10, local_rank=local_rank)
    config._local_rank = local_rank
    dataset = Mind2Web(config)

    # check that task for this dataset is working

    # check that task adapter with processor is working
    processor = FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name)
    task_dataset = TaskAdapter(
        dataset,
        {
            "task_func": task_mind2web,
            "preprocessor": Mind2Web.task_preprocessor,
            "processor": processor,
        },
    )

    if local_rank == 0:
        logger.log("done with task dataset")

    sampler = DistributedSampler(
        task_dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        shuffle=True,
    )

    if local_rank == 0:
        logger.log("done with sampler creation")

    train_dataloader = torch.utils.data.DataLoader(
        task_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=sampler,
    )

    if local_rank == 0:
        logger.log("making model")

    # model = model_config.ModelCls.from_pretrained(model_config.model_name) #,  torch_dtype=torch.float16)
    # model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b", torch_dtype=torch.bfloat16)
    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # model_config.model_extra_info["decoder_layer"],
            transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer,
        },
    )

    if local_rank == 0:
        logger.log("putting model into FSDP")

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        # mixed_precision=MixedPrecision(
        #     param_dtype=torch.bfloat16,
        #     reduce_dtype=torch.bfloat16,
        #     buffer_dtype=torch.bfloat16,
        # ),
        backward_prefetch=None,
        param_init_fn=None,
        # cpu_offload=train_config.cpu_offload,
    )

    max_steps = len(train_dataloader) * train_config.epochs

    optimizer = get_optimizer(model, train_config.lr, train_config.weight_decay)
    scheduler = get_scheduler(local_rank, train_config.scheduler_type, max_steps, optimizer=optimizer)

    # train(train_config.epochs, model=model, dataloader=train_dataloader)
    train(
        epochs=train_config.epochs,
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
        local_rank=local_rank,
    )
