import functools
from dataclasses import dataclass, asdict

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
from pretrain_mm.datasets import get_dataset, Mind2Web, Mind2WebConfig, TaskAdapterProcessor, task_mind2web
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

    batch_size: int = 1
    num_workers_dataloader: int = 8

    weight_decay = 0.0
    lr: float = 2e-05


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
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
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


def train(model, dataloader, optimizer, lr_scheduler,gradient_accumulation_steps):
    dist.barrier()

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
    task_dataset = TaskAdapterProcessor(
        dataset,
        task_func=task_mind2web,
        processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name),
        preprocessor=Mind2Web.task_preprocessor,
    )

    if local_rank == 0:
        logger.log("done with task dataset")

    sampler = DistributedSampler(
                task_dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True,
            )


    train_dataloader = torch.utils.data.DataLoader(
        task_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=sampler,
    )


    model = model_config.ModelCls.from_pretrained(model_config.model_name, **model_config.model_kwargs)
    tokenizer = model_config.ProcessorCls.from_pretrained(model_config.model_name, **model_config.tokenizer_kwargs)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            model_config.model_extra_info["decoder_layer"],
        },
    )

    if local_rank == 0:
        logger.log("putting model into FSDP")

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=train_config.cpu_offload,
    )

    optimizer = get_optimizer(model, train_config.lr, train_config.weight_decay)


