import functools
from dataclasses import dataclass

import torch
from simple_parsing import ArgumentParser
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch import distributed as dist
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from pretrain_mm.distributed.distributed_utils import get_dist_info
from pretrain_mm.distributed.policies import mixed_precision_policy
from pretrain_mm.model.model_utils import setup_model
from pretrain_mm.datasets import DATASETS_AVAILABLE, get_dataset_dir


@dataclass
class TrainConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    auto_wrap_policy: bool = True
    decoder_layer: torch.nn.Module = MistralDecoderLayer

    # dataset
    dataset_name: str = "silatus_websites"
    dataset_dir: str = get_dataset_dir("SILATUS_DATA_DIR")

    # fsdp
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # CPUOffload(offload_params=True) CPUOffload bad https://github.com/pytorch/pytorch/issues/91165
    cpu_offload: CPUOffload = None


def get_dataset(dataset_name: str, dataset_kwargs: dict):
    dataset_info = DATASETS_AVAILABLE[dataset_name]

    dataset = dataset_info["dataset"](**dataset_kwargs)
    return dataset



if __name__ == "__main__":
    parser = ArgumentParser().add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config

    rank, local_rank, world_size = get_dist_info()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model, processor = setup_model(train_config.model_name)

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MistralDecoderLayer,
            },
        ),
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
