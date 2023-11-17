import functools
from dataclasses import dataclass, asdict

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
from pretrain_mm.datasets import get_dataset, get_dataset_dir
from pretrain_mm.utils.config_utils import ModelConfig
from config.fuyu import FuyuConfig


@dataclass
class TrainConfig:
    model_config = FuyuConfig
    auto_wrap_policy: bool = True
    decoder_layer: torch.nn.Module = model_config.model_extra_info["decoder_layer"]

    # dataset
    dataset_name: str = "silatus_websites"
    dataset_dir: str = get_dataset_dir("SILATUS_DATA_DIR")

    # fsdp
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # CPUOffload(offload_params=True) CPUOffload bad https://github.com/pytorch/pytorch/issues/91165
    cpu_offload: CPUOffload = None


if __name__ == "__main__":
    parser = ArgumentParser().add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config

    rank, local_rank, world_size = get_dist_info()
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model = model_config.ModelCls.from_pretrained(model_config.model_name, **model_config.model_kwargs)
    tokenizer = model_config.ProcessorCls.from_pretrained(model_config.model_name, **model_config.tokenizer_kwargs)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            model_config.model_extra_info["decoder_layer"],
        },
    )

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
