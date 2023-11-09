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
from configs.fuyu_config import FuyuConfig


@dataclass
class TrainConfig:
    # model_name: str = "adept/fuyu-8b"

    model_config: ModelConfig = FuyuConfig()
    auto_wrap_policy: bool = True
    decoder_layer: torch.nn.Module = MistralDecoderLayer

    # dataset
    dataset_name: str = "silatus_websites"
    dataset_dir: str = get_dataset_dir("SILATUS_DATA_DIR")

    # fsdp


if __name__ == "__main__":
    parser = ArgumentParser().add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config

    model, processor = setup_model(
        train_config.model_name,
        model_kwargs=train_config.model_config.model_kwargs,
        tokenizer_kwargs=train_config.model_config.tokenizer_kwargs,
    )

    dataset = get_dataset(train_config.dataset_name, dataset_kwargs={"data_dir": train_config.dataset_dir})
    adapted_dataset =
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
