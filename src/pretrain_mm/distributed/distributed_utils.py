import os
import shutil
from typing import Callable


import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def setup() -> None:
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup() -> None:
    dist.destroy_process_group()


def get_dist_info() -> tuple[int, int, int]:
    """
    Returns:
        tuple[int, int, int]: rank, local_rank, world_size that are set by torchrun
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def save_model(local_rank: int, model, tokenizer, outpath: str, current_epoch: int, current_step: int) -> None:
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"/epoch_{current_epoch}/step_{current_step}"
        model.save_pretrained(outpath, state_dict=cpu_state)
        tokenizer.save_pretrained(outpath)


import datasets


datasets.disable_caching()
cache_path = "./tmp/cache"


def dataset_map_multi_worker(dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs) -> datasets.Dataset:
    # saw this: https://twitter.com/jxmnop/status/1716834517909119019
    # https://gist.github.com/jxmorris12/69a730fee174f5309968e984c298f8f2
    # and thought i could use for processing but it seems like its ddp specific and not for mp
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return dataset.map(map_fn, *args, **kwargs)
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache") for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets([datasets.load_from_disk(p) for p in ds_shard_filepaths])
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset
