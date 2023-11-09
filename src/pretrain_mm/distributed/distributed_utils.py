import os
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)


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
