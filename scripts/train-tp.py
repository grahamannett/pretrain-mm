import os
from dataclasses import dataclass

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from config.model_configs import ExperimentConfigModelInfo, ExperimentModelConfigMixin
from pretrain_mm.utils.config_utils import BaseTrainConfig


# note:
# Currently Loaded Modulefiles:
# slurm/slurm/23.02.7
# vim/9.0.2149
# git/2.41.0
# singularity/3.7.3
# tmux/3.2a
# cuda_toolkit/12.3.0
# gcc/10.2.0

layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "self_attn.query_key_value": ColwiseParallel(),
    "self_attn.dense": RowwiseParallel(output_layouts=Shard(1)),  # not clear if this is correct
    "mlp.dense_h_to_4h": ColwiseParallel(),
    "mlp.dense_4h_to_h": RowwiseParallel(),
}


@dataclass
class TrainConfig(BaseTrainConfig, ExperimentModelConfigMixin):
    model: ExperimentConfigModelInfo = ExperimentConfigModelInfo.Fuyu


def setup_model(config: TrainConfig):
    model_info = config.model.resolve()
    model = model_info.model_class.from_pretrained(model_info.model_name)
    processor = model_info.processor.from_pretrained(model_info.model_name)
    return model, processor


def fsdp_main():
    config = TrainConfig.cli()

    model, tokenizer = setup_model(config)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    tp_mesh = init_device_mesh("cuda", (8,))

    for layer_id, transformer_block in enumerate(model.layers):
        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )


if __name__ == "__main__":
    fsdp_main()
