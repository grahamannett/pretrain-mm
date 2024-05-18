import torch
import transformers

from pretrain_mm.model.fuyu import MODEL_ID, FuyuProcessor, FuyuConstants
from pretrain_mm.utils.config_utils import ModelInitInfo


def get_model_config_kwargs(config):
    return {
        "patch_image_out": config.model_image_patch_loss,
        "patch_idx_latent": config.model_patch_idx_latent,
        "patch_gather_continuous_embeddings": config.model_patch_gather_continuous_embeddings,
        **(
            {"num_hidden_layers": 1, "text_config": {"model_type": "persimmon", "num_hidden_layers": 1}}
            if config.model_chop
            else {}
        ),
    }


FuyuInfo = ModelInitInfo(
    model_name=MODEL_ID,
    model_kwargs={"torch_dtype": torch.float16},
    ModelConstants=FuyuConstants,
    ModelCls=transformers.models.fuyu.FuyuForCausalLM,
    # alternative is transformers.models.fuyu.FuyuProcessor but ours is patched
    ProcessorCls=FuyuProcessor,
    model_extra_info={
        "decoder_layer": transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer,
        "lora_target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    },
    get_model_config_kwargs=get_model_config_kwargs,
)
