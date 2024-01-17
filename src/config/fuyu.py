import torch
import transformers

# from pretrain_mm.model.fuyu import FuyuProcessor
from pretrain_mm.model.fuyu.processing import FuyuProcessor
from pretrain_mm.utils.config_utils import ModelInitInfo

FuyuInfo = ModelInitInfo(
    model_name="adept/fuyu-8b",
    model_kwargs={"torch_dtype": torch.float16},
    ModelCls=transformers.models.fuyu.FuyuForCausalLM,
    # alternative is transformers.models.fuyu.FuyuProcessor but ours is patched
    ProcessorCls=FuyuProcessor,
    model_extra_info={
        "decoder_layer": transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer,
        "lora_target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    },
)
