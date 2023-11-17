import torch
import transformers

from pretrain_mm.utils.config_utils import ModelInitInfo, torch_dtype_float16

FuyuInfo = ModelInitInfo(
    model_name="adept/fuyu-8b",
    model_kwargs={**torch_dtype_float16},
    ModelCls=transformers.models.fuyu.FuyuForCausalLM,
    ProcessorCls=transformers.models.fuyu.FuyuProcessor,
)
