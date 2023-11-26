import transformers

from pretrain_mm.utils.config_utils import ModelInitInfo, torch_dtype_float16
from pretrain_mm.model.fuyu import FuyuProcessor

FuyuInfo = ModelInitInfo(
    model_name="adept/fuyu-8b",
    model_kwargs={**torch_dtype_float16},
    ModelCls=transformers.models.fuyu.FuyuForCausalLM,
    # alternative is transformers.models.fuyu.FuyuProcessor but ours is patched
    ProcessorCls=FuyuProcessor,
    model_extra_info={
        "decoder_layer": transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer,
    },
)
