from pretrain_mm.model.paligemma.paligemma_model import (
    MODEL_ID,
    PaliGemmaConfig,
    PaliGemmaConstants,
    PaliGemmaConstantsClass,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from pretrain_mm.utils.config_utils import ModelInitInfo


PaliGemmaInfo = ModelInitInfo(
    model_name=MODEL_ID,
    ModelCls=PaliGemmaForConditionalGeneration,
    ModelConfigCls=PaliGemmaConfig,
    ProcessorCls=PaliGemmaProcessor,
    ModelConstantsCls=PaliGemmaConstantsClass,
    ModelConstants=PaliGemmaConstants,
    model_extra_info={},
    get_model_config_kwargs=None,
)
