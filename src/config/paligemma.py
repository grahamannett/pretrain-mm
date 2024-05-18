from pretrain_mm.model.paligemma import (
    MODEL_ID,
    PaliGemmaConstantsClass,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from pretrain_mm.utils.config_utils import ModelInitInfo


PaliGemmaInfo = ModelInitInfo(
    model_name=MODEL_ID,
    # model_kwargs={"torch_dtype": torch.float16},
    ModelCls=PaliGemmaForConditionalGeneration,
    ProcessorCls=PaliGemmaProcessor,
    ModelConstantsCls=PaliGemmaConstantsClass,
    model_extra_info={},
    get_model_config_kwargs=lambda config: {},
)
