from pretrain_mm.model.paligemma import (
    MODEL_ID,
    PaliGemmaConfig,
    PaliGemmaConstants,
    PaliGemmaConstantsClass,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from pretrain_mm.utils.config_utils import ModelInitInfo


def get_model_config_kwargs(config):
    return {
        # "patch_image_out": config.model_image_patch_loss,
        # "patch_idx_latent": config.model_patch_idx_latent,
        # "patch_gather_continuous_embeddings": config.model_patch_gather_continuous_embeddings,
        **({"text_config": {"model_type": "gemma", "num_hidden_layers": 1}} if config.model_chop else {}),
    }


PaliGemmaInfo = ModelInitInfo(
    model_name=MODEL_ID,
    # model_kwargs={"torch_dtype": torch.float16},
    ModelCls=PaliGemmaForConditionalGeneration,
    ModelConfigCls=PaliGemmaConfig,
    ProcessorCls=PaliGemmaProcessor,
    ModelConstantsCls=PaliGemmaConstantsClass,
    ModelConstants=PaliGemmaConstants,
    model_extra_info={},
    get_model_config_kwargs=lambda config: {},
)
