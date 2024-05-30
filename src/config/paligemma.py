from pretrain_mm.model.paligemma import (
    MODEL_ID,
    PaliGemmaConfig,
    PaliGemmaConstants,
    PaliGemmaConstantsClass,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from pretrain_mm.utils.config_utils import BaseTrainConfig, ModelInitInfo


def get_model_config_kwargs(config) -> dict[str, any]:
    """
    NOT USING THIS BUT KEEPING FOR REFERENCE
    USING THE BELOW modify_model_config_callback INSTEAD TO CHOP MODEL FOR LOCAL DEV
    """
    return {
        # "patch_image_out": config.model_image_patch_loss,
        # "patch_idx_latent": config.model_patch_idx_latent,
        # "patch_gather_continuous_embeddings": config.model_patch_gather_continuous_embeddings,
        **({"text_config": {"model_type": "gemma", "num_hidden_layers": 1}} if config.model_chop else {}),
    }


def modify_model_config_callback(model_config: PaliGemmaConfig, exp_config: BaseTrainConfig) -> PaliGemmaConfig:
    # passing these into PaliGemmaConfig.from_pretrained messes up the hidden_size/vocab size etc
    # without spending more time looking into exact config needed to use this is a workaround
    if exp_config.model_chop:
        model_config.vision_config.num_hidden_layers = exp_config.model_chop
        model_config.text_config.num_hidden_layers = exp_config.model_chop

    return model_config


PaliGemmaInfo = ModelInitInfo(
    model_name=MODEL_ID,
    # model_kwargs={"torch_dtype": torch.float16},
    ModelCls=PaliGemmaForConditionalGeneration,
    ModelConfigCls=PaliGemmaConfig,
    ProcessorCls=PaliGemmaProcessor,
    ModelConstantsCls=PaliGemmaConstantsClass,
    ModelConstants=PaliGemmaConstants,
    model_extra_info={},
    get_model_config_kwargs=None,
    modify_model_config_callback=modify_model_config_callback,
)
