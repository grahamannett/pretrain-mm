from config.fuyu import FuyuInfo
from pretrain_mm import logger

MODEL_ID = "adept/fuyu-8b"


def fuyu_model_kwargs() -> dict:
    logger.warn('Using device_map="auto" and torch_dtype=torch.float16 for model as 24gb GPU wont work otherwise')
    return {
        "device_map": "auto",
        **FuyuInfo.model_kwargs,
    }
