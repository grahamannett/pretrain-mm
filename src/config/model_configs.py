from enum import StrEnum

from pretrain_mm.utils.config_utils import ModelInitInfo

from .fuyu import FuyuInfo
from .paligemma import PaliGemmaInfo

ModelsAvailable = {
    "Fuyu": FuyuInfo,
    "PaliGemma": PaliGemmaInfo,
}

class ExperimentConfigModelInfo(StrEnum):
    Fuyu = "Fuyu"
    PaliGemma = "PaliGemma"

    def resolve(self) -> ModelInitInfo:
        return ModelsAvailable[self]
