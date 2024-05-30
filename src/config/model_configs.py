from enum import StrEnum

from pretrain_mm.utils.config_utils import ModelInitInfo

from .fuyu import FuyuInfo
from .paligemma import PaliGemmaInfo


class ExperimentConfigModelInfo(StrEnum):
    Fuyu = "Fuyu"
    PaliGemma = "PaliGemma"

    def get(self) -> ModelInitInfo:
        return {
            "Fuyu": FuyuInfo,
            "PaliGemma": PaliGemmaInfo,
        }[self]
