from enum import StrEnum

from .fuyu import FuyuInfo
from .paligemma import PaliGemmaInfo


class ExperimentConfigModelInfo(StrEnum):
    Fuyu = "Fuyu"
    PaliGemma = "PaliGemma"

    def get(self):
        return {
            "Fuyu": FuyuInfo,
            "PaliGemma": PaliGemmaInfo,
        }[self]
