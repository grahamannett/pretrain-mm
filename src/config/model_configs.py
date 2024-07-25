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


class ExperimentModelConfigMixin:
    @property
    def model_info(self) -> ModelInitInfo:
        info = ModelsAvailable[self.model_path]

        if name_or_path := getattr(self, "model_name_or_path", None):
            info.model_name = name_or_path

        return info
