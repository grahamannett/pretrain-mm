from torch import nn


from pretrain_mm.utils.config_utils import ModelInitInfo


class Model(nn.Module):
    def __init__(self, config: ModelInitInfo) -> None:
        super().__init__()
