import torch
import torch.nn as nn
from transformers import PretrainedModel

from pretrain_mm.utils.transforms import make_dummy_func_default


class WrappedModel(PretrainedModel):
    def __init__(self, model: PretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._pre_forward_out = make_dummy_func_default(None)
        self._post_forward_out = make_dummy_func_default(None)

    def forward(self, *args, **kwargs):
        if self._pre_forward_out:
            pre_forward_out = self._pre_forward_out(*args, **kwargs)
        kwargs["output_hidden_states"] = True

        model_output = self._model(*args, **kwargs)

        if self._post_forward_out:
            model_output = self._post_forward_out(
                model_output=model_output, pre_forward_out=pre_forward_out, *args, **kwargs
            )

        return model_output


# example functions for pre_forward_out and post_forward_out


def _get_extra_forward(self, image_patches: torch.Tensor, extra_loss: dict, processor: callable):
    if not self.training:
        return None

    if patch_idx := extra_loss.get("patch_idx"):
        if patch_idx >= len(image_patches):
            pass

        return {patch_idx, image_patches[patch_idx]}


def _image_patch_loss_func(self, model_output, pre_forward_out):
    pass
