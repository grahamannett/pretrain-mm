from dataclasses import field
from typing import TypedDict

import torch.nn as nn


class CLMLossKwargs:
    class CLMLossKwargsType(TypedDict):
        reduction: str
        ignore_index: int  # should be -100 generally

    DC_FIELD = field(
        default_factory={
            "reduction": "mean",
            "ignore_index": -100,
        }.copy
    )


class CLMLossAdapter(nn.Module):
    def __init__(self, model_forward: callable, config, **kwargs):
        super().__init__()
        self.model_forward = model_forward
        #
        self.vocab_size: int = config.vocab_size
        self.clm_loss_kwargs: dict = config.causal_lm_loss

    def loss_func(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        loss_fct = nn.CrossEntropyLoss(**self.clm_loss_kwargs)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def forward(
        self,
        **kwargs,
    ):
        if self.training and (labels := kwargs.get("labels", None)) is not None:
            kwargs["labels"] = None

        # original forward just with labels removed
        outputs = self.model_forward(**kwargs)

        if self.training and (labels is not None):
            outputs.loss = self.loss_func(outputs.logits, labels)
        breakpoint()
        return outputs
