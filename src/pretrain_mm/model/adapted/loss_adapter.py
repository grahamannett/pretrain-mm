from dataclasses import field
from typing import TypedDict

import torch
import torch.nn as nn

from pretrain_mm.constants import IGNORE_INDEX


class CLMLossKwargs:
    class CLMLossKwargsType(TypedDict):
        # needs to be json serializable
        reduction: str
        ignore_index: int  # should be -100 generally
        use: bool

        @classmethod
        def factory(cls, reduction="mean", ignore_index=IGNORE_INDEX, use=True):
            return cls(reduction=reduction, ignore_index=ignore_index, use=use)

    DC_FIELD = field(default_factory=CLMLossKwargsType.factory)


class CLMLossAdapter(nn.Module):
    def __init__(self, model_forward: callable, config, **kwargs):
        super().__init__()
        self.model_forward = model_forward
        #
        self.vocab_size: int = config.vocab_size
        self.clm_loss_kwargs: dict = config.causal_lm_loss

    @classmethod
    def use_and_patch_forward(self, parent: nn.Module):
        raise NotImplementedError("todo: implement this")

    def loss_func(self, logits, labels, input_attention_mask: torch.Tensor = None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if input_attention_mask is not None:
            shift_attention_mask = input_attention_mask[..., 1:]
            # not sure if this will work or needs to be done before .contiguous()
            shift_logits = shift_logits[shift_attention_mask != 0]
            shift_labels = shift_labels[shift_attention_mask != 0]

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
        # computing the loss probably 2x! Not ideal and if labels dont work for the model, it will break
        # broken: https://github.com/huggingface/transformers/issues/30753
        # if self.training and (labels := kwargs.get("labels", None)) is not None:
        #     kwargs["labels"] = None

        # original forward just with labels removed
        outputs = self.model_forward(**kwargs)

        # if self.training and (labels is not None):
        if self.training and ("labels" in kwargs):
            outputs.loss = self.loss_func(outputs.logits, kwargs["labels"])

        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ConditionalGenerationLoss:  # not sure if i should make this a mixin or Module
    def forward(self, logits, labels=None, input_attention_mask=None):
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if input_attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                shift_attention_mask = input_attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(logits.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        return loss
