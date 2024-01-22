import torch
from torch.nn import CrossEntropyLoss
from transformers.models.fuyu.modeling_fuyu import FuyuForCausalLM as BaseFuyuForCausalLM

from pretrain_mm.model.fuyu.fuyu_embed import CombineEmbeddings


class FuyuForCausalLM(BaseFuyuForCausalLM):
    @classmethod
    def from_pretrained(cls, *model_args, **kwargs) -> "FuyuForCausalLM":
        model = super().from_pretrained(*model_args, **kwargs)
        model = CombineEmbeddings.patch_gather_embeddings(model)
        return model

    def _custom_loss_func(self, logits: torch.Tensor, labels: torch.Tensor):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss
