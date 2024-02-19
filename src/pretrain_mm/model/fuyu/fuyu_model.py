import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.fuyu.modeling_fuyu import FuyuForCausalLM as BaseFuyuForCausalLM

from pretrain_mm.model.fuyu.fuyu_embed import FuyuPatches
from pretrain_mm.model.model_utils import ModifiedOutputMixin


class FuyuForCausalLM(BaseFuyuForCausalLM, ModifiedOutputMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs) -> "FuyuForCausalLM":
        model = super().from_pretrained(*model_args, **kwargs)
        model = FuyuPatches.patch_gather_embeddings(model)
        return model

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        outputs = super().forward(
            labels=None,  # allow for custom loss
            input_ids=input_ids,
            image_patches=image_patches,
            image_patches_indices=image_patches_indices,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs.loss = self._custom_loss_func(outputs.logits, labels, weight=None, size_average=None, ignore_index=-100)
        return outputs

    def _custom_loss_func(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor = None,
        size_average=None,
        ignore_index: int = -100,
    ):
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss
