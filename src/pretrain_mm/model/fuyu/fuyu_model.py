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

    def patch_lm_forward(self, loss_func_kwargs: dict = {}):
        self._old_lm_forward = self.language_model.forward
        self.language_model.forward = self._lm_forward
        self._loss_func_kwargs = loss_func_kwargs

    def _loss_func(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",  # or "sum"
        label_smoothing: float = 0.0,
    ):
        loss = None
        # Causal language modeling loss
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(
                weight=weight,
                size_average=size_average,
                ignore_index=ignore_index,
                reduction=reduction,
                label_smoothing=label_smoothing,
            )
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss

    def _lm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        This _lm_forward is meant to replace the forward that comes from PersimmonForCausalLM
        so can replace the loss and ammend the lm_head out
        ```"""

        # should call PersimmonModel.forward
        outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        hidden_states = outputs[0]
        logits = self.language_model.lm_head(hidden_states)

        loss = self._loss_func(logits, labels, **self._loss_func_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        breakpoint()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
