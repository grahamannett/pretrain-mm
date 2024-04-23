import torch

# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Linear, L1Loss, Sigmoid
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.fuyu.modeling_fuyu import FuyuConfig as BaseFuyuConfig
from transformers.models.fuyu.modeling_fuyu import FuyuForCausalLM as BaseFuyuForCausalLM

from pretrain_mm import constants, logger
from pretrain_mm.model.fuyu.fuyu_embed import FuyuPatches
from pretrain_mm.model.model_utils import ModifiedOutputMixin


def _chop_model(config: BaseFuyuConfig, num_hidden_layers: int):
    config.text_config.num_hidden_layers = num_hidden_layers
    config.num_hidden_layers = num_hidden_layers
    return config


class LossKey:
    IMAGE_PATCH_LOSS = "image_patch_loss"
    CLM = "clm"

    # other
    LOSS_KW = "loss_kwargs"


class FuyuForCausalLM(BaseFuyuForCausalLM, ModifiedOutputMixin):
    # _do_chop_model = False
    _do_chop_model: bool = False
    _do_patch_forward: bool = False

    _loss_funcs = {LossKey.CLM: {LossKey.LOSS_KW: {}}}

    def __init__(self, config: BaseFuyuConfig, *args, **kwargs):
        if self._do_chop_model:
            # for making model smaller, has to chop the config as that is what is used to init the model
            # if you do it somewhere else, its likely the weights will be initialized for the full model
            logger.warn("CHOPPING MODEL.\nDEBUG ONLY\n" * 5)
            config = _chop_model(config, 1)

        super().__init__(config, *args, **kwargs)
        if self._do_patch_forward:
            self.patch_forward(config)

    @classmethod
    def from_pretrained(cls, *model_args, **kwargs) -> "FuyuForCausalLM":
        """
        this has to be included in the from_pretrained to work with the patch_gather_embeddings iirc
        """
        model = super().from_pretrained(*model_args, **kwargs)
        model = FuyuPatches.patch_gather_embeddings(model)
        return model

    def patch_forward(self, config, loss_func_kwargs: dict = {}):
        self.patch_image_patch_out(config=config)

    def patch_image_patch_out(
        self,
        config,
        # for BCEWithLogitsLoss
        loss_kwargs: dict = {
            "reduction": "mean",
            # "weight": None,
            # "size_average": None,
            # "reduce": None,
            # "pos_weight": None,
        },
    ):
        self.image_patch_out = nn.Sequential(
            nn.Linear(config.hidden_size, config.patch_size * config.patch_size * config.num_channels),
            nn.Tanh(),
        )

        self.image_patch_out.to(self.device)
        self._loss_funcs[LossKey.IMAGE_PATCH_LOSS] = {LossKey.LOSS_KW: loss_kwargs}

    def _clm_loss_func(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor = None,
        size_average=None,
        ignore_index: int = constants.IGNORE_INDEX,
        reduction: str = "mean",  # or "sum"
        label_smoothing: float = 0.0,
    ):
        loss = None
        # Causal language modeling loss
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_func = nn.CrossEntropyLoss(
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
            loss = loss_func(shift_logits, shift_labels)

        return loss

    def _image_patch_loss_func(
        self,
        logits: torch.Tensor,
        # image_patch will be [idx, image_patch]
        image_patch_idx: int = None,
        image_patch: torch.Tensor = None,
        #
    ):
        patch_loss = None
        # Patch language modeling loss
        if image_patch is not None:
            patch_logits = self.image_patch_out(logits)[:, image_patch_idx]
            loss_func = nn.L1Loss(**self._loss_funcs[LossKey.IMAGE_PATCH_LOSS][LossKey.LOSS_KW])
            patch_loss = loss_func(patch_logits, image_patch.to(patch_logits.device))

        return patch_loss

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
        extra_forward_kwargs: dict = None,
        **kwargs,
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

        hidden_states = outputs[0]
        logits = self.language_model.lm_head(hidden_states)

        loss = self._clm_loss_func(logits, labels, **self._loss_funcs[LossKey.CLM][LossKey.LOSS_KW])

        if extra_forward_kwargs and (LossKey.IMAGE_PATCH_LOSS in self._loss_funcs):
            loss += self._image_patch_loss_func(hidden_states, *extra_forward_kwargs["image_patch"])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _get_extra_forward(self, image_patches, **extra_forward):
        # extra forward is for additional losses that ARE ONLY USED IN TRAINING
        if not self.training:
            return None

        if (LossKey.IMAGE_PATCH_LOSS in self._loss_funcs) and (extra_loss := extra_forward.get("extra_loss")):
            patch_idx = extra_loss["patch_idx"]
            # how is this possible?
            if patch_idx >= image_patches.shape[1]:
                return None

            return {
                "image_patch": (patch_idx, image_patches[:, patch_idx]),
            }

        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask=None,  # Optional[torch.Tensor] = None,
        position_ids=None,  # Optional[torch.LongTensor] = None,
        past_key_values=None,  # Optional[List[torch.FloatTensor]] = None,
        inputs_embeds=None,  # Optional[torch.FloatTensor] = None,
        use_cache=None,  # Optional[bool] = None,
        labels=None,  # Optional[torch.Tensor] = None,
        output_attentions=None,  # Optional[bool] = None,
        output_hidden_states=None,  # Optional[bool] = None,
        return_dict=None,  # Optional[bool] = None,
        extra: dict = {},
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        almost identical as the forward from Fuyu, but I want ability to add extra forward kwargs and stash tensors

        and dont want to wrap/unwrap the attentions/hidden states
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # need to parse from extra to get the image patch from image_patches since we only know the x,y
        extra_forward_kwargs = self._get_extra_forward(image_patches=image_patches, **extra)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0)
                    for patch in image_patches
                ]
                inputs_embeds = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                )

        outputs = self._lm_forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
            extra_forward_kwargs=extra_forward_kwargs,
        )

        return outputs
