from dataclasses import make_dataclass
from typing import Any

import torch
from transformers import BatchFeature
from transformers import ProcessorMixin as HFProcessorMixin

from pretrain_mm.constants import IGNORE_INDEX
from pretrain_mm.utils.data_utils import DTObject, get_fields


# enc kwargs related to what special tokens to add
default_enc_kwargs = {
    "add_bos_token": True,
    "add_boa_token": True,
    "label_add_eos_token": True,
}


def _get_tokens_to_mask(constants):
    return [
        constants.image_newline_token,
        constants.image_placeholder_token,
        IGNORE_INDEX,
    ]


class ExtraMetadata(DTObject):
    @classmethod
    def using(cls, extra: dict | bool = None, include_raw: bool = True, **kwargs):
        fields = get_fields(extra) + get_fields(kwargs) if include_raw else get_fields(extra)
        MetadataCls = make_dataclass("Metadata", fields)
        return MetadataCls(**extra, **kwargs)

    @staticmethod
    def on_batch(batch, extra, include_raw, **kwargs):
        batch.extra = {**(extra if isinstance(extra, dict) else {})}
        if include_raw:
            batch.extra.update(**kwargs)
        return batch


# MARK: - TextProcessorMixin
class TextProcessorMixin:
    """
    methods to help with tokenization of text to ids.

    Actually might make sense to merge this with ProcessorMixin below
    """

    _additional_tokens: bool = False
    pad_token_id: int = 0

    @property
    def vocab(self):
        """cant remember if this gave me an error when saving/loading before"""
        return self.tokenizer.vocab

    def _decode(self, outputs: torch.Tensor, **kwargs):
        """helper function to decode outputs"""
        return self.tokenizer.decode(outputs, **kwargs)

    def _ensure_is_id(self, tok: str | int) -> int:
        if isinstance(tok, str):
            tok = self.tokenizer.vocab[tok]
        return tok

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_token_id] = 0
        return attention_mask

    def _tokenize_num_within_tags(self, num_str: str) -> list[int]:
        """helper func for _transform_within_tags in the case where we have a number that is not a bbox or point"""
        if num_str in self.tokenizer.vocab:
            return [self.tokenizer.vocab[num_str]]

        # this is for instances of the number being a float or being > 1000 which is not in the vocab
        return self.tokenizer.encode(num_str, add_special_tokens=False)[1:]

    def add_extra_tokens(self, tokens: list[str], use_flag: bool = True) -> int:
        """
        other option is to use from_pretrained and
        if "additional_tokens" in kwargs:
            proc._additional_tokens = True
        return proc
        """
        num_added = self.tokenizer.add_tokens(tokens)
        if num_added and use_flag:
            self._additional_tokens = True
        return num_added

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, outputs: torch.Tensor, do_post: bool = True, **kwargs) -> str:
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.

        implement this in the subclasses processor if you need to handle special decoding
        """
        if do_post:
            outputs = self.post_process_ids(outputs)

        return self.tokenizer.decode(outputs, **kwargs)

    def full_decode(self, outputs: torch.Tensor, masked: bool = True, **kwargs):
        """
        Decodes the outputs of a model and applies optional masking.

        Args:
            outputs (torch.Tensor): The model outputs to be decoded.
            masked (bool, optional): Whether to apply masking. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the `decode` method.

        Returns:
            The decoded outputs.

        """
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.from_numpy(outputs)
        if masked:
            # mask out IGNORE_INDEX, image_newline_token, and image_placeholder_token
            outputs = self.genmask(outputs)

        outputs = self.decode(outputs, **kwargs)
        return outputs

    def genmask(self, outputs: torch.Tensor, tokens_to_mask: list[str | int] = None):
        """
        Generates a mask for the given outputs tensor, based on the specified tokens to mask.

        Args:
            outputs (torch.Tensor): The tensor containing the outputs.
            tokens_to_mask (list[str | int], optional): The list of tokens to mask. Defaults to None.

        Returns:
            torch.Tensor: The masked outputs tensor.
        """
        tokens_to_mask = tokens_to_mask or _get_tokens_to_mask(self.constants)
        mask = torch.ones(outputs.size(), dtype=torch.bool, device=outputs.device)
        for token in tokens_to_mask:
            if isinstance(token, str):
                token = self.tokenizer.vocab[token]
            mask &= outputs != token
        return outputs[mask]

    def get_inputs_start_idx(
        self, inputs: dict | torch.Tensor, labels: torch.Tensor = None, from_token: str = None, offset: int = 1
    ) -> int:
        """helper function to get the start index for inputs

        assumes batch size is 1

        Args:
            inputs (dict): _description_
            from_token (str, optional): _description_. Defaults to None and will match on boa token.

        Returns:
            int: _description_
        """
        from_token = from_token or self.constants.boa_token

        # this will work for FeatureBatch feature
        inputs = getattr(inputs, "input_ids", inputs)

        if not from_token:
            if (labels is None) or (labels.ndim > 2):
                raise ValueError("from_token or labels (of correct size) must be provided")

            if labels.ndim == 2:
                assert labels.shape[0] == 1, "batch size should be 1"
                labels = labels[0]
            return (labels != IGNORE_INDEX).nonzero().flatten()[0].item()

        # only handle 2d or 1d tensor but 1 sample regardless?
        assert inputs.ndim <= 2, "inputs should be 2d or 1d tensor"

        if inputs.ndim == 2:
            assert inputs.shape[0] == 1, "batch size should be 1"
            inputs = inputs[0]
        return (inputs == self.vocab[from_token]).nonzero().flatten().item() - offset

    def post_process_ids(self, outputs: torch.Tensor, *args, **kwargs):
        """implement in subclass if requires extra post processing on the output ids as Fuyu does

        Args:
            outputs (torch.Tensor): the processor (or model generated) output

        Returns:
            _type_: _description_
        """
        return outputs

    def replace_text_with_tokens(self, raw_text: str, **kwargs) -> str:
        """
        this method is for general use and should be overridden by the processor that uses it

        it is called `raw_text` as just text is not ideal when "Add Next Occurance" is used in vscode
        """
        return raw_text


class ProcessorMixin(HFProcessorMixin):
    enc_kwargs: dict[str, Any] = default_enc_kwargs

    def _attach_extra(self, batch: BatchFeature, extra: bool | dict = False, include_raw: bool = True, **kwargs):
        """
        Attaches extra information to the given batch.
        you want to include raw if you need the raw text label for evaluation without having to decode

        Args:
            batch (BatchFeature): The batch to attach the extra information to.
            extra (bool | dict, optional): The extra information to attach. Defaults to False aka don't attach anything.
            kwargs: Additional keyword arguments to be passed to the `ExtraMetadata.using` method.

        Returns:
            BatchFeature: The batch with the attached extra information.
        """
        return self.create_attachable(batch, extra)(**kwargs)

    def create_attachable(self, batch: BatchFeature, extra: dict | bool = None) -> callable:
        """
        Wonder if it might be better to use like a func vs using attach_extra, but allows currying so can always attach
        later on.  Useful for testing/eval where i dont want the possibility of the raw text/label to be included

        Args:
            batch (BatchFeature): _description_
            extra (dict | bool, optional): _description_. Defaults to None.

        Returns:
            callable: kwargs with key,value as the extra information to attach
        """
        if not extra:
            return lambda: batch

        def func(include_raw: bool = True, **kwargs):
            # batch.extra = ExtraMetadata.using(extra=extra, include_raw=include_raw, **kwargs)
            # i think it makes more sense to just keep as a dict, should be quicker and
            # easier to unpack into eval if needed
            batch.extra = {**extra, **(kwargs if include_raw else {})}
            return batch

        return func

    def encode_sample(
        self,
        sample: dict,
        include_label: bool = True,
        include_text: bool = True,
        # these can override encode_kwargs
        add_bos_token: bool = None,
        add_boa_token: bool = None,
        label_add_eos_token: bool = None,
        label_mask_image_patches: bool = None,
        label_mask_text_ids: bool = None,
        max_length: int = None,
        instruction_spacer: str = "",
    ):
        """Process the input sample to create the sample with output that has labels for training.

        SINCE __call__ should try to mimic the original processor, this method is for containing the logic
        related to going from the sample to the inputs that are needed for the __call__ which can then be used for
        training/eval

        in the case where you want to test generated output you want the inputs to be the encoded inputs without label
        but with boa token

        Args:
        ----
            sample (dict): The input sample containing text, label, and images.

        Returns:
        -------
            dict: The processed output with labels.

        """

        call_kwargs = {
            **self.enc_kwargs,
            "max_length": self.max_length if max_length is None else max_length,
            "extra": sample.get("extra", False),
        }

        # is there a speed difference if i move this outside of here?
        def _patch_kwargs(k: str, v):
            if v is not None:
                call_kwargs[k] = v

        raw_text = sample.get("text")  # text should always be in sample
        raw_image = sample.get("image", None)  # image is not guaranteed to be in the sample
        raw_label = sample.get("label", None)  # label is not guaranteed to be in the sample
        raw_instruction = sample.get("instruction", False)

        if include_text is False:  # may want only image or only instruction
            raw_text = ""

        if include_label is False:
            raw_label = None

        if raw_instruction:
            raw_text = f"{raw_instruction}{instruction_spacer}{raw_text}"

        # if we pass in encode kwargs on the sample then override defaults
        call_kwargs.update(sample.get("encode_kwargs", {}))

        # lastly if we pass in any of the kwargs to encode_kwargs, we want to override the sample and the defaults.
        # this is only useful in the case of eval/test
        _patch_kwargs("add_bos_token", add_bos_token)
        _patch_kwargs("add_boa_token", add_boa_token)
        _patch_kwargs("label_add_eos_token", label_add_eos_token)
        _patch_kwargs("label_mask_image_patches", label_mask_image_patches)
        _patch_kwargs("label_mask_text_ids", label_mask_text_ids)

        # encode with the actual processor
        batch = self.__call__(
            text=raw_text,
            images=raw_image,
            label=raw_label,
            **call_kwargs,
        )

        return batch

    def update(self, **kwargs) -> "ProcessorMixin":
        for k, v in kwargs.items():
            # if the v is a dict then merge it into the existing dict
            if isinstance(v, dict):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)
        return self
