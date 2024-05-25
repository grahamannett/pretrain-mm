from typing import Any

import torch
from transformers import ProcessorMixin as HFProcessorMixin

from pretrain_mm.constants import IGNORE_INDEX


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


# MARK: - TextProcessorMixin
class TextProcessorMixin:
    """methods to help with tokenization of text to ids"""

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

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_inputs_start_idx(self, inputs: dict | torch.Tensor, from_token: str = None, offset: int = 1) -> int:
        """helper function to get the start index for inputs

        assumes batch size is 1

        Args:
            inputs (dict): _description_
            from_token (str, optional): _description_. Defaults to None and will match on boa token.

        Returns:
            int: _description_
        """
        from_token = from_token or self.constants.boa_token

        # this will work for FuyuBatch feature
        inputs = getattr(inputs, "input_ids", inputs)

        # only handle 2d or 1d tensor but 1 sample regardless?
        assert inputs.ndim <= 2, "inputs should be 2d or 1d tensor"

        if inputs.ndim == 2:
            assert inputs.shape[0] == 1, "batch size should be 1"
            inputs = inputs[0]

        return (inputs == self.vocab[from_token]).nonzero().flatten().item() - offset

    def replace_text_with_tokens(self, raw_text: str, **kwargs) -> str:
        """_summary_
        this method is for general use and should be overridden by the processor that uses it

        it is called `raw_text` as just text is not ideal when "Add Next Occurance" is used in vscode
        """
        return raw_text


class ProcessorMixin(HFProcessorMixin):
    pad_token_id: int = 0
    enc_kwargs: dict[str, Any] = default_enc_kwargs

    def full_decode(self, outputs: torch.Tensor, masked: bool = True, **kwargs):
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.from_numpy(outputs)

        if masked:
            # mask out IGNORE_INDEX, image_newline_token, and image_placeholder_token
            outputs = self.genmask(outputs)

        outputs = self.post_process_box_coordinates(outputs)
        outputs = self.tokenizer.decode(outputs, **kwargs)
        return outputs

    def genmask(
        self,
        outputs: torch.Tensor,
        tokens_to_mask: list[str | int] = None,
    ):
        tokens_to_mask = tokens_to_mask or _get_tokens_to_mask(self.constants)
        mask = torch.ones(outputs.size(), dtype=torch.bool, device=outputs.device)
        for token in tokens_to_mask:
            if isinstance(token, str):
                token = self.tokenizer.vocab[token]
            mask &= outputs != token
        return outputs[mask]

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
