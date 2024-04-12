import re
from itertools import chain

import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from pretrain_mm import logger
from pretrain_mm.constants import IGNORE_INDEX
from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstants
from pretrain_mm.model.fuyu.fuyu_image_processor import FuyuImageProcessor
from pretrain_mm.utils.token_tag_utils import TagType, token_box_pattern, token_point_pattern


def coords_raw_to_scaled(coords: list[str], scale_factor: float = 1.0) -> list[str]:
    """
    takes a list of string ints and scales them by a factor then returns a list of string (that are ints) to be tokenized

    goes from full size (e.g. 1920) to 1/2 size (e.g. 960)
    """

    def _scale_fn(val):
        return str(round((float(val) / 2) * scale_factor))

    return [_scale_fn(val) for val in coords]


def coords_scaled_to_raw(coords: list[str], scale_factor: float = 1.0) -> list[str]:
    """
    inverse the scaling of coords_raw_to_scaled (e.g. goes from 1/2 size to full size)
    """

    def _scale_fn(val):
        # need try b/c potential for val not to be a float/int
        try:
            val = str(round((float(val) * 2) * scale_factor))
        except ValueError:
            logger.error(f"could not scale val: {val}")
        return val

    return [_scale_fn(val) for val in coords]


def _iter_pattern_over_str(raw_str: str, pattern: re.Pattern, tag_type: TagType):
    """
    given a raw string, a pattern, and a tag type, iterate over the pattern and return a list of tuples of the form: (str, tag_type).
    tag_type is None if it does not belong to a tag
    """
    last_match_idx = 0
    segmented_arr = []
    for matched in pattern.finditer(raw_str):
        start, end = matched.span()

        # either do this or append each.
        # raw_str[start: end]/matched.group() can be used if i dont want parsed groups
        segs = ((raw_str[last_match_idx:start], None), (matched.groups(), tag_type))
        segmented_arr.extend(segs)

        last_match_idx = end

    if last_match_idx < len(raw_str):
        segmented_arr.append((raw_str[last_match_idx:], None))
    return segmented_arr


def _handle_str_with_pattern(
    base_str: list[tuple[str, TagType | None]],
    pattern: re.Pattern,
    tag: TagType,
) -> list[tuple[str, TagType | None]]:
    replaced_segments = []
    for seg in base_str:
        if not seg[1]:
            seg = _iter_pattern_over_str(seg[0], pattern, tag)
            replaced_segments.extend(seg)
        else:
            replaced_segments.append(seg)
    return replaced_segments


def segment_str(base_str: list[str] | str) -> list[tuple[str, TagType | None]]:
    if isinstance(base_str, str):
        base_str = [(base_str, None)]

    base_str = _handle_str_with_pattern(base_str, token_box_pattern, TagType.BOX)
    base_str = _handle_str_with_pattern(base_str, token_point_pattern, TagType.POINT)

    return base_str


# MARK: - TextTokenizerMixin
class TextTokenizerMixin:
    """methods to help with tokenization of text to ids"""

    tokenizer: callable

    def _ensure_is_id(self, tok: str | int) -> int:
        if isinstance(tok, str):
            tok = self.tokenizer.vocab[tok]
        return tok

    def _get_open_close_tokens(self, seg_type: TagType) -> tuple[str, str]:
        tokens = {
            TagType.BOX: (FuyuConstants.token_bbox_open_string, FuyuConstants.token_bbox_close_string),
            TagType.POINT: (FuyuConstants.token_point_open_string, FuyuConstants.token_point_close_string),
        }[seg_type]

        return [self.tokenizer.vocab[tokens[0]], self.tokenizer.vocab[tokens[1]]]

    def _get_open_close_text(self, seg_type: TagType) -> tuple[str, str]:
        return {
            TagType.BOX: (FuyuConstants.text_repr_bbox_open, FuyuConstants.text_repr_bbox_close),
            TagType.POINT: (FuyuConstants.text_repr_point_open, FuyuConstants.text_repr_point_close),
        }[seg_type]

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_token_id] = 0
        return attention_mask

    def _make_labels(
        self, input_ids: torch.Tensor, label_mask_image: bool = True, label_mask_nonlabel_text: bool = True, **kwargs
    ) -> torch.Tensor:
        labels = input_ids.clone()
        # labels
        return labels

    def replace_text_with_tokens(self, prompt: str, added_extra_tokens: bool = False) -> str:
        prompt = prompt.replace(FuyuConstants.text_repr_point_open, FuyuConstants.token_point_open_string)
        prompt = prompt.replace(FuyuConstants.text_repr_point_close, FuyuConstants.token_point_close_string)
        prompt = prompt.replace(FuyuConstants.text_repr_bbox_open, FuyuConstants.token_bbox_open_string)
        prompt = prompt.replace(FuyuConstants.text_repr_bbox_close, FuyuConstants.token_bbox_close_string)

        # CUSTOM
        if added_extra_tokens is False:
            prompt = prompt.replace(FuyuConstants.text_repr_action_open, FuyuConstants.token_action_open_string)
            prompt = prompt.replace(FuyuConstants.text_repr_action_close, FuyuConstants.token_action_close_string)

        return prompt

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


# class EncodeDataMixin:
#     pass


class FuyuProcessor(ProcessorMixin, TextTokenizerMixin):
    # the original FuyuProcessor has a few bugs that need to be fixed.
    # e.g. image patches indices being longer than input_ids, the box decoding not working, and the combining of the
    # need to test against https://github.com/huggingface/transformers/blob/main/tests/models/fuyu/test_processing_fuyu.py
    # interleaved should be like sample = ["here is the image", image1, "here is another image", image2]

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"
    constants = FuyuConstants

    def __init__(
        self,
        # image_processor has to be above tokenizer for ProcessorMixin class methods to work properly
        # and need the FuyuImageProcessor class to be subclassed for it to work with the llama tokenizer?
        # or something weird. I know i did this intentionally but i am tired and cannot remember exactly
        image_processor,
        tokenizer,
        label_mask_image_patches: bool = True,
        label_mask_text_ids: bool = False,
        # fields related to encoding. included in init so saved when used with save_pretrained
        enc_max_length: int = None,
        # enc kwargs related to what special tokens to add
        enc_kwargs: dict = {
            "add_bos_token": True,
            "add_boa_token": True,
            "label_add_eos_token": True,
        },
        **kwargs,
    ):
        # Use our image processor. the original was buggy at one point and impossible to get HF to merge
        if not isinstance(image_processor, FuyuImageProcessor):
            image_processor = FuyuImageProcessor()

        super().__init__(image_processor=image_processor, tokenizer=tokenizer)

        # --- THESE ARE THE SAME VALUES AS SET IN ORIGINAL FUYU PROCESSOR ---
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1

        # --- BELOW ARE MY CUSTOM VALUES ---
        self.add_bos_token = False
        self.add_boa_token = False

        self.label_mask_image_patches = label_mask_image_patches
        self.label_mask_text_ids = label_mask_text_ids

        # update the tokenizer to use bos_token and eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.update_post_processor()

        self._additional_tokens = False

        # NOTE: was using __getattr__ and @property for some of these before but using that makes saving/loading the
        # tokenizer/processor hit a recursion error if not using hf api.  avoid
        self.convert_ids_to_tokens = self.tokenizer.convert_ids_to_tokens
        self.convert_tokens_to_ids = self.tokenizer.convert_tokens_to_ids
        self.vocab = self.tokenizer.vocab

        self.enc_max_length = enc_max_length
        self.enc_kwargs = enc_kwargs

    def __call__(
        self,
        text: str = None,
        images: Image.Image = None,
        label: str = None,
        # target: str = None,
        add_special_tokens: bool = True,
        add_bos_token: bool = False,
        add_boa_token: bool = False,
        add_eos_token: bool = False,
        label_add_bos_token: bool = False,
        label_add_boa_token: bool = False,
        label_add_eos_token: bool = False,
        label_mask_from: int = None,
        label_mask_text_ids: bool = None,
        label_mask_image_patches: bool = None,
        return_attention_mask: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = None,
        max_length: int = None,
        stride: int = 0,
        pad_to_multiple_of: int = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        scale_factor: float = 1.0,
        is_interleaved: bool = False,  # TODO: implement interleaving of images+text
        extra: bool | dict = False,
        **kwargs,
    ) -> "FuyuBatchFeature":
        if text:
            text_encoding = self.preprocess_text(
                text,
                scale_factor=scale_factor,
                add_bos_token=add_bos_token,
                add_boa_token=add_boa_token,
                add_eos_token=add_eos_token,
            )

            len_base_text_encoding = text_encoding.shape[-1]

        if label:
            label_encoding = self.preprocess_text(
                label,
                scale_factor=scale_factor,
                add_bos_token=label_add_bos_token,
                add_boa_token=label_add_boa_token,
                add_eos_token=label_add_eos_token,
            )
            len_label_encoding = label_encoding.shape[-1]
            text_encoding = torch.cat([text_encoding, label_encoding], dim=0)

        if images:
            image_encoding = self.image_processor.encode_image(images, return_tensors="pt")
            len_image_patches_indices = len(image_encoding.image_patches_indices)
            batch = self._combine_encodings(
                text_encoding=text_encoding,
                image_encoding=image_encoding,
                attention_mask=return_attention_mask if return_attention_mask else None,
            )
        elif images is None:
            # if no images asumme we
            return FuyuBatchFeature(data={"input_ids": text_encoding})

        if label:
            batch["labels"] = batch["input_ids"].clone()

            if label_mask_from is None:
                # if none then means we use the kwargs/defaults
                label_mask_from = 0
            elif label_mask_from == -1:
                # should not mask at all regardless of other settings
                pass

            # use this format so that either label_mask_ can be False and it will override the self.label_mask_ value
            if label_mask_image_patches is None:
                label_mask_image_patches = self.label_mask_image_patches
            if label_mask_image_patches:
                label_mask_from += len_image_patches_indices

            if label_mask_text_ids is None:
                label_mask_text_ids = self.label_mask_text_ids
            if label_mask_text_ids:
                label_mask_from += len_base_text_encoding

            batch["labels"][..., :label_mask_from] = IGNORE_INDEX

        # unsqueeze because this is how the original fuyu processor returns values
        for key, arr in batch.items():
            if max_length and (key != "image_patches"):
                # image_patches length dim is generally -2
                # if all the inputs are single samples could just take use max length on 0th dim?
                arr = arr[..., :max_length]

            # add dim to start for stacking in batch
            batch[key] = arr[None, ...]

        batch = FuyuBatchFeature(data=batch)

        if extra:
            batch = self._extra_attach(batch, images, text, label, extra=extra)

        return batch

    def _combine_encodings(
        self, text_encoding: torch.Tensor, image_encoding: FuyuBatchFeature, attention_mask: bool = None
    ) -> FuyuBatchFeature:
        input_ids = torch.cat([image_encoding.input_ids, text_encoding], dim=0)
        image_patches_indices = torch.cat(
            [image_encoding.image_patches_indices, torch.full_like(text_encoding, -1)], dim=0
        )

        if attention_mask:
            attention_mask = self._make_attention_mask(input_ids)

        # put into dict to be passed to FuyuBatchFeature
        data = {
            "input_ids": input_ids,
            "image_patches": image_encoding.image_patches,
            "image_patches_indices": image_patches_indices,
            "attention_mask": attention_mask,
        }

        return data

    def _extra_attach(
        self, batch: FuyuBatchFeature, images=None, text=None, label=None, extra: dict = None
    ) -> FuyuBatchFeature:
        batch.extra = {
            "image": images,
            "text": text,
            "label": label,
            **(extra if isinstance(extra, dict) else {}),
        }

        return batch

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

    def add_before_after_tokens(
        self, tokens: list[int], before: str | int = None, after: str | int = None
    ) -> list[int]:
        before, after = self._ensure_is_id(before), self._ensure_is_id(after)
        return [tok for tok in chain(*[[before]] + tokens + [[after]]) if tok != None]

    def preprocess_text(
        self,
        text: str,
        scale_factor: float = 1.0,
        add_bos_token: bool = False,
        add_boa_token: bool = False,
        add_eos_token: bool = False,
    ):
        text = self.replace_text_with_tokens(text, added_extra_tokens=self._additional_tokens)

        segments = segment_str(base_str=text)
        tokenized = []
        for seg, seg_type in segments:
            if seg_type:
                seg = coords_raw_to_scaled(seg, scale_factor=scale_factor)
                tok_open, tok_close = self._get_open_close_tokens(seg_type)

                tok_ids = [[tok_open]] + [self._tokenize_num_within_tags(n) for n in seg] + [[tok_close]]
                # fastest way to flatten list of lists
                tokenized.extend(list(chain(*tok_ids)))
            else:
                tok_ids = self.tokenizer.encode(seg, add_special_tokens=False)
                tokenized.extend(tok_ids)

        if add_bos_token:
            tokenized = [self.tokenizer.vocab[FuyuConstants.bos_string]] + tokenized

        if add_boa_token:
            tokenized = tokenized + [self.tokenizer.vocab[FuyuConstants.boa_string]]

        if add_eos_token:
            tokenized = tokenized + [self.tokenizer.vocab[FuyuConstants.eos_string]]

        return torch.tensor(tokenized)

    def post_process_box_coordinates(self, outputs: torch.Tensor, target_sizes: torch.Tensor = None) -> torch.Tensor:
        def transform_raw_to_image_coords_type(tokens: list[int], tag_type: TagType, len_check: int = True):
            tok_open, tok_close = self._get_open_close_tokens(tag_type)
            tag_repr_open, tag_repr_close = self._get_open_close_text(tag_type)

            _open_ids = self.tokenizer.encode(f" {tag_repr_open}", add_special_tokens=False)[1:]
            _close_ids = self.tokenizer.encode(f" {tag_repr_close}", add_special_tokens=False)[1:]

            def _check_for_open_close(tokens) -> bool:
                # check if both open and close tokens are in tokens
                return (tok_open in tokens) and (tok_close in tokens)

            def _issue_between_open_close(s_idx, e_idx) -> bool:
                if len_check is False:
                    return False

                if 0 <= ((e_idx - s_idx) - len_check) <= 1:
                    return False

                # check if there is another open and close token after the current open and close token
                # if there isnt, then we are just going to replace between open and close regardless of len
                # if (tok_open not in toks[s_idx + 1 :]) and (tok_close not in toks[e_idx + 1 :]):
                #     return False

                logger.warn(
                    f"Warning: the length between open and close tokens for {tag_type} is not correct.\n"
                    + f"Expected {len_check } but got {e_idx - s_idx}.\n"
                    + f"From s_idx the output is: {self.tokenizer.decode(tokens[s_idx:])}"
                )

                # get the token index and token ids to replace of whichever is first
                _idx_from, _toks_ids = (e_idx, _close_ids) if (e_idx < s_idx) else (s_idx, _open_ids)
                tokens[_idx_from : _idx_from + 1] = _toks_ids
                return True

            _tries = len(tokens)

            while _check_for_open_close(tokens) and (_tries >= 0):
                s_idx, e_idx = tokens.index(tok_open), tokens.index(tok_close)

                _tries -= 1

                if not _tries:
                    breakpoint()

                if _issue_between_open_close(s_idx, e_idx):
                    continue

                coords = self.tokenizer.convert_ids_to_tokens(tokens[s_idx + 1 : e_idx])
                coords = ", ".join(coords_scaled_to_raw(coords))
                coords = f" {tag_repr_open}{coords}{tag_repr_close}"
                coord_tokens = self.tokenizer.encode(coords, add_special_tokens=False)[1:]  # drop the _ on first token
                tokens[s_idx : e_idx + 1] = coord_tokens

            return tokens

        def _transform_extra_ids(tokens: list[int]):
            """this is for the extra ids that are not bbox or point such as <action>"""
            for tok_idx, token in enumerate(tokens):
                if token in FuyuConstants.replace_extra_ids:
                    tokens = tokens[:tok_idx] + self.tokenizer.encode()

        if not isinstance(outputs, torch.Tensor):
            logger.info("why am i converting to tensor?")
            outputs = torch.tensor(outputs)

        if outputs.ndim > 1:
            if outputs.shape[0] > 1:
                raise NotImplementedError("Batched post processing not implemented yet")
            outputs = outputs[0]

        if target_sizes is None:
            # WARN: i am not actually doing any scaling at this point.
            # WARN: i should just assume all images are scaled to be max height 1080 for time being
            target_sizes = self.image_processor.target_size

        token_list = outputs.tolist()

        # len should be 1 more than expected e.g. 4 + 1
        token_list = transform_raw_to_image_coords_type(token_list, tag_type=TagType.BOX, len_check=4)
        token_list = transform_raw_to_image_coords_type(token_list, tag_type=TagType.POINT, len_check=2)

        token_list = torch.tensor(token_list, dtype=outputs.dtype, device=outputs.device)

        return token_list

    def full_decode(self, outputs: torch.Tensor, mask_image: bool = False, **kwargs):
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.from_numpy(outputs)

        if mask_image:
            outputs = self.genmask(outputs)

        outputs = self.post_process_box_coordinates(outputs)
        outputs = self.tokenizer.decode(outputs, **kwargs)
        return outputs

    def genmask(
        self,
        outputs: torch.Tensor,
        tokens_to_mask: list[str | int] = [
            FuyuConstants.image_newline_string,
            FuyuConstants.image_placeholder_string,
            IGNORE_INDEX,
        ],
    ):
        mask = torch.ones(outputs.size(), dtype=torch.bool, device=outputs.device)
        for token in tokens_to_mask:
            if isinstance(token, str):
                token = self.tokenizer.vocab[token]
            mask &= outputs != token
        return outputs[mask]

    def get_inputs_start_idx(self, inputs: dict | torch.Tensor, from_token: str = None, offset: int = 1) -> int:
        """helper function to get the start index for inputs

        assumes batch size is 1

        Args:
            inputs (dict): _description_
            from_token (str, optional): _description_. Defaults to None and will match on boa token.

        Returns:
            int: _description_
        """
        from_token = from_token or self.constants.boa_string

        # this will work for FuyuBatch feature
        inputs = getattr(inputs, "input_ids", inputs)

        # only handle 2d or 1d tensor but 1 sample regardless?
        assert inputs.ndim <= 2, "inputs should be 2d or 1d tensor"

        if inputs.ndim == 2:
            assert inputs.shape[0] == 1, "batch size should be 1"
            inputs = inputs[0]

        return (inputs == self.vocab[from_token]).nonzero().flatten().item() - offset

    def encode_sample(
        self,
        sample: dict,
        include_label: bool = True,
        include_text: bool = True,
        # these can override encode_kwargs
        add_bos_token: bool = None,
        add_boa_token: bool = None,
        label_add_eos_token: bool = None,
        mask_from: str = None,
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
            "max_length": self.enc_max_length,
            "extra": sample.get("extra", False),
        }

        # is there a speed difference if i move this outside of here?
        def _patch_kwargs(k: str, v):
            if v is not None:
                call_kwargs[k] = v

        raw_text = sample.get("text")  # text + image should always be in sample
        raw_image = sample.get("image")  # image is guaranteed to be in the sample
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

        # encode with the actual processor
        batch = self.__call__(
            text=raw_text,
            images=raw_image,
            label=raw_label,
            **call_kwargs,
        )

        return batch

    def update(self, **kwargs) -> "FuyuProcessor":
        for k, v in kwargs.items():
            # if the v is a dict then merge it into the existing dict
            if isinstance(v, dict):
                getattr(self, k).update(v)
            else:
                setattr(self, k, v)
        return self
