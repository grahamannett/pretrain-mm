import re
from functools import lru_cache
from itertools import chain

import torch
from PIL import Image

# from transformers import ProcessorMixin
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from pretrain_mm import logger
from pretrain_mm.constants import IGNORE_INDEX
from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstants, FuyuConstantsClass
from pretrain_mm.model.fuyu.fuyu_image_processor import FuyuImageProcessor, TFuyuImageProcessor
from pretrain_mm.processor.processor import ProcessorMixin, TextProcessorMixin
from pretrain_mm.processor.tokenizer_constants import SetConstants
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


# @lru_cache
def _get_open_close_text(
    seg_type: TagType,
    constants: FuyuConstantsClass,
) -> tuple[str, str]:
    return {
        TagType.BOX: (constants.repr_bbox_open_text, constants.repr_bbox_close_text),
        TagType.POINT: (constants.repr_point_open_text, constants.repr_point_close_text),
    }[seg_type]


# @lru_cache
def _get_open_close_tokens(
    seg_type: TagType,
    constants: FuyuConstantsClass,
    tokenizer: callable,
) -> tuple[str, str]:
    tokens = {
        TagType.BOX: (constants.bbox_open_string, constants.bbox_close_string),
        TagType.POINT: (constants.point_open_string, constants.point_close_string),
    }[seg_type]

    return [tokenizer.vocab[tokens[0]], tokenizer.vocab[tokens[1]]]


class FuyuTextProcessorMixin(TextProcessorMixin):
    constants: FuyuConstantsClass

    def replace_text_with_tokens(self, raw_text: str, added_extra_tokens: bool = False) -> str:
        raw_text = raw_text.replace(self.constants.repr_point_open_text, self.constants.point_open_string)
        raw_text = raw_text.replace(self.constants.repr_point_close_text, self.constants.point_close_string)
        raw_text = raw_text.replace(self.constants.repr_bbox_open_text, self.constants.bbox_open_string)
        raw_text = raw_text.replace(self.constants.repr_bbox_close_text, self.constants.bbox_close_string)

        # CUSTOM
        if added_extra_tokens is False:
            raw_text = raw_text.replace(self.constants.repr_action_open_text, self.constants.action_open_token)
            raw_text = raw_text.replace(self.constants.repr_action_close_text, self.constants.action_close_token)

        return raw_text


@SetConstants(FuyuConstants)
class FuyuProcessor(ProcessorMixin, TextProcessorMixin):
    # the original FuyuProcessor has a few bugs that need to be fixed.
    # e.g. image patches indices being longer than input_ids, the box decoding not working, and the combining of the
    # need to test against https://github.com/huggingface/transformers/blob/main/tests/models/fuyu/test_processing_fuyu.py
    # interleaved should be like sample = ["here is the image", image1, "here is another image", image2]

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        # image_processor has to be above tokenizer for ProcessorMixin class methods to work properly
        # and need the FuyuImageProcessor class to be subclassed for it to work with the llama tokenizer?
        # or something weird. I know i did this intentionally but i am tired and cannot remember exactly
        image_processor: TFuyuImageProcessor,
        tokenizer,
        label_mask_image_patches: bool = True,
        label_mask_text_ids: bool = False,
        max_length: int = None,
        **kwargs,
    ):
        if not isinstance(image_processor, FuyuImageProcessor):
            # Use our image processor. the original was buggy at one point and impossible to get HF to merge
            image_processor = FuyuImageProcessor(**image_processor.to_dict())

        super().__init__(image_processor=image_processor, tokenizer=tokenizer)

        # --- THESE ARE THE SAME VALUES AS SET IN ORIGINAL FUYU PROCESSOR ---
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
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

        self.max_length = max_length

        if "enc_kwargs" in kwargs:
            self.enc_kwargs = kwargs["enc_kwargs"]

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
            # len_label_encoding = label_encoding.shape[-1]
            text_encoding = torch.cat([text_encoding, label_encoding], dim=0)

        if images:
            image_encoding = self.image_processor.encode_image(
                images,
                return_tensors="pt",
                image_placeholder_id=self.tokenizer.vocab[self.constants.image_placeholder_token],
                image_newline_id=self.tokenizer.vocab[self.constants.image_newline_token],
                **kwargs,
            )
            len_image_patches_indices = len(image_encoding.image_patches_indices)
            batch = self._combine_encodings(
                text_encoding=text_encoding,
                image_encoding=image_encoding,
                attention_mask=return_attention_mask if return_attention_mask else None,
            )
        elif images is None:
            len_image_patches_indices = 0
            batch = {"input_ids": text_encoding}

        if label:
            batch["labels"] = batch["input_ids"].clone()

            if label_mask_from is None:
                # if none then means we use the kwargs/defaults
                label_mask_from = 0
            elif label_mask_from == -1:
                # should not mask at all regardless of other settings
                raise NotImplementedError("Need to implement when label_mask_from == -1")

            # use this format so that either label_mask_ can be False and it will override the self.label_mask_ value
            if label_mask_image_patches is None:
                label_mask_image_patches = self.label_mask_image_patches
            if label_mask_image_patches:
                label_mask_from += len_image_patches_indices
                # label_mask_from += batch['image_patches_indices']

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
            # you want to include raw if you need the raw text label for evaluation without having to decode
            batch = self._extra_attach(batch, extra=extra, images=images, text=text, label=label, include_raw=True)

        return batch

    def _combine_encodings(
        self, text_encoding: torch.Tensor, image_encoding: FuyuBatchFeature, attention_mask: bool = None
    ) -> FuyuBatchFeature:
        """
        Combines the text encoding and image encoding to create a new FuyuBatchFeature.

        Args:
            text_encoding (torch.Tensor): The text encoding tensor.
            image_encoding (FuyuBatchFeature): The image encoding FuyuBatchFeature.
            attention_mask (bool, optional): The attention mask. Defaults to None.

        Returns:
            FuyuBatchFeature: The combined FuyuBatchFeature.

        """
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
        self,
        batch: FuyuBatchFeature,
        extra: dict = None,
        images=None,
        text=None,
        label=None,
        include_raw: bool = True,
    ) -> FuyuBatchFeature:
        """
        Attaches extra information to the given batch.

        Args:
            batch (FuyuBatchFeature): The batch to attach the extra information to.
            images (optional): The images to attach.
            text (optional): The text to attach.
            label (optional): The label to attach.
            extra (optional): Additional extra information to attach.

        Returns:
            FuyuBatchFeature: The batch with the attached extra information.
        """
        batch.extra = {**(extra if isinstance(extra, dict) else {})}

        if include_raw:
            batch.extra.update(image=images, text=text, label=label)

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
                tok_open, tok_close = _get_open_close_tokens(seg_type, self.constants, self.tokenizer)

                tok_ids = [[tok_open]] + [self._tokenize_num_within_tags(n) for n in seg] + [[tok_close]]
                # fastest way to flatten list of lists
                tokenized.extend(list(chain(*tok_ids)))
            else:
                tok_ids = self.tokenizer.encode(seg, add_special_tokens=False)
                tokenized.extend(tok_ids)

        if add_bos_token:
            tokenized = [self.tokenizer.vocab[self.constants.bos_token]] + tokenized

        if add_boa_token:
            tokenized = tokenized + [self.tokenizer.vocab[self.constants.boa_token]]

        if add_eos_token:
            tokenized = tokenized + [self.tokenizer.vocab[self.constants.eos_token]]

        return torch.tensor(tokenized)

    def post_process_box_coordinates(self, outputs: torch.Tensor, target_sizes: torch.Tensor = None) -> torch.Tensor:
        def transform_raw_to_image_coords_type(tokens: list[int], tag_type: TagType, len_check: int = True):
            tok_open, tok_close = _get_open_close_tokens(tag_type, self.constants, self.tokenizer)
            tag_repr_open, tag_repr_close = _get_open_close_text(tag_type, self.constants)

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
                    raise ValueError(f"Could not find open and close tokens for {tag_type}")

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
