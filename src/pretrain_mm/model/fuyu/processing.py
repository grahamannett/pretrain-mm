import re
from itertools import chain
from typing import Optional, Tuple, Union

import torch
from PIL import Image

from transformers import ProcessorMixin
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature, FuyuImageProcessor
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from pretrain_mm import logger
from pretrain_mm.constants import IGNORE_INDEX
from pretrain_mm.model.fuyu.fuyu_constants import FuyuConstants
from pretrain_mm.processor.image_processor import ImageProcessorMixin, ImageInfo, ChannelDimension
from pretrain_mm.processor.image_processor_helpers import patchify_image
from pretrain_mm.utils.token_tag_utils import TagType, token_box_pattern, token_point_pattern


def coords_raw_to_scaled(coords: list[str], scale_factor: float = 1.0) -> list[str]:
    """
    takes a list of string ints and scales them by a factor then returns a list of string (that are ints) to be tokenized
    """

    def _scale_fn(val):
        return str(round((float(val) / 2) * scale_factor))

    return [_scale_fn(val) for val in coords]


def coords_scaled_to_raw(coords: list[str], scale_factor: float = 1.0) -> list[str]:
    """
    inverse the scaling of coords_raw_to_scaled (e.g. goes )
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


class ImageProcessor(FuyuImageProcessor, ImageProcessorMixin):
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_pad = True
        self.do_normalize = True
        self.do_rescale = True
        self.do_resize = True
        self.rescale_factor: float = 1 / 255

        self.image_mean: float = 0.5
        self.image_std: float = 0.5

        self.num_channels = 3
        self.patch_size = 30

        # default size from fuyu
        # default width from original fuyu processor is 1920 but makes context length longer by a fair amount for wide images
        self.target_size = {"height": 1080, "width": 1290}  #   # {"height": 1080, "width": 1920} but 1280 seems

        # dont want these hardcoded but leaving for reference
        self._image_placeholder_id = 71011
        self._image_newline_id = 71019

    def prepare_image(
        self,
        image: list[Image.Image | torch.Tensor],
        data_format: str | ChannelDimension = ChannelDimension.FIRST,
    ) -> Tuple[torch.Tensor, tuple[int, int, int]]:
        """equivalent to preprocess on FuyuImageProcessor

        # TODO: can i do most of this in torch so that its quicker

        Args:
            image (List[Image.Image  |  torch.Tensor]): _description_
            data_format (Optional[Union[str, ChannelDimension]], optional): _description_. Defaults to ChannelDimension.FIRST.

        Returns:
            Tuple[torch.Tensor, tuple[int, int, int]]: _description_
        """
        # the base normalize/rescale/etc rely on numpy
        image_info: ImageInfo
        image, image_info = self._check_image(image, data_format=data_format)

        if self.do_resize:
            # NOTE: resize is a method on FuyuImageProcessor
            image = self._resize(image, target_size=self.target_size, image_size=image_info)

        if self.do_pad:
            target_size = {
                "height": self._calc_target_size(image_info["height"], self.patch_size),
                "width": self._calc_target_size(image_info["width"], self.patch_size),
            }

            # NOTE: pad_image is a method on FuyuImageProcessor
            image = self._pad_image(image, target_size=target_size, image_size=image_info)

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor)

        if self.do_normalize:
            # using normalize from transformers but normalize from torch seems noticeably faster
            image = self.normalize(image, mean=self.image_mean, std=self.image_std)

        # WARN: if i need to enable this remove from _check_image
        # if data_format is not None:
        #     image = to_channel_dimension_format(image, data_format)

        image = torch.from_numpy(image)[None, ...]
        return image, image_info

    def encode_image(
        self,
        image: Image.Image | torch.Tensor | str,
        patch_size: int = None,
        image_placeholder_id: int = None,
        image_newline_id: int = None,
        attach_sizes: bool = False,
        extra: dict = {},
        **kwargs,
    ) -> FuyuBatchFeature:
        """encode is what prepares (i.e. scales/resize/normalize) then transform to patches with image ids/pos ids
        the ids to be used with tokenizer

        Args:
            image (Image.Image | torch.Tensor): _description_
            return_tensors (str, optional): _description_. Defaults to "pt".

        Returns:
            torch.Tensor: _description_
        """

        if isinstance(image, str):
            image = Image.open(image)

        patch_size = patch_size or self.patch_size

        image, image_info = self.prepare_image(image)
        patch_cols = image.shape[-1] // patch_size
        patch_rows = image.shape[-2] // patch_size
        # image_patches = self.patchify_image(image).squeeze(0)
        image_patches = patchify_image(image, patch_dim_h=patch_size, patch_dim_w=patch_size, flatten=True)

        image_ids, image_pos_ids = self._make_image_tokens(
            image_placeholder_id=image_placeholder_id or self._image_placeholder_id,
            image_newline_id=image_newline_id or self._image_newline_id,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            device=image_patches.device,
        )

        if attach_sizes:
            extra = {
                **extra,
                "image_info": image_info,
                "patch_sizes": (patch_cols, patch_rows),
            }

        return FuyuBatchFeature(
            data={
                "image_patches": image_patches,
                "input_ids": image_ids,
                "image_patches_indices": image_pos_ids,
                **extra,
            }
        )


class TextTokenizerInterface:
    """methods to help with tokenization"""

    tokenizer: callable

    def __getattr__(self, name):
        # instead of defining all the methods on the tokenizer, just forward them to the tokenizer
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

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


class FuyuProcessor(ProcessorMixin, TextTokenizerInterface):
    # the original FuyuProcessor has a few bugs that need to be fixed.
    # e.g. image patches indices being longer than input_ids, the box decoding not working, and the combining of the
    # need to test against https://github.com/huggingface/transformers/blob/main/tests/models/fuyu/test_processing_fuyu.py
    # interleaved should be like sample = ["here is the image", image1, "here is another image", image2]

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"
    constants = FuyuConstants

    def __init__(self, image_processor, tokenizer):
        image_processor = ImageProcessor()  # overwrite default image processor
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1

        self.add_bos_token = False
        self.add_boa_token = False

        self.label_mask_text_ids = False
        self.label_mask_image_patches = True

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _batch_from_encodings(
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
        label_mask_from: int = 0,
        label_mask_text_ids: bool = None,
        label_mask_image_patches: bool = None,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        scale_factor: float = 1.0,
        is_interleaved: bool = False,  # TODO: implement interleaving of images+text
        _attach_extra: bool = True,
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

        if images is None:
            return FuyuBatchFeature(data={"input_ids": text_encoding})

        if images:
            image_encoding = self.image_processor.encode_image(images, return_tensors="pt")
            len_image_patches_indices = len(image_encoding.image_patches_indices)
            batch = self._batch_from_encodings(
                text_encoding=text_encoding,
                image_encoding=image_encoding,
                attention_mask=return_attention_mask if return_attention_mask else None,
            )

        if label:
            # batch['labels'] = self._make_labels(input_ids, label_mask_image=len_image_patches_indices if label_mask_)
            batch["labels"] = batch["input_ids"].clone()
            label_mask_image_patches = label_mask_image_patches or self.label_mask_image_patches
            label_mask_text_ids = label_mask_text_ids or self.label_mask_text_ids

            if label_mask_image_patches:
                label_mask_from += len_image_patches_indices
            if label_mask_text_ids:
                label_mask_from += len_base_text_encoding

            batch["labels"][..., :label_mask_from] = IGNORE_INDEX

        # unsqueeze because this is how the original fuyu processor returns values
        for key, arr in batch.items():
            if max_length and key != "image_patches":
                # image_patches length dim is generally -2
                # if all the inputs are single samples could just take use max length on 0th dim?
                arr = arr[..., :max_length]

            # add dim to start for stacking in batch
            batch[key] = arr[None, ...]

        batch = FuyuBatchFeature(data=batch)

        if _attach_extra:
            batch = self._extra_attach(batch, images, text, label)

        return batch

    def _extra_attach(self, batch: FuyuBatchFeature, images=None, text=None, label=None) -> FuyuBatchFeature:
        batch._extra = {
            "image": images,
            "text": text,
            "label": label,
        }
        return batch

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
        text = FuyuConstants.replace_text_with_tokens(text)

        segments = segment_str(base_str=text)
        tokenized = []
        for seg, seg_type in segments:
            if seg_type:
                seg = coords_raw_to_scaled(seg, scale_factor=scale_factor)
                tok_open, tok_close = self._get_open_close_tokens(seg_type)

                tokens = [[tok_open]] + [self._tokenize_num_within_tags(n) for n in seg] + [[tok_close]]
                # fastest way to flatten list of lists
                tokenized.extend(list(chain(*tokens)))
            else:
                tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                tokenized.extend(tokens)

        if add_bos_token:
            tokenized = [self.tokenizer.vocab[FuyuConstants.bos_string]] + tokenized

        if add_boa_token:
            tokenized = tokenized + [self.tokenizer.vocab[FuyuConstants.boa_string]]

        if add_eos_token:
            tokenized = tokenized + [self.tokenizer.vocab[FuyuConstants.eos_string]]

        return torch.tensor(tokenized)

    def post_process_box_coordinates(
        self, outputs: torch.Tensor, do_len_check: bool = False, target_sizes: torch.Tensor = None
    ) -> torch.Tensor:
        def transform_raw_to_image_coords_type(tokens: list[int], tag_type: TagType, len_check: int = True):
            tok_open, tok_close = self._get_open_close_tokens(tag_type)
            tag_repr_open, tag_repr_close = self._get_open_close_text(tag_type)

            def _check_for_open_close(tokens) -> bool:
                # check if both open and close tokens are in tokens
                return (tok_open in tokens) and (tok_close in tokens)

            def _check_if_issue_between_open_close(s_idx, e_idx, toks) -> bool:
                if len_check == False:
                    return False

                if s_idx + (len_check + 1) == e_idx:
                    return False

                # check if there is another open and close token after the current open and close token
                # if there isnt, then we are just going to replace between open and close regardless of len
                if (tok_open not in toks[s_idx + 1 :]) and (tok_close not in toks[e_idx + 1 :]):
                    return False

                logger.warn(
                    f"Warning: the length between open and close tokens for {tag_type} is not correct.\n"
                    + f"Expected {len_check + 1} but got {e_idx - s_idx}.\n"
                    + f"From s_idx the output is: {self.tokenizer.decode(toks[s_idx:])}"
                    # + f"Just replacing open and then continue while loop."
                )
                tokens[s_idx : s_idx + 1] = self.tokenizer.encode(f" {tag_repr_open}", add_special_tokens=False)[1:]
                return True

            while _check_for_open_close(tokens):
                s_idx, e_idx = tokens.index(tok_open), tokens.index(tok_close)

                if _check_if_issue_between_open_close(s_idx, e_idx, tokens):
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


class _FuyuBatchFeature_(FuyuBatchFeature):
    pass
