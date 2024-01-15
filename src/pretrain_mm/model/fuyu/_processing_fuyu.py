"""
wip

"""
import re
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import pad, to_channel_dimension_format
from transformers.image_utils import ChannelDimension, get_image_size
from transformers.models.fuyu.image_processing_fuyu import FuyuBatchFeature
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor as BaseFuyuImageProcessor
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from pretrain_mm.utils.token_tag_utils import TagType, token_box_pattern, token_point_pattern

TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>
BEGINNING_OF_SENTENCE_STRING = "<s>"  # <bos>


class FuyuConstants:
    text_repr_bbox_open = TEXT_REPR_BBOX_OPEN
    text_repr_bbox_close = TEXT_REPR_BBOX_CLOSE
    text_repr_point_open = TEXT_REPR_POINT_OPEN
    text_repr_point_close = TEXT_REPR_POINT_CLOSE

    token_bbox_open_string = TOKEN_BBOX_OPEN_STRING
    token_bbox_close_string = TOKEN_BBOX_CLOSE_STRING
    token_point_open_string = TOKEN_POINT_OPEN_STRING
    token_point_close_string = TOKEN_POINT_CLOSE_STRING

    boa_string: str = BEGINNING_OF_ANSWER_STRING
    bos_string: str = BEGINNING_OF_SENTENCE_STRING

    eos_string: str = "|ENDOFTEXT|"
    image_newline_string: str = "|NEWLINE|"
    image_placeholder_string: str = "|SPEAKER|"

    @classmethod
    def replace_text_with_tokens(cls, prompt: str) -> str:
        prompt = prompt.replace(cls.text_repr_point_open, cls.token_point_open_string)
        prompt = prompt.replace(cls.text_repr_point_close, cls.token_point_close_string)
        prompt = prompt.replace(cls.text_repr_bbox_open, cls.token_bbox_open_string)
        prompt = prompt.replace(cls.text_repr_bbox_close, cls.token_bbox_close_string)
        return prompt


def scale_coords_by_factor(
    values: list[str],
    scale_factor: float = 1.0,
    scale_fn: callable = lambda val, scale: round((val / 2) * scale),
) -> list[str]:
    """
    takes a list of string ints and scales them by a factor then returns a list of string ints to be tokenized
    """
    return [str(scale_fn(int(val), scale=scale_factor)) for val in values]


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


def _tokenize_num_within_tags(num_str: str, tokenizer) -> List[int]:
    """helper func for _transform_within_tags in the case where we have a number that is not a bbox or point"""
    if num_str in tokenizer.vocab:
        return [tokenizer.vocab[num_str]]
    return tokenizer.encode(num_str, add_special_tokens=False)[1:]


class FuyuImageProcessor(BaseFuyuImageProcessor):
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
        self.do_resize = False
        self.rescale_factor: float = 1 / 255

        self.image_mean: float = 0.5
        self.image_std: float = 0.5

        self.num_channels = 3
        self.patch_size = 30

        # default size from fuyu
        self.target_size = {"height": 1080, "width": 1920}

        # dont want these hardcoded but leaving for reference
        self._image_placeholder_id = 71011
        self._image_newline_id = 71019

    def _setup_image_tokens(
        self,
        tokenizer: callable,
        image_placeholder_string: str = FuyuConstants.image_placeholder_string,
        image_newline_string: str = FuyuConstants.image_newline_string,
    ) -> None:
        self._image_placeholder_id = tokenizer(image_placeholder_string, add_special_tokens=False)["input_ids"][0]
        self._image_newline_id = tokenizer(image_newline_string, add_special_tokens=False)["input_ids"][0]

    def _make_image_size_dict(self, image_size: tuple[int, int, int]) -> dict[str, int]:
        return {
            "height": image_size[0],
            "width": image_size[1],
            "channels": image_size[2],
        }

    def _calc_target_size(self, val: int, patch_size: int) -> int:
        if val % patch_size:
            return (val + patch_size) - (val % patch_size)
        return val

    def resize(self, image, size):
        raise NotImplementedError("resize not implemented")

    def pad_image(
        self,
        image: np.ndarray,
        # image_size: tuple[int, int, int]
        size: Dict[str, int],
        original_image_size: Dict[str, int] = None,
        mode: str = "constant",
        constant_values: float = 1.0,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # image_height, image_width = get_image_size(image, input_data_format)
        image_height, image_width = original_image_size["height"], original_image_size["width"]
        target_height, target_width = size["height"], size["width"]
        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width

        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def prepare_image(
        self,
        image: List[Image.Image | torch.Tensor],
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
    ) -> Tuple[torch.Tensor, tuple[int, int, int]]:
        """equivalent to preprocess on FuyuImageProcessor

        Args:
            image (List[Image.Image  |  torch.Tensor]): _description_
            data_format (Optional[Union[str, ChannelDimension]], optional): _description_. Defaults to ChannelDimension.FIRST.

        Returns:
            Tuple[torch.Tensor, tuple[int, int, int]]: _description_
        """
        # the base normalize/rescale/etc rely on numpy
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if isinstance(image, Image.Image):
            image = np.array(image)

        original_image_size = self._make_image_size_dict(image.shape)

        if self.do_resize:
            image = self.resize(image, self.image_size)

        if self.do_pad:
            target_size = {
                "height": self._calc_target_size(original_image_size["height"], self.patch_size),
                "width": self._calc_target_size(original_image_size["width"], self.patch_size),
            }

            image = self.pad_image(image, target_size, original_image_size)

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor)

        if self.do_normalize:
            image = self.normalize(image, mean=self.image_mean, std=self.image_std)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format)

        image = torch.from_numpy(image).unsqueeze(0)
        return image, original_image_size

    def patchify(self, image, patch_height: int = None, patch_width: int = None, flatten: bool = True):
        patch_height = patch_height or self.patch_size
        patch_width = patch_width or self.patch_size

        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)

        # there are cases where we want to flatten the patches but for processing its helpful to not flatten yet
        if flatten:
            patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)

        return patches

    def make_image_tokens(
        self,
        image_placeholder_id: int,
        image_newline_id: int,
        patch_rows: int,
        patch_cols: int,
        # allow image patches to be passed or just device
        image_patches: torch.Tensor = None,
        device: str = None,
        non_index_token: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or image_patches.device

        # placeholder token for each patch
        image_ids = torch.full(
            (patch_rows, patch_cols),
            image_placeholder_id,
            dtype=torch.int32,
            device=image_patches.device,
        )

        image_ids = torch.cat(
            [
                # image ids is placeholder that is used to signify the image patch before combination of text and image
                image_ids,
                #
                # for each patch row signify a newline, e.g. newline ids at the end of each row
                torch.full((patch_rows, 1), image_newline_id, dtype=torch.int32, device=image_patches.device),
            ],
            dim=1,
        )

        # position corresponds to the patch index in the flattened patches
        # e.g. for image with patches such that patches are 3x2: [[1, 2, 3], [4,5,6]]
        image_pos_ids = torch.cat(
            [
                # indexes that correspond to the patch index in the flattened patches
                torch.arange(0, patch_cols * patch_rows).view(patch_rows, patch_cols),
                # non index token at the end of each row that corresponds to the newline token
                torch.full((patch_rows, 1), non_index_token, dtype=torch.int32, device=image_patches.device),
            ],
            dim=1,
        )

        return image_ids.view(-1), image_pos_ids.view(-1)

    def preprocess(
        self,
        image: Image.Image | torch.Tensor,
        patch_size: int = None,
        image_placeholder_id: int = None,
        image_newline_id: int = None,
        **kwargs,
    ) -> FuyuBatchFeature:
        """preprocess is what converts image to patches and gives us ids to be used with tokenizer

        Args:
            image (Image.Image | torch.Tensor): _description_
            return_tensors (str, optional): _description_. Defaults to "pt".

        Returns:
            torch.Tensor: _description_
        """
        patch_size = patch_size or self.patch_size

        image, _ = self.prepare_image(image)
        patch_cols = image.shape[-1] // patch_size
        patch_rows = image.shape[-2] // patch_size
        image_patches = self.patchify(image).squeeze(0)

        image_ids, image_pos_ids = self.make_image_tokens(
            image_placeholder_id=image_placeholder_id or self._image_placeholder_id,
            image_newline_id=image_newline_id or self._image_newline_id,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            image_patches=image_patches,
        )

        return FuyuBatchFeature(
            data={"image_patches": image_patches, "input_ids": image_ids, "image_patches_indices": image_pos_ids}
        )


class FuyuProcessor(ProcessorMixin):
    # the original FuyuProcessor is not good
    # need to test against https://github.com/huggingface/transformers/blob/main/tests/models/fuyu/test_processing_fuyu.py
    # interleaved should be like sample = ["here is the image", image1, "here is another image", image2]

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"
    constants = FuyuConstants

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1

        self.add_bos_token = False
        self.add_boa_token = False

    def __call__(
        self,
        text: str = None,
        images: "PIL.Image.Image" = None,
        # target: str = None,
        add_special_tokens: bool = True,
        add_bos_token: bool = False,
        add_boa_token: bool = False,
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
        **kwargs,
    ) -> "FuyuBatchFeature":
        if text:
            batch = self.preprocess_text(text, scale_factor, add_bos_token, add_boa_token)

        if images:
            image_encoding = self.image_processor.preprocess(images, return_tensors="pt")
            batch = self._combine_modalities(
                text_encoding=batch,
                image_encoding=image_encoding,
                attention_mask=return_attention_mask if return_attention_mask else None,
            )

        return batch

    def _get_open_close_tokens(self, seg_type: TagType) -> tuple[str, str]:
        tokens = {
            TagType.BOX: (FuyuConstants.token_bbox_open_string, FuyuConstants.token_bbox_close_string),
            TagType.POINT: (FuyuConstants.token_point_open_string, FuyuConstants.token_point_close_string),
        }[seg_type]

        return [self.tokenizer.vocab[tokens[0]], self.tokenizer.vocab[tokens[1]]]

    def _tokenize_num_within_tags(self, num_str: str) -> List[int]:
        """helper func for _transform_within_tags in the case where we have a number that is not a bbox or point"""
        if num_str in self.tokenizer.vocab:
            return [self.tokenizer.vocab[num_str]]
        return self.tokenizer.encode(num_str, add_special_tokens=False)[1:]

    def _combine_modalities(
        self, text_encoding: torch.Tensor, image_encoding: FuyuBatchFeature, attention_mask: bool = None
    ) -> FuyuBatchFeature:
        input_ids = torch.cat([image_encoding.input_ids, text_encoding], dim=0)
        image_patches_indices = torch.cat(
            [image_encoding.image_patches_indices, torch.full_like(text_encoding, -1)], dim=0
        )

        if attention_mask:
            attention_mask = self._make_attention_mask(input_ids)

        return FuyuBatchFeature(
            data={
                "input_ids": input_ids,
                "image_patches": image_encoding.image_patches,
                "image_patches_indices": image_patches_indices,
                "attention_mask": attention_mask,
            }
        )

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_token_id] = 0
        return attention_mask

    def preprocess_text(
        self, text: str, scale_factor: float = 1.0, add_bos_token: bool = False, add_boa_token: bool = False
    ):
        text = FuyuConstants.replace_text_with_tokens(text)

        segments = segment_str(base_str=text)
        tokenized = []
        for seg, seg_type in segments:
            if seg_type:
                seg = scale_coords_by_factor(seg, scale_factor=scale_factor)
                tok_open, tok_close = self._get_open_close_tokens(seg_type)
                tokens = [[tok_open]] + [self._tokenize_num_within_tags(n) for n in seg] + [[tok_close]]
                # fastest way to flatten list of lists
                tokens = list(chain(*tokens))
                tokenized.extend(tokens)

            else:
                tokens = self.tokenizer.encode(seg, add_special_tokens=False)
                tokenized.extend(tokens)
        if add_bos_token:
            tokenized = [self.tokenizer.vocab[FuyuConstants.bos_string]] + tokenized

        if add_boa_token:
            tokenized = tokenized + [self.tokenizer.vocab[FuyuConstants.boa_string]]

        return torch.tensor(tokenized)

    # def post_process_box_coordinates(self, outputs: torch.Tensor, target_sizes: torch.Tensor = None):

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """

        def scale_factor_to_fit(original_size, target_size=None):
            height, width = original_size
            if target_size is None:
                max_height = self.image_processor.size["height"]
                max_width = self.image_processor.size["width"]
            else:
                max_height, max_width = target_size
            if width <= max_width and height <= max_height:
                return 1.0
            return min(max_height / height, max_width / width)

        def find_delimiters_pair(tokens, start_token, end_token):
            start_id = self.tokenizer.convert_tokens_to_ids(start_token)
            end_id = self.tokenizer.convert_tokens_to_ids(end_token)

            starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
            ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]

            if torch.any(starting_positions) and torch.any(ending_positions):
                return (starting_positions[0], ending_positions[0])
            return (None, None)

        def tokens_to_boxes(tokens, original_size):
            num_tries = 3
            while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 5:
                    if num_tries == 0:
                        break
                    num_tries -= 1
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        def tokens_to_points(tokens, original_size):
            num_tries = 3
            while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 3:
                    if num_tries == 0:
                        break
                    num_tries -= 1
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                x, y = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        if target_sizes is None:
            target_sizes = ((self.image_processor.size["height"], self.image_processor.size["width"]),) * len(outputs)
        # elif target_sizes.shape[1] != 2:
        elif target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if len(outputs) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as output sequences")

        results = []

        for seq, size in zip(outputs, target_sizes):
            seq = tokens_to_boxes(seq, size)
            seq = tokens_to_points(seq, size)
            results.append(seq)

        return results

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
