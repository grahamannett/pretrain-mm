from functools import lru_cache

from transformers import PreTrainedTokenizer


class TokenizerConstants:
    """tokenizer constants are used to define special tokens and methods for tokenizers.
    helpful to define in one area as abstraction across models/tokenizers

    Returns:
        _type_: _description_
    """

    # special tokens
    str_boa: str
    str_bos: str
    str_eos: str

    # multimodal/image related
    str_image_placeholder: str
    str_image_newline: str

    @classmethod
    @lru_cache
    def _tokenizer(cls):
        raise NotImplementedError("Must implement base_tokenizer method")

    @classmethod
    def get_stop_ids(cls, tokenizer: PreTrainedTokenizer, extra_tokens: list[str] = []) -> list[int]:
        return tokenizer.convert_tokens_to_ids(
            [
                cls.eos_string,
                cls.image_newline_string,
                cls.image_placeholder_string,
                *extra_tokens,
            ],
        )

    @classmethod
    def get_stop_tokens(cls) -> list[str]:
        return [
            cls.eos_string,
            cls.image_newline_string,
            cls.image_placeholder_string,
        ]
