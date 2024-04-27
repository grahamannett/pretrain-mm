from dataclasses import field
from functools import lru_cache

from transformers import PreTrainedTokenizer


class TokenizerConstants:
    """tokenizer constants are used to define special tokens and methods for tokenizers.
    helpful to define in one area as abstraction across models/tokenizers


    Naming schema is like the following:

    fields that end with
        - `_token` are the classic/traditional special tokens related to bos/eos/etc

        - `_string` are tokens that are not typical in a tokenizer or specific to models, e.g.

        - `_text` are text representations that should be swapped out to typically a singluar string/token by explicitly
            replacing either via str.replace or similar mechanims.  if they are not and passed to tokenizer it will
            result in the tokenizer splitting it into multiple tokens.
            the reason for this makes sense typically due to not wanting to add a token to the tokenizer as then it would
            require the model to be resized.

    fields that start with `_` are ignored fields but used for methods


    Using this schema (e.g. `{token_name}_token` or `{token_name}_text`) is mostly because it seems ideal for the
    ability to use methods based on the field name endswith().  Debated doing it the opposite way and using
    `token_{token_name}` but seems like due to IDE/naming conventions makes more sense to do it this way.

    Returns:
        _type_: _description_
    """

    # special tokens
    boa_token: str
    bos_token: str
    eos_token: str

    # multimodal/image related
    image_placeholder_token: str
    image_newline_token: str

    _tokenizer: callable = field(default=None, repr=False)

    @classmethod
    @lru_cache
    def _tokenizer(cls):
        raise NotImplementedError("Must implement base_tokenizer method")

    @classmethod
    def get_stop_ids(cls, tokenizer: PreTrainedTokenizer, extra_tokens: list[str] = []) -> list[int]:
        return tokenizer.convert_tokens_to_ids(
            [
                cls.eos_token,
                cls.image_newline_token,
                cls.image_placeholder_token,
                *extra_tokens,
            ],
        )

    @classmethod
    def get_stop_tokens(cls) -> list[str]:
        return [
            cls.eos_token,
            cls.image_newline_token,
            cls.image_placeholder_token,
        ]
