from dataclasses import field
from functools import lru_cache, wraps

from transformers import PreTrainedTokenizer


class ConstantsMeta(type):
    """metaclass for constants related to tokenizer.

    can use like

    class Processor(ProcessorMixin, TextTokenizerMixin, metaclass=ConstantsMeta, tconstants=FuyuConstants):
        pass

    Args:
        type (_type_): _description_
    """

    def __new__(cls, name: str, bases: tuple[type], dct: dict, tconstants=None, **kwargs):
        def _wrapped_init(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)
            self.tokenizer_const = tconstants
            tconstants.set_tokenizer(self.tokenizer)

        dct["__init__"] = _wrapped_init

        return super().__new__(cls, name, bases, dct)


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

    _tokenizer: PreTrainedTokenizer = field(default=None, repr=False, init=False)

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

    @classmethod
    def set_tokenizer(cls, tokenizer: PreTrainedTokenizer):
        cls._tokenizer = tokenizer

    @classmethod
    def get_id(cls, token: str):
        return cls._tokenizer.vocab[token]


def SetConstants(constants: TokenizerConstants):
    def decorator(cls):
        @wraps(cls, updated=())
        class WrappedProcessor(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenizer_const = constants

            @property
            def constants(self):
                return self.tokenizer_const

        return WrappedProcessor

    return decorator
