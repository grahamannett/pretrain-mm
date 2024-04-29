from dataclasses import dataclass, make_dataclass
from functools import lru_cache, wraps

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

    _refs: dict = {}
    # _tokenizer: PreTrainedTokenizer = None

    def __init_subclass__(cls, frozen: bool = True, *args, **kwargs) -> None:
        dataclass(cls, frozen=frozen, **kwargs)
        return super().__init_subclass__(*args, **kwargs)

    @classmethod
    def dc(cls, subclass: type = None):
        """
        usage like (or can put on __new__):
        ```
        @TokenizerConstants.dc
        class SubConstants:
            boa_token: str = "<s>
        ```

        the reason to use it like such is can control/change the tokens in the subclass more uniformly, but it
        ideally should just be subclassed as that works better for IDE's

        """
        return make_dataclass(subclass.__name__, subclass.__annotations__.items(), bases=(subclass, cls))

    # Class/static methods should not require a _tokenizer or any other instance variables
    @classmethod
    def get_stop_tokens(cls) -> list[str]:
        return [
            cls.eos_token,
            cls.image_newline_token,
            cls.image_placeholder_token,
        ]

    @classmethod
    def _get_tokenizer(cls, tokenizer: PreTrainedTokenizer = None, obj: type = None):
        if not tokenizer:
            if not obj:
                if cls_tokenizer := cls._refs.get("tokenizer"):
                    raise ValueError("Either pass tokenizer or have it set (prefer instance, but can use class)")
                return cls_tokenizer
            return obj.tokenizer
        return tokenizer

    @lru_cache
    def get_all_ids(self, tokenizer: PreTrainedTokenizer = None, skip_ids: list = []) -> dict[str, int]:
        tokenizer = TokenizerConstants._get_tokenizer(tokenizer, self)

        tokens_to_ids: dict[str, int] = {}

        for key, key_type in self.__annotations__.items():
            if key_type is not str:
                continue

            key_attr = getattr(self, key)
            if key.startswith("_") or key_attr not in tokenizer.vocab:
                continue

            tok_id = tokenizer.vocab[key_attr]

            if tok_id in skip_ids:
                continue

            tokens_to_ids[key] = tok_id

        return tokens_to_ids

    @lru_cache
    def get_stop_ids(self, tokenizer: PreTrainedTokenizer = None, extra_tokens: list[str] = []) -> list[int]:
        tokenizer = TokenizerConstants._get_tokenizer(tokenizer, self)

        return tokenizer.convert_tokens_to_ids(self.get_stop_tokens() + extra_tokens)

    @classmethod
    def set_with(cls, obj: PreTrainedTokenizer, consts: "TokenizerConstants" = None):
        consts = consts or cls

        # give the tokenizer/processor a ref to constants
        obj.constants = consts
        # give the constants a ref to the tokenizer
        # consts._tokenizer = obj
        consts._refs["tokenizer"] = obj

    @property
    def tokenizer(self):
        return self._refs.get("tokenizer", None)


class ConstantsMeta(type):
    """metaclass for constants related to tokenizer.

    Initially this seemed like an interesting design pattern but I think its much clearer to use the decorator below

    can use like

    class Processor(ProcessorMixin, TextTokenizerMixin, metaclass=ConstantsMeta, tconstants=FuyuConstants):
        pass

    Args:
        type (_type_): _description_
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type],
        dct: dict,
        consts: TokenizerConstants = None,
        **kwargs,
    ):
        def __init__(self, *args, **kwargs):
            super(self.__class__, self).__init__(*args, **kwargs)
            TokenizerConstants.set_with(obj=self, consts=consts)

        dct["__init__"] = __init__

        return super().__new__(cls, name, bases, dct, **kwargs)


def SetConstants(consts: TokenizerConstants):
    def decorator(cls):
        @wraps(cls, updated=())
        class WrappedProcessor(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                TokenizerConstants.set_with(obj=self, consts=consts)

        return WrappedProcessor

    return decorator
