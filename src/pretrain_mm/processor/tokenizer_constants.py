from dataclasses import dataclass
from functools import cache, wraps

from transformers import PreTrainedTokenizer

from pretrain_mm.utils.class_utils import SingletonMixin


class TokenizerConstants(SingletonMixin):
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

    # special tokens - should be part of tokenizer vocab
    bos_token: str
    eos_token: str
    image_placeholder_token: str

    # equivalent to these tokens are not on paligemma so im not sure how to handle them yet
    boa_token: str = None
    image_newline_token: str = None

    # using dict for refs as TokenizerConstants should be frozen by default but the tokenizer is instantiated after
    _refs: dict = {}

    def __init_subclass__(cls, frozen: bool = True, *args, **kwargs) -> None:
        dataclass(cls, frozen=frozen, **kwargs)
        return super().__init_subclass__(*args, **kwargs)

    # Class/static methods should not require a _tokenizer or any other instance variables
    @classmethod
    @cache
    def get_stop_tokens(cls) -> list[str]:
        return [tok for tok in [cls.eos_token, cls.image_newline_token, cls.image_placeholder_token] if tok]

    @classmethod
    def _get_tokenizer(cls, tokenizer: PreTrainedTokenizer = None, obj: type = None):
        if not tokenizer:
            if not obj:
                if cls_tokenizer := cls._refs.get("tokenizer"):
                    raise ValueError("Either pass tokenizer or have it set (prefer instance, but can use class)")
                return cls_tokenizer
            return obj.tokenizer
        return tokenizer

    @classmethod
    def bind_tokenizer(cls, obj: PreTrainedTokenizer, consts: "TokenizerConstants" = None):
        consts = consts or cls

        # give the tokenizer/processor a ref to constants
        obj.constants = consts
        consts._refs["tokenizer"] = obj

    @property
    def tokenizer(self):
        return self._refs.get("tokenizer", None)

    @cache
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

    @cache
    def get_stop_ids(self, tokenizer: PreTrainedTokenizer = None, extra_tokens: list[str] = []) -> list[int]:
        tokenizer = TokenizerConstants._get_tokenizer(tokenizer, self)

        return tokenizer.convert_tokens_to_ids(self.get_stop_tokens() + extra_tokens)


class ConstantsMeta(type):
    """metaclass for constants related to tokenizer.

    Initially this seemed like an interesting design pattern but I think its much clearer to use the decorator below

    can use like
        > class Processor(ProcessorMixin, TextTokenizerMixin, metaclass=ConstantsMeta, consts=FuyuConstants): ...

    feel like a nicer way to do this would be:
        > class Processor(SetConstants(TokenizerConstants), ProcessorMixin, TextTokenizerMixin): ...

    or the decorator as done below in SetConstants

    Args:
        name (str): name of the class
        bases (tuple[type]): base classes
        dct (dict): dictionary of class attributes
        consts (TokenizerConstants, optional): tokenizer constants to bind. Defaults to None.
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
            TokenizerConstants.bind_tokenizer(obj=self, consts=consts)

        dct["__init__"] = __init__
        return super().__new__(cls, name, bases, dct, **kwargs)


def SetConstants(consts: TokenizerConstants):
    def decorator(cls):
        @wraps(cls, updated=())
        class WrappedProcessor(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                TokenizerConstants.bind_tokenizer(obj=self, consts=consts)

        return WrappedProcessor

    return decorator
