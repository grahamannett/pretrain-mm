from functools import lru_cache

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

# GUESSES ABOUT UNUSED TOKENS
# actually this is all probably nonsense given that 0x01 does not convert to anything
# Generated without using images which probably impacts results but regardless, can use something like
# processor.full_decode(model.generate(input_ids=processor(text="<0x01>")['input_ids'][None, :].to(model.device), max_new_tokens=100)[0])
# <0x05>
# - the generations look like:
#      `\x05 5 ?\n\n\n\n Option D: 5\n\n Option E: 0\n\n Option F: 4\n\n Option G: 3\n\n Option H: 2\n`
# which looks like ranking the options?
#
# <0x06> like
# \x06 1 / 2 \n\n Option D: 1 / 3\n\n Option E: 1 / 4\n\n Option F: 1 / 5\n\n Option G: 1 / 6\n\n Option H: 1 / 7\n\n Option I: 1 / 8\n\n Option J: 1 / 9\n\n Option K:'
#  which is fractions? idk
# <0x10> like
# \x10, a squash, a carrot, a potato, a banana, a melon, a tomato, a pear, a peach, a melon, a cucumber, a pumpkin
#


EOS_STRING = "|ENDOFTEXT|"
IMAGE_NEWLINE_STRING = "|NEWLINE|"
IMAGE_PLACEHOLDER_STRING = "|SPEAKER|"

# CUSTOM REPRS AND TOKENS

# CUSTOM REPRS
TEXT_REPR_ACTION_OPEN = "|ACTION|"
TEXT_REPR_ACTION_CLOSE = "|ENDACTION|"

# CUSTOM TOKENS
# prior to this was using <0x90>
# but the symbols dont even show up in terminal so
# seems like for custom tokens we need to use the 0x00-0x07
# range as the other ones might have actual meaning in embeddings/vocab
TOKEN_ACTION_BEGIN = "<0x06>"
TOKEN_ACTION_END = "<0x07>"


class FuyuConstants:
    # no annotation so wont be in cl.__annotations__
    text_repr_bbox_open = TEXT_REPR_BBOX_OPEN
    text_repr_bbox_close = TEXT_REPR_BBOX_CLOSE
    text_repr_point_open = TEXT_REPR_POINT_OPEN
    text_repr_point_close = TEXT_REPR_POINT_CLOSE

    token_bbox_open_string: str = TOKEN_BBOX_OPEN_STRING
    token_bbox_close_string: str = TOKEN_BBOX_CLOSE_STRING
    token_point_open_string: str = TOKEN_POINT_OPEN_STRING
    token_point_close_string: str = TOKEN_POINT_CLOSE_STRING

    boa_string: str = BEGINNING_OF_ANSWER_STRING
    bos_string: str = BEGINNING_OF_SENTENCE_STRING

    eos_string: str = EOS_STRING
    image_newline_string: str = IMAGE_NEWLINE_STRING
    image_placeholder_string: str = IMAGE_PLACEHOLDER_STRING

    # CUSTOM
    text_repr_action_open: str = TEXT_REPR_ACTION_OPEN
    text_repr_action_close: str = TEXT_REPR_ACTION_CLOSE

    token_action_open_string: str = TOKEN_ACTION_BEGIN
    token_action_close_string: str = TOKEN_ACTION_END

    @classmethod
    def get_extra_tokenizer_tokens(cls, flag: bool = False):
        if not flag:
            return None

        return [cls.text_repr_action_open, cls.text_repr_action_close]


    @classmethod
    @lru_cache
    def get_stop_tokens(cls, processor=None, additional_tokens: list[str] = []):
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained("adept/fuyu-8b", trust_remote_code=True)

        return processor.tokenizer.convert_tokens_to_ids(
            [
                cls.eos_string,
                cls.image_newline_string,
                cls.image_placeholder_string,
                *additional_tokens,
            ],
        )

    @classmethod
    def get_all_ids(cls, processor: callable, skip_ids: list = []) -> dict[str, int]:
        tokens_to_ids: dict[str, int] = {}
        for key, _ in cls.__annotations__.items():
            if key.startswith("_") or cls.__dict__[key] not in processor.vocab:
                continue

            tok_id = processor.vocab[cls.__dict__[key]]
            if tok_id in skip_ids:
                continue

            tokens_to_ids[key] = (cls.__dict__[key], tok_id)

        return tokens_to_ids
