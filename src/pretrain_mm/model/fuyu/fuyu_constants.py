from pretrain_mm.constants import IGNORE_INDEX

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
    def get_all_ids(cls, processor: callable) -> dict[str, int]:
        tokens_to_ids: dict[str, int] = {}
        for key, _ in cls.__annotations__.items():
            if key.startswith("_") or cls.__dict__[key] not in processor.vocab:
                continue
            tokens_to_ids[key] = (cls.__dict__[key], processor.vocab[cls.__dict__[key]])

        return tokens_to_ids
