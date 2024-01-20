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
TEXT_REPR_ACTION_OPEN = "<action>"
TEXT_REPR_ACTION_CLOSE = "</action>"

# CUSTOM TOKENS
TOKEN_ACTION_BEGIN = "<0x90>"
TOKEN_ACTION_END = "<0x91>"


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

    eos_string: str = EOS_STRING
    image_newline_string: str = IMAGE_NEWLINE_STRING
    image_placeholder_string: str = IMAGE_PLACEHOLDER_STRING

    # CUSTOM
    text_repr_action_open: str = TEXT_REPR_ACTION_OPEN
    text_repr_action_close: str = TEXT_REPR_ACTION_CLOSE

    token_action_open_string: str = TOKEN_ACTION_BEGIN
    token_action_close_string = TOKEN_ACTION_END

    @classmethod
    def replace_text_with_tokens(cls, prompt: str) -> str:
        prompt = prompt.replace(cls.text_repr_point_open, cls.token_point_open_string)
        prompt = prompt.replace(cls.text_repr_point_close, cls.token_point_close_string)
        prompt = prompt.replace(cls.text_repr_bbox_open, cls.token_bbox_open_string)
        prompt = prompt.replace(cls.text_repr_bbox_close, cls.token_bbox_close_string)

        # CUSTOM
        prompt = prompt.replace(cls.text_repr_action_open, cls.token_action_open_string)
        prompt = prompt.replace(cls.text_repr_action_close, cls.token_action_close_string)

        return prompt

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
