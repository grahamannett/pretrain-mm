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

    @classmethod
    def get_stop_tokens(cls, processor=None, additional_tokens: list[str] = []):
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained("adept/fuyu-8b", trust_remote_code=True)

        return [
            processor.tokenizer.encode(token, add_special_tokens=False)[0]
            for token in [
                cls.image_placeholder_string,  # self.processor.tokenizer.vocab["|SPEAKER|"],
                cls.image_newline_string,  # self.processor.tokenizer.vocab["|NEWLINE|"],
                cls.eos_string,
                *additional_tokens,
            ]
        ]
