from pretrain_mm.processor.tokenizer_constants import TokenizerConstants


_REPR_BBOX_OPEN_TEXT = "<box>"
_REPR_BBOX_CLOSE_TEXT = "</box>"
_REPR_POINT_OPEN_TEXT = "<point>"
_REPR_POINT_CLOSE_TEXT = "</point>"


_TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
_TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
_TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
_TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
_BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>
_BEGINNING_OF_SENTENCE_STRING = "<s>"  # <bos>

# GUESSES ABOUT UNUSED TOKENS
# this could probably be nonsense given that 0x01 does not convert to anything
# Generated without using images which probably impacts results but regardless, can use something like
#   processor.full_decode(
#       model.generate(
#           input_ids=processor(text="<0x01>")['input_ids'][None, :].to(model.device), max_new_tokens=100
#       )[0]
#   )
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


_EOS_TOKEN = "|ENDOFTEXT|"
_IMAGE_NEWLINE_TOKEN = "|NEWLINE|"
_IMAGE_PLACEHOLDER_STRING = "|SPEAKER|"

# CUSTOM REPRS AND TOKENS

# CUSTOM REPRS
_REPR_ACTION_OPEN_TEXT = "|ACTION|"
_REPR_ACTION_CLOSE_TEXT = "|ENDACTION|"

# CUSTOM TOKENS
# prior to this was using <0x90>
# but the symbols dont even show up in terminal so
# seems like for custom tokens we need to use the 0x00-0x07
# range as the other ones might have actual meaning in embeddings/vocab
_ACTION_OPEN_TOKEN = "<0x06>"
_ACTION_CLOSE_TOKEN = "<0x07>"


# NOTE: TokenizerConstants subclass are frozen by default
class FuyuConstantsClass(TokenizerConstants):
    boa_token: str = _BEGINNING_OF_ANSWER_STRING
    bos_token: str = _BEGINNING_OF_SENTENCE_STRING
    eos_token: str = _EOS_TOKEN

    bbox_open_string: str = _TOKEN_BBOX_OPEN_STRING
    bbox_close_string: str = _TOKEN_BBOX_CLOSE_STRING
    point_open_string: str = _TOKEN_POINT_OPEN_STRING
    point_close_string: str = _TOKEN_POINT_CLOSE_STRING

    image_newline_token: str = _IMAGE_NEWLINE_TOKEN
    image_placeholder_token: str = _IMAGE_PLACEHOLDER_STRING

    repr_bbox_open_text: str = _REPR_BBOX_OPEN_TEXT
    repr_bbox_close_text: str = _REPR_BBOX_CLOSE_TEXT
    repr_point_open_text: str = _REPR_POINT_OPEN_TEXT
    repr_point_close_text: str = _REPR_POINT_CLOSE_TEXT

    # CUSTOM
    repr_action_open_text: str = _REPR_ACTION_OPEN_TEXT
    repr_action_close_text: str = _REPR_ACTION_CLOSE_TEXT

    action_open_token: str = _ACTION_OPEN_TOKEN
    action_close_token: str = _ACTION_CLOSE_TOKEN


FuyuConstants = FuyuConstantsClass()

if __name__ == "__main__":
    fuyuconst = FuyuConstantsClass()

    breakpoint()
