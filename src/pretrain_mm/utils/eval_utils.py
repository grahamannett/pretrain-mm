import re

import torch

# should match ` any_text anytext <box>int, int, int, int</box>`
box_pattern = re.compile(r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*</box>")


def _detokenize_helper_fuyu(tokens: torch.Tensor, processor: callable) -> str:
    post_processed_bbox_tokens = processor.post_process_box_coordinates(tokens)[0]
    decoded_tokens = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
    return decoded_tokens


def mse_bbox(target_pos, sequence, decode_func: callable = None, tokenizer: callable = None):
    """
    this is a metric that can be used if i am training with the bbox task.  model should output the sequence in <box>int, int, int, int<box>
    """

    decoded_str = decode_func(sequence, tokenizer)
    if matched := box_pattern.search(decoded_str):
        matched = matched.groups()
        matched = [int(m) for m in matched]
        return sum((x - y) ** 2 for x, y in zip(target_pos, matched))
    return None
