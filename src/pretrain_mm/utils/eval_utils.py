import re

import torch
from pretrain_mm import logger

# should match ` any_text anytext <box>int, int, int, int</box>`
box_pattern = re.compile(r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*</box>")


def _detokenize_helper_fuyu(tokens: torch.Tensor, processor: callable) -> str:
    post_processed_bbox_tokens = processor.post_process_box_coordinates(tokens)[0]
    decoded_tokens = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
    return decoded_tokens


def bbox_metric(target_pos, sequence, decode_func: callable = None, tokenizer: callable = None) -> float:
    """
    this is a metric that can be used if i am training with the bbox task.  model should output the sequence in <box>int, int, int, int<box>
    """
    try:

        target_strs = box_pattern.match(target_pos)
        target_values = map(int, target_strs.groups())

        decoded_str = decode_func(sequence, tokenizer)
        if matched := box_pattern.search(decoded_str):
            preds = map(int, matched.groups())
            # matched = torch.tensor([int(m) for m in matched])
            max_value = max(torch.max(target_values, preds))

            # normalize
            metric_value = torch.nn.functional.l1_loss(target_values, preds) / max_value
            return metric_value.item()

    except Exception as err:
        logger.warn(f"Error for {target_pos}, computing bbox metric: {err}")
        return 1.0
        # return sum((x - y) ** 2 for x, y in zip(target_pos, matched))

    # torch.nn.functional.l1_loss(torch.tensor([100, 200, 300], dtype=float), torch.tensor([200, 400, 500])) / max(torch.max(targets, preds))
    # return None
