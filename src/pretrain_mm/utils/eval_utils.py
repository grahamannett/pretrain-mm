import re

import torch
from pretrain_mm import logger

# should match ` any_text anytext <box>int, int, int, int</box>`
box_pattern = re.compile(r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*</box>")


def _detokenize_helper_fuyu(tokens: torch.Tensor, processor: callable) -> str:
    post_processed_bbox_tokens = processor.post_process_box_coordinates(tokens)[0]
    decoded_tokens = processor.decode(post_processed_bbox_tokens, skip_special_tokens=True)
    return decoded_tokens


def bbox_metric_from_str(target_str: str, pred_str: str, _print_err_cutoff: int = 30) -> float:
    # y1, x1, y2, x2 = map(int, box_pattern.search(target_str).groups())
    # y1_pred, x1_pred, y2_pred, x2_pred = map(int, box_pattern.search(generated_str).groups())
    # target = [x1, x2, y1, y2]
    # pred = [x1_pred, x2_pred, y1_pred, y2_pred]

    try:
        target = torch.tensor(list(map(int, box_pattern.search(target_str).groups())), dtype=float)
        pred = torch.tensor(list(map(int, box_pattern.search(pred_str).groups())), dtype=float)
        return bbox_metric(target, pred)
    except Exception as err:
        logger.warn(
            f"Error Metric: {err} with target_str: {target_str[-_print_err_cutoff:]} and pred_str: {pred_str[-_print_err_cutoff]}"
        )
        return 1.0


def bbox_metric(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    this is a metric that can be used if i am training with the bbox task.  model should output the sequence in <box>int, int, int, int<box>
    """
    max_value = max(*target, *pred)
    return (torch.nn.functional.l1_loss(torch.tensor(target, dtype=float), torch.tensor(pred)) / max_value).item()
