import re

import torch
from PIL import Image, ImageDraw

from pretrain_mm import logger
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.utils.image_utils import draw_helper
from pretrain_mm.utils.token_tag_utils import TagType, tag_patterns

# should match ` any_text anytext <box>int, int, int, int</box>` and `<point>int, int</point>`


def calculate_metric(target: torch.Tensor, pred: torch.Tensor) -> float:
    """
    this is a metric that can be used if i am training with the bbox task.  model should output the sequence in <box>int, int, int, int<box>
    """
    max_value = max(*target, *pred)
    return (torch.nn.functional.l1_loss(target, pred) / max_value).item()


def box_from_str(text: str, pattern: re.Pattern = tag_patterns[TagType.BOX]) -> list[int] | None:
    if matched := pattern.search(text):
        return list(map(int, matched.groups()))
    return None


def loc_metric_from_str(
    target_str: str,
    pred_str: str,
    pattern_str: str = TagType.BOX,
    _image: Image.Image = None,
    _image_save_path: str = None,
    _score: float = 1.0,
    _print_cutoff: int = 30,
    _tokenizer: callable = None,
) -> float:
    # compute loss based on box.  0 is perfect 1 means not even bbox.
    pattern_to_match: re.Pattern = tag_patterns[pattern_str]

    try:
        target = torch.tensor(box_from_str(target_str, pattern_to_match), dtype=float)
        pred = torch.tensor(box_from_str(pred_str, pattern_to_match), dtype=float)
        # target = torch.tensor(list(map(int, pattern_to_match.search(target_str).groups())), dtype=float)
        # pred = torch.tensor(list(map(int, pattern_to_match.search(pred_str).groups())), dtype=float)
        _score = calculate_metric(target, pred)

        if _image:
            draw = ImageDraw.Draw(_image)
            draw.rectangle(target, outline="green")
            draw.rectangle(pred, outline="red")

            _image.save(_image_save_path)

    except Exception as err:
        # clean up strings befroe output
        # _p_str = pred_str[-_print_cutoff:].replace("\n", "")
        # _t_str = target_str[-_print_cutoff:].replace("\n", "")
        # logger.warn(f"Eval Error\n\ttarget_str:\n{_t_str}\n\tpred_str:\n{_p_str}")
        raise TypeError(f"target_str: {target_str}\npred_str: {pred_str}\n{err}")
    return _score


def eval_compare_cfid(inputs, y_model, x_model, constrain_dist, **kwargs):

    # inputs processed?
    given = inputs["given"]
    full = inputs["full"]
    y_logits = y_model(**full).logits
    x_logits = x_model(**full).logits
