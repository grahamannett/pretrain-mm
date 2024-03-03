import re

import math
import torch
from PIL import Image, ImageDraw

from pretrain_mm import logger
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.utils.token_tag_utils import TagType, tag_patterns
from pretrain_mm.utils.token_tag_utils import box_pattern
from pretrain_mm.utils.generate_utils import generate_helper
import statistics

# should match ` any_text anytext <box>int, int, int, int</box>` and `<point>int, int</point>`

EVAL_BY_COMPLETION_GENERATE_KWARGS = {
    # "max_new_tokens": 5,
    # "return_last_logits": True,
}


def _pattern_to_vals(text: str, pattern: re.Pattern = box_pattern, map_to_type: type = int) -> list[int] | None:
    """convert a string to a list of values based on a box/point pattern

    Args:
        text (str): _description_
        pattern (re.Pattern, optional): _description_. Defaults to box_pattern.
        map_to_type (type, optional): _description_. Defaults to int.

    Returns:
        list[int] | None: list of ints/floats or None
    """
    if matched := pattern.search(text):
        return list(map(map_to_type, matched.groups()))
    return None


def box_distance_fn(output: str | torch.Tensor, label: list[int] | str, decode_func: callable = None) -> float | None:
    if isinstance(label, str):
        label = _pattern_to_vals(text=label, pattern=box_pattern)

    # make sure the output is a string, if its not we need a decode_func (like processor.decode/processor.full_decode)
    if not isinstance(output, str):
        output = decode_func(output)

    # WARNING: might need try/catch here
    if box_vals := _pattern_to_vals(output, pattern=box_pattern):
        metric = math.dist(label, box_vals)
        return metric

    return None


def eval_by_completion(
    model,
    processor,
    dataset: callable = None,
    task_func: callable = None,
    encode_data_func: callable = None,
    num_samples: int = None,
    num_generations: int = 1,
    samples: list = None,
    return_extra: bool = False,
    prepend_str: str = "eval/",
    prepend_str_extra: str = "extra/",
    get_decode_start_idx: callable = None,
    # forward_kwargs: dict = {},  # these are passed to model.forward
    generate_kwargs: dict = dict(  # these are used for generation
        max_new_tokens=10,
        return_extra=True,
        use_past_key_values=False,
        forward_kwargs=dict(),
        # return_last_logits=True,
    ),
):

    # default values
    get_decode_start_idx = get_decode_start_idx or _default_get_start_idx
    n_samples = num_samples or len(samples)
    metrics, all_outputs, per_gen_info = [], [], {}

    # check how we are doing samples
    if samples and num_samples:
        logger.warning_once("Passed in both samples and num_samples/task_func/encode_data_func.  Using samples.")
        assert (num_samples == len(samples)) or (num_samples in [None, 0]), "num_samples should match len(samples)"

    def _default_get_start_idx(*args, **kwargs):
        return -generate_kwargs["max_new_tokens"]

    def _get_encoded_sample(n: int):
        if samples:
            return samples[n]

        while True:
            task_sample, raw_sample, _ = dataset.get_with_transform(task_func, return_extra=True)
            if raw_sample not in [None, False]:
                break

        return {
            "raw": raw_sample,
            "task": task_sample,
            "encoded": encode_data_func(
                task_sample,
                add_bos_token=False,
                add_boa_token=False,
                label_add_eos_token=False,
                include_label=False,
            ),
        }

    for n in range(n_samples):
        sample = _get_encoded_sample(n)
        sample_enc, sample_task = sample["encoded"], sample["task"]
        gen_start_idx = get_decode_start_idx(sample_enc)

        per_gen_info[n] = {
            "attempts": num_generations,
            "success": 0,
            "errors": 0,
        }

        for g_idx in range(num_generations):
            gen_output = generate_helper(
                model,
                model_inputs=sample_enc.to(model.device),
                **generate_kwargs,
            )

            # all_outputs is the general catch all for outputs
            all_outputs.append(gen_output)

            if isinstance(gen_output, dict):
                gen_output = gen_output["input_ids"]

            decoded_output = processor.full_decode(gen_output[..., gen_start_idx:])

            dist_metric = box_distance_fn(decoded_output, sample_task["label"])

            if dist_metric is None:
                per_gen_info[n]["errors"] += 1
                continue

            metrics.append(dist_metric)

    return {
        f"{prepend_str}dist_metric": (sum(metrics) / len(metrics)) if metrics else -100,
        f"{prepend_str}errs": sum([v["errors"] for v in per_gen_info.values()]),
        # use extra on metrics to ignore them in logging
        **(
            {
                f"{prepend_str_extra}distances": metrics,
                f"{prepend_str_extra}outputs": all_outputs,
            }
            if return_extra
            else {}
        ),
    }


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
