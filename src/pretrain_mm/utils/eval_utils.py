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


def box_distance_fn(output: torch.Tensor | str, label: list[int] | str, decode_func: callable = None):
    if isinstance(label, str):
        # WARN: if there is an issue with this we should debug it now
        label = list(map(int, box_pattern.search(label).groups()))

    # decoded_output = decode_func(outputs)
    if not isinstance(output, str):
        output = decode_func(output)

    try:
        if box_match := box_pattern.search(output):
            box_vals = list(map(int, box_match.groups()))
            metric = math.dist(label, box_vals)
            return metric
    except:
        breakpoint()

    return False


def eval_by_completion(
    model,
    processor,
    dataset: callable = None,
    task_func: callable = None,
    encode_data_func: callable = None,
    num_samples: int = None,
    samples: list = None,
    return_extra: bool = False,
    prepend_str: str = "eval/",
    prepend_str_extra: str = "extra/",
    # forward_kwargs: dict = {},  # these are passed to model.forward
    generate_kwargs: dict = dict(  # these are used for generation
        max_new_tokens=10,
        return_extra=True,
        forward_kwargs=dict(),
        # return_last_logits=True,
    ),
):

    if samples and num_samples:
        logger.warning_once("Passed in both samples and num_samples/task_func/encode_data_func.  Using samples.")
        assert (num_samples == len(samples)) or (num_samples in [None, 0]), "num_samples should match len(samples)"

    n_samples = num_samples or len(samples)

    def _get_encoded_sample(n: int):
        if samples:
            return samples[n]

        while True:
            task_sample, raw_sample, _ = dataset.get_with_transform(task_func, return_extra=True)
            if raw_sample not in [None, False]:
                break

        encoded_sample = encode_data_func(
            task_sample,
            add_bos_token=False,
            add_boa_token=False,
            label_add_eos_token=False,
            include_label=False,
        )
        return {"raw": raw_sample, "task": task_sample, "encoded": encoded_sample}

    metrics, all_outputs = [], []
    num_errs = 0

    for n in range(n_samples):
        sample = _get_encoded_sample(n)
        sample_enc, sample_task = sample["encoded"], sample["task"]

        gen_output = generate_helper(
            model,
            model_inputs=sample_enc.to(model.device),
            **generate_kwargs,
        )

        all_outputs.append(gen_output)

        if isinstance(gen_output, dict):
            gen_output = gen_output["input_ids"]

        decoded_output = processor.full_decode(gen_output[..., -generate_kwargs["max_new_tokens"] :])

        if metric := box_distance_fn(output=decoded_output, label=sample_task["label"]):
            metrics.append(metric)
        else:
            num_errs += 1

    return {
        f"{prepend_str}dist_metric": (sum(metrics) / len(metrics)) if metrics else -100,
        f"{prepend_str}errs": num_errs,
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
