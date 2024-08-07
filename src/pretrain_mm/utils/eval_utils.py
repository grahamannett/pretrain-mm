import math
import re
import statistics
from collections import defaultdict
from itertools import chain
from typing import Callable
from enum import StrEnum, auto

import torch
from PIL import Image, ImageDraw

from pretrain_mm import logger
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import TagType, box_pattern, tag_patterns


# should match ` any_text anytext <box>int, int, int, int</box>` and `<point>int, int</point>`
EVAL_BY_COMPLETION_GENERATE_KWARGS = {
    # "max_new_tokens": 5,
    # "return_last_logits": True,
}


class OCREvalCompletion(StrEnum):
    bounding_box: str = auto()  # provide text and have to generate bounding box
    text_extraction: str = auto()  # provide bounding box and have to extract text


class EvalInfo_:
    def __init__(self):
        pass

    def store(self, key, val):
        pass


def remove_label(batch, to_idx):
    batch["attention_mask"] = batch["attention_mask"][..., :to_idx]
    batch["input_ids"], removed_input_ids = batch["input_ids"][..., :to_idx], batch["input_ids"][..., to_idx:]
    if isinstance(batch, dict):
        removed_labels = batch.pop("labels")
    else:
        removed_labels = batch["labels"]
        batch["labels"] = None

    if hasattr(batch, "image_patches_indices"):
        batch["image_patches_indices"] = batch["image_patches_indices"][..., :to_idx]

    return batch, (removed_input_ids, removed_labels)


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


def box_distance_fn(pred: str | torch.Tensor, target: str | list[int], decode_func: callable = None) -> float | None:
    # make sure the output is a string, if its not we need a decode_func (like processor.decode/processor.full_decode)
    if not isinstance(pred, str):
        pred = decode_func(pred)

    if isinstance(target, str):
        target = _pattern_to_vals(text=target, pattern=box_pattern)

    if box_vals := _pattern_to_vals(pred, pattern=box_pattern):
        metric = math.dist(target, box_vals)
        return metric

    return None


def default_gen_gen_output(gen_output, data_holder, **kwargs):
    for k, v in gen_output.items():
        data_holder[k].append(v)


def _get_start_idx(sample=None, gen_kwargs: dict = None):
    return -gen_kwargs["max_new_tokens"]


def _calc_mean(data, key, val=0.0):
    try:
        val = statistics.mean(data[key])
    except TypeError:
        logger.error(f"TypeError calculating mean for {key}")
    except statistics.StatisticsError as err:
        logger.error(f"StatisticsError calculating mean for {key}\n{err}")
    return val


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
    get_decode_start_idx_fn: callable = None,
    metric_fn: Callable[[str, str], float] = box_distance_fn,
    # forward_kwargs: dict = {},  # these are passed to model.forward
    generate_kwargs: dict = dict(  # these are used for generation
        max_new_tokens=10,
        return_extra=True,
        use_past_key_values=False,
        forward_kwargs=dict(),
        # return_last_logits=True,
    ),
):
    if samples and num_samples:
        logger.warning_once("Passed in both samples and num_samples/task_func/encode_data_func.  Using samples.")
        assert (num_samples == len(samples)) or (num_samples in [None, 0]), "num_samples should match len(samples)"

    # default values
    # check how we are doing samples
    # gen_start_idx = get_decode_start_idx_fn(sample_enc) or partial(_get_start_idx, gen_kwargs=generate_kwargs)
    num_samples = len(samples) if samples else num_samples

    def _get_encoded_sample(n: int, _tries: int = 10):
        if samples:
            return samples[n]

        _task_samp = _raw_samp = False

        while not all([_task_samp, _raw_samp]):
            _tries -= 1
            _task_samp, _raw_samp, _i = dataset.get_with_transform(task_func, return_extra=True)
            logger.check_or_fail(_tries > 0, "Failed to get a sample", log_locals=True)

        _enc_samp = encode_data_func(
            _task_samp,
            add_bos_token=False,
            add_boa_token=False,
            label_add_eos_token=False,
            include_label=False,
        )

        return {
            "raw": _raw_samp,
            "task": _task_samp,
            "encoded": _enc_samp,
        }

    def _got_dict_gen(gen_output):
        # gen_output = {"input_ids": gen_output["input_ids"], "logits": gen_output["logits"].float()}
        # return gen_output, gen_output["input_ids"]
        return {"input_ids": gen_output["input_ids"], "logits": gen_output["logits"].float()}

    def _got_tensor_gen(gen_output):
        return {"input_ids": gen_output}

    metric_key = f"{prepend_str_extra}metrics"
    decoded_key = f"{prepend_str_extra}decoded"
    target_key = f"{prepend_str_extra}target"
    # keys with single vals
    # reason to take both averages is sometimes
    sample_avg_metric_key = f"{prepend_str}sample_metric_avg"  # average per sample
    avg_metric_key = f"{prepend_str}metric_avg"  # average of all
    errors_key = f"{prepend_str}errors"

    evals = []

    for sample_idx in range(num_samples):
        sample = _get_encoded_sample(sample_idx)

        sample_eval = sample_eval_by_completion(
            model=model,
            processor=processor,
            sample=sample,
            num_generations=num_generations,
            prepend_str=prepend_str,
            prepend_str_extra=prepend_str_extra,
            get_decode_start_idx_fn=get_decode_start_idx_fn,
            named_key=None,
            keep_decoded_output=True,
            keep_target=True,
            metric_fn=metric_fn,
            generate_kwargs=generate_kwargs,
        )

        evals.append(sample_eval)

    # i dont like the way this is all being done ATM

    combined_data = {
        metric_key: list(chain(*(s[metric_key] for s in evals))),
        decoded_key: list(chain(*(s[decoded_key] for s in evals))),
        target_key: list(chain((s[target_key] for s in evals))),
        # this is
        sample_avg_metric_key: list(s[avg_metric_key] for s in evals if s[avg_metric_key]),
        errors_key: sum(s[errors_key] for s in evals),
    }

    combined_data[sample_avg_metric_key] = _calc_mean(combined_data, sample_avg_metric_key, val=0.0)
    combined_data[avg_metric_key] = _calc_mean(combined_data, metric_key, val=0.0)

    return combined_data


def _get_sample_key(sample, named_key=None) -> tuple[str, str]:
    _get_k = lambda k: filter(lambda v: k in v, sample)

    if named_key:
        # named_key for if the sample has multiple tasks on it
        return f"task.{named_key}", f"encoded.{named_key}"

    # unpack to get 1st key. dont use 'task.' or 'encoded.' as they might not have labels
    task_k, enc_k = next(_get_k("task")), next(_get_k("encoded"))
    return task_k, enc_k


def _process_gen_output():
    pass


def sample_eval_by_completion(
    model,
    processor,
    sample,
    num_generations: int = 1,
    prepend_str: str = "eval/",
    prepend_str_extra: str = "extra/",
    get_decode_start_idx_fn: callable = None,
    named_key: str = None,
    keep_decoded_output: bool = True,  # might not be helpful and should just decode from caller
    keep_target: bool = False,  # sometimes easier to keep label in output rather than in sample
    metric_fn: Callable[[str, str], float] = box_distance_fn,
    # forward_kwargs: dict = {},  # these are passed to model.forward
    generate_kwargs: dict = dict(  # these are used for generation
        max_new_tokens=10,
        return_extra=True,  # return extra e.g. hidden states
        use_past_key_values=False,  # for cache, might speed up
        forward_kwargs=dict(),
        # return_last_logits=True,
    ),
    **kwargs,
):
    task_k, enc_k = _get_sample_key(sample, named_key=named_key)

    sample_enc = sample[enc_k].to(model.device)
    sample_label = sample[task_k]["label"]

    gen_start_idx = get_decode_start_idx_fn(sample_enc) or _get_start_idx(gen_kwargs=generate_kwargs)

    metric_key = f"{prepend_str_extra}metrics"  # all the values
    decoded_key = f"{prepend_str_extra}decoded"
    target_key = f"{prepend_str_extra}target"
    # keys with single vals
    avg_metric_key = f"{prepend_str}metric_avg"  # sample average
    errors_key = f"{prepend_str}errors"  # is errors helpful even?

    gen_vals = defaultdict(list)
    gen_info = {
        metric_key: [],
        errors_key: 0,
    }

    _decoded = []
    _target = []

    for gen_idx in range(num_generations):
        gen_output = generate_helper(
            model,
            model_inputs=sample_enc,
            **generate_kwargs,
        )

        # save the data from latest generation
        for k, v in gen_output.items():
            gen_vals[k].append(v)

        # if "hidden_states" in gen_output:
        if (hidden_states := gen_output.get("hidden_states")) is not None:
            if hidden_states.shape[0] != 1:
                raise ValueError("Hidden states should have batch size of 1")

            # first dim 1 means batch size is 1
            hidden_states = hidden_states[0]

            gen_vals["hs_emb"].append(hidden_states[0])
            gen_vals["hs_last"].append(hidden_states[-1])

        decoded_gen = processor.full_decode(gen_output["input_ids"][..., gen_start_idx:])
        _decoded.append(decoded_gen)

        # should metric fn be after generations?
        if metric_fn:
            if (metric_val := metric_fn(_decoded[-1], sample_label)) is None:
                gen_info[errors_key] += 1
                continue

            gen_info[metric_key].append(metric_val)

    # compute the average of all the metrics
    if metric_fn:
        gen_info[avg_metric_key] = statistics.mean(gen_info[metric_key]) if gen_info[metric_key] else None

    if keep_decoded_output:
        gen_info[decoded_key] = _decoded

    if keep_target:
        # should i keep 1 target for each generation?
        gen_info[target_key] = sample_label

    return {**gen_info, **gen_vals}


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
        logger.warn(f"Eval Error for: {target_str} with {pred_str}")
        raise TypeError(f"target_str: `{target_str}` pred_str: `{pred_str}`\n\t{err}")
    return _score


# MOVED FROM pretrain-task.py
def eval_with_generate(
    model,
    eval_dataset,
    task_processor,
    max_new_tokens: int = 150,
    num_choices: int = 5,
    pattern_str: str = "box",
    temperature: float = 1.0,
    stop_tokens: list[int] = [],
    drop_last_of_input: bool = False,
    include_loss: bool = True,
) -> float:
    """ """
    logger.info("DOING EVAL WITH GENERATE")
    processor = task_processor.processor

    choices = list(range(0, len(eval_dataset)))
    random.shuffle(choices)
    choices = choices[:num_choices]

    acc_metric, loss_metric = [], []
    # Format is like '\x04' + '__' + '\n' + '\x00' => boa + space + newline + box_open
    after_boa = 4

    model.eval()
    for sample_id in choices:
        sample = eval_dataset[sample_id]

        input_for_loss = task_train_dataset.call_transforms(sample).to(model.device)

        boa_idx = input_for_loss.input_ids[0] == processor.vocab[processor.constants.boa_token]

        # include the boa token
        boa_idx = boa_idx.nonzero().view(-1)[0].item() + after_boa

        bos_idx = input_for_loss.input_ids[0] == processor.vocab[processor.constants.bos_token]
        bos_idx = bos_idx.nonzero().view(-1)[0].item()

        input_for_gen = {
            "input_ids": input_for_loss.input_ids[:, :boa_idx],
            "image_patches": input_for_loss.image_patches,
            "image_patches_indices": input_for_loss.image_patches_indices[:, :boa_idx],
            "attention_mask": input_for_loss.attention_mask[:, :boa_idx],
        }

        with torch.no_grad():
            loss = model(**input_for_loss).loss
            loss_metric.append(loss.item())

            gen_output = generate_helper(
                model,
                model_inputs=input_for_gen,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=temperature,
                drop_last_of_input=drop_last_of_input,
            )

        decoded_output = processor.full_decode(gen_output[0, bos_idx:])
        label_decoded = processor.full_decode(input_for_loss.input_ids[0, bos_idx:])

        logger.info(f"\nOutput generated: {decoded_output}")

        acc_val = 1.0
        try:
            acc_val = loc_metric_from_str(
                target_str=label_decoded,
                pred_str=decoded_output,
                pattern_str=pattern_str,
            )
        except TypeError:
            logger.warn("Generate string incompatible")
        except ValueError as err:
            logger.warn(f"ValueError for eval_with_generate: {err}")

        acc_metric.append(acc_val)

    return {"eval/acc_metric": sum(acc_metric) / len(acc_metric), "eval/loss": sum(loss_metric)}
