import math
import re
import statistics
from collections import defaultdict

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
    if samples and num_samples:
        logger.warning_once("Passed in both samples and num_samples/task_func/encode_data_func.  Using samples.")
        assert (num_samples == len(samples)) or (num_samples in [None, 0]), "num_samples should match len(samples)"

    # default values
    get_decode_start_idx = get_decode_start_idx or _default_get_start_idx
    num_samples = len(samples) if samples else num_samples

    gen_info = {"samples": {}, "metrics": []}

    # check how we are doing samples
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

    def _base_gen_info_dict():
        return {"attempts": num_generations, "success": 0, "errors": 0, "generations": {}}

    # flag until i refactor this to be more coherent
    _fix_gen_output = False

    for sample_idx in range(num_samples):
        sample = _get_encoded_sample(sample_idx)
        sample_enc, sample_task = sample["encoded"].to(model.device), sample["task"]
        gen_start_idx = get_decode_start_idx(sample_enc)

        gen_info["samples"][sample_idx] = _base_gen_info_dict()

        for gen_idx in range(num_generations):
            gen_output = generate_helper(
                model,
                model_inputs=sample_enc,
                **generate_kwargs,
            )

            if isinstance(gen_output, dict):
                decode_ids = gen_output["input_ids"]
                _fix_gen_output = True
            elif isinstance(gen_output, torch.Tensor):
                decode_ids = gen_output
            else:
                breakpoint()

            decode_ids = gen_output["input_ids"]
            decoded_output = processor.full_decode(decode_ids[..., gen_start_idx:])
            dist_metric = box_distance_fn(decoded_output, sample_task["label"])

            if dist_metric is None:
                gen_info["samples"][sample_idx]["errors"] += 1
                continue

            gen_info["metrics"].append(dist_metric)

            if _fix_gen_output:
                gen_output = {
                    # should these be cast to numpy?
                    "input_ids": gen_output["input_ids"],
                    # might want these as float16 to save space
                    "logits": gen_output["logits"].float(),
                }

            gen_info["samples"][sample_idx]["generations"][gen_idx] = gen_output

    gen_info[f"{prepend_str}distance"] = statistics.mean(gen_info["metrics"]) if gen_info["metrics"] else None
    gen_info[f"{prepend_str}errors"] = sum([v["errors"] for v in gen_info["samples"].values()])

    return gen_info


def default_gen_gen_output(gen_output, data_holder, **kwargs):
    for k, v in gen_output.items():
        data_holder[k].append(v)


def sample_eval_by_completion(
    model,
    processor,
    sample,
    num_generations: int = 1,
    # prepend_str: str = "eval/",
    # prepend_str_extra: str = "extra/",
    get_decode_start_idx: callable = None,
    get_gen_output_values: callable = default_gen_gen_output,
    # forward_kwargs: dict = {},  # these are passed to model.forward
    generate_kwargs: dict = dict(  # these are used for generation
        max_new_tokens=10,
        return_extra=True,
        use_past_key_values=False,
        forward_kwargs=dict(),
        # return_last_logits=True,
    ),
    **kwargs,
):
    sample_enc, sample_label = sample["encoded"].to(model.device), sample["task"]["label"]

    gen_start_idx = get_decode_start_idx(sample_enc)

    success, errors = 0, 0

    distances = []
    # gen_vals = {"logits": [], "input_ids": [], "hs_emb": [], "hs_last": []}
    gen_vals = defaultdict(list)
    # gen_logits, gen_input_ids, metrics = [], [], []

    for gen_idx in range(num_generations):
        gen_output = generate_helper(
            model,
            model_inputs=sample_enc,
            **generate_kwargs,
        )

        # save the data from latest generation
        gen_vals["logits"].append(gen_output["logits"])
        gen_vals["input_ids"].append(gen_output["input_ids"])

        # if "hidden_states" in gen_output:
        if (hidden_states := gen_output.get("hidden_states")) is not None:
            if hidden_states.shape[0] == 1:
                hidden_states = hidden_states[0]

            gen_vals["hs_emb"].append(hidden_states[0])
            gen_vals["hs_last"].append(hidden_states[-1])

        # calculate distance
        decoded_output = processor.full_decode(gen_output["input_ids"][..., gen_start_idx:])
        if (dist_metric := box_distance_fn(decoded_output, sample_label)) is None:
            errors += 1
            continue

        distances.append(dist_metric)

    # not sure if this will work as i think gen_output can be different sizes depending
    # for k, v in gen_vals.items():
    #     gen_vals[k] = torch.stack(v).squeeze()

    return {
        "success": success,
        "errors": errors,
        "dist": distances,
        **gen_vals,
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
        logger.warn(f"Eval Error for: {target_str} with {pred_str}")
        raise TypeError(f"target_str: `{target_str}` pred_str: `{pred_str}`\n\t{err}")
    return _score


def eval_compare_cfid(inputs, y_model, x_model, constrain_dist, **kwargs):

    # inputs processed?
    given = inputs["given"]
    full = inputs["full"]
    y_logits = y_model(**full).logits
    x_logits = x_model(**full).logits


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

        boa_idx = input_for_loss.input_ids[0] == processor.vocab[FuyuConstants.boa_string]

        # include the boa token
        boa_idx = boa_idx.nonzero().view(-1)[0].item() + after_boa

        bos_idx = input_for_loss.input_ids[0] == processor.vocab[FuyuConstants.bos_string]
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
        except TypeError as err:
            logger.warn(f"Generate string incompatible")
        except ValueError as err:
            logger.warn(f"ValueError for eval_with_generate: {err}")

        acc_metric.append(acc_val)

    return {"eval/acc_metric": sum(acc_metric) / len(acc_metric), "eval/loss": sum(loss_metric)}
