import random

from pretrain_mm import logger
from pretrain_mm.datasets.mind2web.mind2web import M2WAction
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate


def _make_point_str(x1, y1, x2=None, y2=None) -> str:
    x, y = x1, y1

    if x2 and y2:
        x, y = round((x + x2) / 2), round((y1 + y2) / 2)

    return f"<point>{x}, {y}</point>"


def _make_box_str(x1, y1, x2, y2) -> str:
    # FUYU NEEDS IN format: y1, x1, y2, x2 but bounding box comes in form x0, y0, x1, y1,
    return f"<box>{y1}, {x1}, {y2}, {x2}</box>"


_make_next_loc_funcs = {
    "point": _make_point_str,
    "box": _make_box_str,
}


class Mind2WebTaskProcessor:
    @staticmethod
    def preprocessor(sample: dict):
        """
        this is a task preprocessor for the Mind2Web dataset such that it works for the processor meaning it is only image + text
        the output from this MUST be ingestible by the processor
        """
        return {
            "text": sample["text"] + sample["label"],
            "images": sample["image"],
        }

    @staticmethod
    def postprocessor(sample):
        """
        helper function that reshapes the sample that comes from processor as processor gives us a batched sample but
        data collator expects a list of samples
        """
        sample["input_ids"] = sample["input_ids"].squeeze(0)
        sample["attention_mask"] = sample["attention_mask"].squeeze(0)
        sample["image_patches"] = [img.squeeze(0) for img in sample["image_patches"]]
        sample["image_patches_indices"] = sample["image_patches_indices"].squeeze(0)
        return sample


def _alt_format(previous_actions_text):
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"
    return text


def task_mind2web(sample: M2WAction, next_action_loc_type: str = "point") -> dict:
    """
    given a sample from Mind2Web return a dict for the task adapter

    this task is close to clippy targeted format.
    E.g.
    [website-screenshot]
    [text] [next-action]


    Usage can be like

    ```
    task_func = functools.partial(
        task_mind2web, next_action_loc_type=config.loc_type
    )
    ```

    Alternate formats maybe: `text = f"Task: {sample.trajectory.confirmed_task} {previous_actions_text}\nNext Action: "`
    """

    make_loc_func = _make_next_loc_funcs[next_action_loc_type]

    joined_prev_actions = ", ".join(sample.trajectory.action_reprs[: sample.action_idx])
    previous_actions_text = f"Previous Actions: {joined_prev_actions}." if joined_prev_actions != "" else "None."
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if needed the action locator.\n{previous_actions_text}\nNext Action:\n"
    # You are a helpful Web Assistant.
    # Based on the prior actions and the current browser content, respond with the next step you'd take to achieve the OBJECTIVE.
    if len(sample.pos_candidates) > 0:
        # operation = f"{sample.operation.op.lower().capitalize()}" # dont think i should lower case since action_reprs are all CAP
        operation = f"{sample.operation.op}"
        if sample.operation.value != "":
            operation += f" {sample.operation.value}"

        attrs = parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
        x1, y1, x2, y2 = map(int, attrs["bounding_box_rect"])

        loc = make_loc_func(x1, y1, x2, y2)
        next_action = f"{operation} @ {loc}"
    else:
        try:
            operation = f"{sample.operation.op}"
            if sample.operation.value != "":
                operation += f" {sample.operation.value}"
            next_action = operation
        except Exception as err:
            logger.warn(f"Error with {sample.annotation_id} and action idx: {sample.action_idx}.\n{err}")
            next_action = "DONE"
    # else:
    #     next_action = "DONE"

    return {
        "text": text,
        "label": next_action,
        "image": sample.image,
    }
