import random

from pretrain_mm import logger
from pretrain_mm.datasets.mind2web.mind2web import M2WAction
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate

BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>


def _make_point_str(x1, y1, x2=None, y2=None) -> str:
    x, y = x1, y1

    if x2 and y2:
        x, y = round((x + x2) / 2), round((y1 + y2) / 2)

    return f"<point>{x}, {y}</point>"


def _make_box_str(x1, y1, x2, y2) -> str:
    # FUYU NEEDS IN format: y1, x1, y2, x2 but bounding box comes in form x0, y0, x1, y1,
    return f"<box>{y1}, {x1}, {y2}, {x2}</box>"


def limit_loc_int(*args, max_value: int = 999) -> list[int]:
    return (min(a, max_value) for a in args)


_make_next_loc_funcs = {
    "point": _make_point_str,
    "box": _make_box_str,
}


class Mind2WebTaskProcessor:
    eos_token: str = "|ENDOFTEXT|"  # adjust this if needed

    boa_string: str = BEGINNING_OF_ANSWER_STRING
    ignore_index: int = -100
    processor: callable = None

    # drop last since processor adds boa string to all even when its part of training
    drop_last: bool = True

    def __init__(self, processor: callable, ignore_index: int):
        self.processor = processor
        self.eos_token = processor.tokenizer.eos_token
        self.ignore_index = ignore_index

        self.extra_stop_tokens = [
            self.processor.tokenizer.vocab["|SPEAKER|"],
            self.processor.tokenizer.vocab["|NEWLINE|"],
            self.processor.tokenizer.vocab[self.eos_token],
        ]

    @staticmethod
    def postprocessor(sample: dict):
        """
        helper function that reshapes the sample that comes from processor as processor gives us a batched sample but
        data collator expects a list of samples
        """
        sample["input_ids"] = sample["input_ids"].squeeze(0)
        sample["attention_mask"] = sample["attention_mask"].squeeze(0)
        sample["image_patches"] = [img.squeeze(0) for img in sample["image_patches"]]
        sample["image_patches_indices"] = sample["image_patches_indices"].squeeze(0)
        sample["labels"] = sample["labels"].squeeze(0)
        return sample

    def preprocessor(self, sample: dict):
        """
        this is a task preprocessor for the Mind2Web dataset such that it works for the processor meaning it is only image + text
        the output from this MUST be ingestible by the processor
        """
        text = sample["text"]
        text_with_label = text + " \x04 " + sample["label"] + self.eos_token
        return {
            "text": text,
            "label": text_with_label,
            "images": sample["image"],
        }

    def process_func(self, sample: dict) -> dict:
        """
        Process the input sample to generate the processed output with labels.

        Args:
            sample (dict): The input sample containing text, label, and images.

        Returns:
            dict: The processed output with labels.
        """
        input_text = sample["text"]
        input_text_with_label = input_text + self.boa_string + sample["label"] + self.eos_token

        # Sample with image needed to mask out the length of the label
        inputs = self.processor(text=input_text, images=sample["image"]).input_ids
        inputs_with_label = self.processor(text=input_text_with_label, images=sample["image"])

        # since we put boa token into input_with_label and processor does this as well for some reason
        # we need to drop the last bit

        if self.drop_last:
            inputs_with_label.input_ids = inputs_with_label.input_ids[0, :-1]
            inputs_with_label.image_patches_indices = inputs_with_label.image_patches_indices[0, :-1]
            inputs_with_label.attention_mask = inputs_with_label.attention_mask[0, :-1]

        # Mask out instructions/image
        label = inputs_with_label.input_ids.clone()
        label[: inputs.shape[1]] = self.ignore_index

        # Make sure to include labels in the return item
        inputs_with_label["labels"] = label
        return inputs_with_label


def task_mind2web(sample: M2WAction, next_action_loc_type: str = "point", do_limit_loc_int: bool = False) -> dict:
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
    joined_prev_actions = joined_prev_actions if joined_prev_actions != "" else "None"
    previous_actions_text = f"Previous Actions: {joined_prev_actions}"

    text = f"You are presented with a browser screenshot, task objective, and previous actions. Generate the corresponding action and action target.\\n"
    text += f"Task: {sample.trajectory.confirmed_task}. {previous_actions_text}. Next Action: "
    # You are a helpful Web Assistant.
    # Based on the prior actions and the current browser content, respond with the next step you'd take to achieve the OBJECTIVE.
    if len(sample.pos_candidates) > 0:
        # operation = f"{sample.operation.op.lower().capitalize()}" # dont think i should lower case since action_reprs are all CAP
        operation = f"{sample.operation.op}"
        if sample.operation.value != "":
            operation += f" {sample.operation.value}"

        attrs = parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
        x1, y1, x2, y2 = map(int, attrs["bounding_box_rect"])

        if do_limit_loc_int:
            x1, x2, y1, y2 = limit_loc_int(x1, x2, y1, y2)

        loc = make_loc_func(x1, y1, x2, y2)
        next_action = f"{operation} @ {loc}"
    else:
        operation = f"{sample.operation.op}"
        if sample.operation.value != "":
            operation += f" {sample.operation.value}"
        next_action = operation

    return {
        "text": text,
        "label": next_action,
        "image": sample.image,
    }


# ALTERNATIVE TASK/FORMATING BELOW

def _alt_format(previous_actions_text):
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"
    return text
