import random

from pretrain_mm import logger
from pretrain_mm.datasets.mind2web.mind2web import M2WAction
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate

from pretrain_mm import constants

# from pretrain_mm.model.fuyu.processing_fuyu import FuyuConstants


# just split image into 4 sections for now.  potentially can do 9 in future
image_sections = [
    [0, 0, 640, 540],  # top left corner
    [0, 540, 640, 1080],  # bottom left corner
    [640, 0, 1280, 540],  # top right corner
    [640, 540, 1280, 1080],  # bottom right corner
]


def _alt_format(previous_actions_text):
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"
    return text


def parse_action_repr(action_repr: str):
    """
    This function parses the following into a dict:
    '[div]  BMW -> CLICK', '[span]   -> CLICK', '[select]  1992 -> SELECT: 2010', '[button]  Close dialog -> CLICK', '[select]  2024 -> SELECT: 2010', '[combobox]  Sort By -> SELECT: Price: Low to High', '[span]   -> CLICK', '[span]   -> CLICK', '[span]   -> CLICK'
    """
    left_info, right_info = action_repr.split("->")
    left_info = left_info.strip()
    # match the component between [] and the value between []
    html_component = left_info[left_info.index("[") + 1 : left_info.index("]")]
    html_value = left_info[left_info.index("]") + 1 :].strip()
    if html_value == "":
        html_value = None

    # parse right info which is related to action and action value
    right_info = right_info.strip().split(":", 1)
    if len(right_info) == 1:
        action = right_info[0].strip()
        action_value = None
    elif len(right_info) == 2:
        action, action_value = right_info
        action, action_value = action.strip(), action_value.strip()

    return {
        "html_component": html_component,
        "html_value": html_value,
        "action": action,
        "action_value": action_value,
    }


def transform_box_to_cropped_section(coords: tuple[int, int, int, int], image: "Image.Image"):
    """
    given a box in form x1, y1, x2, y2
    return the section of the image that is cropped
    """
    if image.size[0] > 1280 or image.size[1] > 1080:
        logger.warning(f"Image size is {image.size}.  This is larger than 1280x1080.  I should look into this.")
        image = image.crop((0, 0, 1280, 1080))

    section = None
    x1, y1, x2, y2 = coords
    # mid_x, mid_y = round((x1 + x2) / 2), round((y1 + y2) / 2)

    for i_section, (cropped_left, cropped_top, cropped_right, cropped_bottom) in enumerate(image_sections):
        # just use x2+y2 since using middle point can be confusing
        if x2 <= cropped_right and y2 <= cropped_bottom:
            new_x1 = max(x1 - cropped_left, 0)
            new_y1 = max(y1 - cropped_top, 0)
            new_x2 = min(x2 - cropped_left, cropped_right - cropped_left)
            new_y2 = min(y2 - cropped_top, cropped_bottom - cropped_top)
            image = image.crop((cropped_left, cropped_top, cropped_right, cropped_bottom))
            return [new_x1, new_y1, new_x2, new_y2], image, i_section

    return coords, image, section


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
    """
    todo:

    """

    # THESE ARE NEEDED
    boa_string: str
    eos_string: str

    # drop last since processor adds boa string to all even when its part of training
    drop_last: bool = True
    # ignore index is for masking label
    ignore_index: int = constants.IGNORE_INDEX

    def __init__(
        self,
        processor: callable,
        ignore_index: int = ignore_index,
        boa_string: str = None,
        eos_string: str = None,
        loc_before_action_repr: bool = False,
        next_action_loc_type: str = "box",
        crop_image_and_coords: bool = False,
        do_limit_loc_int: bool = False,
    ):
        self.processor = processor
        self.ignore_index = ignore_index

        # these should be part of processor
        self.boa_string = boa_string or processor.constants.boa_string
        self.eos_string = eos_string or processor.constants.eos_string

        self.extra_stop_tokens = [
            self.processor.tokenizer.vocab[token]
            for token in [
                self.processor.constants.image_placeholder_string,  # self.processor.tokenizer.vocab["|SPEAKER|"],
                self.processor.constants.image_newline_string,  # self.processor.tokenizer.vocab["|NEWLINE|"],
                self.eos_string,
            ]
        ]

        # related to creating task

        self.loc_before_action_repr: bool = loc_before_action_repr
        self.next_action_loc_type: str = next_action_loc_type
        self.crop_image_and_coords: bool = crop_image_and_coords
        self.do_limit_loc_int: bool = do_limit_loc_int

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

    def add_stop_token(self, token: str):
        self.extra_stop_tokens.append(self.processor.tokenizer.vocab[token])

    def preprocessor(self, sample: dict):
        """
        this is a task preprocessor for the Mind2Web dataset such that it works for the processor meaning it is only image + text
        the output from this MUST be ingestible by the processor
        """
        text = sample["text"]
        text_with_label = text + f" {self.boa_string} " + sample["label"] + self.eos_string
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
        input_text_with_label = sample["text"] + self.boa_string + sample["label"] + self.eos_string

        # Sample with image needed to mask out the length of the label
        inputs = self.processor(text=sample["text"], images=sample["image"]).input_ids
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

    def task_mind2web(
        self,
        sample: M2WAction,
    ) -> dict:
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

        """
        coords = None

        make_loc_func = _make_next_loc_funcs[self.next_action_loc_type]

        joined_prev_actions = ", ".join(sample.trajectory.action_reprs[: sample.action_idx])
        joined_prev_actions = joined_prev_actions if joined_prev_actions != "" else "None"
        previous_actions_text = f"Previous Actions: {joined_prev_actions}"

        current_action_repr = sample.trajectory.action_reprs[sample.action_idx]

        text = f"You are presented with a browser screenshot, task objective, and previous actions. Generate the corresponding action and action target.\\n"
        text += f"Task: {sample.trajectory.confirmed_task}. {previous_actions_text}."

        if len(sample.pos_candidates) > 0:
            operation = f"{sample.operation.op}"
            if sample.operation.value != "":
                operation += f" {sample.operation.value}"

            attrs = parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
            coords = list(map(int, attrs["bounding_box_rect"]))

            if self.do_limit_loc_int:
                coords = list(limit_loc_int(*coords))

            if self.crop_image_and_coords:
                coords, sample.image, i_section = transform_box_to_cropped_section(coords, sample.image)

            loc = make_loc_func(*coords)
            next_action = f"{operation} @ {loc}"

            # allow either the locator or the action to be the label
            if self.loc_before_action_repr:
                locator = next_action.split(" @ ", 1)[1]
                text += f" Locator: `{locator}` Next Action: "
                label = current_action_repr
            else:
                locator = next_action.split(" @ ", 1)[1]
                text += f" Next Action: `{current_action_repr}` Locator: "
                label = locator

        else:
            operation = f"{sample.operation.op}"
            if sample.operation.value != "":
                operation += f" {sample.operation.value}"
            next_action = operation
            label = next_action

        return {
            "text": text,
            # "label": next_action,
            "label": label,
            "image": sample.image,
            "box": coords,
        }
