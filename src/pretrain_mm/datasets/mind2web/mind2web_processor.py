import random

from bs4 import BeautifulSoup

from pretrain_mm import constants, logger
from pretrain_mm.datasets.mind2web.mind2web import M2WAction
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate
from pretrain_mm.utils.image_utils import transform_box_to_cropped_section


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


def limit_loc_int(*args, max_value: int = 999) -> list[int]:
    return (min(a, max_value) for a in args)


class Mind2WebPretrainProcessor:
    def __init__(self, viewport_size: tuple[int, int] = (1280, 1080)):
        self.viewport_size = viewport_size
        self.task_form = "html-bbox"  # one of 'html-bbox', 'text-bbox',
        self.num_tries = 100
        self.max_text_len = 1_000  # risk of OOM otherwise

    def _make_pretrain(self, sample: M2WAction, parsed_candidate: dict) -> dict:
        x1, y1, x2, y2 = parsed_candidate["attributes"]["bounding_box_rect"]
        node = BeautifulSoup(sample.cleaned_html, "html.parser").find(
            backend_node_id=parsed_candidate["backend_node_id"]
        )

        if len(node.contents) > 5:
            return None

        # bounding_box_label = f"<box>{y1}, {x1}, {y2}, {x2}</box>"
        bbox_label = _make_box_str(x1, y1, x2, y2)

        if self.task_form == "html-bbox":
            instruction = "Given the following HTML provide the bounding box\n"
            text = str(node)

            if len(text) > self.max_text_len:
                return None

            text = f"{instruction}{text}"

            return {"text": text, "label": bbox_label}

        if self.task_from == "text-bbox":
            if node.text == "":
                return None

            instruction = "Given the following text provide the bounding box\n"
            text = node.text
            return {"instruction": instruction, "text": text, "label": bbox_label}

        if self.task_from == "bbox-html":
            instruction = "Given the following bounding box provide the HTML"
            text = bbox_label
            return {"instruction": instruction, "text": text, "label": str(node)}

        return None

    def pretrain_func(self, sample: M2WAction) -> dict:
        """
        pretrain is to generate
        """

        def crop_image_and_cand(image, candidate):
            # for now just increase image size by 1.5 times if candidate is out of viewport
            width, height = self.viewport_size
            if candidate["attributes"]["bounding_box_rect"][3] > height:
                height = int(height * 1.5)

            return image.crop((0, 0, width, height))

        def cand_out_of_viewport(candidate) -> bool:
            if (
                candidate["attributes"]["bounding_box_rect"][2] > self.viewport_size[0]
                or candidate["attributes"]["bounding_box_rect"][3] > self.viewport_size[1]
            ):
                return True
            return False

        def get_and_check() -> dict | None:
            candidate = random.choice(sample.pos_candidates + sample.neg_candidates)
            # convert candidate to dict with bounding box
            parsed_candidate = parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)

            if cand_out_of_viewport(parsed_candidate):
                return None

            output = self._make_pretrain(sample, parsed_candidate)

            if output is None:
                return None

            # crop image to scrolled viewport
            output["image"] = crop_image_and_cand(sample.image.copy(), parsed_candidate)
            return output

        trys = self.num_tries
        inputs_with_labels = None
        while trys and not inputs_with_labels:
            trys -= 1
            inputs_with_labels = get_and_check()

        if trys == 0:
            logger.error("Could not find a candidate that is in the viewport with given number of tries")

        return inputs_with_labels

    def eval_func(self, sample: M2WAction) -> dict:
        pass


class Mind2WebTaskProcessor:
    """ """

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
        self.text_spacer = " "

        self.generate_extra_stop_tokens = [
            # i believe you want this instead of tokenizer.vocab[token] as that includes prefix space
            self.processor.tokenizer.encode(token, add_special_tokens=False)[0]
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
        self.generate_extra_stop_tokens.append(self.processor.tokenizer.vocab[token])

    def process_func(self, sample: dict) -> dict:
        """
        Process the input sample to create the sample with output that has labels for training.

        Args:
            sample (dict): The input sample containing text, label, and images.

        Returns:
            dict: The processed output with labels.
        """

        text = sample["text"]

        if "instruction" in sample:
            text = f"{sample['instruction']}{self.text_spacer}{text}"

        input_text_with_label = text + self.boa_string + sample["label"] + self.eos_string

        # Sample with image needed to mask out the length of the label
        inputs = self.processor(text=text, images=sample["image"]).input_ids
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
