import random
from typing import Callable

import torch
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

from pretrain_mm import constants, logger
from pretrain_mm.datasets.base import create_sample_type_like_dict
from pretrain_mm.datasets.mind2web import ActionOp, M2WAction
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.datasets.pretrain_instructions import InstructionInstances, PretrainTask
from pretrain_mm.processor.tokenizer_constants import TokenizerConstants
from pretrain_mm.utils.bbox_utils import (
    BoundingBox,
    add_margin_to_bbox,
    get_midpoint,
    invalid_or_outside,
    point_within_box,
)
from pretrain_mm.utils.image_utils import transform_box_to_cropped_section
from pretrain_mm.utils.token_tag_utils import TagType
from pretrain_mm.utils.transforms import dummy_func


# MARK: Helper Functions


def action_op_to_str(operation: ActionOp, midpoint: tuple[int, int], tconstants: TokenizerConstants) -> str:
    # for turning a mind2web action into a string, the format of the unaltered previous actions is like

    # midpoint is in the format of (x, y) but fuyu uses (y,x)
    # also some actions do not come with pos_candidate meaning no bounding box.
    # in those cases i believe the action location corresponds to previous action location
    loc_str = ""
    if midpoint:
        loc_str = f"{tconstants.repr_point_open_text}{midpoint[1]}, {midpoint[0]}{tconstants.repr_point_close_text} ->"

    def handle_CLICK():
        return f"{loc_str} CLICK"

    def handle_TYPE():
        return f"{loc_str} TYPE: {operation.value}"

    def handle_SELECT():
        return f"{loc_str} SELECT: {operation.value}"

    def handle_OTHER():
        return f"{loc_str} {operation.op}: {operation.value}"

    _handler = {
        "click": handle_CLICK,
        "type": handle_TYPE,
        "select": handle_SELECT,
    }.get(operation.op.lower(), handle_OTHER)  # should already be lowercase
    return _handler()


# MARK: TaskSample
@create_sample_type_like_dict
class TaskSample:
    image: Image.Image
    text: str
    label: str

    # extra metadata should ONLY be for debugging/tracking
    def use(self, **kwargs):
        # allow chaining of setting attributes for kwargs/metadata
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"Cannot set attribute {k} as it already exists")

            setattr(self, k, v)
        return self


class Mind2WebProcessor:
    def __init__(
        self,
        tokenizer_constants: TokenizerConstants = None,
        crop_image: bool = True,
        viewport_size: tuple[int, int] = constants.VIEWPORT_SIZE,
        *args,
        **kwargs,
    ):
        self.tconstants: TokenizerConstants = tokenizer_constants
        self.viewport_size = viewport_size
        self.crop_image = crop_image

    def __call__(self):
        pass


# MARK: Mind2WebPretrainProcessor
class Mind2WebPretrainProcessor(Mind2WebProcessor):
    def __init__(
        self,
        # task_function: str | Callable = None,
        instruction: str = None,
        skip_include_text: bool = False,
        get_text_from: str = "html",
        ocr_preprocessed: Callable = None,
        ocr_use_gpu: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.next_action_loc_type = "box"
        self.task_form = "html-box"  # one of 'html-bbox', 'text-bbox',
        self.num_tries = 150
        self.max_text_len = 1_000  # risk of OOM otherwise

        # self.task_function = task_function if callable(task_function) else getattr(self, task_function)
        self.instruction = instruction

        # specific task/instruction specific
        self.skip_include_text = skip_include_text
        self.ocr_preprocessed = ocr_preprocessed
        self.ocr_use_gpu = ocr_use_gpu

        # get the textbox from either the html (works poorly) or ocr
        self._setup_text_from(get_text_from)

    def _get_image(self, sample: M2WAction) -> Image.Image:
        image: Image = sample.image

        if self.crop_image:
            image = image.crop((0, 0, *self.viewport_size))

        return image

    def _setup_text_from(self, get_text_from: str) -> None:
        """
        helper method to define if using external tool to get text from image/html
        """
        self._text_from = get_text_from

        def prepare_text_HTML(sample: M2WAction):
            self.soup = BeautifulSoup(sample.cleaned_html, "html.parser")

        def get_text_HTML(cand: dict, **kwargs) -> str:
            node = self.soup.find(backend_node_id=cand["backend_node_id"])
            cleaned_text = node.text.replace("\n", " ").strip()
            cleaned_text = " ".join(cleaned_text.split())
            return cleaned_text

        def get_text_OCR(image: Image.Image, coords: tuple[int, int, int, int], **kwargs):
            raise NotImplementedError("not doing ocr in this anymore")

        self._prepare_text = {
            "html": prepare_text_HTML,
            "ocr": dummy_func,
        }

        # get text is for getting the text for cand from either html or ocr (paddleocr)
        self._get_text = {
            "html": get_text_HTML,
            "ocr": get_text_OCR,
        }

    def _make_include_text(self, text: str, max_length: int = 100, skip_include_text: bool = False) -> str:
        if skip_include_text or text == "":
            return ""

        return f"{text[:max_length]}"
        # open_tag, close_tag = self.tconstants.repr_action_open_text, self.tconstants.repr_action_close_text
        # return f"{open_tag} {text[:max_length]} {close_tag} "

    def _get_text_from_cache(self, ocr_cache, cand_type: str = "pos_candidates", c_idx: int = 0):
        ocr_results = ocr_cache.get(cand_type, {}).get(c_idx, {}).get("before", {}).get("text", {}).get("paddle", [])
        if ocr_results == []:
            return None

        return " ".join(ocr_results)

    def _candidate_to_str(self, cand: dict) -> str:
        pass

    def _add_cand_outline(
        self,
        bounding_box: BoundingBox,
        image: Image.Image = None,
        draw: ImageDraw.Draw = None,
        margin: int = None,
        color: str = "black",
        width: int = 3,
    ) -> ImageDraw.ImageDraw:
        if not draw:
            draw = ImageDraw.Draw(image)

        if margin:
            bounding_box = add_margin_to_bbox(bounding_box, margin=margin)

        draw.rectangle(bounding_box, outline=color, width=width)
        return draw

    def ocr_eval(
        self,
        sample: M2WAction,
        add_cand_outline: bool = False,
        cand_max_len: int = 1000,
        eval_from: str = "bounding_box",
        **kwargs,
    ):
        _outside_kwargs = {
            "viewport_cutoff": 1.1,
            "area_cutoff": 0.5,  # max area size
            "width": self.viewport_size[0],
            "height": self.viewport_size[1],
        }

        image = self._get_image(sample)

        candidate = sample.pos_candidates[0]
        parsed_candidate = m2w_utils.parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)

        node = BeautifulSoup(sample.cleaned_html, "html.parser").find(
            backend_node_id=parsed_candidate["backend_node_id"]
        )

        candidate_text = node.text.replace("\n", " ").strip()[:cand_max_len]
        bounding_box = sample.get_bounding_box(cand_type="pos_candidates", cand_idx=0)

        bounding_box_str = TagType.make(TagType.BOX)(*bounding_box)

        if eval_from == "bounding_box":
            instruction_text = InstructionInstances.ocr_bounding_box_completion.format(text_str=candidate_text)
            label = bounding_box_str
        elif eval_from == "text_ocr":
            instruction_text = InstructionInstances.ocr_bounding_box_completion.format(text_str=bounding_box_str)
            label = candidate_text

        if add_cand_outline:
            # the margin/width are so there is a black box on top of a red box.
            # for human it should be obvious what the target is
            draw = self._add_cand_outline(bounding_box, image=image, color="black", width=6, margin=3)
            draw = self._add_cand_outline(bounding_box, draw=draw, color="red", width=2, margin=1)

        # NOTE:
        # This seems like a very bad design decision, but i am not sure how I can pass metadata forward easily to
        # eval/training for debugging/analysis while also adding additional values for loss
        extra = {
            # metadata is not used for forward
            "meta": {
                "annotation_id": sample.annotation_id,
                "sample": sample,
            }
        }

        return TaskSample(image=image, text=instruction_text, label=label).use(
            encode_kwargs={
                "add_bos_token": True,
                "add_boa_token": True,
                "label_add_eos_token": True,
            },
            extra=extra,
        )

    # MARK: >agent training
    def agent_training(
        self,
        sample: M2WAction,
        include_patch_idx: bool = False,
        add_cand_outline: bool = False,
        **kwargs,
    ) -> TaskSample | None:
        _outside_kwargs = {
            "viewport_cutoff": 1.1,
            "area_cutoff": 0.5,  # max area size
            "width": self.viewport_size[0],
            "height": self.viewport_size[1],
        }

        image = self._get_image(sample)

        if sample.pos_candidates != []:
            bounding_box = sample.get_bounding_box()
            midpoint = get_midpoint(bounding_box, to_int=True)

            if invalid_or_outside(bounding_box, **_outside_kwargs):
                return None

            if add_cand_outline:
                # the margin/width are so there is a black box on top of a red box.
                # for human it should be obvious what the target is
                draw = self._add_cand_outline(bounding_box, image=image, color="black", width=6, margin=3)
                draw = self._add_cand_outline(bounding_box, draw=draw, color="red", width=2, margin=1)
        else:
            # no pos_candidates means we have operation but no location, either need to get from previous action or???
            logger.warn("No pos_candidates")
            midpoint = None

        # the list of previous actions into a string that model can use. avoid list comprehension to read easier
        prev_actions_text = ""
        for prev_act_idx, prev_act in enumerate(sample.action_repr_previous):
            prev_actions_text += f"\n\t{prev_act_idx + 1}: {prev_act.format()}"

        # instruction_text = self.instruction_func.format(
        instruction_text = InstructionInstances.assistant_response.format(
            task=sample.confirmed_task,
            previous_actions=prev_actions_text,
            next_action="",  # add label during encode
        )
        action_op_str = action_op_to_str(sample.operation, midpoint=midpoint, tconstants=self.tconstants)
        label = action_op_str

        # NOTE:
        # This seems like a very bad design decision, but i am not sure how I can pass metadata forward easily to
        # eval/training for debugging/analysis while also adding additional values for loss
        extra = {
            # metadata is not used for forward
            "meta": {
                "annotation_id": sample.annotation_id,
                "sample": sample,
            }
        }

        if include_patch_idx:
            # extra loss will be used in forward
            patch_idx = kwargs.get("image_processor").get_patch_idx_from_midpoint(midpoint, image_size=image.size)
            extra["extra_loss"] = {
                "patch_idx": torch.tensor(patch_idx),
            }

        return TaskSample(image=image, text=instruction_text, label=label).use(
            encode_kwargs={
                "add_bos_token": True,
                "add_boa_token": True,
                "label_add_eos_token": True,
            },
            extra=extra,
        )

    def pretrain_func_generate_possible_actions(
        self,
        sample: M2WAction,
        randomize_order_label: bool | float = False,
        get_from: str = "html",
        cands_range: tuple[int, int] = (3, 10),
        skip_include_text: bool = False,
        add_cand_outline: bool = False,
        include_patch_idx: bool = False,
        image_processor: Callable = None,
        **kwargs,
    ):
        """This pretraining just has the model generate a bunch of bounding boxes for possible actions"""
        # trying to think about what makes most sense
        draw = None
        midpoint = None
        _outside_kwargs = {
            "viewport_cutoff": 1.75,
            "area_cutoff": 0.5,
            "width": self.viewport_size[0],
            "height": self.viewport_size[1],
        }
        image = self._get_image(sample)
        cands_allowed = random.randint(*cands_range)
        instruction = InstructionInstances.generate_num_potential_actions(num_candidates=cands_allowed)

        def _random_order_label():
            if randomize_order_label:
                return random.random() < randomize_order_label
            return False

        boxes_covered = []
        text_label = ""

        self._prepare_text[get_from](sample=sample)

        cands = sample.pos_candidates + sorted(sample.neg_candidates, key=lambda x: random.random())
        cand_types = [1] * len(sample.pos_candidates) + [0] * len(sample.neg_candidates)

        for c_idx, (cand, cand_type) in enumerate(zip(cands, cand_types)):
            parsed_candidate = m2w_utils.parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
            bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]

            # check coords are valid
            if invalid_or_outside(bounding_box, **_outside_kwargs):
                continue

            if any(point_within_box(get_midpoint(bounding_box, to_int=True), b) for b in boxes_covered):
                continue

            if add_cand_outline:
                draw = self._add_cand_outline(bounding_box, image=image, draw=draw, color="black", width=4, margin=2)

            candidate_text = self._get_text[get_from](cand=cand, image=image, coords=bounding_box)
            if cand_type == 0:
                # should do something special if pos cand
                include_text = self._make_include_text(candidate_text, skip_include_text=skip_include_text)
                midpoint = get_midpoint(bounding_box, to_int=True)
            else:
                include_text = self._make_include_text(candidate_text, skip_include_text=skip_include_text)

            # if the candidate text is empty, dont postfix and just use box
            if candidate_text.strip() == "":
                tag_str = TagType.make(TagType.POINT)(*bounding_box)
                text_label += f"\n {tag_str}"
            else:
                tag_str = TagType.make(TagType.BOX)(*bounding_box)

                if _random_order_label():
                    # some amount of swtiching the order of the tag and the text
                    text_label += f"\n {include_text}{tag_str} "
                else:
                    text_label += f"\n {tag_str} {include_text}"

            boxes_covered.append(bounding_box)

            if len(boxes_covered) >= cands_allowed:
                break

        extra = {
            # metadata is not used for forward
            "meta": {
                "annotation_id": sample.annotation_id,
                "action_id": sample.action_uid,
                "action_idx": sample.action_idx,
                "sample": sample,
            }
        }

        if include_patch_idx:
            # extra loss will be used in forward
            if midpoint and image_processor:
                patch_idx = image_processor.get_patch_idx_from_midpoint(midpoint, image_size=image.size)
                extra["extra_loss"] = {
                    "patch_idx": torch.tensor(patch_idx),
                }

        return TaskSample(image=image, text=instruction, label=text_label).use(
            encode_kwargs={
                "add_bos_token": True,
                "add_boa_token": True,
                "label_add_eos_token": True,
                # "include_label": True,
            },
            extra=extra,
        )


class Mind2WebEvalMethod(Mind2WebProcessor):
    def acc_func_complete_box(self, sample: M2WAction):
        if sample.pos_candidates == []:
            return False

        # accuracy is based on single candidate completion
        # instruction = self.instruction_func(num_candidates=1)

        cand = sample.pos_candidates[0]

        parsed_candidate = m2w_utils.parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
        bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]
        _outside_kwargs = {
            "viewport_cutoff": 1.1,
            "area_cutoff": 0.5,
            "width": self.viewport_size[0],
            "height": self.viewport_size[1],
        }

        if invalid_or_outside(bounding_box, **_outside_kwargs):
            return False

        tag_str = TagType.make(self.next_action_loc_type)(*bounding_box)

        # use starting tag as `<tag>y1` for acc metric
        # if we want first 2 coords (basically get the model to complete the box):
        #  -  ','.join(alttag.split(',', 2)[0:2])
        starting_tag = tag_str.split(",", 1)[0]

        # text should be like: <s> instruction <0x04> <s> tag <0x04>
        # text = f"<s> {instruction} \n <0x04> {starting_tag}"
        instruction_text = InstructionInstances.generate_num_potential_actions()
        text = f"{instruction_text} \n <0x04> {starting_tag}"

        image = self._get_image(sample)

        return TaskSample(image=image, text=text, label=tag_str).use(
            encode_kwargs={
                "add_bos_token": True,
                "add_boa_token": False,
                "label_add_eos_token": False,
                "include_label": False,
            },
            extra={"annotation_id": sample.annotation_id},
        )

    def eval_by_complete_text(
        self,
        sample: M2WAction,
        # hacky way but allows the class_getitem to only happen once
        _instruction: PretrainTask = PretrainTask["BaselineBoxToText"](),
    ):
        """_summary_

        Args:
        ----
            sample (M2WAction): _description_
            Instruction (PretrainTask, optional): _description_. Defaults to PretrainTask["BaselineBoxToText"].
            crop_image (bool, optional): _description_. Defaults to True.

        Returns:
        -------
            _type_: _description_

        """
        if sample.pos_candidates == []:
            return False

        _get_from = "html"
        self._prepare_text[_get_from](sample=sample)

        cand = sample.pos_candidates[0]
        _outside_kwargs = {
            "viewport_cutoff": 1.1,
            "area_cutoff": 0.5,
            "width": self.viewport_size[0],
            "height": self.viewport_size[1],
        }

        bounding_box = sample.get_bounding_box()

        if invalid_or_outside(bounding_box, **_outside_kwargs):
            return False

        tag_str = TagType.make(self.next_action_loc_type)(*bounding_box)

        instruction = _instruction(box_str=tag_str)
        candidate_text = self._get_text[_get_from](cand=cand, image=sample.image, coords=bounding_box)

        # if empty string then skip
        if candidate_text.strip() == "":
            return False

        image = self._get_image(sample)

        return TaskSample(image=image, text=instruction, label=candidate_text).use(
            encode_kwargs={
                "add_bos_token": True,
                "add_boa_token": True,
                "label_add_eos_token": False,
                "include_label": False,
            },
            extra={"annotation_id": sample.annotation_id},
        )


# ----------------------------------------
# ----------------------------------------
# ----------------------------------------
# OLD
#  LIKELY WILL BE REINTEGRATED BUT NEED TO
#  MORE CLEARLY UNDERSTAND WHAT IS NEEDED
# ----------------------------------------
# ----------------------------------------
# ----------------------------------------


def limit_loc_int(*args, max_value: int = 999) -> list[int]:
    return (min(a, max_value) for a in args)


# MARK: OLD
# THE PREVIOUS PRETRAIN FUNC
def task_mind2web(
    self,
    sample: M2WAction,
    loc_before_action_repr: bool = False,
    next_action_loc_type: TagType = TagType.BOX,
    crop_image_and_coords: bool = False,
    do_limit_loc_int: bool = False,
) -> dict:
    # related to creating task

    self.next_action_loc_type = next_action_loc_type
    self.loc_before_action_repr = loc_before_action_repr
    self.crop_image_and_coords = crop_image_and_coords
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

    joined_prev_actions = ", ".join(sample.trajectory.action_reprs[: sample.action_idx])
    joined_prev_actions = joined_prev_actions if joined_prev_actions != "" else "None"
    previous_actions_text = f"Previous Actions: {joined_prev_actions}"

    current_action_repr = sample.trajectory.action_reprs[sample.action_idx]

    text = "You are presented with a browser screenshot, task objective, and previous actions. Generate the corresponding action and action target.\\n"
    text += f"Task: {sample.trajectory.confirmed_task}. {previous_actions_text}."

    if len(sample.pos_candidates) > 0:
        operation = f"{sample.operation.op}"
        if sample.operation.value != "":
            operation += f" {sample.operation.value}"

        attrs = m2w_utils.parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
        coords = list(map(int, attrs["bounding_box_rect"]))

        if do_limit_loc_int:
            coords = list(limit_loc_int(*coords))

        if self.crop_image_and_coords:
            coords, sample.image, i_section = transform_box_to_cropped_section(coords, sample.image)

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


def old_pretrain_func(self, sample: M2WAction) -> dict:
    """Pretrain is to generate"""

    def get_and_check() -> dict | None:
        candidate = random.choice(sample.pos_candidates + sample.neg_candidates)
        # convert candidate to dict with bounding box
        parsed_candidate = m2w_utils.parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)

        if m2w_utils.cand_out_of_viewport(parsed_candidate, self.viewport_size):
            return None

        output = self._make_pretrain_sample(sample, parsed_candidate)

        if output is None:
            return None

        # crop image to scrolled viewport
        output["image"] = m2w_utils.crop_image_and_cand(sample.image.copy(), parsed_candidate)
        return output

    trys = self.num_tries
    inputs_with_labels = None
    while trys and not inputs_with_labels:
        trys -= 1
        inputs_with_labels = get_and_check()

    if not trys:
        logger.error("Could not find a candidate that is in the viewport with given number of tries")

    return inputs_with_labels


def _make_pretrain_sample(self, sample: M2WAction, parsed_candidate: dict) -> dict:
    x1, y1, x2, y2 = parsed_candidate["attributes"]["bounding_box_rect"]
    node = BeautifulSoup(sample.cleaned_html, "html.parser").find(backend_node_id=parsed_candidate["backend_node_id"])

    if len(node.contents) > 5:
        return None

    box_label = TagType.make(self.next_action_loc_type)(x1, y1, x2, y2)

    if self.task_form == "html-box":
        # instruction = "When presented with HTML perform OCR to generate the corresponding bounding box. \n "
        # instruction = "Generate the bounding box of 3 potential actions for the screenshot.  Give the action text if relevant. \n"
        instruction_text = self.instruction()

        text = node.text
        text = text.replace("\n", " ")
        # text = str(node) # might want `text.replace(">\n<", "> <")`

        if text.strip() == "":
            return None

        if len(text) > self.max_text_len:
            return None

        return {"text": text, "label": box_label, "instruction": instruction_text}

    if self.task_from == "text-box":
        if node.text == "":
            return None

        instruction = "Given the following text provide the bounding box\n"
        text = node.text
        return {"instruction": instruction, "text": text, "label": box_label}

    if self.task_from == "box-html":
        instruction = "Given the following bounding box provide the HTML"
        text = box_label
        return {"instruction": instruction, "text": text, "label": str(node)}

    return None
