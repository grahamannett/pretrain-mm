import random

from bs4 import BeautifulSoup

from pretrain_mm import constants, logger
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.datasets.mind2web.mind2web import M2WAction
from pretrain_mm.datasets.pretrain_instructions import PretrainTask
from pretrain_mm.model.fuyu import FuyuConstants
from pretrain_mm.utils.image_utils import transform_box_to_cropped_section
from pretrain_mm.utils.token_tag_utils import TagType


def limit_loc_int(*args, max_value: int = 999) -> list[int]:
    return (min(a, max_value) for a in args)


import numpy as np
from paddleocr import PaddleOCR


class Mind2WebPretrainProcessor:
    def __init__(
        self,
        viewport_size: tuple[int, int] = constants.VIEWPORT_SIZE,
        tokenizer_constants: FuyuConstants = FuyuConstants,
        cands_range: tuple[int, int] = (3, 10),
        pretrain_task_name: str = "GenerateNumPotentialActions",
        skip_include_text: bool = False,
        get_text_from: str = "html",
    ):
        self.viewport_size = viewport_size
        self.next_action_loc_type = "box"
        self.task_form = "html-box"  # one of 'html-bbox', 'text-bbox',
        self.num_tries = 150
        self.max_text_len = 1_000  # risk of OOM otherwise
        self.tokenizer_constants = tokenizer_constants

        self.cands_range = cands_range
        self.instruction_func = PretrainTask[pretrain_task_name](num_candidates=self.cands_range[0])
        self.skip_include_text = skip_include_text

        # get the textbox from either the html (works poorly) or ocr
        self._get_text_from = get_text_from
        self._setup_text = {
            "html": self._setup_text_from_html,
            "ocr": self._setup_text,
        }[get_text_from]

        self._get_text = {
            "html": self._get_text_from_html,
            "ocr": self._get_text_from_ocr,
        }[get_text_from]
        # ocr testing
        self.paddleocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    def _make_pretrain_sample(self, sample: M2WAction, parsed_candidate: dict) -> dict:
        x1, y1, x2, y2 = parsed_candidate["attributes"]["bounding_box_rect"]
        node = BeautifulSoup(sample.cleaned_html, "html.parser").find(
            backend_node_id=parsed_candidate["backend_node_id"]
        )

        if len(node.contents) > 5:
            return None

        box_label = TagType.make(self.next_action_loc_type)(x1, y1, x2, y2)

        if self.task_form == "html-box":
            # instruction = "When presented with HTML perform OCR to generate the corresponding bounding box. \n "
            # instruction = "Generate the bounding box of 3 potential actions for the screenshot.  Give the action text if relevant. \n"
            instruction = self.instruction_func()

            text = node.text
            text = text.replace("\n", " ")
            # text = str(node) # might want `text.replace(">\n<", "> <")`

            if text.strip() == "":
                return None

            if len(text) > self.max_text_len:
                return None

            return {"text": text, "label": box_label, "instruction": instruction}

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

    def _setup_text(self, **kwargs):
        pass

    def _setup_text_from_html(self, sample: M2WAction):
        self.soup = BeautifulSoup(sample.cleaned_html, "html.parser")

    def _get_text_from_html(self, cand: dict, **kwargs) -> str:
        node = self.soup.find(backend_node_id=cand["backend_node_id"])
        cleaned_text = node.text.replace("\n", " ").strip()
        cleaned_text = " ".join(cleaned_text.split())
        return cleaned_text

    def _get_text_from_ocr(self, image: "PIL.Image.Image", coords: tuple[int, int, int, int], **kwargs) -> str:
        paddle_result = self.paddleocr.ocr(np.asarray(image.crop(coords)), cls=True)[0]
        paddle_texts, paddle_probs = zip(*[pair[1] for pair in paddle_result]) if paddle_result else ([], [])
        # return " ".join(paddle_texts)
        return self._get_text_from_html(self, cand=kwargs["cand"])

    def _prepare_text(self, text: str) -> str:
        if self.skip_include_text or text == "":
            return ""

        return f"|ACTION| {text} |ENDACTION| "

    def pretrain_func_generate_possible_actions(self, sample: M2WAction):
        """
        this pretraining just has the model generate a bunch of bounding boxes for possible actions
        """

        # trying to think about what makes most sense

        cands_allowed = random.randint(*self.cands_range)

        boxes_covered = []
        text_label = ""
        instruction = self.instruction_func(num_candidates=cands_allowed)

        # self._setup_text(sample=sample)
        self._setup_text_from_html(sample=sample)

        cands = sample.pos_candidates + sorted(sample.neg_candidates, key=lambda x: random.random())
        for c_idx, cand in enumerate(cands):
            parsed_candidate = m2w_utils.parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
            bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]

            if m2w_utils.cand_out_of_viewport(parsed_candidate, self.viewport_size, buffer_amt=1.2):
                continue

            if any(m2w_utils.point_within_box(m2w_utils.get_mid_point(bounding_box), b) for b in boxes_covered):
                continue

            tag_str = TagType.make(self.next_action_loc_type)(*bounding_box)
            candidate_text = self._get_text(cand=cand, image=sample.image, coords=bounding_box)
            include_text = self._prepare_text(candidate_text)

            text_label += f"\n {tag_str} {include_text}"
            boxes_covered.append(bounding_box)

            if len(boxes_covered) >= cands_allowed:
                break

        return {
            "image": sample.image.crop((0, 0, *self.viewport_size)),
            "text": instruction,
            "label": text_label,
        }


class Mind2WebTaskProcessor:
    """
    This Processor Is for general usage regardless of task.
    """

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
        max_length: int = 2048,
        loc_before_action_repr: bool = False,
        next_action_loc_type: TagType = TagType.BOX,
        crop_image_and_coords: bool = False,
        do_limit_loc_int: bool = False,
    ):
        self.processor = processor
        self.ignore_index = ignore_index

        # these should be part of processor
        # REQUIRED
        self.boa_string = boa_string or processor.constants.boa_string
        self.eos_string = eos_string or processor.constants.eos_string
        self.instruction_spacer = ""

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
        self.next_action_loc_type = next_action_loc_type
        self.make_loc_func = TagType.make(self.next_action_loc_type)

        self.loc_before_action_repr: bool = loc_before_action_repr
        self.crop_image_and_coords: bool = crop_image_and_coords
        self.do_limit_loc_int: bool = do_limit_loc_int

        self.max_length = max_length

    @classmethod
    def postprocessor(cls, sample: dict):
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

    def encode_data(
        self,
        sample: dict,
        add_bos_token: bool = True,
        add_boa_token: bool = True,
        label_add_eos_token: bool = True,
        include_label: bool = True,
        include_text: bool = True,
        **kwargs,
    ) -> dict:
        """
        Process the input sample to create the sample with output that has labels for training.

        in the case where you want to test generated output you want the inputs to be the encoded inputs without label but with boa token

        Args:
            sample (dict): The input sample containing text, label, and images.

        Returns:
            dict: The processed output with labels.
        """
        raw_text = sample["text"]
        raw_image = sample["image"]
        raw_label = sample.get("label", None)
        raw_instruction = sample.get("instruction", False)

        if not include_text:
            raw_text = ""

        if raw_instruction:
            raw_text = f"{raw_instruction}{self.instruction_spacer}{raw_text}"

        batch = self.processor.__call__(
            text=raw_text,
            images=raw_image,
            label=raw_label if include_label else None,
            add_bos_token=add_bos_token,
            add_boa_token=add_boa_token,
            label_add_eos_token=label_add_eos_token,
            max_length=self.max_length,
        )
        # TODO: Remove once dataloader runtime error is fixed
        logger.log(f"Done encoding batch")

        return batch

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

            attrs = m2w_utils.parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)[
                "attributes"
            ]
            coords = list(map(int, attrs["bounding_box_rect"]))

            if self.do_limit_loc_int:
                coords = list(limit_loc_int(*coords))

            if self.crop_image_and_coords:
                coords, sample.image, i_section = transform_box_to_cropped_section(coords, sample.image)

            loc = self.make_loc_func(*coords)
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


# OLD
def old_pretrain_func(self, sample: M2WAction) -> dict:
    """
    pretrain is to generate
    """

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
