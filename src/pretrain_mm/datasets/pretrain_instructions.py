from dataclasses import dataclass, fields


class PretrainTask:
    instruction: str

    _debug: bool = False

    def __class_getitem__(cls, task_name: str) -> "PretrainTask":
        for cls in cls.__subclasses__():
            if cls.__name__ == task_name:
                return cls
        raise KeyError(f"PretrainTask {task_name} not found")

    def __repr__(self):
        return self.instruction

    def __str__(self):
        return super().__str__()  # or self.__call__()

    def __call__(self, *args, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return self.format(*args, **kwargs)

    def format(self, *args, **kwargs):
        # if you want to override the call() of the class use format()
        return self.instruction.format(*args, **kwargs)

    def _input_fields(self):
        # return the fields of the class that are used when generating prompt

        def _check_field(f):
            return f.name not in ["instruction", "_debug"] and not f.name.startswith("_")

        return [(f.name, f.type) for f in fields(self) if _check_field(f)]


# can decorate instead
def make_instruction(cls, **kwargs):
    cls = type(cls.__name__, (cls, PretrainTask), {})
    return dataclass(cls)(**kwargs)


class PretrainHTML:
    def __init__(self):
        pass


@dataclass
class AssistantResponse(PretrainTask):
    instruction: str = "Perform OCR for the following task: `{task}`"
    previous_actions_text: str = "\nPrevious Actions:{previous_actions}"
    next_action_text: str = "\nNext Action: {next_action}"

    def format(
        self,
        task: str = "",
        previous_actions: str = "",
        next_action: str = "",
        strip_rpunc: bool = False,
        split_instruction: bool = False,
        **kwargs,
    ):
        # strip last period
        if strip_rpunc and task[-1] in ["?", ".", "!"]:
            task = task[:-1]

        instr_str = self.instruction.format(task=task)

        actions_text = ""
        if previous_actions != "":
            actions_text += self.previous_actions_text.format(previous_actions=previous_actions)

        actions_text += self.next_action_text.format(next_action=next_action)

        if split_instruction:
            return instr_str, actions_text

        return instr_str + actions_text


@dataclass
class GeneratePotentialActions(PretrainTask):
    instruction: str = "Generate the bounding box of potential actions for the screenshot."


@dataclass
class GenerateNumPotentialActions(PretrainTask):
    num_candidates: int = 5
    instruction: str = (
        "Generate the bounding box of {num_candidates} potential action for the page. Give the action text if relevant."  # noqa
    )


@dataclass
class GenerateActionGivenLocator(PretrainTask):
    instruction: str = (
        "Given the current state, generate the action that would be performed for a given locator. {state_str}"
    )
    label: str = "Locator: {locator_str}"


@dataclass
class BaselineBoxToText(PretrainTask):
    instruction: str = (
        # e.g. <box>y1, x1, y2, x2</box>"
        "When presented with a box, perform OCR to extract text contained within it. "
        "If provided with text, generate the corresponding bounding box.\\n{box_str}"
    )


@dataclass
class BaselineTextToBox(PretrainTask):
    instruction: str = (
        "When presented with a box, perform OCR to extract text contained within it. "
        "If provided with text, generate the corresponding bounding box.\\n {text_str}"
    )


@dataclass
class MaskedText(PretrainTask):
    instruction: str = (
        "Provide the {text_or_html_str} for the locator on the given page given the action: {locator_pos} {action_str}"
    )


@dataclass
class GenerateInitialTask(PretrainTask):
    instruction: str = "Given the following page and state, generate the most likely initial task. {state_str}"


@dataclass
class GeneratePreviousAction(PretrainTask):
    instruction: str = (
        "Given the following page and state, generate the action that directly preceded the current action. {state_str}"
        "Next action: {next_action_str}"
    )


@dataclass
class OCRBoundingBoxCompletion(PretrainTask):
    instruction: str = (
        "When presented with a box, perform OCR to extract text contained within it. "
        "If provided with text, generate the corresponding bounding box.\\n {text_str}"  # cant remember  at end <box>
    )


class InstructionInstances:
    assistant_response = AssistantResponse()
    generate_potential_actions = GeneratePotentialActions()
    generate_num_potential_actions = GenerateNumPotentialActions()
    baseline_box_to_text = BaselineBoxToText()
    baseline_text_to_box = BaselineTextToBox()
    ocr_bounding_box_completion = OCRBoundingBoxCompletion()


if __name__ == "__main__":
    cls_type = PretrainTask["GenerateNumPotentialActions"]()
    print(" CLS TYPE: ", cls_type)
    print(" CLS CALL: ", cls_type(num_candidates=10))
