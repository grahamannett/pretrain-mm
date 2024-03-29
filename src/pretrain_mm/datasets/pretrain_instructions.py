from dataclasses import dataclass, fields


class PretrainTask:
    instruction: str

    _debug: bool = False

    def format(self, *args, **kwargs):
        # if you want to override the call() of the class use format()
        return self.instruction.format(*args, **kwargs)

    def __repr__(self):
        return self.instruction

    def __call__(self, *args, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return self.format(*args, **kwargs)

    def __class_getitem__(cls, task_name: str) -> "PretrainTask":
        for cls in cls.__subclasses__():
            if cls.__name__ == task_name:
                return cls
        raise KeyError(f"PretrainTask {task_name} not found")

    def _input_fields(self):
        # return the fields of the class that are used when generating prompt

        def _check_field(f):
            return f.name not in ["instruction", "_debug"] and not f.name.startswith("_")

        return [(f.name, f.type) for f in fields(self) if _check_field(f)]

    def __str__(self):
        if self._debug:
            return super().__str__()
        return self.__call__()


# can decorate instead
def make_instruction(cls, **kwargs):
    cls = type(cls.__name__, (cls, PretrainTask), {})
    return dataclass(cls)(**kwargs)


class PretrainHTML:
    def __init__(self):
        pass


@dataclass
class AssistantResponse(PretrainTask):
    # seems like maybe have 'OCR' in the instruction might help as using <point> in next action
    instruction: str = "Perform OCR for the following task: `{task}`."
    previous_actions_text: str = "\nPrevious Actions:{previous_actions}"
    # think i will generally not want to include the next_action, rather it should be added during
    # encoding so that i can mask out the other parts of instruction
    next_action_text: str = "\nNext Action: {next_action}"

    def format(self, task: str, previous_actions: str, next_action: str = "", strip_rpunc: bool = False, **kwargs):
        # strip last period
        if strip_rpunc and task[-1] in ["?", ".", "!"]:
            task = task[:-1]

        resp_str = self.instruction.format(task=task)

        if previous_actions != "":
            resp_str += self.previous_actions_text.format(previous_actions=previous_actions)

        resp_str += self.next_action_text.format(next_action=next_action)
        return resp_str


@dataclass
class GeneratePotentialActions(PretrainTask):
    instruction: str = "Generate the bounding box of potential actions for the screenshot."


@dataclass
class GenerateNumPotentialActions(PretrainTask):
    num_candidates: int = 1
    instruction: str = (
        "Generate the bounding box of {num_candidates} potential action for the page. Give the action text if relevant."  # noqa
    )


@dataclass
class BaselineBoxToText(PretrainTask):
    instruction: str = (
        "When presented with a box, perform OCR to extract text contained within it. "
        "If provided with text, generate the corresponding bounding box.\\n{box_str}"
        # e.g. <box>y1, x1, y2, x2</box>"
    )


@dataclass
class BaselineTextToBox(PretrainTask):
    instruction: str = (
        "When presented with a box, perform OCR to extract text contained within it. "
        "If provided with text, generate the corresponding bounding box.\\n {text_str}"
    )


if __name__ == "__main__":
    cls_type = PretrainTask["GenerateNumPotentialActions"]()
    print(" CLS TYPE: ", cls_type)
    print(" CLS CALL: ", cls_type(num_candidates=10))
