from dataclasses import dataclass


class PretrainTask:
    instruction: str

    _debug: bool = False

    def __call__(self, *args, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return self.instruction.format(*args, **kwargs)

    def __class_getitem__(cls, task_name: str) -> "PretrainTask":
        for cls in cls.__subclasses__():
            if cls.__name__ == task_name:
                return cls
        raise KeyError(f"PretrainTask {task_name} not found")

    def __str__(self):
        if self._debug:
            return super().__str__()
        return self.__call__()


class PretrainHTML:
    def __init__(self):
        pass


@dataclass
class AssistantResponse(PretrainTask):
    instruction = "You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"


@dataclass
class GenerateNumPotentialActions(PretrainTask):
    num_candidates: int
    instruction = "Generate the bounding box of {num_candidates} potential actions for the screenshot. Give the action text if relevant."


@dataclass
class GeneratePotentialActions(PretrainTask):
    instruction: str = "Generate the bounding box of {num_candidates} potential actions for the screenshot."


if __name__ == "__main__":
    cls_type = PretrainTask["GenerateNumPotentialActions"](num_candidates=3)
    print(cls_type)
    print(cls_type(num_candidates=10))
