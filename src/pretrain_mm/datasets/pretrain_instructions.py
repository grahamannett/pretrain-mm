from dataclasses import dataclass


class PretrainTask:
    instruction: str

    def __call__(self, *args, **kwargs):
        return self.instruction.format(*args, **kwargs)


class PretrainHTML:
    def __init__(self):
        pass


@dataclass
class AssistantResponse(PretrainTask):
    instruction = "You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"


# class GenerateNPotentialActions(PretrainTask):
@dataclass
class GenerateNPotentialActions:
    num_candidates: int
    instruction = f"Generate the bounding box of {num_candidates} potential actions for the screenshot. \n"


@dataclass
class GeneratePotentialActions(PretrainTask):
    instruction: str = "Generate the bounding box of {num_candidates} potential actions for the screenshot. Give the action text if relevant. \n"
