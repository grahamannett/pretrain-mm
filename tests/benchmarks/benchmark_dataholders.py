import dataclasses
import timeit
from typing import NamedTuple, TypedDict

from pretrain_mm.datasets.pretrain_instructions import GenerateNumPotentialActions, InstructionInstances


# Simple benchmark showing that the speed on all of these is ~same.
# TypedDict seems marginally faster but does not allow any of the dataclasses functionality.

PretrainTaskInstance = InstructionInstances.generate_num_potential_actions
PretrainTask = GenerateNumPotentialActions


@dataclasses.dataclass
class TaskDC:
    instruction: str = PretrainTask.instruction
    num_candidates: int = PretrainTask.num_candidates

    def format(self):
        return self.instruction.format(num_candidates=self.num_candidates)


class TaskTuple(NamedTuple):
    instruction: str = PretrainTask.instruction
    num_candidates: int = PretrainTask.num_candidates

    def format(self):
        return self.instruction.format(num_candidates=self.num_candidates)


class TaskDict(TypedDict):
    instruction: str = PretrainTask.instruction
    num_candidates: int = PretrainTask.num_candidates


def format_dict(inst: TaskDict):
    return inst["instruction"].format(num_candidates=inst["num_candidates"])


def benchmark(cls, iterations, method):
    start = timeit.default_timer()
    for i in range(iterations):
        instance = cls(num_candidates=i, instruction=PretrainTask.instruction)
        _ = method(instance)
    end = timeit.default_timer()
    return end - start


def benchmark_dict(cls, iterations, method):
    start = timeit.default_timer()
    for i in range(iterations):
        instance = cls(num_candidates=i, instruction=PretrainTask.instruction)
        _ = method(instance)
    end = timeit.default_timer()
    return end - start


iterations = 1_000_000

print(f"PretrainTask: {benchmark(TaskDC, iterations, lambda inst: inst.format())}")
print(f"PretrainTaskTuple: {benchmark(TaskTuple, iterations, lambda inst: inst.format())}")
print(f"PretrainTaskDict: {benchmark(TaskDict, iterations, format_dict)}")
