from dataclasses import dataclass, field
from os import environ


@dataclass
class DatasetInfo:
    make: type
    sample: type
    tasks: type = None

    dataset_kwargs: dict = field(default=lambda: {})


def get_dataset_dir(env_var: str) -> str:
    return environ.get(env_var, None)
