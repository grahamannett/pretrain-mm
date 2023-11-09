from dataclasses import dataclass, field
from os import environ


@dataclass
class DatasetInfo:
    make: type
    sample: type
    tasks: type = None

    dataset_kwargs: dict = field(default=lambda: {})


def get_dataset_dir(env_var: str) -> str:
    dataset_dir = environ.get(env_var)
    if dataset_dir is None:
        raise ValueError(f"Please set the {env_var} environment variable")
    return dataset_dir
