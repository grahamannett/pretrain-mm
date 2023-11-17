from os import environ
from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    make: type
    sample: type
    tasks: type = None

    dataset_kwargs: dict = field(default=lambda: {})
