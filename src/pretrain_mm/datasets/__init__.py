from dataclasses import dataclass, field
from os import environ

from .silatus_websites import SilatusWebsiteDataset, WebsiteSample
from .task_adapter import TaskAdapter, WebsiteTasks


def get_dataset_dir(env_var: str) -> str:
    dataset_dir = environ.get(env_var)
    if dataset_dir is None:
        raise ValueError(f"Please set the {env_var} environment variable")
    return dataset_dir


@dataclass
class DatasetInfo:
    make: type
    sample: type
    tasks: type = None

    dataset_kwargs: dict = field(default=lambda: {})


DatasetsAvailable = {
    "silatus_websites": DatasetInfo(
        make=SilatusWebsiteDataset,
        sample=WebsiteSample,
        tasks=WebsiteTasks,
        dataset_kwargs={
            "data_dir": get_dataset_dir("SILATUS_DATA_DIR"),
        },
    ),
}
