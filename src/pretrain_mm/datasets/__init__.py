from os import environ

from .dataset_utils import DatasetInfo
from .silatus.silatus_websites import SilatusWebsiteDataset, WebsiteSample
from .task_adapter import TaskAdapter, TaskAdapterProcessor, WebsiteTasks
from .mind2web import Mind2Web, Mind2WebConfig, task_mind2web

DatasetsAvailable = {
    "silatus_websites": DatasetInfo(
        make=SilatusWebsiteDataset,
        sample=WebsiteSample,
        tasks=WebsiteTasks,
        dataset_kwargs={
            "data_dir": environ.get("SILATUS_DATA_DIR", None),
        },
    ),
    "mind2web": DatasetInfo(
        make=Mind2Web,
        task=task_mind2web,
        dataset_kwargs={
            "data_dir": environ.get("MIND2WEB_DATA_DIR", None),
        },
    ),
}


def get_dataset(dataset_name: str, dataset_kwargs: dict = {}):
    dataset_info = DatasetsAvailable[dataset_name]

    dataset_kwargs = {**dataset_info.dataset_kwargs, **dataset_kwargs}
    dataset = dataset_info.make(**dataset_kwargs)
    return dataset, dataset_info
