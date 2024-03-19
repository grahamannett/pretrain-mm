from os import environ

from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebEncoder
from pretrain_mm.datasets.silatus.silatus_websites import SilatusWebsiteDataset, WebsiteSample
from pretrain_mm.datasets.task_adapter import WebsiteTasks
from pretrain_mm.datasets.utils.dataset_utils import DatasetInitHelper


DatasetsAvailable = {
    "silatus_websites": DatasetInitHelper(
        make=SilatusWebsiteDataset,
        sample=WebsiteSample,
        tasks=WebsiteTasks,
        dataset_kwargs={
            "data_dir": environ.get("SILATUS_DATA_DIR", None),
        },
    ),
    "mind2web": DatasetInitHelper(
        make=Mind2Web,
        encoder=Mind2WebEncoder,
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
