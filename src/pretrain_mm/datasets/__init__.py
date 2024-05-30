from os import environ

from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor
from pretrain_mm.datasets.silatus.silatus_websites import SilatusWebsiteDataset, WebsiteSample
from pretrain_mm.datasets.task_adapter import TaskAdapter
from pretrain_mm.datasets.usable_tasks import WebsiteTasks
from pretrain_mm.datasets.utils.dataset_helpers import DatasetInitHelper


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
        dataset_kwargs={
            "data_dir": environ.get("MIND2WEB_DATA_DIR", None),
            "config_cls": Mind2WebConfig,
            "pretrain_processor_cls": Mind2WebPretrainProcessor,
        },
    ),
}


def get_dataset(dataset_name: str, dataset_kwargs: dict = {}):
    dataset_info = DatasetsAvailable[dataset_name]

    dataset_kwargs = {**dataset_info.dataset_kwargs, **dataset_kwargs}
    dataset = dataset_info.make(**dataset_kwargs)
    return dataset, dataset_info
