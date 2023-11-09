from .silatus_websites import SilatusWebsiteDataset, WebsiteSample
from .task_adapter import TaskAdapter, TaskAdapterProcessor, WebsiteTasks
from .dataset_utils import get_dataset_dir, DatasetInfo


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


def get_dataset(dataset_name: str, dataset_kwargs: dict):
    dataset_info = DatasetsAvailable[dataset_name]

    dataset = dataset_info["dataset"](**dataset_kwargs)
    return dataset
