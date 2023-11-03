import unittest

import torch


from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets import DatasetsAvailable, TaskAdapter


class TestDataFuncs(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_task_adapter(self):
        dataset_id = "silatus_websites"
        dataset_info = DatasetsAvailable[dataset_id]
        dataset_data_dir = dataset_info.dataset_kwargs["data_dir"]
        dataset_tasks = dataset_info.tasks
        base_dataset = dataset_info.make(dataset_data_dir)

        dataset = TaskAdapter(base_dataset, dataset_tasks.task_func)


if __name__ == "__main__":
    unittest.main()
