import unittest

import torch

from transformers import AutoTokenizer, AutoProcessor

from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets import DatasetsAvailable, TaskAdapter, TaskAdapterProcessor

from tests.fixtures.processors import mm_processor, text_only_tokenizer


# TODO: Refactor

dataset_id = "silatus_websites"
dataset_info = DatasetsAvailable[dataset_id]
dataset_data_dir = dataset_info.dataset_kwargs["data_dir"]
dataset_tasks = dataset_info.tasks
base_dataset = dataset_info.make(dataset_data_dir)


class TestDataFuncs(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_task_adapter(self):
        """
        test that a task adapter converts a dataset sample to a task
        """
        dataset = TaskAdapter(base_dataset, dataset_tasks.TitleWebsiteTask())

        # test that the task adapter converts dataset sample to a task ("title this website")
        sample = dataset[0]
        self.assertTrue(isinstance(sample.text, str))

    def test_task_adapter_processor(self):
        """test that a task adapter processor converts a dataset sample to a task and processes it"""
        dataset = TaskAdapterProcessor(base_dataset, dataset_tasks.TitleWebsiteTask(), processor=mm_processor)
        sample = dataset[0]

        self.assertTrue(isinstance(sample["input_ids"], torch.Tensor))
        self.assertTrue(isinstance(sample["image_patches"][0], torch.Tensor))

    def test_data_collator(self):
        dataset = TaskAdapter(base_dataset, dataset_tasks.TitleWebsiteTask())
        samples = [dataset[i] for i in range(5)]

        # collator = DataCollator(text_tokenizer=text_only_tokenizer)

        # batch = collator(samples)
        # self.assertTrue(isinstance(batch["input_ids"], torch.Tensor))


class TestInterleavedModalities(unittest.TestCase):
    def test_trajectory(self):
        pass


if __name__ == "__main__":
    unittest.main()
