from typing import Any
import unittest

import torch

from pretrain_mm.datasets.dataloader import DataCollator

from pretrain_mm.datasets.silatus.silatus_websites import SilatusWebsiteDataset, WebsiteSample
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.datasets import DatasetsAvailable
from pretrain_mm.processesor.post_processor import fuyu_post_processor

from tests.fixtures.processors import mm_processor

dataset_info = DatasetsAvailable["silatus_websites"]


class TestSilatusWebsiteDataset(unittest.TestCase):
    def test_dataset(self):
        data_dir = dataset_info.dataset_kwargs["data_dir"]
        dataset = SilatusWebsiteDataset(data_dir=data_dir)

        sample = dataset[0]

        self.assertIsInstance(sample, WebsiteSample)
        self.assertIsNotNone(sample.title)
        self.assertIsNotNone(sample.image)

        self.assertTrue(len(dataset) > 9999)

        last_sample = dataset[-1]
        self.assertIsInstance(last_sample, WebsiteSample)

        task_dataset = TaskAdapterProcessor(dataset, processor=mm_processor, post_processor=fuyu_post_processor)

        sample = task_dataset[0]

        self.assertIsInstance(sample.input_ids, torch.Tensor)
        self.assertTrue(sample.image_patches[0].ndim == 2)

        dataloader = torch.utils.data.DataLoader(
            task_dataset, batch_size=2, collate_fn=DataCollator(pad_token_id=mm_processor.pad_token_id)
        )

        batch = next(iter(dataloader))


if __name__ == "__main__":
    unittest.main()
