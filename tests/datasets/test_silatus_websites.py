import os
import unittest

from pretrain_mm.datasets.silatus_websites import SilatusWebsiteDataset, WebsiteSample
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor

from tests.fixtures.processors import mm_processor


class TestSilatusWebsiteDataset(unittest.TestCase):
    def test_dataset(self):
        data_dir = os.environ.get("SILATUS_DATA_DIR")
        dataset = SilatusWebsiteDataset(data_dir=data_dir)

        sample = dataset[0]

        self.assertIsInstance(sample, WebsiteSample)
        self.assertIsNotNone(sample.title)
        self.assertIsNotNone(sample.image)

        self.assertTrue(len(dataset) > 9999)

        last_sample = dataset[-1]
        self.assertIsInstance(last_sample, WebsiteSample)

        task_dataset = TaskAdapterProcessor(dataset, processor=mm_processor)

        sample = task_dataset[0]
        


if __name__ == "__main__":
    unittest.main()
