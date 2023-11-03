import os
import unittest

from pretrain_mm.datasets.silatus_websites import SilatusWebsiteDataset, WebsiteSample


class TestSilatusWebsiteDataset(unittest.TestCase):
    def test_dataset(self):
        data_dir = os.environ.get("SILATUS_DATA_DIR")
        dataset = SilatusWebsiteDataset(data_dir=data_dir)
        sample = dataset[0]

        self.assertIsInstance(sample, WebsiteSample)
        self.assertIsNotNone(sample.title)
        self.assertIsNotNone(sample.screenshot)

        self.assertTrue(len(dataset) > 9999)

        last_sample = dataset[-1]
        self.assertIsInstance(last_sample, WebsiteSample)


if __name__ == "__main__":
    unittest.main()
