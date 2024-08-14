import unittest

from pretrain_mm.datasets.sampler.weighted_sampler import Dataset, WeightedStagedDataset


class TestWeightedStagedDataset(unittest.TestCase):
    def test_weighted_staged_dataset(self):
        class MockDS(Dataset):
            def __init__(self, name, len=5):
                self.name, self.len = name, len

            def __len__(self):
                return self.len

            def __getitem__(self, idx):
                return (idx, self.name)

        ds1 = MockDS("ds1", len=3)
        ds2 = MockDS("ds2", len=100)
        ds3 = MockDS("ds3", len=100)

        ds = WeightedStagedDataset(
            datasets={
                "ds1": ds1,
                "ds2": ds2,
                "ds3": ds3,
            },
            stages=[
                {"ds1": 9, "ds2": 5, "iters": 10},
                {"ds2": 0.5, "ds3": 0.5, "iters": 20},
                {"ds1": 0.5, "ds2": 0.5, "ds3": 0.5},
            ],
            return_info=True,
        )

        dsi = iter(ds)
        for i, s in enumerate(dsi):
            self.assertEqual(i, s[0])  # Check if the index is correct
            self.assertIn(s[1], ["ds1", "ds2", "ds3"])  # Check if the name is one of the expected values


if __name__ == "__main__":
    unittest.main()
