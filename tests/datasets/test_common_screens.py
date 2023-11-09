import unittest

from pretrain_mm.datasets.common_screens import CommonScreensDatasetInfo, filter_csv


class TestCommonScreensDataset(unittest.TestCase):
    def test_parse_data(self):
        filtered = datasets.filter_csv(local_dev_config.metadata_path, local_dev_config.image_dir, max_lines=1000000)
        self.assertGreater(len(filtered), 0)


if __name__ == "__main__":
    local_dev_config = CommonScreensDatasetInfo(
        image_dir="/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/data/jpeg",
        header_path="/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/metadata/common-screens-with-meta-2022-header.txt",
        metadata_path="/Users/graham/code/clippymm_datasets/pretrain_data/common-screens/s3/metadata/common-screens-with-meta-2022-12.csv",
    )

    filtered = filter_csv(local_dev_config.metadata_path, local_dev_config.image_dir, max_lines=1000)

    with open("/Users/graham/code/clippymm_datasets/pretrain_data/filtered/common-screens-filtered.csv", "w") as f:
        f.writelines(filtered)
