import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import IterableDataset

from pretrain_mm.datasets.s3_datasets import S3Dataset, S3Info

DEFAULT_NUM_IMAGES = 70_333_548  # from `cat $file wc -l`
DEFAULT_HEADERS = [  # or header.replace(" ", "").split(",")
    "site_url",
    "image_url",
    "created_at",
    "website_ip",
    "domain",
    "category",
    "size",
    "country_code",
    "page_lang",
    "harmonicc_pos",
    "pr_pos",
    "n_hosts",
    "ptile",
    "title",
    "description",
    "keywords",
]

CommonScreensS3Info = S3Info(bucket_name="common-screens")


@dataclass
class CommonScreensDatasetInfo:
    """common screens stuff

    https://registry.opendata.aws/comonscreens/

    image_dir: path of jpegs
    header_path: path of header file
    metadata_path: path of metadata file (e.g. the csv of data info )

    notes:

    the metadata csv file seems pretty fucked up/inconsistent. makes it extremely hard to parse and not
    even sure if its worth the effort :\

    """

    image_dir: str
    header_path: str
    metadata_path: str

    headers: list[str] = field(default_factory=lambda: DEFAULT_HEADERS)

    def __post_init__(self):
        if self.headers is None:
            self.headers = self._get_headers()

    def _get_headers(self):
        with open(self.header_path) as f:
            return f.read().split(", ")


class CommonScreensDataset(S3Dataset, IterableDataset):
    # https://registry.opendata.aws/comonscreens/
    #

    # to view image: https://dqh5x5k6xg3n1.cloudfront.net/[img_url]
    headers: list[str] = DEFAULT_HEADERS

    def __init__(self, dataset_info: CommonScreensDatasetInfo, filtered_csv: Optional[str] = None):
        self.dataset_info = dataset_info

    def __iter__(self):
        while True:
            pass
