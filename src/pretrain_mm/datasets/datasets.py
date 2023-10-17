import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import IterableDataset
from tqdm import tqdm

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


def split_csv(csv_file: str, out_dir: str, max_rows: int = 10_000_000) -> None:
    """split csv file into multiple csv files with max_rows rows"""
    _out_idx = 0

    def new_file() -> tuple[open, csv.writer]:
        nonlocal _out_idx
        print(f"creating new file {out_dir}/{_out_idx}.csv")
        outfile = open(f"{out_dir}/{_out_idx}.csv", "w")
        _out_idx += 1
        return outfile, csv.writer(outfile)

    out_file, csv_writer = new_file()
    with open(csv_file, "r") as f:
        csv_reader = csv.reader(f)
        # total_rows = sum(1 for row in csv_reader)
        for i, row in enumerate(csv_reader):
            if (i + 1) % max_rows == 0:
                out_file.close()
                out_file, csv_writer = new_file()
            csv_writer.writerow(row)
        out_file.close()


def clean_image_url(image_url: str, remove_ext: bool = False):
    image_url = image_url.split("/")[-1]
    if remove_ext:
        image_url = image_url.split(".")[0]
    return image_url


def filter_csv(
    csv_file: str, image_dir: str, fieldnames=DEFAULT_HEADERS, max_lines: int = None, skip_lines: list[str] = None
):
    """since there are like 70m images, we filter out those that we do not have

    Args:
        csv_file (str): _description_
        image_dir (str): _description_
    """
    image_dir = Path(image_dir)
    filtered = []

    with open(csv_file, "r", encoding="utf-8") as f:
        # read each line without csv reader
        for i in tqdm(range(DEFAULT_NUM_IMAGES)):
            line = next(f)
            if skip_lines and any([skip_line in line for skip_line in skip_lines]):
                continue
            line_split = line.split('","')
            if (len(line_split) < 2) or ("ipv6.systems" not in line_split[1]):
                continue

            try:
                row_image_url = clean_image_url(line_split[1])
                if (image_dir / row_image_url).exists():
                    filtered.append(line)
            except:
                breakpoint()

            if max_lines and i > max_lines:
                break

    return filtered


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
