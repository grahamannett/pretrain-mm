import csv
from pathlib import Path

from .common_screens_dataset import CommonScreensDataset, CommonScreensDatasetInfo, DEFAULT_HEADERS, DEFAULT_NUM_IMAGES
from tqdm import tqdm


# funcs related to common screens dataset
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
