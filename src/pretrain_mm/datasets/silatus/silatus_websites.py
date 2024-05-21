import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from pretrain_mm.datasets.base import create_sample_type


def change_dict_key(d: dict, old_key: str, new_key: str) -> dict:
    """change a key in a dict"""
    d[new_key] = d.pop(old_key)
    return d


@dataclass
class LocationData:
    x: int
    y: int


@dataclass
class SizeData:
    height: int
    width: int


@dataclass
class PositionData:
    horizontal: int
    vertical: int


class CommonBaseFields:
    def __post_init__(self):
        self.location = LocationData(**self.location)
        self.size = SizeData(**self.size)
        self.position = PositionData(**self.position)


@dataclass
class ImageData(CommonBaseFields):
    alt: str
    location: LocationData
    size: SizeData
    position: PositionData
    is_displayed: bool


@dataclass
class ButtonData(CommonBaseFields):
    text: str
    bg_color: str
    location: LocationData
    size: SizeData
    position: PositionData
    is_displayed: bool


@dataclass
class InputData(CommonBaseFields):
    desc: str
    type: str
    location: LocationData
    size: SizeData
    position: PositionData
    is_displayed: bool


@create_sample_type
class WebsiteSample:
    image: torch.Tensor | str | Image.Image
    title: str
    url: str
    full_url: str
    colors: List[Tuple[int, int, int]]

    # i think everything below here is optional/missing from some samples
    # these first three are lists that contain dicts and i have as dataclasses
    # they are dataclasses to allow easier task creation/pretraining on them for variable task
    images: List[ImageData] = None
    inputs: List[InputData] = None
    buttons: List[ButtonData] = None
    desc: str = None
    navbar: str = None
    text: str = None
    font: str = None
    iframes: List = None

    _folder_path: Path | str = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.images:
            self.images = [ImageData(**image) for image in self.images]
        if self.inputs:
            self.inputs = [InputData(**input) for input in self.inputs]
        if self.buttons:
            self.buttons = [ButtonData(**change_dict_key(button, "bg-color", "bg_color")) for button in self.buttons]


class SilatusWebsiteDataset(Dataset):
    """
    initially found here:
    https://huggingface.co/datasets/silatus/1k_Website_Screenshots_and_Metadata
    but that doesnt have metadata so got: https://silatus.com/datasets/

    Args:
        Dataset (_type_): scrapped websites with some metadata and a screenshot
    """

    __name__ = "SilatusWebsiteDataset"

    folders: List[Path]

    def __init__(
        self,
        data_dir: str,
        read_image_fn: Callable = lambda img: Image.open(img).convert("RGB"),  # lambda img: read_image(str(img)),
        verify_data: bool = False,
        include_folder_path: bool = False,
    ):
        self.data_dir = data_dir
        self.include_folder_path = include_folder_path
        self._read_image_fn = read_image_fn

        self.folders = self._get_folders(verify_image_exists=verify_data)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx: int) -> WebsiteSample:
        return self._get_item(self.folders[idx])

    def _get_item(self, folder: Path) -> WebsiteSample:
        metadata_file = folder / "metadata.json"
        screenshot_file = folder / "screenshot.png"

        with open(metadata_file, "r") as f:
            data = json.load(f)

        screenshot = self._read_image_fn(screenshot_file)
        sample = WebsiteSample(image=screenshot, **data)

        if self.include_folder_path:
            sample._folder_path = str(folder)

        return sample

    def _get_folders(self, verify_image_exists: bool = False) -> List[Path]:
        folders = []
        for metadata_file in Path(self.data_dir).rglob("*.json"):
            folder = metadata_file.parent
            if verify_image_exists:
                if not (folder / "screenshot.png").exists():
                    continue
            folders.append(folder)

        return folders
