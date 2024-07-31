from pretrain_mm.utils.image_utils import convert_normalized_to_pixel_coords


def screen_transform(batch):
    bbox = convert_normalized_to_pixel_coords(batch["bbox"][0])
    return {
        "image": batch["image"],
        "bbox": [bbox],
        "text": batch["instruction"],
    }


class RicoDatasetConfig:
    path: str
    split: str = "train"

    def __init__(self, **kwargs):
        for k in self.__annotations__.keys():
            if not hasattr(self, k):
                setattr(self, k, None)
        self.__dict__.update(kwargs)

    def load_kwargs(self):
        return self.__dict__


# not sure which i prefer, either dict or class
rico_datasets = {
    "ScreenSpot": {
        "load_kwargs": {"path": "rootsautomation/ScreenSpot", "split": "test"},
        "transform": screen_transform,
    },
}


class ScreenSpotConf(RicoDatasetConfig):
    path = "rootsautomation/ScreenSpot"
    split = "test"

    def transform(batch):
        return screen_transform(batch)


# MAIN FOCUS IS SCREENSPOT DATASET.  NOT ACTUALLY RICO DATASET
# https://huggingface.co/datasets/rootsautomation/ScreenSpot
ScreenSpotConfig = RicoDatasetConfig(path="rootsautomation/ScreenSpot", split="test")


# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA
ScreenQAConfig = RicoDatasetConfig(path="rootsautomation/RICO-ScreenQA")
# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Complex
ScreenQAComplexConfig = RicoDatasetConfig(path="rootsautomation/RICO-ScreenQA-Complex")
# https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short
ScreenQAShortConfig = RicoDatasetConfig(path="rootsautomation/RICO-ScreenQA-Short")

# https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words
Screen2WordsConfig = RicoDatasetConfig(path="rootsautomation/RICO-Screen2Words")

# https://huggingface.co/datasets/rootsautomation/RICO-SCA
SCAConfig = RicoDatasetConfig(path="rootsautomation/RICO-SCA")
