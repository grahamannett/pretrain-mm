import random

import torch

# Data Types to Mock
mock_text_sample_info = {
    "input_ids": {
        "shape": (
            1,
            {"min": 100, "max": 200},
        ),
        "range": (0, 1000),
        "dtype": int,
    },
}

mock_image_sample_info = {
    "images": {
        "shape": (
            1,
            3,
            {"min": 500, "max": 700},
            {"min": 500, "max": 700},
        ),
        "dtype": float,
    },
}


def get_shape(s) -> int:
    if isinstance(s, int):
        return s
    elif isinstance(s, dict):
        low, high = s["min"], s["max"]
    elif isinstance(s, tuple):
        low, high = s
    return random.randint(low, high)


def make_random_int_tensor(shape: tuple, range_: tuple[int, int] = (0, 100), dtype=int) -> torch.Tensor:
    return torch.randint(*range_, size=shape, dtype=dtype)


def make_random_float_tensor(shape: tuple, dtype=float, **kwargs) -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype)


make_random = {
    int: make_random_int_tensor,
    float: make_random_float_tensor,
}


class MockDataset:
    def __init__(
        self,
        len: int = 100,
        sample_out: dict = {**mock_text_sample_info, **mock_image_sample_info},
    ):
        self.len = len
        self.sample_out = sample_out

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int):
        out = {}

        for key, modal_info in self.sample_out.items():
            dtype = modal_info.get("dtype", float)
            out[key] = make_random[dtype](
                [get_shape(s) for s in modal_info["shape"]],
                range_=modal_info.get("range"),
                dtype=dtype,
            )

        return out


class MockDatasetWithText(torch.utils.data.Dataset):
    def __init__(
        self,
        # available on linux/mac
        words_file: str = "/usr/share/dict/words",
        max_words: int = 100,
        len: int = 100,
        # allow variable sized images
        image_max: tuple[int, int] = [1000, 1000],
        image_min: tuple[int, int] = [500, 500],
    ):
        self.len = len
        self.max_words = max_words
        self.image_min, self.image_max = image_min, image_max

        with open(words_file, "r") as f:
            self.words = [word.strip() for word in f.readlines()]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.generate()

    def generate(self):
        num_words = random.randint(1, self.max_words)
        text = " ".join([random.choice(self.words) for _ in range(num_words)])
        image_dim = [
            random.randint(self.image_min[0], self.image_max[0]),
            random.randint(self.image_min[1], self.image_max[1]),
        ]
        image = torch.rand(3, *image_dim)
        return {"text": text, "image": image}


if __name__ == "__main__":
    dataset = MockDataset()
    sample = dataset[0]
