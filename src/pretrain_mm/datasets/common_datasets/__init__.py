import datasets
from torch.utils.data import Dataset


class InstructDataset(Dataset):
    def __init__(self, path: str, split: str = "train"):
        self.ds = datasets.load_dataset(path, split)[split]

    def __getitem__(self, idx):
        return self.ds[idx]
