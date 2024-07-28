from torch.utils.data import Dataset


class TextInstructionDataset(Dataset):
    pass
    # for datasets like
    # def __init__(self):
    #     pass


class DollyDataset(Dataset):
    _dataset_name = "mosaicml/dolly_hhrlhf"
