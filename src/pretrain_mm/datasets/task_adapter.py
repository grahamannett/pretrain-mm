from torch.utils.data import Dataset


class TaskAdapter(Dataset):
    def __init__(self, dataset: Dataset, transforms: dict[str, callable] | list[callable] = None, **kwargs) -> None:
        """
        Initializes a TaskAdapter object.

        Args:
            dataset (Dataset): The original dataset to be adapted.
            transforms (dict[str, callable] | list[callable], optional): The transforms to be applied to the dataset. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dataset = dataset

        self._setup_transforms(transforms)

    def __len__(self):
        """
        Returns the length of the adapted dataset.

        Args:
            None

        Returns:
            int: The length of the dataset.

        Raises:
            None
        """
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Retrieves an item from the adapted dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The retrieved item.

        Raises:
            None
        """
        return self.call_transforms(self.dataset[idx], transforms=self.transforms)

    def __repr__(self) -> str:
        """
        Returns a string representation of the TaskAdapter object.

        Args:
            None

        Returns:
            str: The string representation of the object.

        Raises:
            None
        """
        name = getattr(self.dataset, "__name__", self.dataset.__class__.__name__)
        dataset_info = f"TaskAdapter(\n\tdataset:={name},"
        for t_name, t_func in self.transforms.items():
            dataset_info += f"\n\t{t_name}:={t_func.__name__},"
        return dataset_info + "\n)"

    def _setup_transforms(self, transforms: dict[str, callable] | list[callable]) -> dict[str, callable]:
        """
        Sets up the transforms for the dataset.

        Args:
            transforms (dict[str, callable] | list[callable]): The transforms to be applied to the dataset.

        Returns:
            dict[str, callable]: The dictionary of transforms.

        Raises:
            None
        """
        if not isinstance(transforms, (list, dict)):
            transforms = [transforms]

        if isinstance(transforms, list):
            transforms = {f"{idx}_{fn.__name__}": fn for idx, fn in enumerate(transforms)}

        self.transforms = transforms

    def call_transforms(
        self, sample: dict, func_kwargs: list[dict] = None, transforms: dict[str, callable] = None
    ) -> dict:
        """
        Call all transforms on the given sample.

        Args:
            sample (dict): The input sample to apply transforms on.
            func_kwargs (list[dict], optional): A list of keyword arguments for each transform function. Defaults to None.
            transforms (dict[str, callable], optional): The transforms to be applied to the sample. Defaults to None.

        Returns:
            dict: The transformed sample.

        Raises:
            None
        """
        transforms = transforms or self.transforms
        for fn_idx, (fn_name, fn) in enumerate(transforms.items()):
            fn_kwargs = func_kwargs[fn_idx] if func_kwargs else {}
            sample = fn(sample, **fn_kwargs)

            # if the sample return from func is None/False, break
            # this then is handled in collate func
            if not sample:
                break
        return sample

    def _handle_func(self, sample: dict, func: callable, func_name: str = "unknown", func_kwargs: dict = {}) -> dict:
        """
        Handles the execution of a function on a sample.

        Args:
            sample (dict): The input sample to apply the function on.
            func (callable): The function to be applied.
            func_name (str, optional): The name of the function. Defaults to "unknown".
            func_kwargs (dict, optional): Additional keyword arguments for the function. Defaults to {}.

        Returns:
            dict: The transformed sample.

        Raises:
            SystemExit: If there is an issue executing the function on the sample.
        """
        try:
            return func(sample, **func_kwargs)
        except Exception as err:
            raise SystemExit(f"Issue for {func_name} on sample: {sample}|{func_kwargs} with Error: {err}")


class FeedbackDatasetAdapter(Dataset):
    def __init__(
        self,
    ) -> None:
        super().__init__()
