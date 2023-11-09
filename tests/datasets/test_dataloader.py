import unittest

import torch

from transformers import AutoTokenizer, AutoProcessor

from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets import DatasetsAvailable, TaskAdapter, TaskAdapterProcessor

# TODO: Refactor

dataset_id = "silatus_websites"
dataset_info = DatasetsAvailable[dataset_id]
dataset_data_dir = dataset_info.dataset_kwargs["data_dir"]
dataset_tasks = dataset_info.tasks
base_dataset = dataset_info.make(dataset_data_dir)

text_only_tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    model_max_length=512,
    padding_side="left",
    add_eos_token=True,
)

mm_processor = AutoProcessor.from_pretrained(
    "adept/fuyu-8b",
    model_max_length=4096,
)


class TestDataFuncs(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_task_adapter(self):
        dataset = TaskAdapter(base_dataset, dataset_tasks.TitleWebsiteTask())

        # dont tokenize or process anything at this point
        sample = dataset[0]
        self.assertTrue(isinstance(sample["input_ids"], str))

    def test_task_adapter_processor(self):
        dataset = TaskAdapterProcessor(base_dataset, dataset_tasks.TitleWebsiteTask(), processor=mm_processor)
        sample = dataset[0]

        self.assertTrue(isinstance(sample["input_ids"], torch.Tensor))
        self.assertTrue(isinstance(sample["image_patches"][0], torch.Tensor))

    def test_data_collator(self):
        dataset = TaskAdapter(base_dataset, dataset_tasks.TitleWebsiteTask())
        samples = [dataset[i] for i in range(5)]

        collator = DataCollator(text_only_tokenizer)

        batch = collator(samples)
        self.assertTrue(isinstance(batch["input_ids"], torch.Tensor))


if __name__ == "__main__":
    unittest.main()
