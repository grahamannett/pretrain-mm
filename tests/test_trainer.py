import unittest

import torch
import transformers

from pretrain_mm.trainer import LoraDPOTrainer
from pretrain_mm.utils.lora_utils import BaseLoraConfig
from pretrain_mm.datasets.task_adapter import TaskAdapter
from tests.mock.dataset import MockDatasetWithText
from tests.mock.model import MockModel

# from pretrain_mm.datasets.dataloader import DataCollator


def collate_fn(items):
    BatchCls = items[0].__class__
    pad_token_id = 0
    data = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"][0] for item in items], batch_first=True, padding_value=pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"][0] for item in items], batch_first=True, padding_value=pad_token_id
        ),
        "image_patches": [item["image_patches"][0] for item in items],
        "image_patches_indices": torch.nn.utils.rnn.pad_sequence(
            [item["image_patches_indices"][0] for item in items], batch_first=True, padding_value=pad_token_id
        ),
    }
    data["labels"] = data["input_ids"].clone()

    return BatchCls(data=data)


def make_transform_func(processor):
    def func(sample: dict):
        return processor(text=sample["text"], images=sample["image"])

    return func


class TestLoraDPO(unittest.TestCase):
    def test_lora_dpo(self):
        model = MockModel()
        dataset = MockDatasetWithText()
        processor = transformers.AutoProcessor.from_pretrained("adept/fuyu-8b")

        # model.resize_token_embeddings(len(processor.tokenizer))

        task_dataset = TaskAdapter(dataset, transforms=[make_transform_func(processor)])
        trainer = LoraDPOTrainer()

        lora_config = BaseLoraConfig(
            enabled=True, target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        trainer.setup_train(lora_config, model, optimizer, scheduler)

        dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=1, collate_fn=collate_fn)

        batch = next(iter(dataloader))

        trainer.train(dataloader)
