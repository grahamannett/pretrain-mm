import unittest

import torch
from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebConfig, task_mind2web
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm import logger

from config.fuyu import FuyuInfo
from config.dev import get_dev_config


class TestMind2Web(unittest.TestCase):
    def test_mind2web(self):
        mind2web_info = get_dev_config("mind2web")
        train_config = Mind2WebConfig(task_dir=mind2web_info["task_dir"], subset=10, **mind2web_info["train"])
        train_dataset = Mind2Web(train_config)

        test_info = mind2web_info["test"]
        test_config = Mind2WebConfig(
            **test_info,
            task_dir=mind2web_info["task_dir"],
            subset=10,
        )
        test_dataset = Mind2Web(test_config)

        sample = train_dataset[0]
        test_sample = test_dataset[0]

        assert sample.action_uid != test_sample.action_uid

        sample = train_dataset[50]
        # check that task for this dataset is working
        sample_as_task = task_mind2web(sample)
        assert "label" in sample_as_task

        # check that task adapter with processor is working
        task_dataset = TaskAdapterProcessor(
            train_dataset,
            task_func=task_mind2web,
            processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name),
            preprocessor=Mind2Web.task_preprocessor,  # this converts to just text and images, probably should be done in task_func
            postprocessor=Mind2Web.task_postprocessor,  # this is needed as Fuyu processor returns tensors with batch dim already so messes up dataloader
        )

        task_sample = task_dataset[50]

        assert "input_ids" in task_sample

        assert task_sample["input_ids"].ndim == 1
        assert "attention_mask" in task_sample

        assert "image_patches" in task_sample
        image_patches = task_sample["image_patches"]
        assert "image_patches_indices" in task_sample

        dataloader = torch.utils.data.DataLoader(
            task_dataset, batch_size=2, collate_fn=DataCollator()
        )  # , pin_memory=True)

        # batch = next(iter(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            logger.log(batch)

            if batch_idx > 2:
                break
