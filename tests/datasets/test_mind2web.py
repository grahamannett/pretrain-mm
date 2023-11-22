import os
import unittest

import torch
from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebConfig, task_mind2web
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm import logger

from config.fuyu import FuyuInfo
from config.dev import get_dev_config
from pretrain_mm.processor.fuyu.fuyu_processing import FuyuProcessor


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
            preprocessor=Mind2Web.task_preprocessor,
            postprocessor=Mind2Web.task_postprocessor,
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


class TestMind2Web(unittest.TestCase):
    def test_mind2web_samples(self):
        m2w_info = get_dev_config("mind2web")
        batch_size = os.environ.get("BATCH_SIZE", 2)
        num_workers = os.environ.get("N_WORKERS", 0)
        device = os.environ.get("DEVICE", "cuda")

        train_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["train"])
        test_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["test"])

        train_dataset = Mind2Web(train_data_config)
        test_dataset = Mind2Web(test_data_config)

        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

        # check that task adapter with processor is working
        task_train_dataset = TaskAdapterProcessor(
            train_dataset,
            task_func=task_mind2web,
            processor=processor,
            preprocessor=Mind2Web.task_preprocessor,
            postprocessor=Mind2Web.task_postprocessor,
        )

        task_test_dataset = TaskAdapterProcessor(
            test_dataset,
            task_func=task_mind2web,
            processor=processor,
            preprocessor=Mind2Web.task_preprocessor,
            postprocessor=Mind2Web.task_postprocessor,
        )

        with logger.progress() as progress:

            collate_fn = DataCollator(processor.pad_token_id, device="cuda", squeeze=(batch_size != 1))
            dl = torch.utils.data.DataLoader(task_train_dataset, batch_size=2, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=False)
            train_progress = progress.add_task("[red] check all items in the train set", total=len(dl))

            # go through train dataset
            for idx, batch in enumerate(dl):
                progress.update(train_progress, advance=1)
                batch.to(device)


            dl = torch.utils.data.DataLoader(task_test_dataset, batch_size=2, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=False)
            test_progress = progress.add_task("[red] check all items in the test set", total=len(dl))

            for idx, batch in enumerate(dl):
                progress.update(test_progress, advance=1)
                batch.to(device)

