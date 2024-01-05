import os
import time
import unittest

import torch
from datasets import disable_caching

from config.dev import get_dev_config
from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebBase, Mind2WebConfig, Mind2WebTaskProcessor
from pretrain_mm.datasets.mind2web.mind2web import Mind2WebIterable
from pretrain_mm.datasets.task_adapter import TaskAdapter
from pretrain_mm.model.fuyu import FuyuProcessor
from pretrain_mm.utils.testing_utils import TimerMixin

disable_caching()

m2w_info = get_dev_config("mind2web")
task_dir = m2w_info.get("task_dir")

# env args
subset = int(os.environ.get("SUBSET", 10))
batch_size = int(os.environ.get("BATCH_SIZE", 2))
num_workers = int(os.environ.get("N_WORKERS", 0))
device = os.environ.get("DEVICE", "cuda")


class TestMind2Web(unittest.TestCase):
    def test_mind2web(self):
        train_config = Mind2WebConfig(task_dir=task_dir, subset=10, **m2w_info["train"])
        train_dataset = Mind2Web(train_config)

        test_info = m2w_info["test"]
        test_config = Mind2WebConfig(**test_info, task_dir=task_dir, subset=subset)
        test_dataset = Mind2Web(test_config)

        sample = train_dataset[0]
        test_sample = test_dataset[0]
        assert sample.action_uid != test_sample.action_uid

        sample = train_dataset[50]
        # check that task for this dataset is working
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
        task_processor = Mind2WebTaskProcessor(
            processor=processor,
            ignore_index=train_config.IGNORE_INDEX,
            loc_before_action_repr=False,
        )
        sample_as_task = task_processor.task_mind2web(sample)
        assert "label" in sample_as_task

        # check that task adapter with processor is working

        task_transforms = {
            "task_func": task_processor.task_mind2web,
            "processor": task_processor.process_func,
            "postprocessor": Mind2WebTaskProcessor.postprocessor,
        }

        task_dataset = TaskAdapter(train_dataset, transforms=task_transforms)

        task_sample = task_dataset[50]

        assert "input_ids" in task_sample

        assert task_sample["input_ids"].ndim == 1
        assert "attention_mask" in task_sample

        assert "image_patches" in task_sample
        image_patches = task_sample["image_patches"]
        assert "image_patches_indices" in task_sample

        dataloader = torch.utils.data.DataLoader(
            task_dataset,
            batch_size=2,
            collate_fn=DataCollator(processor.pad_token_id, squeeze=False, include_labels=True),
        )

        # batch = next(iter(dataloader))
        for batch_idx, batch in enumerate(dataloader):
            logger.log(batch)

            if batch_idx > 2:
                break

    def test_transforms(self):
        train_config = Mind2WebConfig(task_dir=task_dir, subset=10, **m2w_info["train"])
        train_dataset = Mind2Web(train_config)

        sample = train_dataset[50]
        # check that task for this dataset is working
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
        task_processor = Mind2WebTaskProcessor(
            processor=processor,
            ignore_index=train_config.IGNORE_INDEX,
            loc_before_action_repr=False,
        )
        sample_as_task = task_processor.task_mind2web(sample)
        breakpoint()

        # check that task adapter with processor is working

        task_transforms = {
            "task_func": task_processor.task_mind2web,
            "processor": task_processor.process_func,
            "postprocessor": Mind2WebTaskProcessor.postprocessor,
        }

        task_dataset = TaskAdapter(train_dataset, transforms=task_transforms)

        task_sample = task_dataset[50]


class TestMind2WebIterable(unittest.TestCase):
    def test_train_dataset(self):
        train_config = Mind2WebConfig(task_dir=task_dir, subset=10, **m2w_info["train"])
        dataset = Mind2WebIterable(train_config)
        sample = dataset[0]

    def test_test_dataset(self):
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
        test_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["test"])
        dataset = Mind2WebIterable(test_config)


class TestFlatten(TimerMixin, unittest.TestCase):
    def test_flatten(self):
        data_config = Mind2WebConfig(task_dir=task_dir, subset=subset, **m2w_info["train"])
        data_config.map_num_workers = num_workers
        data_config.map_load_from_cache_file = False
        dataset = Mind2Web(data_config)
        # self.check_timer("made dataset just from indexes")
        self.check_timer(extra_print="Flatten Timer")


@unittest.skip("skip-until-fix")
class TestSamples(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = int(os.environ.get("BATCH_SIZE", 2))
        self.num_workers = int(os.environ.get("N_WORKERS", 0))
        self.device = os.environ.get("DEVICE", "cuda")

        disable_progress = os.environ.get("DISABLE_PROGRESS", False)
        if disable_progress in ["True", "true", "1"]:
            disable_progress = True
        else:
            disable_progress = False
        self.disable_progress = disable_progress

        return super().setUp()

    def test_check_train(self):
        batch_size, num_workers, device = self.batch_size, self.num_workers, self.device
        disable_progress = self.disable_progress
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
        train_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["train"])
        train_data_config.map_num_workers = num_workers

        train_dataset = Mind2Web(train_data_config)

        # check that task adapter with processor is working
        task_train_dataset = TaskAdapter(
            train_dataset,
            {
                "task_func": task_mind2web,
                "preprocessor": Mind2WebTaskProcessor.preprocessor,
                "processor": processor,
                "postprocessor": Mind2WebTaskProcessor.postprocessor,
            },
        )

        _ = task_train_dataset[559]  # known bad - no before image
        _ = task_train_dataset[1322]  # known bad - if indexes filtered this will

        logger.info(f"==> train_dataset len: {len(train_dataset)}")
        with logger.progress(disable=disable_progress) as progress:
            collate_fn = DataCollator(processor.pad_token_id, squeeze=(batch_size != 1))
            dl = torch.utils.data.DataLoader(
                task_train_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )
            train_progress = progress.add_task(f"[red] ✅ train set w/ batch_size: {batch_size}", total=len(dl))

            # go through train dataset
            for idx, batch in enumerate(dl):
                progress.update(train_progress, advance=1)
                batch.to(device)

    def test_check_test(self):
        batch_size, num_workers, device = self.batch_size, self.num_workers, self.device
        disable_progress = self.disable_progress
        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)
        test_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["test"])
        test_data_config.map_num_workers = num_workers

        test_dataset = Mind2Web(test_data_config)

        task_test_dataset = TaskAdapter(
            test_dataset,
            transforms={
                "task_func": task_mind2web,
                "preprocessor": Mind2Web.task_preprocessor,
                "processor": processor,
                "postprocessor": Mind2Web.task_postprocessor,
            },
        )

        _ = task_test_dataset[367]  # previously known bad
        _ = task_test_dataset[368]  # good

        # something wrong i tihnk with this sample?

        with logger.progress(disable=disable_progress) as progress:
            collate_fn = DataCollator(processor.pad_token_id, squeeze=(batch_size != 1))

            dl = torch.utils.data.DataLoader(
                task_test_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=False,
                shuffle=False,
                drop_last=False,
            )
            test_progress = progress.add_task(f"[red] ✅ test set w/ batch_size: {batch_size}", total=len(dl))

            for batch in dl:
                progress.update(test_progress, advance=1)
                batch.to(device)
