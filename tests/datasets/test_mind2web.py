import os
import unittest
import time

import torch
from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebBase, Mind2WebConfig, task_mind2web
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor
from pretrain_mm.datasets.dataloader import DataCollator
from pretrain_mm import logger

from config.fuyu import FuyuInfo
from config.dev import get_dev_config
from pretrain_mm.processor.fuyu.fuyu_processing import FuyuProcessor


from datasets import disable_caching

m2w_info = get_dev_config("mind2web")
task_dir = m2w_info["task_dir"]


class TestMind2Web(unittest.TestCase):
    def test_mind2web(self):
        subset = 10

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


class TestFlatten(unittest.TestCase):
    def setUp(self):
        self.subset = int(os.environ.get("SUBSET", 10))
        self.start_time = time.perf_counter()
        logger.info(f"[yellow]FUNC:{self._testMethodName}")

    def check_timer(self, name: str = None):
        time_now = time.perf_counter()
        elapsed_time = time_now - self.start_time
        start_str = f"[red]⏰[{name}] " if name else "[red] ⏰"
        logger.info(f"{start_str}: {elapsed_time} seconds")

        # reset timer
        self.start_time = time.perf_counter()

    def test_flatten(self):
        # batch_size = int(os.environ.get("BATCH_SIZE", 2))
        # num_workers = int(os.environ.get("N_WORKERS", 0))
        # device = os.environ.get("DEVICE", "cuda")
        # self.check_timer("got env vars")

        data_config = Mind2WebConfig(task_dir=task_dir, subset=self.subset, **m2w_info["train"])
        dataset = Mind2Web(data_config)
        # self.check_timer("made dataset just from indexes")
        self.check_timer("==>FROM INDEXES TIMER")
        # sample = dataset[0]
        # breakpoint()

    def test_flatten_map(self):
        train_data_config = Mind2WebConfig(task_dir=task_dir, **m2w_info["train"])
        dataset = Mind2WebBase(train_data_config)
        dataset = dataset.dataset.select(range(self.subset))

        def flatten_data(data, idx, rank):
            # print(f"IDX: {idx} - RANK {rank}")
            output = {
                "subdomain": [],
                "website": [],
                "annotation_id": [],
                "action_reprs": [],
                "actions": [],
                "domain": [],
                "confirmed_task": [],
                "act_idx": [],
            }

            for samp_idx, (action_reprs, actions) in enumerate(zip(data["action_reprs"], data["actions"])):
                for act_idx, (action_repr, action) in enumerate(zip(action_reprs, actions)):
                    output["subdomain"].append(data["subdomain"][samp_idx])
                    output["website"].append(data["website"][samp_idx])
                    output["annotation_id"].append(data["annotation_id"][samp_idx])
                    output["action_reprs"].append(action_repr)
                    output["actions"].append(action)
                    output["domain"].append(data["domain"][samp_idx])
                    output["confirmed_task"].append(data["confirmed_task"][samp_idx])
                    output["act_idx"].append(act_idx)

            return output

        flat_dataset = dataset.map(
            flatten_data,
            batched=True,
            num_proc=10,
            remove_columns=dataset.column_names,
            with_indices=True,
            with_rank=True,
            load_from_cache_file=False,
        )

        self.check_timer("==>DATASET.MAP ")

        def flatten_data_idxs(data, idx):
            # vals = []
            # for _i, actions in enumerate(data["actions"]):
            #     for act_idx in range(len(actions)):
            #         vals.append([idx[_i], act_idx])
            # return {"indexes": vals}

            return {
                "indexes": [
                    [idx[_i], act_idx] for _i, actions in enumerate(data["actions"]) for act_idx in range(len(actions))
                ]
            }

        flat_dataset_idxs = dataset.map(
            flatten_data_idxs,
            batched=True,
            num_proc=10,
            with_indices=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )
        self.check_timer("==>DATASET.MAP.INDEXS")

        # breakpoint()
        # return {"idxs": [idx, list(range(len(data["actions"])))]}


class TestSamples(unittest.TestCase):
    def test_check_samples(self):
        batch_size = int(os.environ.get("BATCH_SIZE", 2))
        num_workers = int(os.environ.get("N_WORKERS", 0))
        device = os.environ.get("DEVICE", "cuda")

        train_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["train"])
        # train_data_config.subset = 200

        train_dataset = Mind2Web(train_data_config)

        processor = FuyuProcessor.from_pretrained(FuyuInfo.model_name)

        # check that task adapter with processor is working
        task_train_dataset = TaskAdapterProcessor(
            train_dataset,
            task_func=task_mind2web,
            processor=processor,
            preprocessor=Mind2Web.task_preprocessor,
            postprocessor=Mind2Web.task_postprocessor,
        )

        # out = task_train_dataset[1322]
        # breakpoint()

        test_data_config = Mind2WebConfig(task_dir=m2w_info["task_dir"], **m2w_info["test"])
        test_dataset = Mind2Web(test_data_config)

        task_test_dataset = TaskAdapterProcessor(
            test_dataset,
            task_func=task_mind2web,
            processor=processor,
            preprocessor=Mind2Web.task_preprocessor,
            postprocessor=Mind2Web.task_postprocessor,
        )

        # something wrong i tihnk with this sample?

        with logger.progress() as progress:
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

            # dl = torch.utils.data.DataLoader(
            #     task_test_dataset,
            #     batch_size=batch_size,
            #     collate_fn=collate_fn,
            #     num_workers=num_workers,
            #     pin_memory=True,
            #     shuffle=False,
            #     drop_last=False,
            # )
            # test_progress = progress.add_task(f"[red] ✅ test set w/ batch_size: {batch_size}", total=len(dl))

            # for idx, batch in enumerate(dl):
            #     progress.update(test_progress, advance=1)
            #     batch.to(device)

    def test_compare_cache(self):
        pass
