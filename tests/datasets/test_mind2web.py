import unittest

from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebConfig, task_mind2web
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor

from config.fuyu import FuyuInfo
from config.dev import get_dev_config


class TestMind2Web(unittest.TestCase):
    def test_mind2web(self):
        mind2web_info = get_dev_config("mind2web")
        config = Mind2WebConfig(task_dir=mind2web_info["task_dir"], subset=10)
        dataset = Mind2Web(config)

        sample = dataset[50]
        # check that task for this dataset is working
        sample_as_task = task_mind2web(sample)
        assert "label" in sample_as_task

        # check that task adapter with processor is working
        task_dataset = TaskAdapterProcessor(
            dataset,
            task_func=task_mind2web,
            processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name),
            preprocessor=Mind2Web.task_preprocessor,
        )

        task_sample = task_dataset[50]

        assert "input_ids" in task_sample
        assert task_sample["input_ids"].ndim == 2
        assert "attention_mask" in task_sample

        assert "image_patches" in task_sample
        image_patches = task_sample["image_patches"]
        assert "image_patches_indices" in task_sample
