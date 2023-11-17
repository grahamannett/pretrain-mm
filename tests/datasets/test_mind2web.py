import unittest

from pretrain_mm.datasets.mind2web import Mind2Web, Mind2WebConfig, task_mind2web
from pretrain_mm.datasets.task_adapter import TaskAdapterProcessor

from config.fuyu import FuyuInfo


class TestMind2Web(unittest.TestCase):
    def test_mind2web(self):
        config = Mind2WebConfig(subset=10)
        dataset = Mind2Web(config)

        sample = dataset[50]
        # check that task for this dataset is working
        sample_as_task = task_mind2web(sample)
        assert "label" in sample_as_task

        # check that task adapter with processor is working
        task_dataset = TaskAdapterProcessor(
            dataset, task_func=task_mind2web, processor=FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name)
        )

        task_sample = task_dataset[50]
        assert "text" in task_sample
        assert task_sample["images"].size == (1280, 1080)
