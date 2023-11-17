import os
from dataclasses import dataclass, field

from pretrain_mm import logger

@dataclass
class DatasetInitHelper:
    make: type
    sample: type = None

    task: callable = None
    tasks: type = None

    dataset_kwargs: dict = field(default_factory=dict)



@dataclass
class DatasetConfig:

    # fsdp related
    fsdp_enabled: bool = False
    local_rank: int = None


    def __post_init__(self):
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))


class DatasetProgressMixin:
    _progress_bars = {}
    _task_ids = {}

    def _progress_start(self, amt: int, desc: str = "[cyan]Flattening dataset...") -> None:

        if not self.config.show_progress:
            return

        self.progress = logger.progress()

        self.traj_task = self.progress.add_task(desc, total=amt)

    def _progress_update(self, advance: int = 1):
        if not self.config.show_progress:
            return

        self.progress.update(self.traj_task, advance=advance)

    def _progress_end(self):

        if not hasattr(self, self.progress):
            return

        self.progress.stop()