import base64
import random
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Literal, NamedTuple

import PIL
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

from pretrain_mm.datasets.dataset_utils import DatasetConfig
from pretrain_mm.datasets.mind2web.mind2web_utils import ReturnFromTypes, read_json


# test set is not available online but have it here:
#    /data/graham/code/mind2web/data/Mind2Web/data/test_set

# === === === === ===
# create functions to use on hf dataset (for filtering/breaking apart actions primarily)


def make_map_idx_batched_fn(
    task_dir: str, screenshot_file: str, filter_before: bool = True, filter_after: bool = True
) -> callable:
    """
    """

    def filter_actions_fn(data: dict, indexes: List[int]):
        filtered_indexes = []
        for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
            json_data = read_json(f"{task_dir}/task/{ann_id}/{screenshot_file}", use_cache=True)
            for act_idx, _ in enumerate(actions):
                before_screenshot = json_data[act_idx]["before"]["screenshot"]
                _ = json_data[act_idx]["after"]["screenshot"]
                if before_screenshot != "":
                    filtered_indexes.append([indexes[idx], act_idx])

        return {"indexes": filtered_indexes}

    return filter_actions_fn


def make_map_filter_batched_actions_fn(
    task_dir: str, screenshot_file: str, filter_when: Literal["before", "after"] = "before"
) -> callable:
    """
    this should be used like
    dataset.map(
        make_map_filter_batched_actions_fn(task_dir, screenshot_file, filter_when="before"),
        batched=False,
        with_indices=False,

    )
    """

    # dont use indexes here
    def filter_actions_fn(data: dict):
        annotation_id: str = data["annotation_id"]
        json_data = read_json(f"{task_dir}/task/{annotation_id}/{screenshot_file}", use_cache=True)

        filtered_actions = []
        for action in data["actions"]:
            screenshot = json_data[action["action_idx"]][filter_when]["screenshot"]
            if screenshot == "":
                continue
            filtered_actions.append(action)
        data["actions"] = filtered_actions
        return data

    return filter_actions_fn


# === === === === ===
# Dataclasses/Sample Related


class ActionOp(NamedTuple):
    op: str  # not certain yet all vals here but at least 'SELECT', 'CLICK', 'TYPE'
    original_op: str  # seems like this is one of 'SELECT', 'CLICK', 'TYPE', 'HOVER'
    value: str


@dataclass
class Mind2WebConfig(DatasetConfig):
    #
    dataset_path: str = "osunlp/Mind2Web"
    data_files: str = None  # needed for test data
    split: str = "train"  # for test we will need to read the files from

    show_progress: bool = True

    #
    task_dir: str = "/data/graham/datasets/mind2web/data"
    screenshot_file: str = "processed/screenshot.json"

    viewport_size: tuple[int, int] = (1280, 1080)  # {"width": 1280, "height": 1080}
    crop_image: bool = True
    include_html: bool = False

    # subset allows for testing quicker
    subset: int = None


@dataclass
class M2WAction:
    action_idx: int

    action_uid: str
    operation: ActionOp

    pos_candidates: List[dict]
    # dont show these b/c there are us ually 100+
    neg_candidates: List[dict] = field(default_factory=list, repr=False)

    # probably not using these as i am using screenshot
    cleaned_html: str = field(default=None, repr=False)
    raw_html: str = field(default=None, repr=False)

    # info needed from trajectory
    annotation_id: str = None  # field(default=None, repr=False)
    image: PIL.Image.Image = None  # field(default=None, init=False)

    # primarily for typing
    trajectory: "M2WTrajectory" = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.operation = ActionOp(**self.operation)


@dataclass
class M2WTrajectory:
    action_reprs: List[str]

    annotation_id: str
    confirmed_task: str
    website: str

    domain: str
    subdomain: str

    trajectory_idx: int = None
    actions: List[M2WAction] = field(default=None, repr=False)


# === === === === ===
# Actual Datasets


class Mind2WebBase(Dataset):
    """
    base class for Mind2Web, doesnt split off actions
    """

    def __init__(self, config: Mind2WebConfig, **kwargs):
        self.config = config
        self._use_cache = True

        self.dataset = load_dataset(
            self.config.dataset_path,
            data_files=self.config.data_files,
            split=self.config.split,
        )

        self.disable_progress = getattr(self.config, "disable_progress", False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> M2WTrajectory:
        traj = self.dataset[idx]
        traj["actions"] = [M2WAction(**action) for action in traj["actions"]]
        return M2WTrajectory(**traj)

    def _load_json_data(self, annotation_id: str) -> dict:
        return read_json(
            f"{self.config.task_dir}/task/{annotation_id}/{self.config.screenshot_file}", use_cache=self._use_cache
        )

    def _process_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        if self.config.crop_image:
            image = image.crop((0, 0, self.config.viewport_size[0], self.config.viewport_size[1]))
        return image

    def screenshot_from_json_data(
        self, json_data: dict, action_id: int, return_from: ReturnFromTypes = "before"
    ) -> PIL.Image.Image:
        """
        # might want to include warning
        #     logger.warn(f"Error loading image for (ann-id, action-idx, err): {annotation_id} {action_id} {err}")

        """
        action_data = json_data[action_id]
        image_str = action_data[return_from]["screenshot"]
        image = PIL.Image.open(BytesIO(base64.b64decode(image_str)))
        return image

    def _get_action_from_trajectory(self, trajectory: dict, action_idx: int, return_from: str) -> M2WAction:
        json_data = self._load_json_data(trajectory["annotation_id"])
        action = M2WAction(
            action_idx=action_idx,
            annotation_id=trajectory["annotation_id"],
            image=self._process_image(self.screenshot_from_json_data(json_data, action_idx, return_from=return_from)),
            **trajectory["actions"][action_idx],
        )
        return action


class Mind2Web(Mind2WebBase):
    """
    Mind2Web dataset
    avoiding preprocessing for now as for training i believe bottleneck will be model
    """

    def __init__(self, config: Mind2WebConfig, return_from: ReturnFromTypes = "before", **kwargs):
        super().__init__(config)
        self.return_from = return_from


        map_fn = make_map_idx_batched_fn(self.config.task_dir, self.config.screenshot_file)
        self.dataset_idxs = self.dataset.map(
            map_fn,
            batched=True,
            with_indices=True,
            remove_columns=self.dataset.column_names,
            num_proc=self.config.map_num_workers,
            load_from_cache_file=self.config.map_load_from_cache_file,
        )

    def __len__(self):
        return len(self.dataset_idxs)

    def __getitem__(self, idx: int) -> M2WAction:
        t_idx, action_idx = self.dataset_idxs[idx]["indexes"]
        trajectory = self.dataset[t_idx]

        action = self._get_action_from_trajectory(
            trajectory=trajectory, action_idx=action_idx, return_from=self.return_from
        )
        action.trajectory = M2WTrajectory(trajectory_idx=t_idx, **trajectory)
        return action


class Mind2WebIterable(Mind2WebBase, IterableDataset):
    def __init__(
        self, config: Mind2WebConfig, num_iters: int = None, return_from: ReturnFromTypes = "before", **kwargs
    ):
        super().__init__(config)

        self.return_from = return_from
        self.num_iters = num_iters

    def __iter__(self):
        for _ in range(self.num_iters):
            yield self._sample()

    def _sample(self):
        while True:
            t_idx = random.randint(0, len(self.dataset) - 1)
            trajectory = self.dataset[t_idx]
            action_idx = random.randint(0, trajectory)
            try:
                sample = self._get_action_from_trajectory(
                    trajectory=trajectory,
                    action_idx=action_idx,
                    return_from=self.return_from,
                )
                sample.trajectory = M2WTrajectory(trajectory_idx=t_idx, **trajectory)
                return sample
            except:
                pass
