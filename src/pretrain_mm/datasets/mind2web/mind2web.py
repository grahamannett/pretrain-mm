import base64
import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from typing import List, Literal, NamedTuple

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from pretrain_mm import logger
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate
from pretrain_mm.datasets.dataset_utils import DatasetConfig


@lru_cache(maxsize=10)
def read_json(filename: str) -> dict:
    with open(filename) as f_in:
        return json.load(f_in)


# test set is not available online but have it here:
#    /data/graham/code/mind2web/data/Mind2Web/data/test_set


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


class ActionOp(NamedTuple):
    op: str  # not certain yet all vals here but at least 'SELECT', 'CLICK', 'TYPE'
    original_op: str  # seems like this is one of 'SELECT', 'CLICK', 'TYPE', 'HOVER'
    value: str


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
    annotation_id: str = field(default=None, repr=False)
    image: Image.Image = field(default=None, init=False)

    # trajectory
    traj: "M2WTrajectory" = field(default=None, repr=False, init=False)

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

    actions: List[M2WAction] = field(default=None, repr=False)


class Mind2WebBase(Dataset):
    """
    base class for Mind2Web, doesnt split off actions
    """

    def __init__(self, config: Mind2WebConfig, **kwargs):
        self.config = config

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

    @staticmethod
    def task_preprocessor(sample: dict):
        """
        this is a task preprocessor for the Mind2Web dataset such that it works for the processor
        """
        return {
            "text": sample["text"] + sample["label"],
            "images": sample["image"],
        }

    @staticmethod
    def task_postprocessor(sample):
        sample["input_ids"] = sample.input_ids.squeeze(0)
        sample["attention_mask"] = sample.attention_mask.squeeze(0)
        sample["image_patches"] = [img.squeeze(0) for img in sample.image_patches]
        sample["image_patches_indices"] = sample.image_patches_indices.squeeze(0)
        return sample

    def _load_json_data(self, annotation_id: str) -> dict:
        return read_json(f"{self.config.task_dir}/task/{annotation_id}/{self.config.screenshot_file}")

    def _process_image(self, image: Image.Image) -> Image.Image:
        if self.config.crop_image:
            image = image.crop((0, 0, self.config.viewport_size[0], self.config.viewport_size[1]))
        return image

    def load_screenshot_from_task_dir(
        self, annotation_id: str, action_id: int, return_from: Literal["after", "before"] = "before"
    ) -> Image.Image:
        """
        return from should be one of 'before' or 'after'
        """

        json_data = self._load_json_data(annotation_id)
        action_data = json_data[action_id]
        image_str = action_data[return_from]["screenshot"]
        image = Image.open(BytesIO(base64.b64decode(image_str)))
        return image


class Mind2Web(Mind2WebBase):
    """
    Mind2Web dataset
    avoiding preprocessing for now as for training i believe bottleneck will be model
    """

    def __init__(self, config: Mind2WebConfig, **kwargs):
        super().__init__(config)
        self.dataset_idxs = self._flatten_dataset()

    def __len__(self):
        return len(self.dataset_idxs)

    def __getitem__(self, idx: int) -> M2WAction:
        t_idx, action_idx = self.dataset_idxs[idx]
        traj = self.dataset[t_idx]
        annotation_id = traj["annotation_id"]

        # poping actions so if we print traj we dont see them
        actions = traj.pop("actions")
        raw_action = actions[action_idx]

        action = M2WAction(action_idx=action_idx, annotation_id=annotation_id, **raw_action)
        action.image = self._process_image(self.load_screenshot_from_task_dir(annotation_id, action_idx))

        # include trajectory for task adapter
        traj = M2WTrajectory(**traj)
        action.traj = traj

        return action

    def _flatten_dataset(self) -> list[tuple[int, int]]:
        """
        go from dataset of trajectories to dataset of actions
        """

        pbar_desc = "[cyan]Flattening dataset..."
        pbar_amount = self.config.subset or len(self.dataset)
        flat_idxs = []

        with logger.progress(disable=self.disable_progress) as progress:
            traj_task = progress.add_task(pbar_desc, total=pbar_amount)

            for t_idx, traj in enumerate(self.dataset):
                # if we are subsetting then break early - this is for testing
                if self.config.subset and t_idx >= self.config.subset:
                    break

                for action_idx, action in enumerate(traj["actions"]):
                    flat_idxs.append([t_idx, action_idx])

                progress.update(traj_task, advance=1)

        return flat_idxs


def task_mind2web(sample: M2WAction) -> dict:
    """
    given a sample from Mind2Web return a dict for the task adapter

    this task is close to clippy targeted format.
    E.g.
    [website-screenshot]
    [text] [next-action]
    """
    previous_actions_text = ", ".join(sample.traj.action_reprs[: sample.action_idx])
    text = f"Task: {sample.traj.confirmed_task} Previous Actions {previous_actions_text}\nNext Action:"

    if len(sample.pos_candidates) > 0:
        operation = f"{sample.operation.op} {sample.operation.value}"
        attrs = parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
        box = "<box>" + ", ".join([str(int(v)) for v in attrs["bounding_box_rect"]]) + "</box>"
        next_action = f"{operation} @ {box}"
    else:
        next_action = "DONE"

    return {
        "text": text,
        "label": next_action,
        "image": sample.image,
    }
