import base64
import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from typing import List, Literal, NamedTuple

import PIL
from datasets import load_dataset
from torch.utils.data import Dataset

from pretrain_mm import logger
from pretrain_mm.datasets.dataset_utils import DatasetConfig
from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate, return_from_type


@lru_cache(maxsize=128)
def _read_json(filename: str) -> dict:
    with open(filename) as f_in:
        return json.load(f_in)


def read_json(filename: str, use_cache: bool = True) -> dict:
    # if use_cache:
    func = _read_json if use_cache else _read_json.__wrapped__
    return func(filename)


# test set is not available online but have it here:
#    /data/graham/code/mind2web/data/Mind2Web/data/test_set

# === === === === ===
# create functions to use on hf dataset (for filtering/breaking apart actions primarily)


def make_map_idx_batched_fn(
    task_dir: str, screenshot_file: str, filter_before: bool = True, filter_after: bool = True
) -> callable:
    def filter_actions_fn(data: dict, indexes: List[int]):
        filtered_indexes = []
        for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
            json_data = read_json(f"{task_dir}/task/{ann_id}/{screenshot_file}", use_cache=True)
            for act_idx, action in enumerate(actions):
                before_screenshot = json_data[act_idx]["before"]["screenshot"]
                after_screenshot = json_data[act_idx]["after"]["screenshot"]
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
    annotation_id: str = field(default=None, repr=False)
    image: PIL.Image.Image = field(default=None, init=False)

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
        self, json_data: dict, action_id: int, return_from: return_from_type = "before"
    ) -> PIL.Image.Image:
        """
        return from should be one of 'before' or 'after'
        """
        action_data = json_data[action_id]
        image_str = action_data[return_from]["screenshot"]
        image = PIL.Image.open(BytesIO(base64.b64decode(image_str)))
        return image


class Mind2Web(Mind2WebBase):
    """
    Mind2Web dataset
    avoiding preprocessing for now as for training i believe bottleneck will be model
    """

    def __init__(self, config: Mind2WebConfig, **kwargs):
        super().__init__(config)

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
        annotation_id = trajectory["annotation_id"]

        # poping actions so if we print trajectory we dont see them
        # actions = trajectory.pop("actions")
        actions = trajectory["actions"]

        # issue before was action was messed up
        try:
            raw_action = actions[action_idx]
        except Exception as err:
            logger.warn(f"Could not access sample action at: {idx} | annotation-id: {annotation_id}")

        action = M2WAction(action_idx=action_idx, annotation_id=annotation_id, **raw_action)
        json_data = self._load_json_data(annotation_id)

        # ive seen error with image before but not sure if there are others
        try:
            image = self.screenshot_from_json_data(json_data, action_idx, return_from="before")
        except Exception as err:
            logger.warn(f"Error loading image for (ann-id, action-idx, err): {annotation_id} {action_idx} {err}")

        action.image = self._process_image(image)

        # include trajectory for task adapter/debugging/log
        trajectory = M2WTrajectory(**trajectory)
        trajectory.trajectory_idx = t_idx
        action.trajectory = trajectory

        return action


class Mind2WebIterable(Mind2WebBase):
    def __init__(self, config: Mind2WebConfig, return_from: str = "before", **kwargs):
        super().__init__(config)
        self.return_from = return_from

        self.dataset = self.dataset.map(
            make_map_filter_batched_actions_fn(
                self.config.task_dir, self.config.screenshot_file, filter_when=return_from
            ),
            batched=False,
            with_indices=False,
            num_proc=self.config.map_num_workers,
            load_from_cache_file=self.config.map_load_from_cache_file,
        )

    def __getitem__(self, idx: int):
        trajectory = self.dataset[idx]
        annotation_id = trajectory["annotation_id"]

        # need to either filter out actions with empty before/after or handle it by draw a new one
        idx_choices = random.shuffle(list(range(0, len(trajectory["actions"]))))
        action_idx = idx_choices.pop()
        json_data = self._load_json_data(annotation_id)

        while idx_choices and not self._action_check(action_idx, json_data):
            action_idx = idx_choices.pop()

        if not idx_choices:
            logger.warn(f"no valid actions for {annotation_id}")
            raise ValueError(f"no valid actions for {annotation_id}")

        try:
            image = self.screenshot_from_json_data(json_data, action_idx, return_from=self.return_from)
        except Exception as err:
            logger.warn(f"Error loading image for {annotation_id} {action_idx} {err}")

        action = M2WAction(action_idx, trajectory["actions"][action_idx], image=image)
        trajectory = M2WTrajectory(**trajectory)
        action.trajectorty = trajectory
        return action

    def __len__(self):
        return len(self.dataset)


def task_mind2web(sample: M2WAction) -> dict:
    """
    given a sample from Mind2Web return a dict for the task adapter

    this task is close to clippy targeted format.
    E.g.
    [website-screenshot]
    [text] [next-action]

    # Previously was using
    # text = f"Task: {sample.trajectory.confirmed_task} {previous_actions_text}\nNext Action: "
    """

    joined_prev_actions = ", ".join(sample.trajectory.action_reprs[: sample.action_idx])
    previous_actions_text = f"Previous Actions: {joined_prev_actions}." if joined_prev_actions != "" else "None."
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if needed the action locator.\n{previous_actions_text}\nNext Action:\n"
    # You are a helpful Web Assistant.
    # Based on the prior actions and the current browser content, respond with the next step you'd take to achieve the OBJECTIVE.
    if len(sample.pos_candidates) > 0:
        # operation = f"{sample.operation.op.lower().capitalize()}" # dont think i should lower case since action_reprs are all CAP
        operation = f"{sample.operation.op}"
        if sample.operation.value != "":
            operation += f" {sample.operation.value}"

        # FUYU NEEDS IN format: y1, x1, y2, x2 but bounding box comes in form x0, y0, x1, y1,
        attrs = parse_candidate(random.choice(sample.pos_candidates), parse_bounding_box=True)["attributes"]
        x1, y1, x2, y2 = map(int, attrs["bounding_box_rect"])
        # therefore:
        box = f"<box>{y1}, {x1}, {y2}, {x2}</box>"
        next_action = f"{operation} @ {box}"
    else:
        try:
            operation = f"{sample.operation.op}"
            if sample.operation.value != "":
                operation += f" {sample.operation.value}"
            next_action = operation
        except Exception as err:
            logger.warn(f"Error with {sample.annotation_id} and action idx: {sample.action_idx}.\n{err}")
            next_action = "DONE"
    # else:
    #     next_action = "DONE"

    return {
        "text": text,
        "label": next_action,
        "image": sample.image,
    }


def _alt_format(previous_actions_text):
    text = f"You are a helpful web assistant. Based on the prior actions and the current browser content, respond with the next action and if necessary action position.\n{previous_actions_text}\nNext Action:\n"
    return text
