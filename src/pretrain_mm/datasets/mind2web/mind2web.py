import base64
import random
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Literal, NamedTuple

import PIL
from bs4 import BeautifulSoup
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

from pretrain_mm import DEBUG, logger
from pretrain_mm.datasets.dataset_utils import DatasetConfig
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils

# test set is not available online but have it here:
#    /data/graham/code/mind2web/data/Mind2Web/data/test_set

# === === === === ===
# create functions to use on hf dataset (for filtering/breaking apart actions primarily)


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
        json_data = m2w_utils.read_json(f"{task_dir}/task/{annotation_id}/{screenshot_file}", use_cache=True)

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
    data_files: str = "**/*.json"  # needed for test data
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

    _mode: str = None

    def __init__(self, config: Mind2WebConfig, **kwargs):
        self.config = config
        self._use_cache = True

        if not config.task_dir:
            logger.warn(f"Task Dir is empty, assume we are in test mode/without data")
            self._mode = "localdev"

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

    def _get_raw(self, idx: int) -> dict:
        return self.dataset[idx]

    def process_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        if self.config.crop_image:
            image = image.crop((0, 0, self.config.viewport_size[0], self.config.viewport_size[1]))
        return image

    def screenshot_from_json_data(
        self, json_data: dict, action_idx: int, return_from: m2w_utils.ReturnFromTypes = "before"
    ) -> PIL.Image.Image:
        """
        # might want to include warning
        #     logger.warn(f"Error loading image for (ann-id, action-idx, err): {annotation_id} {action_idx} {err}")
        """
        if self._mode == "localdev":
            return PIL.Image.new("RGB", self.config.viewport_size)

        action_data = json_data[action_idx]
        image_str = action_data[return_from]["screenshot"]
        image = PIL.Image.open(BytesIO(base64.b64decode(image_str)))
        return image

    def get_screenshot_for_idxs(self, t_idx: int, a_idx: int = 0, return_from: m2w_utils.ReturnFromTypes = "before"):
        """helper function for getting screenshot for a trajectory/action idx"""
        trajectory = self.dataset[t_idx]
        data = m2w_utils.read_json(
            f"{self.config.task_dir}/task/{trajectory['annotation_id']}/{self.config.screenshot_file}"
        )
        image = self.screenshot_from_json_data(data, action_idx=a_idx, return_from=return_from)
        return image

    def get_action_from_trajectory(self, trajectory: dict, action_idx: int, return_from: str) -> M2WAction:
        json_data = {}
        if self._mode != "localdev":
            json_data = m2w_utils.read_json(
                f"{self.config.task_dir}/task/{trajectory['annotation_id']}/{self.config.screenshot_file}",
                self._use_cache,
            )

        action = M2WAction(
            action_idx=action_idx,
            annotation_id=trajectory["annotation_id"],
            image=self.process_image(self.screenshot_from_json_data(json_data, action_idx, return_from=return_from)),
            **trajectory["actions"][action_idx],
        )

        return action


class Mind2Web(Mind2WebBase):
    """
    Mind2Web dataset
    avoiding preprocessing for now as for training i believe bottleneck will be model
    """

    def __init__(self, config: Mind2WebConfig, return_from: m2w_utils.ReturnFromTypes = "before", **kwargs):
        super().__init__(config)
        self.return_from = return_from

        self._make_dataset_idxs()
        # map_fn = make_map_idx_batched_fn(self.config.task_dir, self.config.screenshot_file)

    def __len__(self):
        return len(self.dataset_idxs)

    def __getitem__(self, idx: int) -> M2WAction:
        t_idx, action_idx = self.dataset_idxs[idx]["indexes"]
        trajectory = self.dataset[t_idx]

        action = self.get_action_from_trajectory(
            trajectory=trajectory, action_idx=action_idx, return_from=self.return_from
        )
        action.trajectory = M2WTrajectory(trajectory_idx=t_idx, **trajectory)
        return action

    def _filter_candidates(
        self,
        screenshot_margin: float = 1.0,  # was 1.5
        max_area: float = 1e5,
        # enforce_clickable: bool = True,
    ):
        """used for pretrain objective
        filter out elements that are larger than 1e5 (~300*300 so 1/4th of 1200x1200)
        filter out elements that are further than 1.5x the viewport size

        """
        if DEBUG:
            self.config.map_num_workers = 1

        width, height = self.config.viewport_size

        def candidate_ok(
            candidate: dict, screenshot_margin: float = screenshot_margin, max_area: float = max_area, html_tree=None
        ) -> bool:
            # if enforce_clickable and not candidate["attributes"]["is_clickable"]:
            #     return False

            bbox = candidate["attributes"]["bounding_box_rect"]

            box_area = m2w_utils.get_bounding_box_area(bbox)
            mid_x, mid_y = m2w_utils.get_mid_point(bbox)

            if (mid_x > (width * screenshot_margin)) or (mid_y > (height * screenshot_margin)):
                return False

            if box_area > max_area:
                return False

            if html_tree:
                # check if the node has a bounding box and if it does and is -1 it means hidden so we dont want that
                node = html_tree.find(backend_node_id=candidate["backend_node_id"])
                # breakpoint()
                if not m2w_utils.check_dirty_node(node):
                    return False
                if not m2w_utils.check_node_has_text(node):
                    return False

            return True

        # ensure that all candidates are ok.  meaning it is within the viewport and not too large
        # if more restrictions are needed, add to `candidate_ok`
        def map_fn(data: dict):
            for a_idx, action in enumerate(data["actions"]):
                for s_idx, subaction in enumerate(action):
                    html_tree = BeautifulSoup(subaction["raw_html"], "html.parser")
                    # use copy since process_candidate modifies the dict
                    neg_cands = [
                        x
                        for x in subaction["neg_candidates"]
                        if candidate_ok(m2w_utils.parse_candidate(x.copy(), True), html_tree=html_tree)
                    ]
                    pos_cands = [
                        x
                        for x in subaction["pos_candidates"]
                        if candidate_ok(m2w_utils.parse_candidate(x.copy(), True), html_tree=html_tree)
                    ]

                    action[s_idx]["neg_candidates"] = neg_cands
                    action[s_idx]["pos_candidates"] = pos_cands

            return data

        self.dataset = self.dataset.map(
            map_fn,
            batched=True,
            num_proc=self.config.map_num_workers,
            load_from_cache_file=self.config.map_load_from_cache_file,
        )

    def setup_pretrain(self, **kwargs):
        self._filter_candidates(**kwargs)

    def _make_dataset_idxs(self):
        """
        make indexes for dataset as some actions dont have before screenshots
        also makes it so that if task_dir is empty we just use all the actions since

        """

        def filter_actions_fn(data: dict, indexes: List[int]):
            filtered_indexes = []
            for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
                # if we are in localdev mode we just use all the actions
                if self._mode == "localdev":
                    filtered_indexes.extend([indexes[idx], act_idx] for act_idx in range(len(actions)))
                    continue

                json_data = m2w_utils.read_json(
                    f"{self.config.task_dir}/task/{ann_id}/{self.config.screenshot_file}", use_cache=True
                )
                for act_idx, _ in enumerate(actions):
                    before_screenshot = json_data[act_idx]["before"]["screenshot"]
                    _ = json_data[act_idx]["after"]["screenshot"]
                    if before_screenshot != "":
                        filtered_indexes.append([indexes[idx], act_idx])

            return {"indexes": filtered_indexes}

        self.dataset_idxs = self.dataset.map(
            filter_actions_fn,
            batched=True,
            with_indices=True,
            remove_columns=self.dataset.column_names,
            num_proc=self.config.map_num_workers,
            load_from_cache_file=self.config.map_load_from_cache_file,
        )


class Mind2WebIterable(Mind2WebBase, IterableDataset):
    def __init__(
        self, config: Mind2WebConfig, num_iters: int = None, return_from: m2w_utils.ReturnFromTypes = "before", **kwargs
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
                sample = self.get_action_from_trajectory(
                    trajectory=trajectory,
                    action_idx=action_idx,
                    return_from=self.return_from,
                )
                sample.trajectory = M2WTrajectory(trajectory_idx=t_idx, **trajectory)
                return sample
            except:
                pass
