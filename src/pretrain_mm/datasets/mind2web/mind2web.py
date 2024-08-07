import os
import random

import PIL
from datasets import load_dataset
from torch.utils.data import IterableDataset

from pretrain_mm import DEBUG, logger
from pretrain_mm.datasets.mind2web import mind2web_preprocess_data
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.datasets.mind2web.mind2web_datatypes import M2WAction, M2WTrajectory, Mind2WebConfig, ReturnFromTypes
from pretrain_mm.datasets.utils.dataset_helpers import Dataset
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.utils.json_utils import read_json


# test set is not available online but have it here:
#    /data/graham/code/mind2web/data/Mind2Web/data/test_set

# === === === === ===
# create functions to use on hf dataset (for filtering/breaking apart actions primarily)


def make_map_filter_batched_actions_fn(
    task_dir: str, screenshot_file: str, filter_when: ReturnFromTypes = ReturnFromTypes.before
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
# Actual Datasets


class Mind2WebBase(Dataset):
    """
    base class for Mind2Web, doesnt split off actions
    """

    _mode: str = None
    _on_get_attach: dict = None

    def __init__(self, config: Mind2WebConfig = None, **kwargs):
        # allow empty/auto config
        self.config = config or Mind2WebConfig(**kwargs)
        self.disable_progress = getattr(self.config, "disable_progress", False)

        self.dataset = load_dataset(
            self.config.dataset_path,
            data_files=self.config.data_files,
            # m2w test dataset still uses 'train', needs specific dataset_path and data_files
            split="train",
        )

        # allow disabling progress as on slurm it buffers/prints bad

        if not config.task_dir:
            logger.warn(f"{self.__class__.__name__}.task_dir is empty, assume we are in test mode/without data")
            self._mode = "localdev"

        # allow subset or DEBUG to make ds shorter
        if self.config.subset:
            if (_env_ds_len := os.environ.get("DS_LEN", None)) or self.config.subset:
                _ds_len = _env_ds_len or self.config.subset
                logger.warn(f"For {self.__class__.__name__} Using subset of {_ds_len}")
                self.dataset = self.dataset.select(range(0, int(_ds_len)))

        if DEBUG:
            logger.warn("Debugging Mode is on using 1 worker for map[slow]")
            self.config.map_num_workers = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> M2WTrajectory:
        traj = self.dataset[idx]
        traj["actions"] = [M2WAction(**action) for action in traj["actions"]]
        trajectory = M2WTrajectory(**self._include_json_filepath(traj))

        self._attach_on_get(trajectory)
        return trajectory

    def _include_json_filepath(self, traj: dict) -> dict:
        traj["json_filepath"] = f"{self.config.task_dir}/task/{traj['annotation_id']}/{self.config.screenshot_file}"
        return traj

    def _get_raw(self, idx: int) -> dict:
        return self.dataset[idx]

    @staticmethod
    def parse_candidate(candidate: dict, as_copy: bool = True, **kwargs):
        """helper function so you dont need to import mind2web_utils"""
        if as_copy:
            candidate = candidate.copy()

        return m2w_utils.parse_candidate(candidate, **kwargs)

    def set_attach(self, **kwargs):
        self._on_get_attach = kwargs
        return self

    def _attach_on_get(self, obj):
        obj._attached = self._on_get_attach
        return obj

    def get_image_for_sample(self, action: M2WAction, return_from: ReturnFromTypes = None) -> PIL.Image.Image:
        if self._mode == "localdev":
            return PIL.Image.new("RGB", self.config.viewport_size)

        return_from = return_from or action.return_from

        image = action.load_image_from_filepath(
            task_dir=self.config.task_dir,
            screenshot_file=self.config.screenshot_file,
            return_from=return_from,
        )

        # preprocessing of image that happens before the task processor applied
        # which is where normalization/resizing/patchification happens
        if self.config.crop_image:
            image = image.crop((0, 0, self.config.viewport_size[0], self.config.viewport_size[1]))

        return image

    def get_image_for_idx(
        self, t_idx: int, a_idx: int = 0, return_from: ReturnFromTypes = ReturnFromTypes.before, use_cache: bool = True
    ) -> tuple[PIL.Image.Image, dict]:
        """helper function for getting screenshot for a trajectory/action idx"""
        trajectory = self.dataset[t_idx]
        annotation_id = trajectory["annotation_id"]
        json_data_filepath = f"{self.config.task_dir}/task/{annotation_id}/{self.config.screenshot_file}"

        json_data = read_json(json_data_filepath, use_cache=use_cache)
        image = read_image_from_b64(json_data[a_idx][return_from])

        return image, json_data

    def get_action_from_trajectory(
        self, trajectory: M2WTrajectory, action_idx: int, return_from: ReturnFromTypes
    ) -> M2WAction:
        return M2WAction.from_trajectory(action_idx=action_idx, trajectory=trajectory, return_from=return_from)


class Mind2Web(Mind2WebBase):
    """
    Mind2Web dataset, the difference between this and base is that this is focused on actions
    versus the base is meant as something that can be subclassed for possibly multiple actions
    at once or various preprocessing

    avoiding preprocessing for now as for training i believe bottleneck will be model

    """

    def __init__(
        self,
        config: Mind2WebConfig = None,
        return_from: ReturnFromTypes = ReturnFromTypes.before,
        ensure_pos_candidates: bool = True,
        **kwargs,
    ):
        """Mind2Web is the dataset for Mind2Web where each sample is an action rather than a trajectory

        Args:
            config (Mind2WebConfig): _description_
            return_from (ReturnFromTypes, optional): _description_. Defaults to "before".
        """

        # load the original dataset
        super().__init__(config=config, **kwargs)

        self.ensure_pos_candidates = ensure_pos_candidates

        self.return_from = return_from
        self._make_dataset_idxs()

    def __len__(self):
        return len(self.dataset_idxs)

    def __getitem__(self, idx: int, return_from: ReturnFromTypes = None) -> M2WAction:
        return_from = return_from or self.return_from
        t_idx, action_idx = self.dataset_idxs[idx]["indexes"]

        try:
            trajectory = self.dataset[t_idx]
        except Exception as err:
            logger.warn(f"Got err: {err}")
            breakpoint()

        # create  base trajectory object/dataclass
        trajectory = M2WTrajectory(trajectory_idx=t_idx, **self._include_json_filepath(trajectory))

        # create action object/dataclass, contains trajectory
        action = self.get_action_from_trajectory(
            trajectory=trajectory,
            action_idx=action_idx,
            return_from=return_from,
        )

        action.image = self.get_image_for_sample(action, return_from=return_from)

        if self.config.attach_config_to_sample:
            action._config = self.config
            action._idx = idx

        self._attach_on_get(action)
        return action

    def _make_dataset_idxs(self):
        """
        make indexes for dataset as some actions dont have before screenshots
        also makes it so that if task_dir is empty we just use all the actions since

        """

        _ensure_pos = self.ensure_pos_candidates
        self._ignored_idxs = []

        def filter_actions_fn(data: dict, indexes: list[int]):
            filtered_indexes = []
            for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
                # if we are in localdev mode we just use all the actions
                if self._mode == "localdev":
                    filtered_indexes.extend([indexes[idx], act_idx] for act_idx in range(len(actions)))
                    continue

                json_data = read_json(
                    f"{self.config.task_dir}/task/{ann_id}/{self.config.screenshot_file}", use_cache=True
                )

                for act_idx, action in enumerate(actions):
                    # check before screenshot, after would be like
                    # _ = json_data[act_idx]["after"]["screenshot"]
                    if json_data[act_idx]["before"]["screenshot"] == "":
                        # NOTE: none of these ignored are saved since its done in .map
                        self._ignored_idxs.append([idx, ann_id, act_idx, "no_before_screenshot"])
                        #     {ann_id: {"action_idx": act_idx, "ann_idx": idx, "issue": "no_before_screenshot"}}
                        # )
                        continue

                    if _ensure_pos and action["pos_candidates"] == []:
                        self._ignored_idxs.append([idx, ann_id, act_idx, "no_pos_candidates"])
                        # {ann_id: {"action_idx": act_idx, "ann_idx": idx, "issue": "no_pos_candidates"}}
                        continue

                    filtered_indexes.append([indexes[idx], act_idx])

            # print(f"Len of ignored: {len(self._ignored_idxs)}")
            return {"indexes": filtered_indexes}

        self.dataset_idxs = self.dataset.map(
            filter_actions_fn,
            batched=True,
            with_indices=True,
            remove_columns=self.dataset.column_names,
            num_proc=self.config.map_num_workers,
            load_from_cache_file=self.config.map_load_from_cache_file,
            desc=f"Making dataset idxs from: {filter_actions_fn.__name__}",
        )

    def _filter_candidates(
        self,
        **kwargs,
        # screenshot_margin: float = 1.0,  # was 1.5
        # max_area: float = 1e5,
        # enforce_clickable: bool = True,
    ):
        """used for pretrain objective
        filter out elements that are larger than 1e5 (~300*300 so 1/4th of 1200x1200)
        filter out elements that are further than 1.5x the viewport size

        """
        if DEBUG:
            self.config.map_num_workers = 1

        valid_candidates_map = mind2web_preprocess_data.valid_candidates_map
        batch_size = 256

        self.dataset = self.dataset.map(
            valid_candidates_map,
            batched=True,
            batch_size=batch_size,
            num_proc=self.config.map_num_workers,
            with_rank=True,
            load_from_cache_file=self.config.map_load_from_cache_file,
            # writer_batch_size=2000,
        )

    def setup_pretrain(self, **kwargs):
        self._filter_candidates(**kwargs)
        return self


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
                sample = self.get_action_from_trajectory(
                    trajectory=trajectory,
                    action_idx=action_idx,
                    return_from=self.return_from,
                )
                sample.trajectory = M2WTrajectory(trajectory_idx=t_idx, **self._include_json_filepath(trajectory))
                return sample
            except:
                pass
