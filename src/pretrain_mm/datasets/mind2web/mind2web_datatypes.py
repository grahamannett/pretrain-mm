import re
from dataclasses import dataclass, field
from enum import StrEnum, auto

from PIL.Image import Image

from pretrain_mm.datasets.mind2web.mind2web_utils import parse_candidate
from pretrain_mm.datasets.utils.dataset_utils import DatasetConfig
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.utils.json_utils import read_json


# this pattern should match things like the following:
#   '[textbox]  Search -> TYPE: black sleeping bag', '[button]  Search -> CLICK', '[textbox]  Upper Bound -> TYPE: 40'
# into groups of target_type, target_value, action_type, action_value (4 groups)
action_repr_pattern = re.compile(r"\[([^]]+)\]\s*(.*?)\s*->\s*(\w+):?\s*(.*)")


# === === === === ===
# Dataclasses/Sample Related


class ReturnFromTypes(StrEnum):
    before = auto()
    after = auto()

    def flip(self) -> "ReturnFromTypes":
        """flip return from before to after and vice versa"""
        return {ReturnFromTypes.after: ReturnFromTypes.before, ReturnFromTypes.before: ReturnFromTypes.after}[self]


class ActionType(StrEnum):
    type = auto()
    click = auto()
    select = auto()


# class ActionOp(NamedTuple): change to use dataclass so can post init
@dataclass
class ActionOp:
    op: str | ActionType  # not certain yet all vals here but at least 'SELECT', 'CLICK', 'TYPE'
    original_op: str  # seems like this is one of 'SELECT', 'CLICK', 'TYPE', 'HOVER'
    value: str

    def __post_init__(self):
        self.op = ActionType[self.op.lower()] if isinstance(self.op, str) else self.op


@dataclass
class ActionRepresentation:
    raw: str = field(repr=False)

    # values to parse into
    target_type: str = None
    target_value: str = None  # this is optional
    # after `->`
    op_type: ActionType = None
    op_value: str = None  # this is optional

    def __post_init__(self):
        self.parse()

    def parse(self):
        match = action_repr_pattern.match(self.raw)

        if match is None:
            raise ValueError(f"Could not parse action representation: {self.raw}")

        self.target_type, target_value, op_type, op_value = match.groups()

        # these may be empty strings so set to None
        self.target_value = target_value or None
        self.op_value = op_value or None

        self.op_type = ActionType[op_type.lower()]

    def format(self, cb: callable = None) -> str:
        """format the action representation back into a string

        if you want to format the way ActionRepr is printed in task instructions, use this
        """
        if cb:
            return cb(self)

        # return f"[{self.target_type}] {self.target_value} -> {self.op_type}: {self.op_value}"
        return self.raw


@dataclass
class Mind2WebConfig(DatasetConfig):
    #
    dataset_path: str = "osunlp/Mind2Web"
    data_files: str = "**/*.json"  # needed for test data
    split: str = "train"  # for test we will need to read the files from

    show_progress: bool = True

    # this is where the images are as b64/mhtml
    # e.g. "/data/graham/datasets/mind2web/data"
    task_dir: str = None
    screenshot_file: str = "processed/screenshot.json"

    viewport_size: tuple[int, int] = (1280, 1080)  # {"width": 1280, "height": 1080}
    crop_image: bool = False
    include_html: bool = False

    # subset allows for testing/debugging quicker
    subset: int = None
    attach_config_to_sample: bool = False
    json_data_use_cache: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.task_dir is None:
            self._init_from_dev_config(ensure_set=["task_dir"])


@dataclass
class M2WAction:
    action_idx: int

    action_uid: str
    operation: ActionOp

    pos_candidates: list[dict]
    # dont show these b/c there are us ually 100+
    neg_candidates: list[dict] = field(default_factory=list, repr=False)

    # probably not using these as i am using screenshot
    cleaned_html: str = field(default=None, repr=False)
    raw_html: str = field(default=None, repr=False)

    # info needed from trajectory
    # annotation_id: str = None  # field(default=None, repr=False)
    image: Image = None  # field(default=None, init=False)

    # primarily for typing
    trajectory: "M2WTrajectory" = field(default=None, repr=False, init=True)
    return_from: ReturnFromTypes = field(default=ReturnFromTypes.before, repr=False)

    def __post_init__(self):
        self.operation = ActionOp(**self.operation)

    @classmethod
    def from_trajectory(cls, action_idx: int, trajectory: "M2WTrajectory", **action_kwargs) -> "M2WAction":
        action_data = trajectory.actions[action_idx]
        return cls(
            action_idx=action_idx,
            # annotation_id=trajectory.annotation_id,
            trajectory=trajectory,
            **action_data,
            **action_kwargs,  # include last to allow for overriden values
        )

    @property
    def action_repr(self) -> ActionRepresentation:
        return ActionRepresentation(self.trajectory.action_reprs[self.action_idx])

    @property
    def action_repr_previous(self) -> list[ActionRepresentation]:
        return [ActionRepresentation(o) for o in self.trajectory.action_reprs[: self.action_idx]]

    @property
    def annotation_id(self) -> str:
        return self.trajectory.annotation_id

    @property
    def confirmed_task(self) -> str:
        return self.trajectory.confirmed_task

    @property
    def trajectory_idx(self) -> int:
        return self.trajectory.trajectory_idx

    def get_bounding_box(self, cand_type: str = "pos_candidates", cand_idx: int = 0) -> tuple[int, int, int, int]:
        candidate = getattr(self, cand_type)[cand_idx]
        parsed_candidate = parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)
        return parsed_candidate["attributes"]["bounding_box_rect"]

    def load_image_from_filepath(
        self,
        task_dir: str = Mind2WebConfig.task_dir,
        screenshot_file: str = Mind2WebConfig.screenshot_file,
        return_from: ReturnFromTypes = None,
        use_cache: bool = True,
    ) -> Image:
        json_data = self.trajectory._get_json_data(
            annotation_id=self.annotation_id,
            screenshot_file=screenshot_file,
            task_dir=task_dir,
            use_cache=use_cache,
        )
        action_data = json_data[self.action_idx]

        # set these values on the instance if you need them for flip
        self.screenshot_file = screenshot_file
        self.task_dir = task_dir
        self.return_from = return_from or self.return_from

        return read_image_from_b64(action_data[return_from]["screenshot"])

    def flip_image_return_from(self) -> Image:
        self.return_from = self.return_from.flip()
        self.image = self.load_image_from_filepath(self.task_dir, self.screenshot_file, self.return_from)
        return self.image


@dataclass
class M2WTrajectory:
    action_reprs: list[str] | list[ActionRepresentation]

    annotation_id: str
    confirmed_task: str
    website: str

    domain: str
    subdomain: str

    trajectory_idx: int = None
    actions: list[M2WAction] = field(default=None, repr=False)

    # json_filepath is mostly used for data labeling/tagging
    json_filepath: str = field(default=None, repr=False)

    def __len__(self):
        return len(self.actions)

    def _get_json_data(
        self,
        annotation_id: str = None,
        screenshot_file: str = Mind2WebConfig.screenshot_file,
        task_dir: str = Mind2WebConfig.task_dir,
        use_cache: bool = Mind2WebConfig.json_data_use_cache,
    ) -> dict:
        """since the json data has bounding box preproecssed it might be worth using rather than parsing from the dataset"""
        return M2WTrajectory.get_json_data(
            annotation_id=annotation_id or self.annotation_id,
            screenshot_file=screenshot_file,
            task_dir=task_dir,
            use_cache=use_cache,
        )

    @staticmethod
    def get_json_data(
        annotation_id: str = None,
        screenshot_file: str = Mind2WebConfig.screenshot_file,
        task_dir: str = Mind2WebConfig.task_dir,
        use_cache: bool = Mind2WebConfig.json_data_use_cache,
        json_filepath: str = None,
    ) -> dict:
        """since the json data has bounding box preproecssed it might be worth using rather than parsing from the dataset"""
        json_filepath = json_filepath or f"{task_dir}/task/{annotation_id}/{screenshot_file}"
        return read_json(json_filepath, use_cache)


if __name__ == "__main__":
    # json_file = "/data/graham/datasets/mind2web/data/raw_dump/task/4f395aad-6f10-4055-932a-d2af443e6bfa/processed/screenshot.json"
    # json_data = read_json(json_file)
    # before_image = read_image_from_b64(json_data[0]["before"]["screenshot"])
    # breakpoint()
    return_from = ReturnFromTypes("before")
    assert return_from == ReturnFromTypes.before

    return_from_flipped = return_from.flip()
    breakpoint()
    assert return_from == ReturnFromTypes.after
