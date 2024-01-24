from dataclasses import dataclass, field
from typing import Literal, NamedTuple, TypeAlias

from PIL.Image import Image

from pretrain_mm.datasets.dataset_utils import DatasetConfig
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.utils.json_utils import read_json

ReturnFromTypes: TypeAlias = Literal["after", "before"]

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
    crop_image: bool = False
    include_html: bool = False

    # subset allows for testing/debugging quicker
    subset: int = None
    attach_config_to_sample: bool = False
    json_data_use_cache: bool = True


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
    annotation_id: str = None  # field(default=None, repr=False)
    image: Image = None  # field(default=None, init=False)

    # primarily for typing
    trajectory: "M2WTrajectory" = field(default=None, repr=False, init=True)

    def __post_init__(self):
        self.operation = ActionOp(**self.operation)

    @classmethod
    def from_trajectory(cls, action_idx: int, trajectory: "M2WTrajectory") -> "M2WAction":
        action_data = trajectory.actions[action_idx]
        return cls(
            action_idx=action_idx,
            annotation_id=trajectory.annotation_id,
            trajectory=trajectory,
            **action_data,
        )

    def load_image_from_filepath(
        self,
        task_dir: str = Mind2WebConfig.task_dir,
        screenshot_file: str = Mind2WebConfig.screenshot_file,
        return_from: ReturnFromTypes = "before",
        use_cache: bool = True,
    ) -> Image:
        json_data = self.trajectory.get_json_data(
            annotation_id=self.annotation_id,
            screenshot_file=screenshot_file,
            task_dir=task_dir,
            use_cache=use_cache,
        )
        action_data = json_data[self.action_idx]
        return read_image_from_b64(action_data[return_from]["screenshot"])


@dataclass
class M2WTrajectory:
    action_reprs: list[str]

    annotation_id: str
    confirmed_task: str
    website: str

    domain: str
    subdomain: str

    trajectory_idx: int = None
    actions: list[M2WAction] = field(default=None, repr=False)

    def get_json_data(
        self,
        annotation_id: str = None,
        screenshot_file: str = Mind2WebConfig.screenshot_file,
        task_dir: str = Mind2WebConfig.task_dir,
        use_cache: bool = Mind2WebConfig.json_data_use_cache,
    ) -> dict:
        """since the json data has bounding box preproecssed it might be worth using rather than parsing from the dataset"""
        return read_json(f"{task_dir}/task/{annotation_id or self.annotation_id}/{screenshot_file}", use_cache)
