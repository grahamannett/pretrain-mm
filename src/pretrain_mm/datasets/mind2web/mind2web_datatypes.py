from dataclasses import dataclass, field
from typing import NamedTuple


from PIL.Image import Image
from pretrain_mm.datasets.dataset_utils import DatasetConfig

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
    trajectory: "M2WTrajectory" = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.operation = ActionOp(**self.operation)


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
