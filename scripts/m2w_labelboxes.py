import time
from dataclasses import dataclass
from pathlib import Path
import os

from pretrain_mm.utils.json_utils import read_json
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils

import paddleocr
import easyocr

from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser, Serializable
from tinydb import Query, TinyDB
import torch

from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets.mind2web import M2WAction, Mind2Web
from pretrain_mm.utils.ocr_helper import OCRLabeler

parse_candidates = m2w_utils.parse_candidate
read_image_from_b64_ = read_image_from_b64
read_json_ = read_json


@dataclass
class DataLabelConfig(Serializable):
    label_data_file: str = "labeled_data.json"
    outdir: str = "output"
    # need to save image each time
    current_image_saved_path: str = "output/tmp_current_image.png"

    data_split: str = "test"
    return_from: str = "before"
    candidate_type: str = "pos_candidates"
    candidate_idx: int = 0

    skip_seen: bool = False
    # since sequential might want to shuffle
    shuffle_idxs: bool = False

    # for debugging - allow skipping after drawing but slow down
    sleep_and_skip_input: int = -1


class DataLabeling:
    def __init__(
        self,
        label_data_file: str = DataLabelConfig.label_data_file,
        outdir: str = DataLabelConfig.outdir,
        split: str = DataLabelConfig.data_split,
        current_image_saved_path: str = DataLabelConfig.current_image_saved_path,
        candidate_idx: int = DataLabelConfig.candidate_idx,
        return_from: str = DataLabelConfig.return_from,
        candidate_type: str = DataLabelConfig.candidate_type,
        skip_seen: bool = DataLabelConfig.skip_seen,
    ):
        self.split = split  # i think i want to incorporate this into the label_data_file name
        self.label_data_file = label_data_file  # the good data
        self.current_image_saved_path = current_image_saved_path  # where each image is that I will label

        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

        # self.user_input_prompt = "[bold italic cyan]>>> Provide the label for this bounding box: "

        # related to if we have seen the sample before
        self.skip_seen = skip_seen

        # data might be important for future if including negative cands
        self.candidate_type = candidate_type
        self.candidate_idx = candidate_idx
        self.return_from = return_from

        # should have used tinydb or similar to start with as dont have to think about saving/loading and get build in search
        self.db = TinyDB(self.label_filepath)

        self.labeled_table = self.db.table("labeled")
        self.issue_table = self.db.table("issue")
        self.tables = {"labeled": self.labeled_table, "issue": self.issue_table}

        self._setup()

    @property
    def label_filepath(self):
        return self.outdir / self.label_data_file

    def __atexit__(self):
        self.save()

    def _setup(self):
        pass

    def draw_bounding_box(
        self, sample: M2WAction, draw_rectangle_kwargs: dict = {"outline": "red", "width": 3}
    ) -> None:
        image: Image.Image = sample.image

        # image needs crop likely since it will be the full browser window not viewport so hard to see
        image_w, image_h = image.size
        image = image.crop((0, 0, image_w, VIEWPORT_SIZE_DICT["height"]))

        draw = ImageDraw.Draw(image)
        draw.rectangle(sample.bounding_box, **draw_rectangle_kwargs)
        image.save(self.current_image_saved_path)

    def check_if_sample_in_seen(self, sample: M2WAction) -> tuple[bool, TinyDB.table, int]:
        Data = Query()

        ann_id = sample.annotation_id
        action_idx = sample.action_idx

        already_labeled = self.labeled_table.search((Data.annotation_id == ann_id) & (Data.action_idx == action_idx))
        already_issued = self.issue_table.search((Data.annotation_id == ann_id) & (Data.action_idx == action_idx))

        if already_labeled != []:
            return True, self.labeled_table, already_labeled[0].doc_id

        if already_issued != []:
            return True, self.issue_table, already_issued[0].doc_id

        return False, None, None


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"


def make_map_fn(task_dir: str, screenshot_file: str, area_threshold: float = 0.5):
    # want to output the data so it can be used like:
    # labels["annotation_id"][action_idx]["pos_candidates"][candidate_idx]["before"]["text"]['paddle']

    WIDTH, HEIGHT = VIEWPORT_SIZE_DICT["width"], VIEWPORT_SIZE_DICT["height"]
    out_of_view: float = 2.0

    def _setup_and_check_candidate(parsed_cand: dict) -> dict:
        # dunno if i should return as dict or the box but then make another dict in walrus
        BAD_CANDIDATE = {}

        # box will be in x1, y1, x2, y2 format
        bounding_box = parsed_cand["attributes"]["bounding_box_rect"]
        bbox_width, bbox_height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

        # check if bounding box area is 0 (or negative) -> if so the ocr tools will fail
        if (bbox_width <= 0) or (bbox_height <= 0):
            return BAD_CANDIDATE

        # if more than 2x the height/width then its probably not even rendered
        if (bounding_box[2] > WIDTH * out_of_view) or (bounding_box[3] > HEIGHT * out_of_view):
            return BAD_CANDIDATE

        # if bbox area is more than area threshold (based on constants viewport)the viewport then its probably not a good candidate
        if (bbox_width * bbox_height) >= (area_threshold * WIDTH * HEIGHT):
            return BAD_CANDIDATE

        return {"bounding_box": bounding_box}

    def make_candidate_data(cand, labeler, before_image, after_image, cand_idx):

        parsed_cand = parse_candidates(cand.copy(), parse_bounding_box=True, to_int=True)

        if ocr_results := _setup_and_check_candidate(parsed_cand):
            # before and after text/probabilities like {"text": {"paddle": }, "prob": {"paddle": }}
            ocr_results["before"] = labeler(before_image.crop(ocr_results["bounding_box"]))
            ocr_results["after"] = labeler(after_image.crop(ocr_results["bounding_box"]))

        return {"cand_idx": cand_idx, "backend_node_id": cand["backend_node_id"], **ocr_results}

    def _map_fn(data: dict, rank: int):
        rank = rank if rank != None else 0

        # BUG: Doesnt change regardless of me setting here: os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        device = str(rank % torch.cuda.device_count())

        # NOTE: - easyocr is buggy and only works if on gpu 0 since it does not cast buffer
        # BUG: these dont seem to really use seperate gpus...
        paddleinst = paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)
        easyinst = easyocr.Reader(["en"], gpu=True)

        ocr_labeler = OCRLabeler(use_paddle=paddleinst, use_easy=easyinst)

        outdata = {
            "annotation_id": [],
            "actions": [],
        }

        # keys in data are:
        # ['confirmed_task', 'domain', 'actions', 'annotation_id', 'subdomain', 'website', 'action_reprs']
        # annotation_id are list of str, actions are list of dicts
        for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):

            json_filepath = f"{task_dir}/task/{ann_id}/{screenshot_file}"
            json_data = read_json_(json_filepath, use_cache=True)

            # pyarrow requires dicts to have keys of type str or byes, cant use int
            # so makes no sense to use dict with action_idx as keys since would have to convert to str
            # similarily with cand_data['pos_candidates'] below
            action_data = []

            for act_idx, action in enumerate(actions):

                before_image = read_image_from_b64_(json_data[act_idx]["before"]["screenshot"])
                after_image = read_image_from_b64_(json_data[act_idx]["after"]["screenshot"])

                img_width, img_height = before_image.size

                cand_data = {
                    "pos_candidates": [],
                    "neg_candidates": [],
                }

                for cand_idx, cand in enumerate(action["pos_candidates"]):
                    cand_idx_data = make_candidate_data(cand, ocr_labeler, before_image, after_image, cand_idx)
                    cand_data["pos_candidates"].append(cand_idx_data)

                for cand_idx, cand in enumerate(action["neg_candidates"]):
                    cand_idx_data = make_candidate_data(cand, ocr_labeler, before_image, after_image, cand_idx)
                    cand_data["neg_candidates"].append(cand_idx_data)

                action_data.append(cand_data)

            outdata["annotation_id"].append(ann_id)
            outdata["actions"].append(action_data)

        return outdata

    return _map_fn


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DataLabelConfig, dest="data_label_config")

    args = parser.parse_args()

    config: DataLabelConfig = args.data_label_config

    data_labeler = DataLabeling(
        label_data_file=config.label_data_file,
        outdir=config.outdir,
        split=config.data_split,
        current_image_saved_path=config.current_image_saved_path,
        skip_seen=config.skip_seen,
        return_from=config.return_from,
        candidate_type=config.candidate_type,
        candidate_idx=config.candidate_idx,
    )

    from multiprocess import set_start_method

    train_ds = Mind2Web(split="train")
    test_ds = Mind2Web(split="test")

    map_fn_train_ds = make_map_fn(train_ds.config.task_dir, train_ds.config.screenshot_file)
    map_fn_test_ds = make_map_fn(test_ds.config.task_dir, test_ds.config.screenshot_file)  # , ocr_labeler=ocr_labeler)

    train_ds = train_ds.dataset.select(range(8))
    test_ds = test_ds.dataset.select(range(8))

    logger.info(f"size of train dataset {len(train_ds)}")
    logger.info(f"size of test dataset {len(test_ds)}")

    t1 = time.perf_counter()

    num_proc = int(os.environ.get("NUM_PROC", 1))

    set_start_method("spawn")

    test_cand_ds = test_ds.map(
        map_fn_test_ds,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        remove_columns=["confirmed_task", "domain", "subdomain", "website", "action_reprs"],
        with_rank=True,
    )

    t2 = time.perf_counter()
    train_cand_ds = train_ds.map(
        map_fn_train_ds,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        remove_columns=["confirmed_task", "domain", "subdomain", "website", "action_reprs"],
        with_rank=True,
    )

    t3 = time.perf_counter()
    print(f"Time to map each: {t2-t1} | {t3 - t2}")

    test_cand_ds.push_to_hub("besiktas/m2w-cands", split="test")
    train_cand_ds.push_to_hub("besiktas/m2w-cands", split="train")

    # t2 = time.perf_counter()

    # task_dir = dataset.config.task_dir
    # screenshot_file = dataset.config.screenshot_file

    # parse_candidates = m2w_utils.parse_candidate
    # read_image_from_b64_ = read_image_from_b64
    # read_json_ = read_json

    # labeler = OCRLabeler(use_zip=False)
