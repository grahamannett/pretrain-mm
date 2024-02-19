import time
from dataclasses import dataclass
from pathlib import Path

from multiprocess import set_start_method
from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser, Serializable
from tinydb import Query, TinyDB

import paddleocr
from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets.mind2web import M2WAction, Mind2Web
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.utils.image_utils import read_image_from_b64
from pretrain_mm.utils.json_utils import read_json
from pretrain_mm.utils.ocr_helper import MultiOCRLabeler

BAD_CANDIDATE = {}
WIDTH, HEIGHT = VIEWPORT_SIZE_DICT["width"], VIEWPORT_SIZE_DICT["height"]


def invalid_bounding_box(bounding_box: tuple[int, int, int, int]) -> bool:
    if bounding_box is None:
        return True

    # some of the bounding boxes in html had negative values
    if any([(x < 0) for x in bounding_box]):
        return True

    # check if the x2,y2 are actual values
    if (bounding_box[2] <= 0) or (bounding_box[3] <= 0):
        return True

    # make sure the box has area otherwise the ocr tools will fail
    if (bounding_box[0] == bounding_box[2]) or (bounding_box[1] == bounding_box[3]):
        return True

    return False


def bounding_box_outside(
    bounding_box: tuple[int, int, int, int], viewport_cutoff: float = None, area_cutoff: float = None
) -> bool:
    bbox_width, bbox_height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

    # if more than 2x the height/width then its probably not even rendered
    if viewport_cutoff and (
        (bounding_box[2] > WIDTH * viewport_cutoff) or (bounding_box[3] > HEIGHT * viewport_cutoff)
    ):
        return True

    # if bbox area is more than area threshold (based on constants viewport)the viewport then its probably not a good candidate
    if (bbox_width * bbox_height) >= (area_cutoff * WIDTH * HEIGHT):
        return True

    return False


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
    use_subset: bool = False

    # for map
    num_proc: int = 8
    batch_size: int = 256
    max_candidates: int = 250
    writer_batch_size: int = 64

    dataset_name_out: str = "m2w-cands"


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


def make_map_fn(
    task_dir: str,
    screenshot_file: str,
    area_cutoff: float = 0.5,
    viewport_cutoff: float = 1.75,
    max_candidates: int = 100,
    _print_info: bool = False,
):
    """make a map function for the m2w dataset that uses ocr to label candidates

    Args:
        task_dir (str): _description_
        screenshot_file (str): _description_
        area_threshold (float, optional): _description_. Defaults to 0.5.
        viewport_threshold (float, optional): _description_. Defaults to 1.75.
        max_candidates (int, optional): most candidates are useless.  often there are 1k+. Defaults to 200.

    Returns:
        _type_: _description_
    """
    # want to output the data so it can be used like:
    # labels["annotation_id"][action_idx]["pos_candidates"][candidate_idx]["before"]["text"]['paddle']

    # NOTE: these must be redefined like this such that datasets.map can cache them
    _parse_candidate = m2w_utils.parse_candidate
    _read_image_from_b64 = read_image_from_b64
    _read_json = read_json

    def _setup_and_check_candidate(parsed_cand: dict) -> dict:
        # dunno if i should return as dict or the box but then make another dict in walrus

        # box will be in x1, y1, x2, y2 format
        bbox = parsed_cand["attributes"]["bounding_box_rect"]
        # bbox_width, bbox_height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

        if invalid_bounding_box(bbox) or bounding_box_outside(bbox, viewport_cutoff, area_cutoff):
            return BAD_CANDIDATE

        return {"bounding_box": bbox}

    def make_candidate_data(cand, labeler, json_data_act_idx, cand_idx):

        parsed_cand = _parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
        if ocr_results := _setup_and_check_candidate(parsed_cand):
            try:
                # dont read the image until bounding box is checked as its slow...
                before_image = _read_image_from_b64(json_data_act_idx["before"]["screenshot"])
                after_image = _read_image_from_b64(json_data_act_idx["after"]["screenshot"])
                ocr_results["before"] = labeler(before_image.crop(ocr_results["bounding_box"]))
                ocr_results["after"] = labeler(after_image.crop(ocr_results["bounding_box"]))
            except Exception as err:
                ocr_results = BAD_CANDIDATE

        return {"cand_idx": cand_idx, "backend_node_id": cand["backend_node_id"], **ocr_results}

    # NOTE:
    #   - easyocr is buggy and only works on gpu set by CUDA_VISIBLE_DEVICES
    #   - paddleocr is not serializable so breaks datasets.map cache
    # use_easy = False  # or easyocr.Reader(lang_list=["en"], gpu=True)
    # use_paddle = True  # or like paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)

    def _map_fn(data: dict, rank: int):
        use_paddle = paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)
        ocr_labeler = MultiOCRLabeler(use_paddle=use_paddle, use_easy=False)

        rank = 0 if rank is None else rank  # if not using num_proc

        outdata = {
            "annotation_id": [],
            "actions": [],
        }

        # keys in data are:
        # ['confirmed_task', 'domain', 'actions', 'annotation_id', 'subdomain', 'website', 'action_reprs']
        # annotation_id: list[str], actions: list[dict]
        for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):

            json_filepath = f"{task_dir}/task/{ann_id}/{screenshot_file}"
            json_data = _read_json(json_filepath, use_cache=False)

            def _append_to_cand_data(cand_data, act_idx, cand_idx, cand):
                try:
                    cand_idx_data = make_candidate_data(cand, ocr_labeler, json_data[act_idx], cand_idx)
                    cand_data.append(cand_idx_data)
                except Exception as err:
                    logger.error(f"ERROR @ {ann_id} {err}")
                return cand_data

            # pyarrow requires dicts to have keys of str/byes, cant use int keys. same for pos/neg cands
            action_data = []

            for act_i, action in enumerate(actions):

                _cands = {"pos_candidates": [], "neg_candidates": []}

                for cand_i, cand in enumerate(action["pos_candidates"]):
                    _cands["pos_candidates"] = _append_to_cand_data(_cands["pos_candidates"], act_i, cand_i, cand)
                    # cand_idx_data = make_candidate_data(cand, ocr_labeler, json_data[act_i], cand_idx)
                    # _cand_data["pos_candidates"].append(cand_idx_data)

                for cand_i, cand in enumerate(action["neg_candidates"]):
                    _cands["neg_candidates"] = _append_to_cand_data(_cands["neg_candidates"], act_i, cand_i, cand)
                    # cand_idx_data = make_candidate_data(cand, ocr_labeler, json_data[act_i], cand_idx)
                    # _cand_data["neg_candidates"].append(cand_idx_data)

                    if cand_i > max_candidates:
                        break

                action_data.append(_cands)

            outdata["annotation_id"].append(ann_id)
            outdata["actions"].append(action_data)

        return outdata

    return _map_fn


def mp(dataset, num_proc):
    # def create_chunks(dataset, dataset_outdir: str):
    #     pass
    idxs = list(range(len(dataset)))
    chunks = [idxs[i : i + num_proc] for i in range(0, len(idxs), num_proc)]

    # for chunk in chunks


def flatten_actions(data: dict, rank: int):
    out_data = {
        "annotation_id": [],
        "actions": [],
        "confirmed_task": [],
        "domain": [],
        "subdomain": [],
        "website": [],
        "action_reprs": [],
    }
    for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
        for act_i, action in enumerate(actions):
            out_data["annotation_id"].append(ann_id)
            out_data["actions"].append(action)
            out_data["confirmed_task"].append(data["confirmed_task"][idx])
            out_data["domain"].append(data["domain"][idx])
            out_data["subdomain"].append(data["subdomain"][idx])
            out_data["website"].append(data["website"][idx])
            out_data["action_reprs"].append(data["action_reprs"][idx][act_i])
    return out_data


def _setup_and_check_candidate(parsed_cand: dict, viewport_threshold=None, area_threshold=None) -> dict:
    # dunno if i should return as dict or the box but then make another dict in walrus

    # box will be in x1, y1, x2, y2 format
    bounding_box = parsed_cand["attributes"]["bounding_box_rect"]
    bbox_width, bbox_height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

    # check if bounding box area is 0 (or negative) -> if so the ocr tools will fail
    if invalid_bounding_box(bounding_box):
        return BAD_CANDIDATE

    # if more than 2x the height/width then its probably not even rendered
    if viewport_threshold and (
        (bounding_box[2] > WIDTH * viewport_threshold) or (bounding_box[3] > HEIGHT * viewport_threshold)
    ):
        return BAD_CANDIDATE
    # if bbox area is more than area threshold (based on constants viewport)the viewport then its probably not a good candidate
    if (bbox_width * bbox_height) >= (area_threshold * WIDTH * HEIGHT):
        return BAD_CANDIDATE

    return


def map_on_pos_cands(data: dict, rank: int, cand_type: str, task_dir: str, screenshot_file: str):

    _parse_candidate = m2w_utils.parse_candidate
    _read_image_from_b64 = read_image_from_b64
    _read_json = read_json

    def make_candidate_data(cand, labeler, json_data, act_idx, cand_idx):
        parsed_cand = _parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
        bbox = parsed_cand["attributes"]["bounding_box_rect"]

        if invalid_bounding_box(bbox) or bounding_box_outside(bbox, viewport_cutoff=1.75, area_cutoff=0.5):
            ocr_results = BAD_CANDIDATE
        else:
            ocr_results = {"bounding_box": bbox}
            # dont read the image until bounding box is checked as its slow...
            before_image = json_data[act_idx]["before"]["screenshot"]

            if before_image != "":
                before_image = _read_image_from_b64(before_image)
                ocr_results["before"] = labeler(before_image.crop(bbox))

            after_image = json_data[act_idx]["after"]["screenshot"]
            if after_image != "":
                after_image = _read_image_from_b64(after_image)
                ocr_results["after"] = labeler(after_image.crop(bbox))

        return {"cand_idx": cand_idx, "backend_node_id": cand["backend_node_id"], **ocr_results}

    use_paddle = paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)
    ocr_labeler = MultiOCRLabeler(use_paddle=use_paddle, use_easy=False)

    out_data = {
        "annotation_id": [],
        "actions": [],
        "action_reprs": [],
        "candidates": [],
        # "confirmed_task": [],
        # "domain": [],
        # "subdomain": [],
        # "website": [],
    }
    for idx, (ann_id, actions) in enumerate(zip(data["annotation_id"], data["actions"])):
        json_filepath = f"{task_dir}/task/{ann_id}/{screenshot_file}"
        json_data = _read_json(json_filepath, use_cache=False)

        for act_i, action in enumerate(actions):
            for cand_i, cand in enumerate(action[cand_type]):
                try:
                    cand_data = make_candidate_data(cand, ocr_labeler, json_data, act_i, cand_i)
                except Exception as err:
                    logger.log(f"ERROR @ {ann_id} act_i: {act_i} with Error: {err}")
                    continue

                cand_data["cand_type"] = cand_type
                out_data["candidates"].append(cand_data)
                out_data["annotation_id"].append(ann_id)
                out_data["actions"].append(action)
                out_data["action_reprs"].append(data["action_reprs"][idx][act_i])

                # out_data["confirmed_task"].append(data["confirmed_task"][idx])
                # out_data["domain"].append(data["domain"][idx])
                # out_data["subdomain"].append(data["subdomain"][idx])
                # out_data["website"].append(data["website"][idx])
    return out_data


def to_processes_json(dataset):
    out_info = {}
    pbar = logger.progress(start=True, time_remaining=True)
    pbar_task = pbar.add_task("Generating list of annotations to process", total=len(dataset))

    for idx, sample in enumerate(dataset):
        ann_id = sample["annotation_id"]
        out_info[ann_id] = {"actions": [[] for _ in range(len(sample["actions"]))], "done": False, "dataset_idx": idx}
        pbar.update(pbar_task, advance=1)
    pbar.stop()

    return out_info


def process_dataset(dataset, task_dir, screenshot_file, cutoff=None):

    outdata = {}
    baddata = []
    n_err = 0

    pbar = logger.progress(start=True, time_remaining=True, ensure_exit=True)
    pbar_task = pbar.add_task("Processing dataset", total=cutoff or len(dataset))

    use_paddle = paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)
    ocr_labeler = MultiOCRLabeler(use_paddle=use_paddle, use_easy=False)

    for idx in range(len(dataset)):
        sample = dataset[idx]

        ann_id = sample["annotation_id"]

        if ann_id not in outdata:
            outdata[ann_id] = {"actions": {}}

        json_filepath = f"{task_dir}/task/{sample['annotation_id']}/{screenshot_file}"
        json_data = read_json(json_filepath, use_cache=False)

        for a_idx, action in enumerate(sample["actions"]):
            for c_idx, cand in enumerate(action["pos_candidates"]):

                parsed_cand = m2w_utils.parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
                bbox = parsed_cand["attributes"]["bounding_box_rect"]

                try:
                    if invalid_bounding_box(bbox):
                        raise ValueError("Bounding box area is 0")

                    before_image = json_data[a_idx]["before"]["screenshot"]
                    after_image = json_data[a_idx]["after"]["screenshot"]

                    if before_image == "" or after_image == "":
                        raise ValueError("No image data")

                    before_image = read_image_from_b64(before_image)
                    after_image = read_image_from_b64(after_image)

                    before_results = ocr_labeler(before_image.crop(bbox))
                    after_results = ocr_labeler(after_image.crop(bbox))

                    results = {
                        "before": before_results,
                        "after": after_results,
                    }

                    if a_idx not in outdata[ann_id]["actions"]:
                        # for the candidates, using list since the order seems arbitrary (e.g. keys dont matter)
                        outdata[ann_id]["actions"][a_idx] = {"pos_candidates": []}

                    outdata[ann_id]["actions"][a_idx]["pos_candidates"].append(results)

                except Exception as err:
                    n_err += 1
                    logger.warn(f"Err[{n_err}] ann: {ann_id} act: {a_idx} cand: {c_idx} with Error:{err}")
                    baddata.append(
                        {
                            "annotation_id": ann_id,
                            "action_idx": a_idx,
                            "cand_idx": c_idx,
                            "json_filepath": json_filepath,
                            "candidate": cand,
                            "bbox": bbox,
                            # "error": err,
                        }
                    )

        pbar.update(pbar_task, advance=1)
    pbar.stop()

    return outdata, baddata


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

    train_ds = Mind2Web(split="train")
    test_ds = Mind2Web(split="test")

    import torch

    train_ocr = torch.load("output/processed/train_ds_raw_output.pt")

    # to_process_train_ds = to_processes_json(train_ds.dataset)
    # breakpoint()

    # train_actions_dataset = train_ds.dataset.map(
    #     flatten_actions,
    #     batched=True,
    #     batch_size=config.batch_size,
    #     num_proc=config.num_proc,
    #     with_rank=True,
    # )

    task_dir = train_ds.config.task_dir
    screenshot_file = train_ds.config.screenshot_file

    # DEBUG WITH BELOW:
    # sample_idx = 196
    # sample = train_ds.dataset[sample_idx]
    # json_filepath = f"{task_dir}/task/{sample['annotation_id']}/{screenshot_file}"
    # json_data = read_json(json_filepath, use_cache=False)

    # use_paddle = paddleocr.PaddleOCR(lang="en", use_angle_cls=True, use_gpu=True, show_log=False)
    # ocr_labeler = MultiOCRLabeler(use_paddle=use_paddle, use_easy=False)

    # for a_idx, act in enumerate(sample["actions"]):
    #     for c_idx, cand in enumerate(act["pos_candidates"]):
    #         parsed_cand = m2w_utils.parse_candidate(
    #             act["pos_candidates"][c_idx].copy(), parse_bounding_box=True, to_int=True
    #         )
    #         bounding_box = parsed_cand["attributes"]["bounding_box_rect"]

    #         before_image = read_image_from_b64(json_data[a_idx]["before"]["screenshot"])
    #         after_image = read_image_from_b64(json_data[a_idx]["after"]["screenshot"])

    #         if a_idx > 0:
    #             breakpoint()

    #         before_results = ocr_labeler(before_image.crop(bounding_box))
    #         after_results = ocr_labeler(after_image.crop(bounding_box))
    #         logger.log(f"got results for a_idx: {a_idx} {c_idx}")

    # breakpoint()

    # train_ds_output, train_baddata = process_dataset(
    #     train_ds.dataset,
    #     task_dir=task_dir,
    #     screenshot_file=screenshot_file,
    # )

    # test_ds_output, test_baddata = process_dataset(
    #     test_ds.dataset,
    #     task_dir=test_ds.config.task_dir,
    #     screenshot_file=test_ds.config.screenshot_file,
    # )

    # import torch

    # torch.save(train_ds_output, "output/processed/train_ds_raw_output.pt")
    # torch.save(train_baddata, "output/processed/train_baddata.pt")

    # torch.save(test_ds_output, "output/processed/test_ds_raw_output.pt")
    # torch.save(test_baddata, "output/processed/test_baddata.pt")

    logger.log("DONE WITH THE ATTEMPT ABOVE")

    # train_actions_dataset = train_ds.dataset
    # train_actions_dataset = train_actions_dataset.select(range(300))

    # train_pos_cands_dataset = train_actions_dataset.map(
    #     map_on_pos_cands,
    #     batched=True,
    #     batch_size=config.batch_size,
    #     writer_batch_size=config.writer_batch_size,
    #     num_proc=config.num_proc,
    #     remove_columns=["confirmed_task", "domain", "subdomain", "website"],
    #     with_rank=True,
    #     fn_kwargs={
    #         "task_dir": train_ds.config.task_dir,
    #         "screenshot_file": train_ds.config.screenshot_file,
    #         "cand_type": "pos_candidates",
    #     },
    # )
    # logger.log(f"done with m2w-pos-cands/train")
    # train_pos_cands_dataset.save_to_disk(f"output/m2w-pos-cands/train")
    # breakpoint()

    # # DO SAME FOR TEST DATASET
    # train_actions_dataset = test_ds.dataset

    # train_pos_cands_dataset = train_actions_dataset.map(
    #     map_on_pos_cands,
    #     batched=True,
    #     batch_size=config.batch_size,
    #     writer_batch_size=config.writer_batch_size,
    #     num_proc=config.num_proc,
    #     remove_columns=["confirmed_task", "domain", "subdomain", "website"],
    #     with_rank=True,
    #     fn_kwargs={
    #         "task_dir": test_ds.config.task_dir,
    #         "screenshot_file": test_ds.config.screenshot_file,
    #         "cand_type": "pos_candidates",
    #     },
    # )
    # logger.log(f"done with m2w-pos-cands/test")
    # train_pos_cands_dataset.save_to_disk(f"output/m2w-pos-cands/test")

    map_fn_train_ds = make_map_fn(
        train_ds.config.task_dir,
        train_ds.config.screenshot_file,
        max_candidates=config.max_candidates,
    )

    map_fn_test_ds = make_map_fn(
        test_ds.config.task_dir,
        test_ds.config.screenshot_file,
        max_candidates=config.max_candidates,
    )

    # subset for debugging/testing
    train_for_map = train_ds.dataset.select(range(2)) if config.use_subset else train_ds.dataset
    test_for_map = test_ds.dataset.select(range(2)) if config.use_subset else test_ds.dataset

    logger.info(f"size of train dataset {len(train_ds)}")
    logger.info(f"size of test dataset {len(test_ds)}")

    set_start_method("spawn")

    train_cand_ds = train_for_map.map(
        map_fn_train_ds,
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_proc,
        remove_columns=["confirmed_task", "domain", "subdomain", "website", "action_reprs"],
        with_rank=True,
    )

    train_cand_ds.save_to_disk(f"output/hfmap/{config.dataset_name_out}/train")
    train_cand_ds.push_to_hub(f"besiktas/{config.dataset_name_out}", split="train")

    test_cand_ds = test_for_map.map(
        map_fn_test_ds,
        batched=True,
        batch_size=config.batch_size,
        num_proc=config.num_proc,
        remove_columns=["confirmed_task", "domain", "subdomain", "website", "action_reprs"],
        with_rank=True,
    )

    test_cand_ds.save_to_disk(f"output/hfmap/{config.dataset_name_out}/test")
    train_cand_ds.push_to_hub(f"besiktas/{config.dataset_name_out}", split="test")

    # # try:
    # #     # test_cand_ds.save_to_disk("output/m2w-cands/test")
    # # except:
    # #     logger.warn("ERROR SAVING")
    # #     breakpoint()

    # # try:
    # #     test_cand_ds.push_to_hub("besiktas/m2w-cands", split="test")
    # # except:
    # #     logger.warn("ERROR UPLOADING")
    # #     breakpoint()

    # # logger.info("ALL GOOD")
    # # breakpoint()

    # # t2 = time.perf_counter()

    # # task_dir = dataset.config.task_dir
    # # screenshot_file = dataset.config.screenshot_file

    # # parse_candidates = m2w_utils.parse_candidate
    # # read_image_from_b64_ = read_image_from_b64
    # # read_json_ = read_json

    # # labeler = OCRLabeler(use_zip=False)
