from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

from bs4 import Tag
from PIL.Image import Image

from pretrain_mm import constants, logger
from pretrain_mm.utils.bbox_utils import BoundingBox


if TYPE_CHECKING:
    from pretrain_mm.datasets.mind2web.mind2web_datatypes import M2WAction


def parse_bounding_box_rect(bounding_box_rect: str, to_int: bool = True) -> BoundingBox:
    """
    The bounding box from osunlp/Mind2Web is in the format of x,y,width,height.
    this parse method is specific to mind2web. other datasets may use different format (floats) or encoding (patch num)
    we generally want x1,y1,x2,y2 for bounding box ease of use (although some bounding boxes are in y1,x1,y2,x2 format)
    """
    x1, y1, width, height = map(float, bounding_box_rect.split(","))
    x2, y2 = x1 + width, y1 + height

    if to_int:
        x1, x2, y1, y2 = map(round, [x1, x2, y1, y2])

    return x1, y1, x2, y2


# midpoints = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
#             bounding_box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def check_dirty_node(node: Tag) -> bool:
    """
    check if the node has a bounding box and if it does and is -1 it means hidden so we dont want that
    """

    if "bounding_box_rect" not in node.attrs or node["bounding_box_rect"] == "-1,-1,-1,-1":
        return False

    # some of the nodes that contents are only '\n' are pos candidates
    if len(node.contents) <= 1:
        return True

    for content in node.contents:
        if isinstance(content, Tag):
            if not check_dirty_node(content):
                return False
    return True


def check_node_has_text(node: Tag) -> bool:
    if node.text.strip() == "":
        return False
    return True


def parse_candidate(
    candidate: dict,
    parse_bounding_box: bool = True,
    to_int: bool = False,
    as_copy: bool = False,
) -> List[dict]:
    """
    use for pos_candidates and neg_candidates on mind2web dataset

    pos_candidates always seems to have dict keys:
        dict_keys(['attributes', 'backend_node_id', 'is_original_target', 'is_top_level_target', 'tag'])
    neg_candidates always seems to have dict keys:
        dict_keys(['attributes', 'backend_node_id', 'tag'])


    attributes can vary but generally looks like:
    dict_keys(['backend_node_id', 'bounding_box_rect', 'class', 'is_clickable', 'data_pw_testid_buckeye_candidate'])
    pos example

    dict_keys(['backend_node_id', 'bounding_box_rect', 'id', 'class', 'data_pw_testid_buckeye_candidate'])

    seems like:
         'backend_node_id', 'bounding_box_rect', 'class'
    are consistent in both pos and neg candidates
    """

    if as_copy:
        candidate = candidate.copy()

    candidate["attributes"] = json.loads(candidate["attributes"])
    if parse_bounding_box:
        candidate["attributes"]["bounding_box_rect"] = parse_bounding_box_rect(
            candidate["attributes"]["bounding_box_rect"], to_int=to_int
        )

    return candidate


def cand_out_of_viewport(candidate: dict, viewport_size: tuple[int, int], buffer_amt: float = 1.0) -> bool:
    bounding_box = candidate["attributes"]["bounding_box_rect"]
    if (bounding_box[2] > round(viewport_size[0] * buffer_amt)) or (
        bounding_box[3] > round(viewport_size[1] * buffer_amt)
    ):
        return True
    return False


def get_all_candidates_in_view(self, sample: "M2WAction", viewport_size: tuple[int, int] = constants.VIEWPORT_SIZE):
    in_viewport = []

    for candidate in sample.pos_candidates:
        parsed_candidate = parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)

        if not cand_out_of_viewport(parsed_candidate, viewport_size, buffer_amt=1.5):
            in_viewport.append((parsed_candidate, 1))

    for candidate in sample.neg_candidates:
        parsed_candidate = parse_candidate(candidate.copy(), parse_bounding_box=True, to_int=True)

        if not cand_out_of_viewport(parsed_candidate, viewport_size, buffer_amt=1.5):
            in_viewport.append((parsed_candidate, 0))

    return in_viewport, sample


def bounding_box_to_point(x1, y1, x2, y2) -> tuple[float, float]:
    """
    Converts the coordinates of a bounding box to the center point.

    Args:
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (float): The y-coordinate of the bottom-right corner of the bounding box.

    Returns:
        tuple[float, float]: The center point of the bounding box.
    """
    return (x1 + x2) / 2, (y1 + y2) / 2


# ----
# Image Utils
# ----


def crop_image_and_cand(
    image: Image, candidate: dict[str, str | dict], viewport_size: tuple[int, int] = constants.VIEWPORT_SIZE
):
    # for now just increase image size by 1.5 times if candidate is out of viewport
    start_height = 0
    width, height = viewport_size
    if candidate["attributes"]["bounding_box_rect"][3] > height:
        adj_height = int(height * 0.5)
        start_height = adj_height
        height += adj_height

        # since bounding box comes from html we need to adjust it to be in the cropped image
        candidate["attributes"]["bounding_box_rect"][1] -= adj_height
        candidate["attributes"]["bounding_box_rect"][3] -= adj_height

        # remove after debug
        _bbox = candidate["attributes"]["bounding_box_rect"]
        _cropped_to = [0, start_height, width, height]
        logger.info(f"Changed image and bbox, {_bbox} and {_cropped_to}")

    return image.crop((0, start_height, width, height))
