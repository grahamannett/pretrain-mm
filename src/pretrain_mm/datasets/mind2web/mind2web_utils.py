import json
from functools import lru_cache
from numbers import Number
from typing import List, Literal, TypeAlias

from bs4 import Tag
from PIL import Image, ImageDraw

ReturnFromTypes: TypeAlias = Literal["after", "before"]


@lru_cache(maxsize=128)
def _read_json(filename: str) -> dict:
    with open(filename) as f_in:
        return json.load(f_in)


def read_json(filename: str, use_cache: bool = True) -> dict:
    # if use_cache:
    func = _read_json if use_cache else _read_json.__wrapped__
    return func(filename)


def box_task(bounding_box_rect):
    return "<box>" + ", ".join([v for v in bounding_box_rect]) + "</box>"


def parse_bounding_box_rect(bounding_box_rect: str, to_int: bool = False) -> tuple[Number, Number, Number, Number]:
    """
    The bounding box from osunlp/Mind2Web is in the format of x,y,width,height
    we generally want x1,y1,x2,y2 for bounding box ease of use (although some bounding boxes are in y1,x1,y2,x2 format)
    """
    x1, y1, width, height = map(float, bounding_box_rect.split(","))
    x2, y2 = x1 + width, y1 + height

    if to_int:
        x1, x2, y1, y2 = map(round, [x1, x2, y1, y2])

    return x1, y1, x2, y2


def check_dirty_node(node: Tag) -> bool:
    """
    check if the node has a bounding box and if it does and is -1 it means hidden so we dont want that
    """

    if "bounding_box_rect" not in node.attrs or node["bounding_box_rect"] == "-1,-1,-1,-1":
        return False

    for content in node.contents:
        if isinstance(content, Tag):
            if not check_dirty_node(content):
                return False
    return True


def check_node_has_text(node: Tag) -> bool:
    if node.text.strip() == "":
        return False
    return True


def parse_candidate(candidate: str, parse_bounding_box: bool = True, to_int: bool = False) -> List[dict]:
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

    candidate["attributes"] = json.loads(candidate["attributes"])
    if parse_bounding_box:
        candidate["attributes"]["bounding_box_rect"] = parse_bounding_box_rect(
            candidate["attributes"]["bounding_box_rect"], to_int=to_int
        )

    return candidate


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
# Unused
# ----


def flip_return_from(return_from: ReturnFromTypes) -> ReturnFromTypes:
    """flip return from before to after and vice versa"""
    return {"after": "before", "before": "after"}[return_from]


def parse_action_repr(action_repr: str):
    """
    This function parses the following into a dict:
    '[div]  BMW -> CLICK', '[span]   -> CLICK', '[select]  1992 -> SELECT: 2010', '[button]  Close dialog -> CLICK', '[select]  2024 -> SELECT: 2010', '[combobox]  Sort By -> SELECT: Price: Low to High', '[span]   -> CLICK', '[span]   -> CLICK', '[span]   -> CLICK'
    """
    left_info, right_info = action_repr.split("->")
    left_info = left_info.strip()
    # match the component between [] and the value between []
    html_component = left_info[left_info.index("[") + 1 : left_info.index("]")]
    html_value = left_info[left_info.index("]") + 1 :].strip()
    if html_value == "":
        html_value = None

    # parse right info which is related to action and action value
    right_info = right_info.strip().split(":", 1)
    if len(right_info) == 1:
        action = right_info[0].strip()
        action_value = None
    elif len(right_info) == 2:
        action, action_value = right_info
        action, action_value = action.strip(), action_value.strip()

    return {
        "html_component": html_component,
        "html_value": html_value,
        "action": action,
        "action_value": action_value,
    }
