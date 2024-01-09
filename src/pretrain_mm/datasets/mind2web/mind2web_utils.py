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


def check_action_screenshot(action: dict, return_from: ReturnFromTypes) -> bool:
    """not sure if i should try to load the actual screenshot?"""
    if action[return_from]["screenshot"] == "":
        return False
    return True


def flip_return_from(return_from: ReturnFromTypes) -> ReturnFromTypes:
    """flip return from before to after and vice versa"""
    return {"after": "before", "before": "after"}[return_from]


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


def draw_bounding_box(image: Image.Image, coords: tuple[int, int, int, int], color: str = "red", outfile: str = None):
    """
    Draws a bounding box on the given image.

    Args:
        image (PIL.Image.Image): The image to draw the bounding box on.
        coords (tuple[int, int, int, int]): The coordinates of the bounding box in the format (x1, y1, x2, y2).
        color (str, optional): The color of the bounding box outline. Defaults to "red".
        outfile (str, optional): The path to save the image with the bounding box drawn. Defaults to None.

    Returns:
        PIL.Image.Image: The image with the bounding box drawn.
    """
    x1, y1, x2, y2 = coords
    assert x2 > x1 and y2 > y1, "Check coords"

    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color)

    if outfile:
        image.save(outfile)

    return image
