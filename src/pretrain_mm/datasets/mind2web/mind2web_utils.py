import json
from typing import List
from PIL import Image, ImageDraw
from datasets import load_dataset

dataset = load_dataset("osunlp/Mind2Web")


def parse_bounding_box_rect(bounding_box_rect: str) -> tuple[float, float, float, float]:
    """
    The bounding box from osunlp/Mind2Web is in the format of x,y,width,height
    """
    x1, y1, width, height = map(float, bounding_box_rect.split(","))
    x2, y2 = x1 + width, y1 + height
    return x1, y1, x2, y2


def parse_candidate(candidate: str, parse_bounding_box: bool = True) -> List[dict]:
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
            candidate["attributes"]["bounding_box_rect"]
        )

    return candidate


def bounding_box_to_point(x1, y1, x2, y2) -> tuple[float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2


def draw_bounding_box(image: Image.Image, coords: tuple[int, int, int, int], color: str = "red", outfile: str = None):
    x1, y1, x2, y2 = coords
    assert x2 > x1 and y2 > y1, "Check coords"

    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color)

    if outfile:
        image.save(outfile)

    return image
