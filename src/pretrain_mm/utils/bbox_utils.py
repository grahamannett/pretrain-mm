from typing import Callable, Union

from pretrain_mm.constants import VIEWPORT_SIZE_DICT


Number = Union[int, float]
BoundingBox = tuple[Number, Number, Number, Number]
MidPoint = tuple[Number, Number]


def invalid_bounding_box(bounding_box: BoundingBox) -> bool:
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

    # check if the box is inverted
    if (bounding_box[2] <= bounding_box[0]) or (bounding_box[3] <= bounding_box[1]):
        return True

    return False


def bounding_box_outside(
    bounding_box: BoundingBox,
    viewport_cutoff: float = None,
    area_cutoff: float = None,
    width: int = VIEWPORT_SIZE_DICT["width"],
    height: int = VIEWPORT_SIZE_DICT["height"],
) -> bool:
    # assume the bounding box is in the format of x1,y1,x2,y2
    bbox_w, bbox_h = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

    # if more than 2x the height/width then its probably not even rendered
    if viewport_cutoff and (
        (bounding_box[2] > (width * viewport_cutoff)) or (bounding_box[3] > (height * viewport_cutoff))
    ):
        return True

    # if bbox area is more than area threshold (based on constants viewport)the viewport then its probably not a good candidate
    if (bbox_w * bbox_h) >= (area_cutoff * width * height):
        return True

    return False


# THE ABOVE 2 ARE USED TOGETHER SO FREQUENTLY
def invalid_or_outside(bounding_box: BoundingBox, **kwargs) -> bool:
    return invalid_bounding_box(bounding_box) or bounding_box_outside(bounding_box, **kwargs)


def get_bounding_box_area(bbox: BoundingBox) -> Number:
    """
    find the area of a bounding box
    in format of x1,y1,x2,y2
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


# POINT RELATED
def get_midpoint(
    bbox: BoundingBox,
    to_int: bool | Callable = True,  # seems like default being true is gonna be less error prone
) -> MidPoint:
    """
    find the mid point of a bounding box
    """
    midpoint = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    if to_int:
        if callable(to_int):
            midpoint = to_int(midpoint)
        else:
            midpoint = tuple(map(round, midpoint))
    return midpoint


def point_within_box(point: tuple[Number, Number], bbox: BoundingBox) -> bool:
    """
    check if a point is within a bounding box
    """
    return (bbox[0] <= point[0] <= bbox[2]) and bbox[1] <= point[1] <= bbox[3]
