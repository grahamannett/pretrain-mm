from pretrain_mm.constants import VIEWPORT_SIZE_DICT


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
    bounding_box: tuple[int, int, int, int],
    viewport_cutoff: float = None,
    area_cutoff: float = None,
    WIDTH: int = VIEWPORT_SIZE_DICT["width"],
    HEIGHT: int = VIEWPORT_SIZE_DICT["height"],
) -> bool:
    bbox_width, bbox_height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

    # if more than 2x the height/width then its probably not even rendered
    if viewport_cutoff and (
        (bounding_box[2] > (WIDTH * viewport_cutoff)) or (bounding_box[3] > (HEIGHT * viewport_cutoff))
    ):
        return True

    # if bbox area is more than area threshold (based on constants viewport)the viewport then its probably not a good candidate
    if (bbox_width * bbox_height) >= (area_cutoff * WIDTH * HEIGHT):
        return True

    return False
