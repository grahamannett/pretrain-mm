import re
from enum import StrEnum, auto

# token_bbox_regex = re.compile(r"<0x00>(\d+(?:,\s*\d+)*)<0x01>")
# token_point_regex = re.compile(r"<0x02>(\d+(?:,\s*\d+)*)<0x03>")


# match 4 numbers separated by commas between the tags
token_box_pattern = re.compile(r"<0x00>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*<0x01>")
# match 2 numbers separated by commas between the tags
token_point_pattern = re.compile(r"<0x02>\s*(\d+),\s*(\d+)\s*<0x03>")

# should be same as above but patch before token replacement
box_pattern = re.compile(r"<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*</box>")
point_pattern = re.compile(r"<point>(\d+),\s*(\d+)\s*</point>")

tag_patterns = {
    "box": box_pattern,
    "point": point_pattern,
}


class TagType(StrEnum):
    BOX = auto()
    POINT = auto()

    @classmethod
    def make(cls, loc_type: str):
        return {
            cls.POINT: make_point_str,
            cls.BOX: make_box_str,
        }[loc_type]


def make_point_str(x1: int, y1: int, x2: int = None, y2: int = None, /, do_round: bool = True) -> str:
    x, y = x1, y1

    if x2 and y2:
        x, y = round((x + x2) / 2), round((y1 + y2) / 2)

    if do_round:
        x, y = round(x), round(y)

    return f"<point>{x}, {y}</point>"


def make_box_str(x1: int, y1: int, x2: int, y2: int) -> str:
    # FUYU NEEDS IN format: y1, x1, y2, x2 but bounding box comes in form x0, y0, x1, y1,
    return f"<box>{y1}, {x1}, {y2}, {x2}</box>"
