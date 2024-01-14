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
