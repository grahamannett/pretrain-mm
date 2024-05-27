import re
from enum import StrEnum, auto


# token_bbox_regex = re.compile(r"<0x00>(\d+(?:,\s*\d+)*)<0x01>")
# token_point_regex = re.compile(r"<0x02>(\d+(?:,\s*\d+)*)<0x03>")


# match 4 numbers separated by commas between the tags
token_box_pattern = re.compile(r"<0x00>\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*<0x01>")
# match 2 numbers separated by commas between the tags
token_point_pattern = re.compile(r"<0x02>\s*(\d+),\s*(\d+)\s*<0x03>")

# should be same as above but patch before token replacement
box_pattern = re.compile(r"<box>\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*</box>")
point_pattern = re.compile(r"<point>\s*(\d+),\s*(\d+)\s*</point>")

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


def _iter_pattern_over_str(raw_str: str, pattern: re.Pattern, tag_type: TagType):
    """
    given a raw string, a pattern, and a tag type, iterate over the pattern and return a list of tuples of the form:
        (str, tag_type).
    tag_type is None if it does not belong to a tag
    """
    last_match_idx = 0
    segmented_arr = []
    for matched in pattern.finditer(raw_str):
        start, end = matched.span()

        # either do this or append each.
        # raw_str[start: end]/matched.group() can be used if i dont want parsed groups
        segs = ((raw_str[last_match_idx:start], None), (matched.groups(), tag_type))
        segmented_arr.extend(segs)

        last_match_idx = end

    if last_match_idx < len(raw_str):
        segmented_arr.append((raw_str[last_match_idx:], None))
    return segmented_arr


def _handle_str_with_pattern(
    base_str: list[tuple[str, TagType | None]],
    pattern: re.Pattern,
    tag: TagType,
) -> list[tuple[str, TagType | None]]:
    replaced_segments = []
    for seg in base_str:
        if not seg[1]:
            seg = _iter_pattern_over_str(seg[0], pattern, tag)
            replaced_segments.extend(seg)
        else:
            replaced_segments.append(seg)
    return replaced_segments


def segment_str(
    base_str: list[str] | str,
    box_pattern: re.Pattern = token_box_pattern,
    point_pattern: re.Pattern = token_point_pattern,
) -> list[tuple[str, TagType | None]]:
    if isinstance(base_str, str):
        base_str = [(base_str, None)]

    base_str = _handle_str_with_pattern(base_str, box_pattern, TagType.BOX)
    base_str = _handle_str_with_pattern(base_str, point_pattern, TagType.POINT)

    return base_str
