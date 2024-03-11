import re

from pretrain_mm.constants import VIEWPORT_SIZE
from pretrain_mm.utils.eval_utils import _pattern_to_vals

rel_position_pattern = re.compile(r"\[(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]")


def convert_bounding_box_from_relative(input_str, viewport_size: tuple[int, int] = VIEWPORT_SIZE):
    """
    Convert a bounding box from a string to a list of floats

    format from llava is like "[0.76, 0.58, 0.88, 0.62]" where the numbers are proportional to the viewport size

    map first to flota so we get base value e.g. 0.76 then multiply by viewport size and round to get pixel value
    """
    if vals := _pattern_to_vals(input_str, pattern=rel_position_pattern, map_to_type=float):
        return [round(val * viewport_size[i % 2]) for i, val in enumerate(vals)]
    return None
