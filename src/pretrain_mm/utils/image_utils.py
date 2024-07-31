from base64 import b64decode
from functools import cache
from io import BytesIO
from typing import TypeAlias

from PIL import Image, ImageDraw

from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE
from pretrain_mm.utils.token_tag_utils import box_pattern


NormalizedCoords: TypeAlias = tuple[float, float] | tuple[float, float, float, float]
PixelCoords: TypeAlias = tuple[int, int] | tuple[int, int, int, int]


class ImageSections:
    _memoized = {}
    # default will be this:
    # image_sections = [
    #     [0, 0, 640, 540],  # top left corner
    #     [0, 540, 640, 1080],  # bottom left corner
    #     [640, 0, 1280, 540],  # top right corner
    #     [640, 540, 1280, 1080],  # bottom right corner
    # ]

    def __init__(self, width: int = 1280, height: int = 1080, sections: list[list[int]] = None):
        self.sections = sections or ImageSections.make_sections(width, height)

    @cache
    @staticmethod
    def make_sections(width: int, height: int):
        return [
            [0, 0, width // 2, height // 2],  # top left corner
            [0, height // 2, width // 2, height],  # bottom left corner
            [width // 2, 0, width, height // 2],  # top right corner
            [width // 2, height // 2, width, height],  # bottom right corner
        ]

    @classmethod
    def from_image_size(cls, image_size: tuple[int, int]):
        if image_size not in cls._memoized:
            cls._memoized[image_size] = cls(*image_size)
        return cls._memoized[image_size]

    def __iter__(self):
        return iter(self.sections)


def draw_helper(
    image: Image.Image,
    box: list[int] = None,
    box_str: str = None,
    box_format: str = "html-mind2web",
    savefile: str = None,
    draw: ImageDraw.ImageDraw = None,
    outline: str = "red",
    width: int = 3,
    **kwargs,
):
    if not draw:
        draw = ImageDraw.Draw(image)

    if box_str:
        # box_format = "html-mind2web"
        box = list(map(int, box_pattern.search(box_str).groups()))

    # if you pass it in as x1,y1,x2,y2 already from box
    if box_format == "xy":
        x1, y1, x2, y2 = box

    if box_format == "html-mind2web":
        # presume box string comes in format x,y,width,height
        x, y, width, height = box
        x1, y1, x2, y2 = x, y, x + width, y + height
    if box_format == "fuyu":
        # fuyu format is <box>y1, x1, y2, x2</box>
        y1, x1, y2, x2 = box

    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

    if savefile:
        image.save(savefile)

    return draw, image


def read_image_from_b64(image_bytes: str) -> Image.Image:
    """
    Read an image from a base64 encoded string.

    Args:
        image_bytes: Base64 encoded image.

    Returns:
        PIL image.
    """
    return Image.open(BytesIO(b64decode(image_bytes)))


def draw_bounding_box(
    image: Image.Image,
    coords: tuple[int, int, int, int],
    color: str = "red",
    outfile: str = None,
) -> Image.Image:
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


def transform_box_to_cropped_section(
    coords: PixelCoords,
    image: Image.Image,
) -> tuple[PixelCoords, Image.Image, int]:
    """
    Given a box in the form x1, y1, x2, y2, return the section of the image that is cropped.
    """
    if image.size[0] > 1280 or image.size[1] > 1080:
        logger.warning(f"Image size is {image.size}, larger than 1280x1080 expected. Look into sample.")
        image = image.crop((0, 0, 1280, 1080))

    x1, y1, x2, y2 = coords

    image_sections = ImageSections.from_image_size(image.size)

    for section_idx, (cropped_left, cropped_top, cropped_right, cropped_bottom) in enumerate(image_sections):
        if x2 <= cropped_right and y2 <= cropped_bottom:
            new_x1 = max(x1 - cropped_left, 0)
            new_y1 = max(y1 - cropped_top, 0)
            new_x2 = min(x2 - cropped_left, cropped_right - cropped_left)
            new_y2 = min(y2 - cropped_top, cropped_bottom - cropped_top)
            image = image.crop((cropped_left, cropped_top, cropped_right, cropped_bottom))
            return [new_x1, new_y1, new_x2, new_y2], image, section_idx

    return coords, image, None


def interleave_vals(*vals) -> list[int]:
    """
    Flattens the given x and y coordinates into a single list.

    Args:
        xs (list): The x-coordinates.
        ys (list): The y-coordinates.

    Returns:
        list: A flattened list of coordinates, where each x and y pair is interleaved.
    """
    return [v for group in zip(*vals) for v in group]


def convert_normalized_to_pixel_coords(
    coords: NormalizedCoords,
    image_size: PixelCoords = VIEWPORT_SIZE,
) -> PixelCoords:
    """Converts from 0-1 normalized coordinates to pixel coordinates.

    Args:
        coords (NormalizedCoords): The normalized coordinates to convert.
        image_size (PixelCoords, optional): The size of the image in pixels. Defaults to VIEWPORT_SIZE.

    Returns:
        PixelCoords: The converted pixel coordinates.

    Examples:
        >>> convert_normalized_to_pixel_coords([0.8697916666666666, 0, 0.9072916666666667, 0.05555555555555555])
        [1113, 0, 1161, 60]

    """
    width, height = image_size
    pixel_xs = [int(x * width) for x in coords[::2]]
    pixel_ys = [int(y * height) for y in coords[1::2]]
    return interleave_vals(pixel_xs, pixel_ys)


def convert_pixel_coords_to_normalized(
    coords: PixelCoords,
    image_size: PixelCoords = VIEWPORT_SIZE,
) -> NormalizedCoords:
    """Converts from pixel coordinates to 0-1 normalized coordinates.

    Args:
        coords (PixelCoords): The pixel coordinates to convert.
        image_size (PixelCoords, optional): The size of the image in pixels. Defaults to VIEWPORT_SIZE.

    Returns:
        NormalizedCoords: The converted normalized coordinates.

    Examples:
        >>> convert_pixel_coords_to_normalized([1113, 0, 1161, 60])
        [0.8697916666666666, 0.0, 0.9070312500000000, 0.0555555555555556]

    """
    width, height = image_size
    normalized_xs = [x / width for x in coords[::2]]
    normalized_ys = [y / height for y in coords[1::2]]
    return interleave_vals(normalized_xs, normalized_ys)
