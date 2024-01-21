from base64 import b64decode
from io import BytesIO

from PIL import Image, ImageDraw
from pretrain_mm import logger

# Define image sections
image_sections = [
    [0, 0, 640, 540],  # top left corner
    [0, 540, 640, 1080],  # bottom left corner
    [640, 0, 1280, 540],  # top right corner
    [640, 540, 1280, 1080],  # bottom right corner
]


def read_image_from_b64(image_bytes: str) -> Image.Image:
    """
    Read an image from a base64 encoded string.

    Args:
        image_bytes: Base64 encoded image.

    Returns:
        PIL image.
    """
    return Image.open(BytesIO(b64decode(image_bytes)))


def draw_all_bounding_boxes(sample: dict, outfile: str = None, transforms: list[callable] = []) -> Image.Image:
    """ """

    # sample = train_dataset[0]

    # draw sample as potential errors from samples quickest to find here
    # sample = pretrain_task_processor.pretrain_func_generate_possible_actions(train_dataset[2000])
    # samp = task_processor.process_func(sample)

    # breakpoint()
    # cands, sample = m2w_utils.get_all_candidates_in_view(train_dataset[1000])
    # # breakpoint()

    # image = sample.image
    # draw = ImageDraw.Draw(image)
    # for c_idx, (cand, c_t) in enumerate(cands):
    #     x1, y1, x2, y2 = cand["attributes"]["bounding_box_rect"]
    #     draw.rectangle([(x1, y1), (x2, y2)], outline="blue" if c_t == 1 else "red", width=3)
    #     draw.text((x1, y1), f"{c_idx}", fill="yellow")

    # image.save("out1.png")
    # breakpoint()
    pass


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
    coords: tuple[int, int, int, int],
    image: Image.Image,
) -> tuple[tuple[int, int, int, int], Image.Image, int]:
    """
    Given a box in the form x1, y1, x2, y2, return the section of the image that is cropped.
    """
    if image.size[0] > 1280 or image.size[1] > 1080:
        logger.warning(f"Image size is {image.size}. This is larger than 1280x1080. I should look into this.")
        image = image.crop((0, 0, 1280, 1080))

    x1, y1, x2, y2 = coords

    for i_section, (cropped_left, cropped_top, cropped_right, cropped_bottom) in enumerate(image_sections):
        if x2 <= cropped_right and y2 <= cropped_bottom:
            new_x1 = max(x1 - cropped_left, 0)
            new_y1 = max(y1 - cropped_top, 0)
            new_x2 = min(x2 - cropped_left, cropped_right - cropped_left)
            new_y2 = min(y2 - cropped_top, cropped_bottom - cropped_top)
            image = image.crop((cropped_left, cropped_top, cropped_right, cropped_bottom))
            return [new_x1, new_y1, new_x2, new_y2], image, i_section

    return coords, image, None
