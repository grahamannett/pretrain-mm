import json
from typing import List

from PIL import Image, ImageDraw

from pretrain_mm import logger


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


def collect_screenshots_for_all_dataset(dataset: "Mind2WebBase", dir_out: str, crop_image: bool = True) -> None:
    """
    Collects all screenshots for a given dataset and saves them to the given directory.

    Args:
        dataset (datasets.Dataset): The dataset to collect screenshots for.
        dir_out (str): The directory to save the screenshots to.
        crop_image (bool, optional): Whether to crop the screenshot to the bounding box. Defaults to True.
    """
    import os

    dataset.config.crop_image = crop_image

    traj: M2WTrajectory
    act: M2WAction

    with logger.progress() as progress:
        dataset_task = progress.add_task("[cyan]Trajectories...", total=len(dataset))
        action_task = progress.add_task(
            "[blue]Actions...",
        )

        for t_i, traj in enumerate(dataset):
            # traj dir will have a before/after for each action in the trajectory
            traj_dir = f"{dir_out}/{traj.annotation_id}"
            os.makedirs(traj_dir, exist_ok=True)

            progress.reset(action_task, len(traj.actions), description=f"[blue]Actions...{traj.annotation_id}")

            for a_i, act in enumerate(traj.actions):
                before_screenshot = dataset.load_screenshot_from_task_dir(act.annotation_id, a_i, return_from="before")
                after_screenshot = dataset.load_screenshot_from_task_dir(act.annotation_id, a_i, return_from="after")

                before_screenshot = dataset._process_image(before_screenshot)
                after_screenshot = dataset._process_image(after_screenshot)

                before_screenshot.save(f"{traj_dir}/{act.action_uid}_before.png")
                after_screenshot.save(f"{traj_dir}/{act.action_uid}_after.png")

                # sub progress update
                progress.update(action_task, advance=1)

            # main progress update
            progress.update(dataset_task, advance=1)
