import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw
from simple_parsing import ArgumentParser, Serializable, choice


from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm import logger
from pretrain_mm.datasets.mind2web import Mind2Web, M2WAction


@dataclass
class DataLabelConfig(Serializable):
    label_data_file: str = "labeled_data.json"
    outdir: str = "output"
    # need to save image each time
    current_image_saved_path: str = "output/tmp_current_image.png"
    data_split: str = "test"


class DataLabeling:
    def __init__(
        self,
        label_data_file: str = DataLabelConfig.label_data_file,
        outdir: str = DataLabelConfig.outdir,
        split: str = DataLabelConfig.data_split,
        current_image_saved_path: str = DataLabelConfig.current_image_saved_path,
    ):
        self.split = split  # i think i want to incorporate this into the label_data_file name
        self.label_data_file = label_data_file  # the good data
        self.current_image_saved_path = current_image_saved_path  # where each image is that I will label

        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)

        self._labeled_samples = []
        self._issue_samples = []

        self.user_input_prompt = "[bold italic cyan]>>> Provide the label for this bounding box: "

        logger.info(f"Open up the image at the path: {self.current_image_saved_path}")

    @property
    def label_filepath(self):
        return self.outdir / self.label_data_file

    def __atexit__(self):
        self.save()

    def load(self):
        with open(self.label_filepath, "r") as f:
            self.labeled_samples = json.load(f)

    def save(self):
        with open(self.label_filepath, "w") as f:
            json.dump(self.labeled_samples, f)

    def to_google_sheets(self):
        raise NotImplementedError

    def label_bounding_box_for_sample(self, sample) -> None:
        # user_input = UserInput(self.user_input_prompt).get()
        self.draw_bounding_box(sample)
        user_input = self.get_user_input()
        user_input, user_response_status = self.parse_user_input(user_input)

        if user_response_status == "done":
            self.gather_for_save(user_input, sample)

    def draw_bounding_box(self, sample: M2WAction) -> None:

        _draw_rectangle_kwargs = {"outline": "red", "width": 3}

        image: Image.Image = sample.image

        # image needs crop likely since it will be the full browser window not viewport so hard to see
        image_w, image_h = image.size
        image = image.crop((0, 0, image_w, VIEWPORT_SIZE_DICT["height"]))

        draw = ImageDraw.Draw(image)
        draw.rectangle(sample.bounding_box, **_draw_rectangle_kwargs)
        image.save(self.current_image_saved_path)

    def get_user_input(self):
        user_input = logger.ask(prompt=self.user_input_prompt)
        # if i want to retry/check do it in parse_user_input
        return user_input

    def parse_user_input(self, user_input: str) -> tuple[str, str]:
        # should be one of None, "skip", "done"
        input_status = "done"
        if user_input.lower() in ["!none", "!skip"]:
            input_status = "skip"

        if user_input.lower() == "!quit":
            exit()

        return user_input, input_status

    def gather_for_save(self, user_input: str, sample: M2WAction, sample_group: str = "labeled"):
        to_save = {
            "labeled_input": user_input,
            "pos_candidate_idx": 0,
            "annotation_id": sample.annotation_id,
            "action_idx": sample.action_idx,
            "trajectory_idx": sample.trajectory.trajectory_idx,
            "bounding_box": sample.bounding_box,
            "json_filepath": sample.trajectory.json_filepath,
            "split": self.split,
        }

        append_to = {
            "labeled": self._labeled_samples,
            "issue": self._issue_samples,
        }[sample_group]

        append_to.append(to_save)


def prepare_sample(sample: M2WAction):
    parsed_candidate = Mind2Web.parse_candidate(sample.pos_candidates[0], as_copy=True, to_int=True)
    sample.bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]
    return sample


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(DataLabelConfig, dest="data_label_config")

    args = parser.parse_args()

    config: DataLabelConfig = args.data_label_config

    data_labeler = DataLabeling(
        label_data_file=config.label_data_file,
        outdir=config.outdir,
        split=config.data_split,
        current_image_saved_path=config.current_image_saved_path,
    )

    dataset = Mind2Web(split="test")

    sample = dataset[0]

    output = data_labeler.label_bounding_box_for_sample(prepare_sample(dataset[0]))
    output = data_labeler.label_bounding_box_for_sample(prepare_sample(dataset[100]))
