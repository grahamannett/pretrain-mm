from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.utils.bbox_utils import bounding_box_outside, invalid_bounding_box


BAD_CANDIDATE = {}
WIDTH, HEIGHT = VIEWPORT_SIZE_DICT["width"], VIEWPORT_SIZE_DICT["height"]

# keep this so that it can be cached for datasets.map
parse_candidate = m2w_utils.parse_candidate


def valid_candidates_map(
    data: dict,
    rank: int = None,
    viewport_cutoff: float = 1.75,
    area_cutoff: float = 0.5,
    width: int = WIDTH,
    height: int = HEIGHT,
):
    _outside_kwargs = {
        "viewport_cutoff": viewport_cutoff,
        "area_cutoff": area_cutoff,
        "width": width,
        "height": height,
    }

    def _skip_box(_cand):
        parsed_candidate = parse_candidate(_cand.copy(), parse_bounding_box=True, to_int=True)
        bbox = parsed_candidate["attributes"]["bounding_box_rect"]
        return invalid_bounding_box(bbox) or bounding_box_outside(bbox, **_outside_kwargs)

    for idx, actions in enumerate(data["actions"]):
        for act_idx, action in enumerate(actions):
            pos_candidates = []
            neg_candidates = []

            for cand in action["pos_candidates"]:
                if _skip_box(cand):
                    continue

                pos_candidates.append(cand)

            for cand in action["neg_candidates"]:
                if _skip_box(cand):
                    continue

                neg_candidates.append(cand)

            # data["actions"][idx][act_idx]["pos_candidates"] = pos_candidates
            # data["actions"][idx][act_idx]["neg_candidates"] = neg_candidates
            action["pos_candidates"] = pos_candidates
            action["neg_candidates"] = neg_candidates

    return data
