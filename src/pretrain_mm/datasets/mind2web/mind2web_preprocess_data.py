from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.utils.bbox_utils import bounding_box_outside, invalid_bounding_box


BAD_CANDIDATE = {}
WIDTH, HEIGHT = VIEWPORT_SIZE_DICT["width"], VIEWPORT_SIZE_DICT["height"]

parse_candidate = m2w_utils.parse_candidate


def valid_candidates_map(data: dict, rank: int = None):
    for idx, actions in enumerate(data["actions"]):
        for act_idx, action in enumerate(actions):
            pos_candidates = []
            neg_candidates = []

            for cand in action["pos_candidates"]:
                parsed_candidate = parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
                bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]

                if invalid_bounding_box(bounding_box) or bounding_box_outside(
                    bounding_box,
                    viewport_cutoff=1.75,
                    area_cutoff=0.5,
                    width=WIDTH,
                    height=HEIGHT,
                ):
                    continue

                # parsed_candidate["parsed"] = True
                # pos_candidates.append(parsed_candidate)
                # # pos_candidates.append(cand.copy())
                pos_candidates.append(cand)

            for cand in action["neg_candidates"]:
                parsed_candidate = parse_candidate(cand.copy(), parse_bounding_box=True, to_int=True)
                bounding_box = parsed_candidate["attributes"]["bounding_box_rect"]

                if invalid_bounding_box(bounding_box) or bounding_box_outside(
                    bounding_box,
                    viewport_cutoff=1.75,
                    area_cutoff=0.5,
                    width=WIDTH,
                    height=HEIGHT,
                ):
                    continue

                # parsed_candidate["parsed"] = True
                # neg_candidates.append(parsed_candidate)
                # neg_candidates.append(cand.copy())
                neg_candidates.append(cand)

            # data["actions"][idx][act_idx]["pos_candidates"] = pos_candidates
            # data["actions"][idx][act_idx]["neg_candidates"] = neg_candidates
            action["pos_candidates"] = pos_candidates
            action["neg_candidates"] = neg_candidates

    return data
