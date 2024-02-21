from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets.mind2web import mind2web_utils as m2w_utils
from pretrain_mm.utils.bbox_utils import invalid_bounding_box, bounding_box_outside

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
                    WIDTH=WIDTH,
                    HEIGHT=HEIGHT,
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
                    WIDTH=WIDTH,
                    HEIGHT=HEIGHT,
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


#
# width, height = self.config.viewport_size

# set these here so that they can be used in map_fn and are hashable for caching
# get_bounding_box_area = m2w_utils.get_bounding_box_area
# get_mid_point = m2w_utils.get_mid_point
# check_dirty_node = m2w_utils.check_dirty_node
# check_node_has_text = m2w_utils.check_node_has_text
# parse_candidate = m2w_utils.parse_candidate

# def candidate_ok(
#     candidate: dict, screenshot_margin: float = screenshot_margin, max_area: float = max_area, html_tree=None
# ) -> bool:
#     # if enforce_clickable and not candidate["attributes"]["is_clickable"]:
#     #     return False

#     bbox = candidate["attributes"]["bounding_box_rect"]

#     box_area = get_bounding_box_area(bbox)
#     mid_x, mid_y = get_mid_point(bbox)

#     if (mid_x > (width * screenshot_margin)) or (mid_y > (height * screenshot_margin)):
#         return False

#     if box_area > max_area:
#         return False

#     if html_tree:
#         # check if the node has a bounding box and if it does and is -1 it means hidden so we dont want that
#         node = html_tree.find(backend_node_id=candidate["backend_node_id"])
#         if not check_dirty_node(node):
#             return False
#         if not check_node_has_text(node):
#             return False

#     return True

# # ensure that all candidates are ok.  meaning it is within the viewport and not too large
# # if more restrictions are needed, add to `candidate_ok`
# def map_fn(data: dict):
#     for a_idx, action in enumerate(data["actions"]):
#         for s_idx, subaction in enumerate(action):
#             html_tree = BeautifulSoup(subaction["raw_html"], "html.parser")
#             # use copy since process_candidate modifies the dict
#             neg_cands = [
#                 x
#                 for x in subaction["neg_candidates"]
#                 if candidate_ok(parse_candidate(x.copy(), True), html_tree=html_tree)
#             ]
#             pos_cands = [
#                 x
#                 for x in subaction["pos_candidates"]
#                 if candidate_ok(parse_candidate(x.copy(), True), html_tree=html_tree)
#             ]

#             action[s_idx]["neg_candidates"] = neg_cands
#             action[s_idx]["pos_candidates"] = pos_cands

#     return data
