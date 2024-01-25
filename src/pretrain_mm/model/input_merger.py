import torch
from itertools import chain


def make_placeholder_idxs(input_ids: torch.Tensor, placeholder_id: int, flatten: bool = True):
    """for all the image, go through and create the indexes
    e.g.
    for idxs:     0   1   2   3    4  5  6  7    8   9  10  11  12
    input_ids = [[1, 99, 99, 99,  99, 5, 6, 99, 99, 99, 10, 11,  0]]
    we want [[0, 1, 5], [0, 7, 10]]

    Args:
        input_ids (torch.Tensor): _description_
        token_id (int): _description_
    """
    all_idxs = []
    for batch_idx, input_id in enumerate(input_ids):
        sample_idxs = []

        start, end = None, None
        for t_idx, token_id in enumerate(input_id):
            if (start == None) and (token_id == placeholder_id):
                start = t_idx
            if token_id == placeholder_id:
                end = t_idx + 1
            if (start != None) and (token_id != placeholder_id):
                sample_idxs.append((batch_idx, start, end))
                start, end = None, None
        if start != None:
            sample_idxs.append((batch_idx, start, end))
        all_idxs.append(sample_idxs)

    if flatten:
        all_idxs = list(chain(*all_idxs))

    return all_idxs


if __name__ == "__main__":
    input_ids = torch.randint(0, 99, (4, 100))
    placeholder_id = 100

    # example where images would be
    input_ids[0, 5:10] = placeholder_id
    input_ids[0, 50:60] = placeholder_id

    input_ids[1, 3:60] = placeholder_id
    input_ids[1, 62:70] = placeholder_id

    # check if it gets end
    input_ids[2, 50:] = placeholder_id

    # check if it gets beginning
    input_ids[3, 0:10] = placeholder_id

    image_idxs = make_placeholder_idxs(input_ids, placeholder_id, flatten=False)
    assert len(image_idxs) == 4

    image_idxs = make_placeholder_idxs(input_ids, placeholder_id)
    assert len(image_idxs) == 6
