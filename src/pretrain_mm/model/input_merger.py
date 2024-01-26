import torch


def make_placeholder_idxs(input_ids: torch.Tensor, placeholder_id: int) -> torch.Tensor:
    """for all the image, go through and create the indexes
    e.g.
    for idxs:     0   1   2   3    4  5  6  7    8   9  10  11  12
    input_ids = [[1, 99, 99, 99,  99, 5, 6, 99, 99, 99, 10, 11,  0]]
    we want [[0, 1, 5], [0, 7, 10]]

    Args:
        input_ids (torch.Tensor): _description_
        token_id (int): _description_
    """
    # Step 1: Identify placeholder positions
    is_placeholder = input_ids == placeholder_id

    # Add False at the beginning and end for boundary detection
    bound_pad = torch.tensor([False for _ in range(input_ids.shape[0])])[:, None]
    padded = torch.cat([bound_pad, is_placeholder, bound_pad], dim=1)

    # Step 2: Find sequence boundaries
    diffs = torch.diff(padded.int())

    # Step 3: Extract indices
    starts_batch, starts = (diffs == 1).nonzero(as_tuple=True)
    ends_batch, ends = (diffs == -1).nonzero(as_tuple=True)
    # if ends isnt supposed to be incluse should do ends -= 1

    if (starts_batch != ends_batch).any():
        raise ValueError("Mismatched starts and ends.  Probably means I have a bug")

    return torch.stack([starts_batch, starts, ends], dim=1)
