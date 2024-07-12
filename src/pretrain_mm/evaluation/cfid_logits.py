from collections import defaultdict
from itertools import chain
from pathlib import Path

import torch
from rich.live import Live

from pretrain_mm import logger
from pretrain_mm.metrics.metrics import cfid, fid


def _get_all_generations_files(base_dir: str, return_sorted: bool = False):
    """returns all the generation files that are per sample for each model checkpoint

    Args:
        base_dir (str): _description_
    """

    files_by_model = {}

    files_iter = Path(base_dir).rglob("generations*.pt")
    # im trying to think if there is a reason we would not want to sort the files
    files_iter = sorted(files_iter) if return_sorted else files_iter

    for file in files_iter:
        if not file.name.endswith(".pt"):  # conditions we want to skip?
            logger.info(f"Skipping file: {file}")
            continue

        parent_dir = file.parent.name
        if parent_dir not in files_by_model:
            files_by_model[parent_dir] = []

        file_str = str(file.resolve())
        files_by_model[parent_dir].append(file_str)
    return files_by_model


def _combine_logits(
    logits: list[list[torch.Tensor]],
    min_dim: list[tuple[int, int]] = [],  # use tuple but could probably use dict as well
    pad_combine: bool = False,
    # , use_pad: bool = False
) -> list[torch.Tensor]:
    """
    combine nested list of logits into one.  since we may have different shapes, we use the smallest for each dim (along with
    optional arg min dim) rather than nested/padding

    Args:
        logits (list[list[torch.Tensor]]): _description_
        min_dim (list[tuple[int, int]], optional): _description_. Defaults to [].

    Returns:
        list[torch.Tensor]: _description_
    """
    logits = list(chain.from_iterable(logits))

    shapes = [l.shape for l in logits]

    # get the min shape for each dim, not sure if there is a way to do this more dynamically
    d_min_max = {i: {"min": min(s), "max": max(s)} for i, s in enumerate(zip(*shapes))}
    assert d_min_max[0]["max"] == 1, "Each logit tensor should be single tensor"

    # allow the min_dim override values
    for i, v in min_dim:
        d_min_max[i]["min"] = min(v, d_min_max[i]["min"])

    if pad_combine:
        tmp_logits = torch.zeros(len(logits), d_min_max[1]["max"], d_min_max[2]["max"], dtype=logits[0].dtype)
        for i, logit in enumerate(logits):
            tmp_logits[i, : logit.shape[1], : logit.shape[2]] = logit
        logits = tmp_logits

    else:
        logits = [l[:, -d_min_max[1]["min"] :, :] for l in logits]
        # below allows other dims to be shortened but i think thats bad
        # get the last n elements from each dim
        # logits = [l[-d_min_max[0]["min"] :, -d_min_max[1]["min"] :, -d_min_max[2]["min"] :] for l in logits]
    return logits


def _gather_logits_from_files(
    files: list[str],
    min_dim: list[tuple[int, int]] = [],
    max_files: int = None,
    cat_logits: bool = True,
) -> torch.Tensor:
    logits = []

    def _load_logits(f):
        return torch.load(f)["logits"]

    if max_files:
        files = files[:max_files]

    logits = [_load_logits(f) for f in files]

    if isinstance(logits[0], list):
        logits = _combine_logits(logits, min_dim=min_dim)

    if cat_logits and isinstance(logits, list):
        logits = torch.cat(logits, dim=0)

    return logits


"""
compute fid/cfid/augmented cfid

y_true ~= base model logits == y_logits
y_predict ~= trained model logits == y_hat_logits
x_true ~= base model conditioned on sequence (e.g. no constraint and no generation) == x_logits
"""


def compute_logit_scores(config: "Config"):
    # will likely need to shorten logits as OOM
    min_dims = [(1, config.cfid_seq_len)]  # (dim, val)

    # can also use: plot_data_file = config.get_plot_data()['generations']['outfiles']
    gen_files = _get_all_generations_files(config.sample_save_base / "generations", return_sorted=True)

    base_files = gen_files["base_model"]
    cond_base_files = gen_files["cond_base_model"]
    checkpoint_keys = [k for k in gen_files.keys() if "checkpoint_" in k]

    _y_logits = _gather_logits_from_files(base_files, min_dim=min_dims, max_files=config.max_files)
    _x_logits = _gather_logits_from_files(cond_base_files, min_dim=min_dims, max_files=config.max_files)
    logger.info("Got base model logits.")

    score_str = []
    # scores_out = {}
    scores_out = defaultdict(list)

    # cast to float and move to cuda
    y_logits, x_logits = _y_logits.float().cuda(), _x_logits.float().cuda()
    y_logits, x_logits = y_logits.transpose(1, 2), x_logits.transpose(1, 2)

    table = logger.use_table(title="CFID/FID Scores")
    table.add_column("CHKPT", justify="right", style="cyan")

    for t in [f"CFID{i+1}" for i in range(10)] + [f"FID{i+1}" for i in range(8)]:
        table.add_column(t, style="magenta")

    def _chkpt_iter(live):
        for k_idx, key in enumerate(checkpoint_keys):
            live.console.print(f"Got checkpoint data #{k_idx} | {key}")
            yield k_idx, key, _gather_logits_from_files(gen_files[key], min_dims, config.max_files)

    with Live(table, refresh_per_second=1) as live:
        for c_idx, checkpoint_name, _y_hat_logits in _chkpt_iter(live):
            y_hat_logits = _y_hat_logits.float().cuda().transpose(1, 2)

            # conditioned on the base model
            logit_scores1 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-1, f_dim=-1)
            logit_scores2 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-1, f_dim=-2)
            logit_scores3 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-2, f_dim=-2)
            logit_scores4 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-2, f_dim=-1)
            logit_scores5 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-1, f_dim=0)
            logit_scores6 = cfid(y_logits, y_hat_logits, x_logits, mean_dim=-2, f_dim=0)

            # conditioned on the trained model
            logit_scores7 = cfid(x_logits, y_logits, y_hat_logits, mean_dim=-1, f_dim=-1)
            logit_scores8 = cfid(x_logits, y_logits, y_hat_logits, mean_dim=-1, f_dim=-2)
            logit_scores9 = cfid(x_logits, y_logits, y_hat_logits, mean_dim=-2, f_dim=-1)
            logit_scores10 = cfid(x_logits, y_logits, y_hat_logits, mean_dim=-2, f_dim=-2)

            cfid_scores_mean = [
                logit_scores1.mean().item(),
                logit_scores2.mean().item(),
                logit_scores3.mean().item(),
                logit_scores4.mean().item(),
                logit_scores5.mean().item(),
                logit_scores6.mean().item(),
                logit_scores7.mean().item(),
                logit_scores8.mean().item(),
                logit_scores9.mean().item(),
                logit_scores10.mean().item(),
            ]

            fid_scores1 = fid(y_hat_logits, x_logits, mean_dim=-1, f_dim=-1)
            fid_scores2 = fid(y_hat_logits, y_logits, mean_dim=-1, f_dim=-1)

            fid_scores3 = fid(y_hat_logits, x_logits, mean_dim=-1, f_dim=-2)
            fid_scores4 = fid(y_hat_logits, y_logits, mean_dim=-1, f_dim=-2)

            fid_scores5 = fid(y_hat_logits, x_logits, mean_dim=-2, f_dim=-1)
            fid_scores6 = fid(y_hat_logits, y_logits, mean_dim=-2, f_dim=-1)

            fid_scores7 = fid(y_hat_logits, x_logits, mean_dim=-2, f_dim=-2)
            fid_scores8 = fid(y_hat_logits, y_logits, mean_dim=-2, f_dim=-2)

            fid_scores_mean = [
                fid_scores1.mean().item(),
                fid_scores2.mean().item(),
                fid_scores3.mean().item(),
                fid_scores4.mean().item(),
                fid_scores5.mean().item(),
                fid_scores6.mean().item(),
                fid_scores7.mean().item(),
                fid_scores8.mean().item(),
            ]

            scores_mean = cfid_scores_mean + fid_scores_mean

            scores_mean_strs = [f"{s:.2f}" for s in scores_mean]

            score_str.append([" ".join(scores_mean_strs)])

            table.add_row(f"chkpt{c_idx}", *scores_mean_strs)

            for _i, _score in enumerate(cfid_scores_mean):
                scores_out[f"cfid{_i}"].append(_score)

            for _i, _score in enumerate(fid_scores_mean):
                scores_out[f"fid{_i}"].append(_score)

    logger.info(f"Scores: {score_str}")

    plot_data = config.get_plot_data()
    plot_data["logit_scores"] = dict(scores_out)  # get rid of defaultdict
    config.save_plot_data(plot_data)
