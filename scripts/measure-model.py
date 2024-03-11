import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from rich.live import Live
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from pretrain_mm import logger
from pretrain_mm.constants import IGNORE_INDEX, VIEWPORT_SIZE
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, pretrain_instructions
from pretrain_mm.datasets.mind2web.mind2web_processor import Mind2WebPretrainProcessor, Mind2WebTaskProcessor
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.model.fuyu import MODEL_ID, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.config_utils import BaseTrainConfig, BaseWandBConfig, LocalDataConfig
from pretrain_mm.utils.eval_utils import eval_by_completion, sample_eval_by_completion
from pretrain_mm.utils.generate_utils import generate_helper

dataset_host_info = get_dev_config("mind2web")


@dataclass
class WandBConfig(BaseWandBConfig):
    group: str = "eval/pretrain-fuyu"
    job_type: str = "pretrain-eval"


@dataclass
class Config:
    cmd: str  # dont use choice("make_samples", "model_process_samples_from_file", "calculate_data", default="make_samples")
    # cmd: str = choice(COMMANDS.keys(), default="make_samples")

    base_model: str = MODEL_ID
    model_path: str = None  # "/data/graham/models/pretrain-mm/fuyu/actiontag-random-order/checkpoint_1"

    processor_path: str = "/data/graham/models/pretrain-mm/fuyu/actiontag-random-order/processor"
    model_subdir_name: str = None  # subdir to where generations will be saved out.  if not passed, uses model_path.name

    # plot related info
    plot_infofile: str = "output/plot_infofile.pt"  # might switch to tinydb if needed
    reset_plot_data: bool = False

    multi_models: bool = False

    # processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    sample_save_base: str = "output/saved_samples/task_box_accuracy/"
    task_samples_file: str = "task_samples.pt"

    device_map: str = "auto"
    device: str = "cuda"  # for tensors when doing non model related

    # input related
    instruction = pretrain_instructions.GenerateNumPotentialActions(num_candidates=1)

    input_max_length: int = 2500
    viewport_size: tuple[int, int] = VIEWPORT_SIZE

    # generate related
    max_new_tokens: int = 10
    use_force_words: bool = False
    # temperature
    temperature: float = 1.0
    eval_num_samples: int = 2
    random_samples: bool = True
    num_generations_per_sample: int = 1
    dataset_for_samples: str = choice("train", "test", default="train")

    data_subset: int = None

    # using past_key_values seems like it might generate different results
    use_past_key_values: bool = False
    output_hidden_states: bool = False  # if we want the penultimate hidden states for umap

    cfid_seq_len: int = 100
    max_files: int = None

    def __post_init__(self):
        self.sample_save_base = Path(self.sample_save_base)

        # if we give model_path and not model_subdir_name, then use the name of the model_path
        if self.model_path and (self.model_subdir_name == None) and (_path := Path(self.model_path)).exists():
            self.model_subdir_name = _path.name
            logger.info(f"Using model_subdir_name name from model_path: {_path.name}")

        self._check_for_plot_infofile()

    def _check_for_plot_infofile(self, default_plot_data: dict = {}):
        if self.reset_plot_data:
            # if reset and data exists, print the prior data to screen before reset (just using empty dict)
            if _prev_plot_data := Path(self.plot_data).exists():
                _prev_plot_data = torch.load(_prev_plot_data)
                logger.info(_prev_plot_data)

            logger.warn(f"RESET PLOT DATA")
            torch.save(default_plot_data, self.plot_infofile)

        if not Path(self.plot_infofile).exists():
            torch.save(default_plot_data, self.plot_infofile)
            logger.info(f"Created plot_infofile: {self.plot_infofile}")

    def save_plot_data(self, data: dict):
        torch.save(data, self.plot_infofile)
        logger.info(f"Saved plot data to: {self.plot_infofile}")

    def get_plot_data(self):
        return torch.load(self.plot_infofile)


def _maybe_sort(_dict, return_sorted: bool = False):
    if return_sorted:
        # sort the checkpoint files AND the keys
        return {k: sorted(_dict[k]) for k in sorted(_dict.keys())}
    return _dict


def validate_sample(raw_sample):
    return raw_sample.pos_candidates != []


def get_model_func(p, device_map: str = "auto"):
    return FuyuForCausalLM.from_pretrained(p, device_map=device_map, torch_dtype=torch.bfloat16)


def get_extra_token_related(
    processor: FuyuProcessor,
    use_force_words: bool,
    skip_ids: list[int] = [],  # or [262144, 262145]
):
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    force_words_ids = []

    if use_force_words:
        force_words_ids = [
            v[1] for v in FuyuConstants.get_all_ids(processor, skip_ids=skip_ids).values()
        ] + processor.tokenizer.convert_tokens_to_ids([str(i) for i in range(999)])
        force_words_ids = list(set(force_words_ids))
        force_words_ids.sort()

    return stop_tokens, force_words_ids

    # return {
    #     "stop_tokens": stop_tokens,
    #     "force_words_ids": force_words_ids,
    # }


def main(config: Config):
    raise KeyError(f"Enter a valid command: {COMMANDS.keys()}")


def create_samples_from_dataset(
    dataset,
    num_samples=Config.eval_num_samples,
    random_samples=True,
    validate_sample: callable = validate_sample,
):
    samples, idxs_used, idxs_bad = [], [], []

    def _iter():
        _gen = range(len(dataset))

        if random_samples:
            _gen = sorted(_gen, key=lambda x: random.random())

        for idx in _gen:
            yield idx

    idx_iter = _iter()

    while len(samples) < num_samples:
        if (idx := next(idx_iter, None)) is None:
            break

        sample = dataset[idx]
        if validate_sample(sample):
            samples.append(sample), idxs_used.append(idx)
        else:
            idxs_bad.append(idx)

    bad_idxs_str = f" | Bad Indices: {idxs_bad}" if idxs_bad else ""
    logger.info(f"Using the following indices: {idxs_used}{bad_idxs_str}")

    # zip these then sort based on idx and then unzip
    samples_idxs = sorted(zip(samples, idxs_used), key=lambda x: x[0])
    samples, idxs_used = zip(*samples_idxs)

    return samples, {"used": idxs_used, "bad": idxs_bad}


def process_samples(samples: list[dict], processor_func: callable, save_path: str = None):

    samples_ = [processor_func(sample) for sample in samples]
    samples_ = [s for s in samples_ if s not in [False, None]]

    return samples_


def get_all_logits(
    model, encoder, samples, max_new_tokens: int, generate_kwargs: dict = {}, collect_base_logits: bool = False
):
    gen_outputs, gen_logits, base_logits = [], [], []

    return_vals = {
        "outputs": [],
        "logits": [],
        **({"base_logits": []} if collect_base_logits else {}),
    }

    for sample in samples:
        model_inputs = encoder(sample)
        model_inputs = model_inputs.to(model.device)

        output, logits = generate_helper(
            model,
            model_inputs=model_inputs,
            max_new_tokens=max_new_tokens,
            # return_last_logits=True,
            **generate_kwargs,
        )

        return_vals["outputs"].append(output.detach().cpu())
        return_vals["logits"].append(logits.detach().cpu())

        if collect_base_logits:
            with torch.no_grad():
                model_outputs = model(**model_inputs)

            return_vals["base_logits"].append(model_outputs.logits.detach().cpu())

    return return_vals


def make_samples(config: Config):

    _m2w_config_kwargs = {
        "task_dir": dataset_host_info["task_dir"],
        "attach_config_to_sample": True,
        "subset": config.data_subset,
    }

    ds_config = Mind2WebConfig(
        **_m2w_config_kwargs,
        **dataset_host_info["train"],
    )
    test_data_config = Mind2WebConfig(
        **_m2w_config_kwargs,
        **dataset_host_info["test"],
    )

    dataset = Mind2Web(config=ds_config)
    test_dataset = Mind2Web(test_data_config)

    dataset.setup_pretrain()
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(config.processor_path)

    task_processor = Mind2WebTaskProcessor(processor=processor, max_length=config.input_max_length)
    pretrain_task_processor = Mind2WebPretrainProcessor(viewport_size=config.viewport_size)

    task_processor = Mind2WebTaskProcessor(
        processor=processor,
        ignore_index=IGNORE_INDEX,
        max_length=config.input_max_length,
        encode_kwargs={"label_mask_text_ids": True},
    )

    def process_func(raw_sample) -> dict:
        task_sample = pretrain_task_processor.acc_func_complete_box(raw_sample)

        if isinstance(task_sample, bool):
            return False

        task_sample["_extra"] = {
            "action_idx": raw_sample.action_idx,
            "trajectory_idx": raw_sample.trajectory_idx,
            "action_uid": raw_sample.action_uid,
            "annotation_id": raw_sample.annotation_id,
        }

        enc_sample = task_processor.encode_data(
            task_sample,
            add_bos_token=False,
            add_boa_token=False,
            label_add_eos_token=False,
            include_label=False,
        )

        return {
            "raw": raw_sample,
            "task": task_sample,
            "encoded": enc_sample,
        }

    base_samples, idx_info = create_samples_from_dataset(
        {"train": dataset, "test": test_dataset}[config.dataset_for_samples],
        num_samples=config.eval_num_samples,
        validate_sample=validate_sample,
        random_samples=config.random_samples,
    )
    samples = process_samples(
        base_samples,
        processor_func=process_func,
        save_path=config.sample_save_base,
    )

    # save data+processors
    save_data = {
        "samples": samples,
        "idx_info": idx_info,
    }

    other_save_data = {
        "task_processor": task_processor,
        "pretrain_task_processor": pretrain_task_processor,
    }

    task_samples_file = config.sample_save_base / config.task_samples_file
    other_samples_file = config.sample_save_base / "extra_sample_info.pt"

    torch.save(save_data, task_samples_file)
    torch.save(other_save_data, other_samples_file)
    logger.info(f"SAVED DATA TO: {task_samples_file} has keys: {list(save_data.keys())}")


def model_process_samples_from_file(config: Config):
    """
    use this function to take the given saved samples, and then use a specific model to generate completions and
    the related logits.  the logits are the main values of interest and we want the logits over the whole output space
    """

    assert config.model_subdir_name is not None, "Need to pass model_subdir_name or have it set in config __post_init__"

    generation_meta_path = config.sample_save_base / "generations" / config.model_subdir_name / f"generation_meta.pt"
    generation_meta = {
        "samples": {"dist": {}},
        "outfiles": [],
    }  # this is NOT for large data, just metrics related to all generations and per sample generations

    model = get_model_func(config.model_path)

    proc = FuyuProcessor.from_pretrained(config.processor_path)
    stop_tokens, force_words_ids = get_extra_token_related(proc, use_force_words=config.use_force_words)

    samples_data = torch.load(config.sample_save_base / config.task_samples_file)
    samples = samples_data["samples"]

    def get_decode_start_idx(inp) -> int:
        return (inp["input_ids"][0] == proc.vocab[proc.constants.boa_string]).nonzero().flatten().item() - 1

    generate_kwargs = {
        "stop_tokens": stop_tokens,
        "force_words_ids": force_words_ids,
        "return_extra": True,
        "max_new_tokens": config.max_new_tokens,
        "use_past_key_values": config.use_past_key_values,
        "forward_kwargs": {
            "output_hidden_states": config.output_hidden_states,
        },
    }

    # save after every generation since otherwise files are 10GB+ which is slow to load i think
    for sample_idx, sample in enumerate(samples):
        sample_generation_info = sample_eval_by_completion(
            model=model,
            processor=proc,
            sample=sample,
            num_generations=config.num_generations_per_sample,
            prepend_str="",
            prepend_str_extra="",
            get_decode_start_idx=get_decode_start_idx,
            generate_kwargs=generate_kwargs,
        )

        breakpoint()

        # output file contains logits/input_ids
        output_path = (
            config.sample_save_base / "generations" / config.model_subdir_name / f"generations_base_{sample_idx}.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # you want to save metrics/distance and then sample related stuff possibly to generation_meta,
        # lose the sample_idx as not clear if i need, alt is generation_meta["samples"][f"{sample_idx}.dist"]
        generation_meta["samples"]["dist"][sample_idx] = sample_generation_info["dist"]
        generation_meta["outfiles"].append(str(output_path.resolve()))

        # WARN: Do not change the dtype on the logits as otherwise filesize increases dramatically (at least if it was bfloat16)
        # ALSO: does not seem like it saves much space/loads much quicker if i seperate the logits and input_ids,
        # prboably because the logits are so big (250k x 1000)
        torch.save(sample_generation_info, output_path)

    # save data for all the samples for this model and update plot info
    try:
        generation_meta["mean_dist"] = statistics.mean(chain.from_iterable(generation_meta["samples"]["dist"].values()))
    except:
        logger.error(f"got error with {generation_meta['samples']['dist']}")
        generation_meta["mean_dist"] = None

    torch.save(generation_meta, generation_meta_path)
    plot_data = config.get_plot_data()

    # update the generations with this one
    plot_data["generations"] = {**plot_data.get("generations", {}), config.model_subdir_name: generation_meta}
    config.save_plot_data(plot_data)

    logger.info(f"DONE:[magenta]/[{config.model_subdir_name}][/magenta]\t mean dist:{generation_meta['mean_dist']}")


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
        _logits = torch.zeros(len(logits), d_min_max[1]["max"], d_min_max[2]["max"], dtype=logits[0].dtype)
        for i, l in enumerate(logits):
            _logits[i, : l.shape[1], : l.shape[2]] = l
        logits = _logits

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


def compute_logit_scores(config: Config):

    # will likely need to shorten logits as OOM
    min_dims = [(1, config.cfid_seq_len)]  # (dim, val)

    # can also use: plot_data_file = config.get_plot_data()['generations']['outfiles']
    gen_files = _get_all_generations_files(config.sample_save_base / "generations", return_sorted=True)

    base_files = gen_files["base_model"]
    cond_base_files = gen_files["cond_base_model"]
    checkpoint_keys = [k for k in gen_files.keys() if "checkpoint_" in k]

    _y_logits = _gather_logits_from_files(base_files, min_dim=min_dims, max_files=config.max_files)
    _x_logits = _gather_logits_from_files(cond_base_files, min_dim=min_dims, max_files=config.max_files)
    logger.info(f"Got base model logits.")

    score_str = []
    # scores_out = {}
    scores_out = defaultdict(list)

    # cast to float and move to cuda
    y_logits, x_logits = _y_logits.float().cuda(), _x_logits.float().cuda()
    y_logits, x_logits = y_logits.transpose(1, 2), x_logits.transpose(1, 2)

    table = logger.get_table(title="CFID/FID Scores")
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


def plot_logit_scores(config: Config):

    acc_vals = [
        876.9435106,
        878.9634703,
        822.9013076,
        810.0338167,  # 791.0338167,
        800.912719,
        795.2998151,  # 852.2998151,
        760.2795084,  # 795.2795084,
        744.7206856,
        705.318847,
        700.1046792,  # 716.1046792,
    ]

    plot_data = config.get_plot_data()
    scores = plot_data["logit_scores"]

    cmap = mpl.colormaps["tab10"]

    plot_keys = [
        "cfid0",
        "cfid1",
        "cfid2",
        "cfid3",
        "cfid4",
        "cfid5",
        "cfid6",
        "cfid7",
        "cfid8",
        "cfid9",
        "fid0",
        "fid1",
        "fid2",
        "fid3",
        "fid4",
        "fid5",
        "fid6",
        "fid7",
    ]
    plot_keys1 = ["cfid0", "cfid1", "cfid2", "cfid3"]
    plot_keys2 = ["cfid4", "cfid5", "cfid6"]
    plot_keys3 = ["cfid7", "cfid8", "cfid9"]
    plot_keys4 = ["fid0", "fid1", "fid2", "fid3"]
    plot_keys5 = ["fid4", "fid5", "fid6", "fid7"]

    def make_plot(keys, out):
        fig, ax1 = plt.subplots()
        ax1.plot(acc_vals, label="acc", color="black")

        axs = []
        # for i, (k, v) in enumerate(scores.items()):
        for i, k in enumerate(keys):
            v = scores[k]
            ax = ax1.twinx()
            axs.append(ax)
            ax.set_ylabel(k)
            ax.plot(v, label=k, color=cmap(random.random()))
            if i > 4:
                break

        fig.tight_layout()
        fig.savefig(f"output/plots/{out}.png")

    make_plot(plot_keys1, "logit_scores1")
    make_plot(plot_keys2, "logit_scores2")
    make_plot(plot_keys3, "logit_scores3")
    make_plot(plot_keys4, "logit_scores4")
    make_plot(plot_keys5, "logit_scores5")
    breakpoint()

    # color = 'tab:red'
    # ax1.set_xlabel('time (s)')
    # ax1.set_ylabel('exp', color=color)
    # ax1.plot(t, data1, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    # ax2.plot(t, data2, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()


def umap_examine(config: Config):
    logger.info(f"Doing UMAP")
    import umap

    files = _get_all_generations_files(config.sample_save_base / "generations")


COMMANDS = {
    "main": main,
    "make_samples": make_samples,
    "model_process_samples_from_file": model_process_samples_from_file,
    "umap_examine": umap_examine,
    "compute_logit_scores": compute_logit_scores,
    "plot_logit_scores": plot_logit_scores,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")
    args = parser.parse_args()

    config: Config = args.config
    wandb_config: WandBConfig = args.wandb_config

    COMMANDS[config.cmd](config)
