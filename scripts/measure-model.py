import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torchmetrics
import torchmetrics.text as textmetrics
from simple_parsing import ArgumentParser, choice

from config.dev import get_dev_config
from pretrain_mm import logger
from pretrain_mm.constants import IGNORE_INDEX, VIEWPORT_SIZE
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, pretrain_instructions
from pretrain_mm.datasets.mind2web.mind2web_processor import Mind2WebEncoder, Mind2WebPretrainProcessor
from pretrain_mm.evaluation.cfid_logits import _get_all_generations_files, compute_logit_scores
from pretrain_mm.model.fuyu import MODEL_ID, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.config_utils import FromConfig, WandBConfig
from pretrain_mm.utils.eval_utils import sample_eval_by_completion
from pretrain_mm.utils.generate_utils import StopOnToken
from pretrain_mm.utils.ocr_helper import PaddleOCRLabeler
from pretrain_mm.utils.token_tag_utils import TagType


dataset_host_info = get_dev_config("mind2web")


def _is_bad_sample(sample):
    return sample in [False, None]


def _get_model(path, device_map: str = "auto", get_processor: bool | str = True):
    """
    HELPER FUNCTION
    """
    model = FuyuForCausalLM.from_pretrained(path, device_map=device_map, torch_dtype=torch.bfloat16)

    if get_processor:
        path = get_processor if isinstance(get_processor, str) else path
        processor = FuyuProcessor.from_pretrained(path)
        model = (model, processor)
    return model


def _to_alphanumeric(text: str) -> str:
    # remove any non alphanumeric/space character
    return re.sub(r"[^a-zA-Z0-9 ]", "", text)


@dataclass
class WandBConfig(WandBConfig):
    group: str = "eval/pretrain-fuyu"
    job_type: str = "pretrain-eval"


@dataclass
class Config(FromConfig.Base):  # make it so its serializable
    cmd: list[str] | str  # List[str] | str  # should be one of COMMANDS.keys() which is initialized

    base_model: str = MODEL_ID
    model_path: str = MODEL_ID  #  None  # "/data/graham/models/pretrain-mm/fuyu/actiontag-random-order/checkpoint_1"

    processor_path: str = MODEL_ID  # "/data/graham/models/pretrain-mm/fuyu/actiontag-random-order/processor"
    model_subdir_name: str = None  # subdir to where generations will be saved out.  if not passed, uses model_path.name

    # plot related info
    plot_infofile: str = "output/plot_infofile.pt"  # might switch to tinydb if needed
    reset_plot_data: bool = False

    # processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    sample_save_base: str = "output/saved_samples/task_box_accuracy/"
    task_samples_file: str = "task_samples.pt"

    device_map: str = "auto"
    device: str = "cuda"  # for tensors when doing non model related

    # input related
    instruction = pretrain_instructions.GenerateNumPotentialActions(num_candidates=1)
    task_gen_func: str | list[str] = "eval_by_complete_text"  # or "acc_func_complete_box"

    input_max_length: int = 2500
    viewport_size: tuple[int, int] = VIEWPORT_SIZE

    # generate related
    max_new_tokens: int = 20
    use_force_words: bool = False
    # temperature
    temperature: float = 1.0
    eval_num_samples: int = 2
    random_samples: bool = True
    num_generations_per_sample: int = 1

    # dataset related
    dataset_for_samples: str = choice("train", "test", default="train")
    task_dir: str = dataset_host_info["task_dir"]

    # if we want to limit the number of samples we process/use
    data_subset: int = None

    # for ocr baselines
    ocr_results_from: str = "paddleocr"
    max_ocr_results: int = None

    # using past_key_values seems like it might generate different results
    use_past_key_values: bool = False
    output_hidden_states: bool = False  # if we want the penultimate hidden states for umap

    cfid_seq_len: int = 100
    max_files: int = None

    def __post_init__(self):
        self.sample_save_base = Path(self.sample_save_base)

        # if we give model_path and not model_subdir_name, then use the name of the model_path
        if self.model_path and (self.model_subdir_name is None) and (_path := Path(self.model_path)).exists():
            self.model_subdir_name = _path.name
            logger.info(f"Using model_subdir_name name from model_path: {_path.name}")

        self._check_for_plot_infofile()

    def _check_for_plot_infofile(self, default_plot_data: dict = {}):
        if self.reset_plot_data:
            # if reset and data exists, print the prior data to screen before reset (just using empty dict)
            if _prev_plot_data := Path(self.plot_data).exists():
                _prev_plot_data = torch.load(_prev_plot_data)
                logger.info(_prev_plot_data)

            logger.warn("RESET PLOT DATA")
            torch.save(default_plot_data, self.plot_infofile)

        if not Path(self.plot_infofile).exists():
            torch.save(default_plot_data, self.plot_infofile)
            logger.info(f"Created plot_infofile: {self.plot_infofile}")

    def save_plot_data(self, data: dict):
        torch.save(data, self.plot_infofile)
        logger.info(f"Saved plot data to: {self.plot_infofile}")

    def get_plot_data(self):
        return torch.load(self.plot_infofile)

    def make_dataset_config_kwargs(self, dataset_split: str = "train"):
        # might be m2w specfic right now but anything you need to init dataset confit
        return {
            "subset": self.data_subset,
            "task_dir": self.task_dir,
            "attach_config_to_sample": True,
            **dataset_host_info[dataset_split],
        }


def _get_dataset(config: "Config", dataset_split: str = "train", return_config: bool = False):
    ds_config = Mind2WebConfig(**config.make_dataset_config_kwargs(dataset_split))
    ds = Mind2Web(config=ds_config)
    if return_config:
        ds = (ds, ds_config)
    return ds


@dataclass
class OCRResult:
    bbox: list[int]
    text: str
    confidence: float

    def bbox_margin(self, margin: int = 2):
        # the box on paddleocr is extremely tight, so add a few pixels
        bbox = [self.bbox[0] - margin, self.bbox[1] - margin, self.bbox[2] + margin, self.bbox[3] + margin]
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        return bbox


def _baseline_ocr_generate_text(model, samples, generate_kwargs):
    


def baseline_ocr(config: Config, conf_threshold: float = 0.8):
    ds = _get_dataset(config, "train")
    # load surya tooling

    sample = ds[0]

    model, processor = _get_model(config.model_path, get_processor=config.processor_path)

    text_template = pretrain_instructions.BaselineBoxToText()

    ocr_labeler = PaddleOCRLabeler()

    generate_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
        pad_token_id=processor.pad_token_id,
        stopping_criteria=[StopOnToken(FuyuConstants.get_stop_tokens())],
    )

    def transform_fn(sample):
        sample.image = sample.image.crop((0, 0, *config.viewport_size))
        return sample

    pbar = logger.progress(ensure_exit=True, start=True, disable=False)
    page_task = pbar.add_task("[blue] processing page", total=config.eval_num_samples)

    gen_strs = []
    ocr_strs = []

    samples = [ds.get_with_transform(transform_fn) for _ in range(config.eval_num_samples)]
    ocr_results = [[OCRResult(*r) for r in ocr_labeler(sample.image)] for sample in samples]
    breakpoint()

    for n in range(config.eval_num_samples):
        # will draw sample randomly
        sample = ds.get_with_transform(transform_fn)

        ocr_results = [OCRResult(*r) for r in ocr_labeler(sample.image)]

        num_ocr_results = config.max_ocr_results or len(ocr_results)
        ocr_task = pbar.add_task("[green] OCR", total=num_ocr_results)

        page_gen_strs = []
        page_ocr_strs = []  # ocr here means a non training model :)

        for res_idx, ocr_result in enumerate(ocr_results):
            pbar.update(ocr_task, advance=1)

            if ocr_result.confidence < conf_threshold:
                continue

            if config.max_ocr_results and (res_idx >= config.max_ocr_results):
                break

            bounding_box = ocr_result.bbox_margin()
            box_str = TagType.make(TagType.BOX)(*bounding_box)
            text = text_template(box_str=box_str)

            model_inputs = processor(text=text, images=sample.image, add_bos_token=True, add_boa_token=True)
            gen_start_idx = processor.get_inputs_start_idx(model_inputs.input_ids, offset=-1)  # from after boa token

            gen_out = model.generate(**model_inputs, **generate_kwargs)
            gen_str = processor.full_decode(gen_out[0, gen_start_idx:])

            page_ocr_strs.append(ocr_result.text)
            page_gen_strs.append(gen_str)

        gen_strs.append(page_gen_strs)
        ocr_strs.append(page_ocr_strs)

        pbar.update(page_task, advance=1)

    pbar.stop()

    def _clean_str(s):
        for c in [FuyuConstants.image_newline_string, FuyuConstants.eos_string]:
            s = s.replace(c, "")
        return s

    infolm = textmetrics.infolm.InfoLM(
        "google/bert_uncased_L-2_H-128_A-2", idf=False, verbose=False, information_measure="l2_distance"
    )
    bertscore = textmetrics.BERTScore()
    matcherr = textmetrics.MatchErrorRate()
    werr = textmetrics.WordErrorRate()

    # per page stats
    vals = defaultdict(list)

    for idx in range(len(gen_strs)):
        _gen = [_clean_str(s) for s in gen_strs[idx]]
        _ocr = ocr_strs[idx]

        vals["infolm"].append(infolm(_gen, _ocr))
        vals["matcherr"].append(matcherr(_gen, _ocr))
        vals["werr"].append(werr(_gen, _ocr))

        # bert is dict and should be averaged
        vals["raw/bertscores"].append(bertscore(_gen, _ocr))
        vals["bertscores"] = {k: v.mean() for k, v in vals["raw/bertscores"][-1].items()}

    infolm_fig, infolm_axs = infolm.plot(vals["infolm"])
    matcherr_fig, matcherr_axs = matcherr.plot(vals["matcherr"])
    werr_fig, werr_axs = werr.plot(vals["werr"])
    bert_fig, bert_axs = bertscore.plot(vals["bertscores"])

    plot_dir = "output/plots/baseline_metrics"
    infolm_fig.savefig(f"{plot_dir}/infolm.png")
    matcherr_fig.savefig(f"{plot_dir}/matcherr.png")
    werr_fig.savefig(f"{plot_dir}/werr.png")
    bert_fig.savefig(f"{plot_dir}/bert.png")
    breakpoint()


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


def main(config: Config):
    raise KeyError(f"Enter a valid command: {COMMANDS.keys()}")


def generate_samples_from_dataset(
    dataset,
    # process data func has to return either dict or False
    process_data_func: callable,
    num_samples=Config.eval_num_samples,
    random_samples=True,
):
    samples, idxs_bad = [], []

    # iterator over dataset in random order
    def _iter():
        _gen = range(len(dataset))

        if random_samples:
            _gen = sorted(_gen, key=lambda x: random.random())

        for idx in _gen:
            yield dataset[idx], idx

    for raw_sample, idx in _iter():
        if _is_bad_sample(sample := process_data_func(raw_sample, idx)):
            idxs_bad.append(idx)
            continue

        # means we got a sample that fully works
        samples.append(sample)

        if len(samples) >= num_samples:
            break

    idxs_bad_str = f" | Bad Indices: {idxs_bad}" if idxs_bad else ""
    idxs_good_str = f" | Good Indices: {' '.join([str(s['idx']) for s in samples])}"

    logger.info(f"Using: {idxs_good_str} {idxs_bad_str}")

    samples = sorted(samples, key=lambda x: x["idx"])

    return samples, {"idxs_bad": idxs_bad}


def make_samples(config: Config):
    dataset, test_dataset = _get_dataset(config), _get_dataset(config, "test")
    dataset.setup_pretrain()
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(config.processor_path)

    task_processor = Mind2WebEncoder(processor=processor, max_length=config.input_max_length)
    pretrain_task_processor = Mind2WebPretrainProcessor(viewport_size=config.viewport_size)

    task_processor = Mind2WebEncoder(
        processor=processor,
        ignore_index=IGNORE_INDEX,
        max_length=config.input_max_length,
        encode_kwargs={"label_mask_text_ids": True},
    )

    task_func = getattr(pretrain_task_processor, config.task_gen_func)

    logger.info(f"Generate: {config.eval_num_samples} samples, task function: {config.task_gen_func}")

    def process_func(
        raw_sample,
        idx: int,
        enc_kwargs={
            "add_bos_token": False,
            "add_boa_token": False,
            "label_add_eos_token": False,
            "include_label": False,
        },
    ) -> dict | bool:
        """
        come in as raw sample, and try to
        """
        # task_sample = pretrain_task_processor.acc_func_complete_box(raw_sample)
        # if not (task_sample := pretrain_task_processor.eval_by_complete_text(raw_sample)):
        if not (task_sample := task_func(raw_sample)):
            return

        enc_kwargs.update(getattr(task_sample, "encode_kwargs", {}))

        enc_sample = task_processor.encode_data(task_sample, **enc_kwargs)

        task_sample["_extra"] = {
            "action_idx": raw_sample.action_idx,
            "trajectory_idx": raw_sample.trajectory_idx,
            "action_uid": raw_sample.action_uid,
            "annotation_id": raw_sample.annotation_id,
        }

        return {
            "idx": idx,
            "raw": raw_sample,
            f"task.{config.task_gen_func}": task_sample,
            f"encoded.{config.task_gen_func}": enc_sample,
        }

    samples, idx_info = generate_samples_from_dataset(
        {"train": dataset, "test": test_dataset}[config.dataset_for_samples],
        process_data_func=process_func,
        num_samples=config.eval_num_samples,
        random_samples=config.random_samples,
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


def evaluate_samples(config: Config):
    samples_data = torch.load(config.sample_save_base / config.task_samples_file)
    samples = samples_data["samples"]

    model, proc = _get_model(config.model_path, get_processor=config.processor_path)
    stop_tokens, force_words_ids = get_extra_token_related(proc, use_force_words=config.use_force_words)

    generate_kwargs = {
        "stop_tokens": stop_tokens,
        "force_words_ids": force_words_ids,
        "return_extra": False,  # dont need
        "max_new_tokens": config.max_new_tokens,
        "use_past_key_values": config.use_past_key_values,
        "forward_kwargs": {
            "output_hidden_states": config.output_hidden_states,
        },
    }

    # information_measure:
    # Literal['kl_divergence', 'alpha_divergence', 'beta_divergence', 'ab_divergence', 'renyi_divergence',
    # 'l1_distance', 'l2_distance', 'l_infinity_distance', 'fisher_rao_distance'])
    infolm = torchmetrics.text.infolm.InfoLM(
        "google/bert_uncased_L-2_H-128_A-2",
        idf=False,
        verbose=False,
        information_measure="l2_distance",
    )

    def metric_fn(pred, target):
        # for infoLM one of:
        # 'kl_divergence', 'alpha_divergence', 'beta_divergence', 'ab_divergence', 'renyi_divergence',
        # 'l1_distance', 'l2_distance', 'l_infinity_distance', 'fisher_rao_distance'
        return infolm(pred, target).item()
        # return torchmetrics.functional.text.bleu_score(pred, target).item()

    eval_info = []
    for sample_idx, sample in enumerate(samples):
        sample_eval_info = sample_eval_by_completion(
            model=model,
            processor=proc,
            sample=sample,
            num_generations=config.num_generations_per_sample,
            prepend_str="",
            prepend_str_extra="",
            get_decode_start_idx=proc.get_inputs_start_idx,
            metric_fn=metric_fn,
            generate_kwargs=generate_kwargs,
        )

        eval_info.append(sample_eval_info)

    metric_mean = statistics.mean(x["metrics"][0] for x in eval_info)
    print(f"got metric mean: {metric_mean}")


def model_process_samples_from_file(config: Config):
    """
    use this function to take the given saved samples, and then use a specific model to generate completions and
    the related logits.  the logits are the main values of interest and we want the logits over the whole output space
    """

    assert config.model_subdir_name is not None, "Need to pass model_subdir_name or have it set in config __post_init__"

    generation_meta_path = config.sample_save_base / "generations" / config.model_subdir_name / "generation_meta.pt"
    generation_meta = {
        "samples": {"metric": {}},
        "outfiles": [],
    }  # this is NOT for large data, just metrics related to all generations and per sample generations

    model = get_model_func(config.model_path)

    proc = FuyuProcessor.from_pretrained(config.processor_path)
    stop_tokens, force_words_ids = get_extra_token_related(proc, use_force_words=config.use_force_words)

    samples_data = torch.load(config.sample_save_base / config.task_samples_file)
    samples = samples_data["samples"]

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
            get_decode_start_idx=proc.get_inputs_start_idx,
            generate_kwargs=generate_kwargs,
        )

        # output file contains logits/input_ids
        output_path = (
            config.sample_save_base / "generations" / config.model_subdir_name / f"generations_base_{sample_idx}.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # you want to save metrics/distance and then sample related stuff possibly to generation_meta,
        # lose the sample_idx as not clear if i need, alt is generation_meta["samples"][f"{sample_idx}.dist"]
        generation_meta["samples"]["distance"][sample_idx] = sample_generation_info["dist"]
        generation_meta["outfiles"].append(str(output_path.resolve()))

        # WARN: Do not change the dtype on the logits as otherwise filesize increases dramatically (at least if it was bfloat16)
        # ALSO: does not seem like it saves much space/loads much quicker if i seperate the logits and input_ids,
        # prboably because the logits are so big (250k x 1000)
        torch.save(sample_generation_info, output_path)

    # save data for all the samples for this model and update plot info
    try:
        generation_meta["mean_metrics"] = statistics.mean(
            chain.from_iterable(generation_meta["samples"]["metrics"].values())
        )
    except:
        logger.error(f"got error with {generation_meta['samples']['dist']}")
        generation_meta["mean_metric"] = None

    torch.save(generation_meta, generation_meta_path)
    plot_data = config.get_plot_data()

    # update the generations with this one
    plot_data["generations"] = {**plot_data.get("generations", {}), config.model_subdir_name: generation_meta}
    config.save_plot_data(plot_data)

    logger.info(f"DONE:[magenta]/[{config.model_subdir_name}][/magenta]\t mean dist:{generation_meta['mean_dist']}")


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
    logger.info("Done with all plots")

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
    logger.info("Doing UMAP")

    files = _get_all_generations_files(config.sample_save_base / "generations")


COMMANDS = {
    "main": main,
    "baseline_ocr": baseline_ocr,
    "make_samples": make_samples,
    "model_process_samples_from_file": model_process_samples_from_file,
    "umap_examine": umap_examine,
    "compute_logit_scores": compute_logit_scores,
    "plot_logit_scores": plot_logit_scores,
    "evaluate_samples": evaluate_samples,
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(WandBConfig, dest="wandb_config", prefix="wandb.")

    # this fixes the nargs for the first arg to be *
    # parser._wrappers[0].fields[0].arg_options["nargs"] = "*"
    args = parser.parse_args()

    config: Config = args.config
    wandb_config: WandBConfig = args.wandb_config

    if isinstance(config.cmd, str):
        config.cmd = [config.cmd]

    logger.info(f"Doing the following commands in order: {config.cmd}")
    for cmd in config.cmd:
        COMMANDS[cmd](config)
