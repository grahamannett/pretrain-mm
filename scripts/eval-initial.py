import os
from dataclasses import dataclass
from functools import partial

import torch
import torchmetrics

from config.dev import get_dev_config
from config.model_configs import ExperimentConfigModelInfo, ExperimentModelConfigMixin
from pretrain_mm import logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebPretrainProcessor, TaskAdapter
from pretrain_mm.metrics.bbox_metric import BBoxDistance
from pretrain_mm.utils.config_utils import BaseTrainConfig, FromConfig, WandBConfig
from pretrain_mm.utils.eval_utils import OCREvalCompletion, remove_label
from pretrain_mm.utils.generate_utils import StopOnToken


@dataclass
class WandBConfigExp(WandBConfig):
    group: str = "testing/pretrain-fuyu"
    job_type: str = "eval"
    tags: tuple[str, ...] = ("eval", "mind2web")


@dataclass
class EvalConfig(BaseTrainConfig, ExperimentModelConfigMixin):
    wandb: WandBConfig = FromConfig[WandBConfigExp]

    # since slurm seems to fuck up progress bar (so cant see in wandb/log.o%job)
    batch_log_every: int = False  # log
    num_iters: int = False  # num iters if not going through full dataset

    model_path: ExperimentConfigModelInfo = ExperimentConfigModelInfo.Fuyu

    output_dir: str = None  # "output/model_output"

    # dataset
    dataset_name: str = "mind2web"
    get_text_from: str = "ocr"

    data_subset: int = None
    batch_size: int = 1

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    eval_iters: int = 1000
    eval_from: OCREvalCompletion = OCREvalCompletion.bounding_box  #

    # generate kwargs
    max_new_tokens: int = 20
    temperature: float = 1.0
    do_sample: bool = True

    max_length: int = 2700
    add_cand_outline: bool = False
    skip_include_text: bool = False
    # MARK: mask related
    label_mask_text_ids: bool = True
    label_mask_image_patches: bool = True

    def __post_init__(self):
        if isinstance(self.dl_disable_progress, str):
            self.dl_disable_progress = self.dl_disable_progress.lower() == "true"


config = EvalConfig.cli()

# not entirely necessary to make these vars but was previously using simple-parsing
wandb_config: WandBConfig = config.wandb
model_info = config.model_info

ModelInfo = config.model_info
ModelConstants = ModelInfo.ModelConstants

ModelConfigCls = ModelInfo.ModelConfigCls
ModelCls = ModelInfo.ModelCls
ModelProcessorCls = ModelInfo.ProcessorCls


logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
logger.tools.check_exp_config(config=config, exp_type="initial-eval")

m2w_info = get_dev_config(config.dataset_name)

train_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    subset=config.data_subset,
    **m2w_info["train"],
)

test_data_config = Mind2WebConfig(
    task_dir=m2w_info["task_dir"],
    subset=config.data_subset,
    **m2w_info["test"],
)

train_dataset = Mind2Web(train_data_config)
test_dataset = Mind2Web(test_data_config)

model_kwargs = {"torch_dtype": getattr(torch, config.dtype, torch.float16), "device_map": config.device}
processor = ModelProcessorCls.from_pretrained(model_info.model_name, **model_info.tokenizer_kwargs)
model = ModelCls.from_pretrained(model_info.model_name, **model_kwargs)
# this goes from raw sample -> sample in task format
task_processor: Mind2WebPretrainProcessor = Mind2WebPretrainProcessor(
    get_text_from=config.get_text_from,
    tokenizer_constants=ModelConstants,
)


encode_func = partial(
    processor.encode_sample,
    max_length=config.max_length,
    truncation=True,
)

ocr_bounding_box_completion = partial(task_processor.ocr_eval, eval_from=config.eval_from)

transforms = {
    "task": ocr_bounding_box_completion,  # or agent_train_func
    "encode": encode_func,
}
train_dataset_adapter = TaskAdapter(
    train_dataset,
    transforms=transforms,
)


def eval_ocr_bounding_box(
    model,
    dataset,
    transforms,
    generate_kwargs: dict = {
        "max_new_tokens": EvalConfig.max_new_tokens,
        "temperature": EvalConfig.temperature,
        "do_sample": EvalConfig.do_sample,
    },
    max_new_tokens: int = 7,
    do_sample: bool = True,
    temperature: float = 0.1,
    max_iters: int = 1000,
    eval_from: str = EvalConfig.eval_from,
):
    stopping_criteria: list[callable] = [StopOnToken(ModelConstants.get_stop_ids(tokenizer=processor.tokenizer))]
    skipped_batches = []

    metric_fns = torchmetrics.MetricCollection(
        [
            torchmetrics.text.WordErrorRate(),
            torchmetrics.text.CharErrorRate(),
            torchmetrics.text.EditDistance(),
        ]
    )

    if eval_from == OCREvalCompletion.bounding_box:
        metric_fns.add_metrics(BBoxDistance())

    def check_stop():
        return metric_fns.BBoxDistance.total.item() >= max_iters

    def fix_gen_label(gen_label, eot_str="|ENDOFTEXT|"):
        if eot_str in gen_label:
            gen_label = gen_label.split(eot_str)[0]
        return gen_label

    def _batch_transform(batch):
        for transform in transforms:
            batch = transform(batch)
            if batch is None:
                return None
        return batch

    for b_idx, batch in enumerate(dataset):
        if (batch := _batch_transform(batch)) is None:
            skipped_batches.append(b_idx)
            continue

        try:
            boa_idx = processor.get_inputs_start_idx(batch.input_ids, labels=batch.labels, offset=-1)
            batch, (rem_ids, rem_lab) = remove_label(batch, to_idx=boa_idx)
        except:  # noqa: E722
            continue

        batch = batch.to(model.device)

        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

        generated_label = output[:, boa_idx:]
        decoded_generated_label = fix_gen_label(processor.full_decode(generated_label))

        text_target = batch.extra["label"]

        _ = metric_fns(decoded_generated_label, text_target)

        if check_stop():
            break

    logger.info(f"Num-Skipped Batches: {len(skipped_batches)}")
    logger.info("Run metrics.\n--------------------------------")
    metric_vals = metric_fns.compute()

    for metric_name, metric_val in metric_vals.items():
        metric_name = f"[{metric_name}]"
        logger.info(f"{metric_name:20} {metric_val}")


eval_ocr_bounding_box(
    model,
    train_dataset,
    transforms=[ocr_bounding_box_completion, encode_func],
    max_iters=config.eval_iters,
)
