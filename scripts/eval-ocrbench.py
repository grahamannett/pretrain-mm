import os
from dataclasses import dataclass

import datasets
import torch

# from simple_parsing import ArgumentParser, choice
from config.model_configs import ExperimentConfigModelInfo, ExperimentModelConfigMixin

# from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.utils.config_utils import BaseTrainConfig, FromConfig, WandBConfig


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

    # model_id: str = FuyuInfo.model_name  # "adept/fuyu-8b"
    # model_config = FuyuInfo
    model_path: ExperimentConfigModelInfo = ExperimentConfigModelInfo.Fuyu
    model_name_or_path: str = None

    output_dir: str = None  # "output/model_output"

    # dataset
    dataset_name: str = "echo840/OCRBench"
    dataset_split: str = "test"  # only test is available for OCRBench
    get_text_from: str = "ocr"

    data_subset: int = None
    batch_size: int = 1

    dl_disable_progress: bool | str = os.environ.get("DL_DISABLE_PROGRESS", False)
    dl_num_workers: int = 4
    dl_pin_memory: bool = True

    eval_iters: int = 1000
    eval_from: str = "bounding_box"

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
# allow for FT/checkpointed models

ModelConfigCls = ModelInfo.ModelConfigCls
ModelCls = ModelInfo.ModelCls
ModelProcessorCls = ModelInfo.ProcessorCls

logger.tools.setup_wandb(wandb_config=wandb_config, config=config)
logger.tools.check_exp_config(config=config, exp_type="initial-eval")
dataset = datasets.load_dataset(config.dataset_name, split=config.dataset_split)


# model_chop = {"num_hidden_layers": 1, "text_config": {"model_type": "persimmon", "num_hidden_layers": 1}}
# model_config = ModelConfigCls.from_pretrained(model_info.model_name) if ModelConfigCls else None
# model = ModelCls.from_pretrained(model_info.model_name, config=model_config, device_map=config.device)
model_kwargs = {"torch_dtype": getattr(torch, config.dtype, torch.float16), "device_map": config.device}
processor = ModelProcessorCls.from_pretrained(model_info.model_name, **model_info.tokenizer_kwargs)
model = ModelCls.from_pretrained(config.model_name)
# this goes from raw sample -> sample in task format

breakpoint()

# encode_func = partial(
#     processor.encode_sample,
#     # label_mask_image_patches=config.label_mask_image_patches,
#     # label_mask_text_ids=config.label_mask_text_ids,
#     max_length=config.max_length,
#     truncation=True,
# )

# ocr_bounding_box_completion = partial(task_processor.ocr_eval, eval_from=config.eval_from)

# transforms = {
#     "task": ocr_bounding_box_completion,  # or agent_train_func
#     "encode": encode_func,
# }
# train_dataset_adapter = TaskAdapter(
#     train_dataset,
#     transforms=transforms,
# )

# sample = train_dataset_adapter[0]


# def eval_ocr_bounding_box(
#     model,
#     dataset,
#     transforms,
#     generate_kwargs: dict = {
#         "max_new_tokens": EvalConfig.max_new_tokens,
#         "temperature": EvalConfig.temperature,
#         "do_sample": EvalConfig.do_sample,
#     },
#     max_new_tokens: int = 7,
#     do_sample: bool = True,
#     temperature: float = 0.1,
#     max_iters: int = 1000,
#     eval_from: str = EvalConfig.eval_from,
# ):
#     stopping_criteria: list[callable] = [StopOnToken(ModelConstants.get_stop_ids(tokenizer=processor.tokenizer))]

#     wer = torchmetrics.text.WordErrorRate()
#     cer = torchmetrics.text.CharErrorRate()
#     edt = torchmetrics.text.EditDistance()

#     wer_vals = 0
#     cer_vals = 0
#     edt_vals = 0
#     dist_vals = 0
#     dist_val_min, dist_val_max = math.inf, -math.inf
#     dist_num_seen = 0
#     num_seen = 0

#     def _batch_transform(batch):
#         for transform in transforms:
#             batch = transform(batch)
#             if batch is None:
#                 return None
#         return batch

#     for b_idx, batch in enumerate(dataset):
#         if (batch := _batch_transform(batch)) is None:
#             logger.info(f"Skipping batch: {b_idx}")
#             continue

#         try:
#             boa_idx = processor.get_inputs_start_idx(batch.input_ids, labels=batch.labels, offset=-1)
#             batch, (rem_ids, rem_lab) = remove_label(batch, to_idx=boa_idx)
#         except:
#             continue

#         batch = batch.to(model.device)

#         output = model.generate(
#             **batch,
#             max_new_tokens=max_new_tokens,
#             do_sample=do_sample,
#             temperature=temperature,
#             pad_token_id=processor.pad_token_id,
#             stopping_criteria=stopping_criteria,
#         )
#         # decoded_output = processor.full_decode(output)

#         generated_label = output[:, boa_idx:]
#         decoded_generated_label = processor.full_decode(generated_label)
#         if "|ENDOFTEXT|" in decoded_generated_label:
#             decoded_generated_label = decoded_generated_label.split("|ENDOFTEXT|")[0]

#         text_target = batch.extra["label"]

#         wer_vals += wer(decoded_generated_label, text_target).item()
#         cer_vals += cer(decoded_generated_label, text_target).item()
#         edt_vals += edt(decoded_generated_label, text_target).item()

#         dist_val = box_distance_fn(decoded_generated_label, text_target)
#         if isinstance(dist_val, float):
#             if dist_val < dist_val_min:
#                 dist_val_min = dist_val
#             if dist_val > dist_val_max:
#                 dist_val_max = dist_val
#             dist_vals += dist_val
#             dist_num_seen += 1
#             dist_val_print = f"{dist_val:.4f}"
#         else:
#             dist_val_print = "[red]DIST-ERR[/red]"

#         num_seen += 1

#         if num_seen > max_iters:
#             break

#         logger.info(
#             f"[B-IDX][{b_idx}]Dist-{dist_val_print}  ||  Gen:{decoded_generated_label}  ||  Target:{text_target}"
#         )

#         # metric_val = torchmetrics.functional.text.word_error_rate(decoded_generated_label, text_target)
#         # distance_val = box_distance_fn(decoded_generated_label, text_target)
#         # breakpoint()
#         # logger.info(f"Got metric val: {metric_val}")

#     logger.info("run metrics")
#     logger.info(f"wer vals: {wer_vals / num_seen}")
#     logger.info(f"wer vals.compute: {wer.compute()}")

#     logger.info(f"cer vals: {cer_vals / num_seen}")
#     logger.info(f"cer vals.compute: {cer.compute()}")
#     logger.info(f"edt vals: {edt_vals / num_seen}")
#     logger.info(f"edt vals.compute: {edt.compute()}")

#     logger.info(f"dist vals: {dist_vals / dist_num_seen}")


# eval_ocr_bounding_box(
#     model,
#     train_dataset,
#     transforms=[ocr_bounding_box_completion, encode_func],
#     max_iters=config.eval_iters,
# )
