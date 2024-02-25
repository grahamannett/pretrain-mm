from dataclasses import dataclass

from pathlib import Path

import torch
import random
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM

from config.dev import get_dev_config
from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE_DICT, IGNORE_INDEX
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, pretrain_instructions
from pretrain_mm.datasets.mind2web.mind2web_processor import Mind2WebPretrainProcessor, Mind2WebTaskProcessor
from pretrain_mm.datasets.task_adapter import TaskAdapter
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.model.fuyu import MODEL_ID, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import box_pattern
from pretrain_mm.utils.eval_utils import eval_by_completion, EVAL_BY_COMPLETION_GENERATE_KWARGS


@dataclass
class Config:
    model_path: str = "/data/graham/models/pretrain-mm/fuyu/latest"
    multi_models: bool = False
    base_model: str = MODEL_ID
    # processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    sample_save_path: str = "output/saved_samples/task_box_accuracy/samples.pt"

    device_map: str = "auto"

    # input related
    instruction = pretrain_instructions.GenerateNumPotentialActions(num_candidates=1)

    input_max_length: int = 2500

    # generate related
    max_new_tokens: int = 10
    # temperature
    temperature: float = 1.0
    eval_num_samples: int = 2


def _create_samples_from(dataset, num_samples: int, random=True, p_func: callable = None):
    pass


def create_samples_from_dataset(
    dataset, num_samples=Config.eval_num_samples, random_draw=True, validate_sample: callable = lambda x: True
):
    samples, idxs_used, idxs_bad = [], [], []

    idxs = list(range(len(dataset)))

    if random_draw:
        random.shuffle(idxs)

    while (len(samples) != num_samples) and idxs:

        idx = idxs.pop(0)

        if validate_sample(sample := dataset[idx]):
            samples.append(sample)
            idxs_used.append(idx)
        else:
            idxs_bad.append(idx)

    bad_idxs_str = f" | Bad Indices: {idxs_bad}" if idxs_bad else ""
    logger.info(f"Using the following indices: {idxs_used}{bad_idxs_str}")

    return samples


def process_samples(samples: list[dict], processor_func: callable, save_path: str = None):

    samples_ = [processor_func(sample) for sample in samples]
    samples_ = [s for s in samples_ if s not in [False, None]]

    # should i save each sample independently or all together?
    if save_path:
        torch.save(samples_, save_path)

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


def calculate_data(config):
    data = torch.load("outputs/gen_logits/data.pt")

    x_logits = torch.cat(data["base"]["base_logits"])
    y_logits = torch.cat(data["base"]["logits"])
    yh_logits = torch.cat(data["trained"]["logits"])

    # limit vals as they are 1000+ seq_len which means will result in a matrix of 1000x1000
    x_logits, y_logits, yh_logits = x_logits[:, :100], y_logits[:, :100], yh_logits[:, :100]
    x_logits, y_logits, yh_logits = x_logits.float(), y_logits.float(), yh_logits.float()
    x_logits, y_logits, yh_logits = x_logits.transpose(1, 2), y_logits.transpose(1, 2), yh_logits.transpose(1, 2)

    scores = cfid(y_logits, yh_logits, x_logits, mean_dim=-2, f_dim=-2)

    breakpoint()


def distance_metric_each_epoch(config, models_dir, get_model: callable, eval_kwargs: dict):

    for m_idx, model_path in enumerate(models_dir):
        model = get_model(model_path)
        eval_metrics = eval_by_completion(
            model,
            processor=processor,
            samples=samples,
            return_extra=True,
            generate_kwargs=dict(
                force_words_ids=force_words_ids,
                stop_tokens=stop_tokens,
                return_extra=True,
                max_new_tokens=config.max_new_tokens,
                forward_kwargs=dict(
                    output_hidden_states=True,
                ),
            ),
        )


def create_data_func()

def main(config):

    m2w_info = get_dev_config("mind2web")
    ds_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        # subset=config.data_subset,
        **m2w_info["train"],
    )
    test_data_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        # subset=config.data_subset,
        **m2w_info["test"],
    )

    dataset = Mind2Web(config=ds_config)
    test_dataset = Mind2Web(test_data_config)

    dataset.setup_pretrain()
    test_dataset.setup_pretrain()

    processor = FuyuProcessor.from_pretrained(f"{config.model_path}/processor")
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    force_words_ids = [v[1] for v in FuyuConstants.get_all_ids(processor, skip_vals=[262144, 262145]).values()]
    force_words_ids += processor.tokenizer.convert_tokens_to_ids([str(i) for i in range(999)])
    force_words_ids = list(set(force_words_ids))

    task_processor = Mind2WebTaskProcessor(processor=processor, max_length=config.input_max_length)
    # pretrain_task_processor = Mind2WebPretrainProcessor(viewport_size=(1290, 1080))
    pretrain_task_processor = Mind2WebPretrainProcessor()

    task_processor = Mind2WebTaskProcessor(
        processor=processor,
        ignore_index=IGNORE_INDEX,
        max_length=config.input_max_length,
        encode_kwargs={"label_mask_text_ids": True},
    )

    transforms = {
        "pretrain_task": pretrain_task_processor.prepare_for_generate,
        # "encode": task_processor.encode_data,
    }

    task_dataset = TaskAdapter(dataset, transforms=transforms)
    # base_model = FuyuForCausalLM.from_pretrained(config.base_model, device_map=config.device_map, torch_dtype=torch.bfloat16)

    # samples = create_samples_from_dataset(task_dataset, num_samples=config.eval_num_samples)
    def validate_sample(raw_sample):
        return raw_sample.pos_candidates != []

    def p_func(raw_sample) -> dict:
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

    samples = process_samples(
        create_samples_from_dataset(test_dataset, num_samples=config.eval_num_samples, validate_sample=validate_sample),
        processor_func=p_func,
        save_path=config.sample_save_path,
    )

    def get_model_func(p):
        return FuyuForCausalLM.from_pretrained(p, device_map=config.device_map, torch_dtype=torch.bfloat16)

    def save_outputs(obj, base_dir, output_file):
        output_file = Path(base_dir) / output_file
        torch.save(obj, output_file)
        logger.info(f"Saved to {output_file}")

    if config.multi_models:
        model_dirs = [p for p in Path(config.model_path).iterdir() if p.name != "processor"]
        model_dirs.sort()

    to_save = {}

    for m_idx, model_path in enumerate(model_dirs):
        model = get_model_func(model_path)
        model_path_info = model_path.name

        logger.info(f"doing model from path: {model_path_info}")

        eval_metrics = eval_by_completion(
            model,
            processor=processor,
            samples=samples,
            return_extra=True,
            prepend_str="",
            prepend_str_extra="",
            generate_kwargs=dict(
                # force_words_ids=force_words_ids,
                stop_tokens=stop_tokens,
                return_extra=True,
                max_new_tokens=config.max_new_tokens,
                forward_kwargs=dict(
                    output_hidden_states=True,
                ),
            ),
        )
        # save file should be like
        # outputs['all_outputs'][sample_n]['logits']
        logger.info(f"Got Eval Metrics: {eval_metrics['dist_metric']}")
        to_save[model_path_info] = eval_metrics

    torch.save(to_save, "output/logits-data-test.pt")
    logger.info(f"THIS IS GREAT IT RAN!")

    #     model = FuyuForCausalLM.from_pretrained(model_dirs[0], device_map=config.device_map, torch_dtype=torch.bfloat16)
    #     logger.info(f"Using the following models: {model_dirs[0]}")
    #     # model = AutoModelForCausalLM.from_pretrained(config.model_path, device_map=config.device_map, torch_dtype=torch.bfloat16)

    # eval_metrics = eval_by_completion(
    #     model,
    #     processor=processor,
    #     samples=samples,
    #     # below arent needed if passing in samples
    #     # dataset=test_dataset,
    #     # task_func=pretrain_task_processor.acc_func_complete_box,
    #     # num_samples=config.eval_num_samples,
    #     return_extra=True,
    #     generate_kwargs=dict(
    #         force_words_ids=force_words_ids,
    #         stop_tokens=stop_tokens,
    #         return_extra=True,
    #         max_new_tokens=config.max_new_tokens,
    #         forward_kwargs=dict(
    #             output_hidden_states=True,
    #         ),
    #     ),
    # )
    # breakpoint()

    # to_save = {}

    # MOVE THIS TO ANOTHER FUNC LATER

    # torch.save(eval_metrics, f"output/measure_metrics/{model_path_info}.pt")

    # vals = get_all_logits(
    #     model=model,
    #     encoder=task_processor.encode_data,
    #     samples=samples,
    #     max_new_tokens=config.max_new_tokens,
    #     generate_kwargs={"force_words_ids": force_words_ids},
    #     collect_base_logits=True,
    # )

    # to_save["base"] = vals

    # model = FuyuForCausalLM.from_pretrained(config.model_path, device_map=config.device_map, torch_dtype=torch.bfloat16)

    # vals = get_all_logits(
    #     model=model,
    #     encoder=task_processor.encode_data,
    #     samples=samples,
    #     max_new_tokens=config.max_new_tokens,
    #     collect_base_logits=True,
    # )

    # to_save["trained"] = vals

    # torch.save(to_save, "outputs/gen_logits/data.pt")

    # breakpoint()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config

    main(config)
    # calculate_data(config)
