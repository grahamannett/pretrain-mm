from dataclasses import dataclass

import torch
import random
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM

from config.dev import get_dev_config
from pretrain_mm import logger
from pretrain_mm.constants import VIEWPORT_SIZE_DICT
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, pretrain_instructions
from pretrain_mm.datasets.mind2web.mind2web_processor import Mind2WebPretrainProcessor, Mind2WebTaskProcessor
from pretrain_mm.datasets.task_adapter import TaskAdapter
from pretrain_mm.metrics.metrics import cfid, fid
from pretrain_mm.model.fuyu import MODEL_ID, FuyuConstants, FuyuForCausalLM, FuyuProcessor
from pretrain_mm.utils.generate_utils import generate_helper
from pretrain_mm.utils.token_tag_utils import box_pattern


@dataclass
class Config:
    model_path: str = "/data/graham/models/pretrain-mm/fuyu/latest"
    base_model: str = MODEL_ID
    # processor_path: str = "/data/graham/models/pretrain-mm/fuyu/mag-pretrain/processor"  # or MODEL_ID

    device_map: str = "auto"

    # input related
    instruction = pretrain_instructions.GenerateNumPotentialActions(num_candidates=1)

    input_max_length: int = 2500

    # generate related
    max_new_tokens: int = 10
    # temperature
    temperature: float = 1.0
    num_samples: int = 2


def create_samples_from_dataset(dataset, num_samples=Config.num_samples, random_draw=True):
    idxs = list(range(len(dataset)))

    if random_draw:
        random.shuffle(idxs)

    idxs = idxs[:num_samples]

    logger.info(f"Using the following indices: {idxs}")

    samples = [dataset[idx] for idx in idxs]

    return samples


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
            return_last_logits=True,
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


def gather_data(config):
    # instruct = f"{config.instruction(num_candidates=1)}{FuyuConstants.boa_string} \n {FuyuConstants.token_bbox_open_string}"

    m2w_info = get_dev_config("mind2web")
    ds_config = Mind2WebConfig(
        task_dir=m2w_info["task_dir"],
        # subset=config.data_subset,
        **m2w_info["train"],
    )

    dataset = Mind2Web(config=ds_config)

    processor = FuyuProcessor.from_pretrained(f"{config.model_path}/processor")
    stop_tokens = FuyuConstants.get_stop_tokens(processor)

    force_words_ids = [v[1] for v in FuyuConstants.get_all_ids(processor, skip_vals=[262144, 262145]).values()]
    force_words_ids += processor.tokenizer.convert_tokens_to_ids([str(i) for i in range(999)])
    force_words_ids = list(set(force_words_ids))

    pretrain_task_processor = Mind2WebPretrainProcessor(viewport_size=(1290, 1080))
    task_processor = Mind2WebTaskProcessor(processor=processor, max_length=config.input_max_length)

    transforms = {
        "pretrain_task": pretrain_task_processor.prepare_for_generate,
        # "encode": task_processor.encode_data,
    }

    task_dataset = TaskAdapter(dataset, transforms=transforms)

    samples = create_samples_from_dataset(task_dataset, num_samples=config.num_samples)

    model = FuyuForCausalLM.from_pretrained(config.base_model, device_map=config.device_map, torch_dtype=torch.bfloat16)

    to_save = {}

    vals = get_all_logits(
        model=model,
        encoder=task_processor.encode_data,
        samples=samples,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={"force_words_ids": force_words_ids},
        collect_base_logits=True,
    )

    to_save["base"] = vals

    model = FuyuForCausalLM.from_pretrained(config.model_path, device_map=config.device_map, torch_dtype=torch.bfloat16)

    vals = get_all_logits(
        model=model,
        encoder=task_processor.encode_data,
        samples=samples,
        max_new_tokens=config.max_new_tokens,
        collect_base_logits=True,
    )

    to_save["trained"] = vals

    torch.save(to_save, "outputs/gen_logits/data.pt")

    breakpoint()

    # get the trained model and get logits
    # model = FuyuForCausalLM.from_pretrained(config.model_path, device_map=config.device_map, torch_dtype=torch.bfloat16)

    # gen_inp = gen_inp.to(model.device)

    # gen_out, gen_logits = generate_helper(
    #     model, model_inputs=gen_inp.to(model.device), max_new_tokens=config.max_new_tokens, return_last_logits=True
    # )

    # gen_logits = gen_logits.detach().cpu()

    # b_model = FuyuForCausalLM.from_pretrained(
    #     config.base_model, device_map=config.device_map, torch_dtype=torch.bfloat16
    # )

    # gen_inp = gen_inp.to(b_model.device)
    # output, logits = generate_helper(
    #     b_model,
    #     model_inputs=gen_inp.to(b_model.device),
    #     max_new_tokens=config.max_new_tokens,
    #     return_last_logits=True,
    #     force_words_ids=force_words_ids,
    # )
    # output, logits = output.detach().cpu(), logits.detach().cpu()

    # breakpoint()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    config = parser.parse_args().config
    # gather_data(config)
    calculate_data(config)
