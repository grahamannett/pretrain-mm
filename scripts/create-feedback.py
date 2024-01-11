from dataclasses import dataclass
from simple_parsing import ArgumentParser

import transformers

from pretrain_mm.model.fuyu.processing_fuyu import FuyuProcessor
from pretrain_mm.utils.config_utils import BaseConfig
from pretrain_mm.utils.eval_utils import box_pattern


@dataclass
class Config(BaseConfig):
    model_id: str = "adept/fuyu-8b"
    model_output_dir: str = "/data/graham/models/pretrain-mm/fuyu/masked_output"
    processor_output_dir: str = "/data/graham/models/pretrain-mm/fuyu/processor"

    dataset_name: str = "mind2web"

    max_new_tokens: int = 25
    temperature: float = 0.5


def feedback_dataset(sample, dataset, evaluate_function):
    for sample in dataset:
        generated_output = sample
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    config: Config = args.config

    processor = FuyuProcessor.from_pretrained(config.model_id, trust_remote_code=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        # config.model_id,
        config.model_output_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    feedback_dataset(dataset, model, score_func)
