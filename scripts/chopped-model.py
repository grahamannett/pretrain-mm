from dataclasses import dataclass

from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, FuyuConfig, FuyuForCausalLM, FuyuProcessor


@dataclass
class Config:
    base_model: str = "adept/fuyu-8b"
    saved_model_name: str = "chopped-fuyu"
    num_layers: int = 4


DEFAULT_CONFIG = Config()


def make_smaller(config: Config = DEFAULT_CONFIG):
    model_config = FuyuConfig.from_pretrained("adept/fuyu-8b")

    model_config.num_hidden_layers = config.num_layers
    model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", config=model_config, device_map="auto")

    model.save_pretrained(config.saved_model_name)


def use_smaller(config: Config = DEFAULT_CONFIG):
    model = AutoModelForCausalLM.from_pretrained(config.saved_model_name)


if __name__ == "__main__":
    parser = ArgumentParser().add_arguments(Config, dest="config")
    args = parser.parse_args()
    make_smaller(args.config)
