from dataclasses import dataclass

from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, FuyuConfig, FuyuForCausalLM, FuyuProcessor

parser = ArgumentParser()


@dataclass
class Config:
    base_model: str = "adept/fuyu-8b"
    saved_model_name: str = "chopped-fuyu"
    num_layers: int = 4

    cmd: str = "make_smaller"


def make_smaller(config: "Config"):
    model_config = FuyuConfig.from_pretrained("adept/fuyu-8b")

    # values to patch
    model_config.num_hidden_layers = config.num_layers
    model_config.text_config.num_hidden_layers = config.num_layers

    model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", config=model_config, device_map="auto")

    model.save_pretrained(config.saved_model_name)


def use_smaller(config: "Config"):
    model = AutoModelForCausalLM.from_pretrained(config.saved_model_name)


cmds = {
    make_smaller.__name__: make_smaller,
    use_smaller.__name__: use_smaller,
}


# global FN_AVAILABLE
# FN_AVAILABLE = locals()

if __name__ == "__main__":
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    config = args.config
    cmd = cmds[config.cmd]
    breakpoint()
    # make_smaller(args.config)
    # cmd = config.cmd(config)
