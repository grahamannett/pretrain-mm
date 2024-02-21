from dataclasses import asdict, dataclass

from simple_parsing import ArgumentParser

from pretrain_mm.model.fuyu import MODEL_ID, FuyuForCausalLM

parser = ArgumentParser()


@dataclass
class Config:
    base_model: str = MODEL_ID
    saved_model_name: str = "tmp/chopped/chopped-fuyu"
    num_layers: int = 1
    cmd: str = "make_smaller"


def make_smaller(
    base_model: str,
    num_layers: int = 1,
    saved_model_name: str = None,
    save_model: bool = False,
    reload_model: bool = False,
    model_load_kwargs: dict = {},
    **kwargs,
):
    # load model as is
    model = FuyuForCausalLM.from_pretrained(base_model, **model_load_kwargs)
    # values to patch
    # these all seem to need to be replaced for model to actually shrink
    model.config.num_hidden_layers = num_layers
    model.config.text_config.num_hidden_layers = num_layers
    model.language_model.config.num_hidden_layers = num_layers
    model.language_model.model.layers = model.language_model.model.layers[:num_layers]

    if save_model:
        model.save_pretrained(saved_model_name)

    if reload_model:
        assert save_model and saved_model_name, "Must save model and have saved_model_name to reload it"
        new_model = FuyuForCausalLM.from_pretrained(saved_model_name, **model_load_kwargs)
        return new_model

    return model


def use_smaller(config: "Config"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(config.saved_model_name)


if __name__ == "__main__":
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    config = args.config

    cmds = {
        make_smaller.__name__: make_smaller,
        use_smaller.__name__: use_smaller,
    }
    cmd = cmds[config.cmd]
    cmd(**config.asdict())
