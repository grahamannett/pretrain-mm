import torch
import transformers


def _default_shorten_model(model: torch.nn.Module, num_layers: int = 2) -> torch.nn.Module:
    get_layers = {
        "FuyuForCausalLM": lambda model: model.language_model.model.layers,
    }

    if isinstance(model, transformers.models.fuyu.modeling_fuyu.FuyuForCausalLM):
        model.language_model.model.layers = model.language_model.model.layers[:num_layers]
    if isinstance(model, transformers.models.mistral.modeling_mistral.MistralForCausalLM):
        model.model.layers = model.model.layers[:num_layers]
    return model


def setup_model_for_dev(
    model: torch.nn.Module, num_layers: int = 4, transform_model_func: callable = _default_shorten_model
) -> torch.nn.Module:
    model = transform_model_func(model, num_layers)
    return model


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def setup_model(
    model_name: str,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
    ModelCls: callable = transformers.AutoModelForCausalLM,
    ProcessorCls: callable = transformers.AutoTokenizer,
):
    model = ModelCls.from_pretrained(model_name, **model_kwargs)
    tokenizer = ProcessorCls.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer
