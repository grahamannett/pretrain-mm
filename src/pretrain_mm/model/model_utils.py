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
    model_load: bool = False,
):
    model = ModelCls.from_pretrained(model_name, **model_kwargs)
    tokenizer = ProcessorCls.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer


# TODO: is adding in_features even helpful?
def change_linear_features_by(layer: torch.nn.Linear, out_features: int = 1) -> torch.nn.Linear:
    if out_features < 0:
        raise ValueError("in_features and out_features must be positive")

    _features = (layer.in_features, layer.out_features)

    def _p_check(p):
        if isinstance(p, torch.nn.Parameter):
            return p.data
        return p

    def _concat_params(p, sz):
        p = _p_check(p)
        if any(map(lambda x: x <= 0, sz)):  # for negatives we should drop or what
            raise ValueError("shape must be positive")
        return torch.cat([p, torch.rand(*sz, dtype=p.dtype, device=p.device)], dim=0)

    with torch.no_grad():
        layer.weight.data = _concat_params(layer.weight.data, (out_features, _features[0]))

        if layer.bias:
            layer.bias.data = _concat_params(layer.bias.data, (1,))

    # change layer fields to match
    layer.out_features = layer.weight.data.shape[0]

    return layer
