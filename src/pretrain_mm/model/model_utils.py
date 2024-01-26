import torch


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


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


class ModifiedOutputMixin:
    def increase_output_size(
        self,
        layer: torch.nn.Linear,
        increase_by: int = 1,
        patch_vocab: bool = True,
        #
        _patch_vocab_key: str = "vocab_size",
        _patch_config_callback: callable = None,
    ):
        change_linear_features_by(layer, out_features=increase_by)

        if patch_vocab:
            # still might be problematic when model has a text_config instance on it where that holds the vocab_size as well
            self.config.__dict__[_patch_vocab_key] += increase_by

        if _patch_config_callback:
            _patch_config_callback(self.config)
