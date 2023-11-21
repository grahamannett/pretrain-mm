def dev_load_model(model_name, model_kwargs, ModelCls, train_config):
    if train_config.chop_model > 0:
        model = ModelCls.from_pretrained(model_name, **model_kwargs)
        model.language_model.model.layers = model.language_model.model.layers[: train_config.chop_model]
        model.to("cuda")
    else:
        model = ModelCls.from_pretrained(model_name, device_map="auto", **model_kwargs)
    return model
