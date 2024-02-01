import torch


def dev_load_model(model_name, model_kwargs, ModelCls, train_config):
    if train_config.chop_model > 0:
        model = ModelCls.from_pretrained(model_name, **model_kwargs)
        model.language_model.model.layers = model.language_model.model.layers[: train_config.chop_model]
        model.to("cuda")
    else:
        model = ModelCls.from_pretrained(model_name, device_map="auto", **model_kwargs)
    return model


def make_profiler(
    dir_name: str = "./output/profiler",
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
    start: bool = True,
):
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name),
        record_shapes=True,
        with_stack=True,
    )

    # allow not started to be used as ctx
    if start:
        prof.start()

    return prof
