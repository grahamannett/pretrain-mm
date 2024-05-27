import torch

from pretrain_mm import logger


def get_bnb_optim(optim_name: str):
    import bitsandbytes as bnb

    OptimCls = getattr(bnb.optim, optim_name)

    def func(params, **kwargs):
        logger.warn(
            f"Using Optimizer from BitsAndBytes: {OptimCls.__name__}.\n"
            + "Likely could be stability issues during training."
        )
        return OptimCls(params=params, **kwargs)

    return func


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types: list[torch.nn.Module]) -> list[str]:
    """
    Get the names of the parameters in the model, excluding the forbidden layer types.

    Args:
        model (torch.nn.Module): The model.
        forbidden_layer_types (list[torch.nn.Module]): The forbidden layer types.

    Returns:
        list[str]: The names of the parameters.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


OPTIM_AVAIL = {
    "sgd": (torch.optim.SGD,),
    "adamw": (torch.optim.AdamW,),
}


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    optimizer_type: str = "adamw",
    use_groups: bool = True,
    eps: float = 1e-8,
    momentum: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Get the optimizer for the model.

    Args:
        model (torch.nn.Module): The model.
        learning_rate (float): The learning rate.
        weight_decay (float): The weight decay.
        betas (tuple[float, float], optional): The betas for AdamW optimizer. Defaults to (0.9, 0.95).
        optimizer_type (str, optional): The type of optimizer. Defaults to "adamw".
        use_groups (bool, optional): Whether to use different groups for different layer types. Its recommended for
            many optims + layer combos e.g. layernorm + adamw. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    # General optimizer kwargs
    shared_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    # Type Specific kwargs
    adam_kwargs = {
        **shared_kwargs,
        "betas": betas,
        "eps": eps,
    }

    sgd_kwargs = {
        **shared_kwargs,
        "momentum": momentum,
    }

    bnb_kwargs = {"optim_bits": 8}

    _optimizers = {
        "sgd": (torch.optim.SGD, sgd_kwargs),
        "adamw": (torch.optim.AdamW, adam_kwargs),
        "adam8bit": (get_bnb_optim("Adam8bit"), {**adam_kwargs, **bnb_kwargs}),
        "adamw8bit": (get_bnb_optim("AdamW8bit"), {**adam_kwargs, **bnb_kwargs}),
    }

    try:
        optimizer_cls, optimizer_kwargs = _optimizers[optimizer_type.lower()]
    except KeyError:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    if use_groups:
        # decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in get_parameter_names(model, [torch.nn.LayerNorm]) if "bias" not in name]

        parameters = [
            {
                "params": (p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad),
                "weight_decay": weight_decay,
            },
            {
                "params": (p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad),
                "weight_decay": 0.0,
            },
        ]

    else:
        parameters = model.parameters()

    return optimizer_cls(params=parameters, **optimizer_kwargs)


def get_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.10,
    num_warmup_steps: int = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the scheduler for the optimizer.

    Args:
        scheduler_type (str): The type of scheduler.
        optimizer (torch.optim.Optimizer): The optimizer.
        num_training_steps (int): The number of training steps.
        warmup_ratio (float, optional): The warmup ratio. Defaults to 0.10.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The scheduler.
    """
    # WARNING: the transformers scheduler seems to be broken.  looking at wandb logs the learning rate does not follow what is expected
    # scheduler= transformers.get_scheduler(
    #     name=scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    # )
    num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-9)
    return scheduler


def show_optim_info(optimizer, scheduler, num_training_steps, num_warmup_steps: int = None, warmup_ratio: float = None):
    # num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    # if num_warmup_steps:
    num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    logger.info(f"[WARMUP STEPS]: {num_warmup_steps}")
    logger.info(f"[TRAIN STEPS]: {num_training_steps}")
    logger.info(f"[SCHEDULER]: {scheduler.__class__.__name__}")
    logger.info(f"[OPTIMIZER]: {optimizer.__class__.__name__}")
