import torch
import transformers

from pretrain_mm import logger


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

    try:
        optimizer_cls, optimizer_kwargs = {
            "sgd": (torch.optim.SGD, sgd_kwargs),
            "adamw": (torch.optim.AdamW, adam_kwargs),
        }[optimizer_type.lower()]
    except KeyError:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    if use_groups:
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

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
    num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-9)

    # scheduler= transformers.get_scheduler(
    #     name=scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps,
    # )
    return scheduler


def show_optim_info(optimizer, scheduler, num_training_steps, num_warmup_steps: int = None, warmup_ratio: float = None):
    # num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    # if num_warmup_steps:
    num_warmup_steps = num_warmup_steps or round(num_training_steps * warmup_ratio)
    logger.info(f"[WARMUP STEPS]: {num_warmup_steps}")
    logger.info(f"[TRAIN STEPS]: {num_training_steps}")
    logger.info(f"[SCHEDULER]: {scheduler.__class__.__name__}")
    logger.info(f"[OPTIMIZER]: {optimizer.__class__.__name__}")
