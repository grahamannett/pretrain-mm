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
    if "use_sgd" in kwargs:
        return torch.optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get("momentum", None),
            weight_decay=weight_decay,
        )

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": (p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad),
            "weight_decay": weight_decay,
        },
        {
            "params": (p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad),
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
        weight_decay=weight_decay,
    )


def get_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.10,
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
    num_warmup_steps = round(num_training_steps * warmup_ratio)

    logger.info(f"[WARMUP STEPS]: {num_warmup_steps}")
    logger.info(f"[TRAIN STEPS]: {num_training_steps}")
    logger.info(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
