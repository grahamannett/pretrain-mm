import torch
import transformers

from pretrain_mm import logger


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types: list[torch.nn.Module]):
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
    model: torch.nn.Module, learning_rate: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)
) -> torch.optim.Optimizer:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
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
    scheduler_type: str, optimizer: torch.optim.Optimizer, num_training_steps: int, warmup_ratio: float = 0.1
) -> torch.optim.lr_scheduler.LRScheduler:
    # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    num_warmup_steps = round(num_training_steps * warmup_ratio)

    # logger.info(f"[MAX STEPS]: {num_training_steps}")
    logger.info(f"[WARMUP STEPS]: {num_warmup_steps}")
    logger.info(f"[TRAIN STEPS]: {num_training_steps}")
    logger.info(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
