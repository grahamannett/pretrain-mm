import functools
import math
from dataclasses import dataclass

# from accelerate import Accelerator, FullyShardedDataParallelPlugin
# from accelerate.state import AcceleratorState
import accelerate
import torch
import transformers

from simple_parsing import ArgumentParser
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from config.fuyu import FuyuInfo
from pretrain_mm import logger
from pretrain_mm.datasets import Mind2Web, Mind2WebConfig, Mind2WebTaskProcessor, TaskAdapter, task_mind2web


@dataclass
class TrainConfig:
    model_config = FuyuInfo
    auto_wrap_policy: bool = True
    decoder_layer: torch.nn.Module = model_config.model_extra_info["decoder_layer"]

    # dataset
    dataset_name: str = "mind2web"
    dataset_dir: str = "/bsuhome/gannett/scratch/datasets/mind2web/raw_dump"

    epochs: int = 2
    batch_size: int = 1
    num_workers_dataloader: int = 8

    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    lr: float = 2e-05
    clip_gradients: bool = True
    scheduler_type: str = "cosine"


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
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
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def get_scheduler(accelerator, scheduler_type: str, max_steps: int, optimizer: torch.optim.Optimizer):
    if scheduler_type.lower() in ["steplr", "step_lr"]:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # warmup_steps = get_warmup_steps(max_steps)
    warmup_steps = math.ceil(max_steps * 0.05)

    accelerator.print(f"[WARMUP STEPS]: {warmup_steps}")
    accelerator.print(f"[WARMUP STEPS]: {warmup_steps}")
    accelerator.print(f"[MAX STEPS]: {max_steps}")
    accelerator.print(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def train(epochs, model, dataloader, optimizer, scheduler, train_config, accelerator: accelerate.Accelerator):
    model.train()

    loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(logits, labels):
        b, l, c = logits.shape

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, c)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits.float(), shift_labels)
        return loss

    dataloader, optimizer, scheduler = accelerator.prepare(dataloader, optimizer, scheduler)

    for epoch in range(0, epochs):
        current_epoch = epoch + 1

        progress = logger.progress(disable=accelerator.is_local_main_process == False)
        traj_task = progress.add_task("[cyan]Training Step", total=len(dataloader))

        for step, batch in enumerate(dataloader):
            current_step = step + 1

            batch["input_ids"] = batch["input_ids"].squeeze(0)
            batch["attention_mask"] = batch["attention_mask"].squeeze(0)
            batch["image_patches"] = [b.squeeze(0) for b in batch["image_patches"]]
            batch["image_patches_indices"] = batch["image_patches_indices"].squeeze(0)

            # inputs = {
            #     "input_ids": batch["input_ids"].to(model.device),
            #     "attention_mask": batch["attention_mask"].to(model.device),
            #     # "image_patches": [b.to(model.device) for b in batch["image_patches"]],
            #     # "image_patches_indices": batch["image_patches_indices"].to(model.device),
            # }

            # inputs["labels"] = torch.clone(inputs["input_ids"]).to(model.device)

            # forward
            model_output = model(**batch)
            logits = model_output.logits

            loss = compute_loss(logits, batch["input_ids"])

            # backward
            loss.backward()

            # clipping
            if accelerator.sync_gradients:
                # if train_config.clip_gradients:
                accelerator.clip_grad_norm_(model.parameters(), train_config.gradient_clipping)
                # model.clip_grad_norm_(train_config.gradient_clipping).item()

                # grad_norm = clip_model_gradients(model, gradient_clipping)
            accelerator.backward(loss)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            # loss = loss.detach()

            # avg loss over all processes
            # loss = get_all_reduce_mean(loss).item()

            progress.update(traj_task, advance=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainConfig, dest="train_config")
    args = parser.parse_args()
    train_config: TrainConfig = args.train_config
    model_config = train_config.model_config

    fsdp_plugin = accelerate.FullyShardedDataParallelPlugin(
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                # model_config.model_extra_info["decoder_layer"],
                transformers.models.persimmon.modeling_persimmon.PersimmonDecoderLayer,
            },
        )
        # state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        # optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )

    deepspeed_plugin = accelerate.DeepSpeedPlugin(
        gradient_accumulation_steps=1,
        zero3_init_flag=False,
        zero_stage=3,
        offload_optimizer_device="cpu",
        offload_param_device="cpu",
    )
    accelerator = accelerate.Accelerator(deepspeed_plugin=deepspeed_plugin)
    accelerate.state.AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    accelerate.state.AcceleratorState()

    # accelerator = Accelerator(gradient_accumulation_steps=1, fsdp_plugin=fsdp_plugin)
    # accelerator = Accelerator()
    # AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = "auto"
    config = Mind2WebConfig(
        task_dir=train_config.dataset_dir, subset=10, is_local_main_process=accelerator.is_local_main_process
    )
    dataset = Mind2Web(config)
    # check that task for this dataset is working

    # check that task adapter with processor is working
    processor = FuyuInfo.ProcessorCls.from_pretrained(FuyuInfo.model_name)
    task_dataset = TaskAdapter(
        dataset,
        task_func=task_mind2web,
        processor=processor,
        preprocessor=Mind2WebTaskProcessor.preprocessor,
    )

    if accelerator.is_local_main_process:
        logger.log("done with task dataset")

    accelerator.print("done with sampler creation")

    train_dataloader = torch.utils.data.DataLoader(
        task_dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
    )

    accelerator.print("making model")

    # model = model_config.ModelCls.from_pretrained(model_config.model_name) #,  torch_dtype=torch.float16)
    # model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b", torch_dtype=torch.bfloat16)
    model = transformers.models.fuyu.FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

    model = accelerator.prepare(model)

    accelerator.print("using accelerator")
    max_steps = len(train_dataloader) * train_config.epochs

    optimizer = get_optimizer(model, train_config.lr, train_config.weight_decay)
    scheduler = get_scheduler(accelerator, train_config.scheduler_type, max_steps, optimizer=optimizer)

    # train(train_config.epochs, model=model, dataloader=train_dataloader)
    train(
        epochs=train_config.epochs,
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
        accelerator=accelerator,
    )
