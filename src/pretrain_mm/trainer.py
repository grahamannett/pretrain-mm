import functools
import math
import os
import random
import uuid
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import transformers
import wandb

from dotenv import load_dotenv

from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.persimmon.modeling_persimmon import PersimmonDecoderLayer


from pretrain_mm.datasets.multipack_sampler import MultipackDistributedBatchSampler
from pretrain_mm.datasets.supervised_dataset import (
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    DataCollatorForSupervisedDataset,
    SupervisedDataset,
)

load_dotenv()


# model_name = "mistralai/Mistral-7B-v0.1"
# MODEL_DECODER_LAYER = MistralDecoderLayer

model_name = "adept/fuyu-8b"
MODEL_DECODER_LAYER = PersimmonDecoderLayer


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def setup_model(model_name, max_length):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=os.environ["HF_TOKEN"],
    )

    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        pad_token=DEFAULT_PAD_TOKEN,
        trust_remote_code=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
):
    if local_rank == 0:
        print("RUNNING EVAL")

    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "labels": batch["labels"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        wandb.log(
            {
                "val_loss": val_loss,
            }
        )

    return val_loss


def get_dataloader(
    use_multipack_sampler,
    max_length,
    dataset,
    world_size,
    local_rank,
    shuffle,
    seed,
    collator,
    batch_size,
):
    if use_multipack_sampler:
        lengths = np.array([len(tokens["input_ids"]) for tokens in dataset])
        sampler = MultipackDistributedBatchSampler(
            batch_max_length=batch_size * max_length,
            lengths=lengths,
            num_replicas=world_size,
            rank=local_rank,
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            pin_memory=True,
            collate_fn=collator,
            batch_sampler=sampler,
        )
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=shuffle,
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            batch_size=batch_size,
            collate_fn=collator,
            sampler=sampler,
        )

    return sampler, loader


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


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)


def clip_model_gradients(model, max_grad_norm):
    return model.clip_grad_norm_(max_grad_norm).item()


def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)

    if local_rank == 0:
        print(f"[WARMUP STEPS]: {warmup_steps}")
        print(f"[MAX STEPS]: {max_steps}")
        print(f"[SCHEDULER]: {scheduler_type}")

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )


def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"/epoch_{current_epoch}/step_{current_step}"
        model.save_pretrained(outpath, state_dict=cpu_state)
        tokenizer.save_pretrained(outpath)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    scheduler_type = "cosine"
    seed = 10  # set your seed
    transformers.set_seed(seed)

    run_id = str(uuid.uuid4())
    output_dir = f"./outputs/{model_name}/{run_id}"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 1524  # adjust as needed
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 1  # adjust as needed
    validation_batch_size = 1  # adjust as needed
    epochs = 3  # adjust as needed
    acc_steps = 0  # TODO: not implemented here yet
    lr = 2e-05  # adjust as needed
    weight_decay = 0.0  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    use_multipack_sampler = True  # whether to use the multipack sampler or torch sampler

    model, tokenizer = setup_model(model_name, max_length)
    # num_params = sum([p.numel() for p in model.parameters()])
    # auto_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={
    #         MODEL_DECODER_LAYER,
    #     },
    # )

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, # or
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # or
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        # cpu_offload=None, # or CPUOffload(True)
        # cpu_offload=CPUOffload(True),
        # limit_all_gathers=True,
        # sync_module_states=True,
    )

    model = FSDP(model, **fsdp_config)

    for layer in model.language_model.model.layers.parameters():
        layer.requires_grad = False

    optimizer = get_optimizer(model, lr, weight_decay)
    # optimizer = get_optimizer(model.language_model.lm_head, lr, weight_decay)

    train_ds = ["data/train.jsonl"]
    val_ds = ["data/validation.jsonl"]
    train_dataset = SupervisedDataset(train_on_inputs, tokenizer, train_ds)
    val_dataset = SupervisedDataset(train_on_inputs, tokenizer, val_ds)
    collator = DataCollatorForSupervisedDataset(tokenizer)

    train_sampler, train_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        val_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        validation_batch_size,
    )

    if use_multipack_sampler:
        total_steps_per_epoch = train_sampler.num_batches()
    else:
        total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="mistral-7b",
            name=run_id,
            mode="disabled",
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": ",".join(train_ds),
                "validation": ",".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "use_multipack_sampler": use_multipack_sampler,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": acc_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    dist.barrier()

    loss_fct = torch.nn.functional.nll_loss

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
        # Flatten the tokens - using FSDP not clear to me how you calculate this since using similar to mistral fails
        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]).to(model.device)
        labels = labels.reshape(labels.shape[0] * labels.shape[1]).to(model.device)

        # logits = logits.view(-1, model.config.vocab_size)
        # labels = labels.view(-1)
        # Enable model parallelism
        # labels = labels.to(logits.device)
        loss = loss_fct()
        return loss

    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_loader),
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )

        for step, batch in pbar:
            current_step = step + 1

            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # forward
            # if local_rank == 0:
            #     print("DOING FORWARD")
            # outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            outputs = model(input_ids=inputs["input_ids"])

            # reshape
            logits = outputs.logits[..., :-1, :]
            logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])

            labels = inputs["labels"][..., 1:]
            labels = labels.reshape(labels.shape[0] * labels.shape[1])

            loss = torch.nn.functional.nll_loss(logits, labels)

            # backward
            loss.backward()

            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    loss,
                    grad_norm,
                    scheduler,
                )

            # runs eval 2x an epoch, adjust as needed
            if should_run_eval(total_steps_per_epoch, 2, current_step):
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                )

                # saves model 2x an epoch, adjust as needed above
                save_model(
                    local_rank,
                    model,
                    tokenizer,
                    output_dir,
                    current_epoch,
                    current_step,
                )

                model.train()

    # save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final")
