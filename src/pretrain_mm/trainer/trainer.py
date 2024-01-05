from dataclasses import asdict, is_dataclass

import os
import torch

from pretrain_mm import logger
from pretrain_mm.datasets.dataloader import Batch
from pretrain_mm.utils import lora_utils
from pretrain_mm.utils.config_utils import BaseTrainConfig
from pretrain_mm.model.fuyu.embed_fuyu import get_embeddings


class CallbackHandler:
    def __init__(self, callbacks: dict):
        self.cb = callbacks

    def __getattr__(self, item):
        return self.cb.get(item, [])


class Trainer(object):
    def __init__(self, config: BaseTrainConfig = BaseTrainConfig(), callbacks: dict = {}):
        self.config = self._parse_config(config)
        self.callbacks = CallbackHandler(callbacks)

    def _parse_config(self, config: BaseTrainConfig):
        self.output_dir = config.output_dir
        self.save_every = config.save_every
        self.num_iters = config.num_iters
        self.epochs = config.epochs
        self.grad_accum_steps = config.grad_accum_steps
        self.gradient_clipping = config.gradient_clipping
        return config

        # if is_dataclass(config):
        #     return self._parse_config(asdict(config))
        # for key, val in config.items():
        #     setattr(self, key, val)

    @property
    def last_lr(self):
        return self.scheduler.get_last_lr()[0]

    def save_model(self, epoch: int = None):
        if self.output_dir is None:
            return

        output_path = f"{self.output_dir}"
        if self.save_every == "epoch":
            output_path += f"/epoch_{epoch}"

        self.model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    def train_step(self, model, batch: Batch):
        batch.to(model.device)
        outputs = model(**batch)

        loss = outputs.loss / self.grad_accum_steps
        loss.backward()

        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)

        return loss.item()

    def post_train_step(self, log_fn: callable = None):
        if log_fn:
            log_fn()

    def setup_train(self, model=None, optimizer=None, scheduler=None, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _do_callbacks(self, cbs: list[callable], **kwargs):
        for cb in cbs:
            cb(**kwargs)

    def train(
        self,
        train_dataloader,
        model: torch.nn.Module = None,
        optimizer=None,
        scheduler=None,
        post_train_step_log_fn: callable = None,
    ):
        model = model or self.model
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler

        def do_grad_accum_step(batch_idx: int) -> bool:
            # handle this first
            if batch_idx == 0:
                return False  # dont do it for batch 0
            if (
                (batch_idx % self.grad_accum_steps == 0)
                or (batch_idx == self.num_iters)
                or (batch_idx == len(train_dataloader) - 1)
            ):
                return True
            return False

        for epoch in range(self.epochs):
            # setup for train/batch loop
            self.batch_loss, self.epoch_loss = 0, 0
            model.train()

            for batch_idx, batch in enumerate(train_dataloader):
                self.batch_loss += self.train_step(model=model, batch=batch)

                if do_grad_accum_step(batch_idx):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.epoch_loss += self.batch_loss
                    self.batch_loss = 0

                self._do_callbacks(self.callbacks.train_step, model=model, batch_idx=batch_idx, trainer=self)

                if self.num_iters and (self.num_iters < batch_idx):
                    break

            self._do_callbacks(self.callbacks.train_epoch, model=model, epoch=epoch, trainer=self)
            self.save_model(epoch=epoch)


class LoraDPOTrainer(Trainer):
    def setup_train(
        self,
        lora_config: lora_utils.BaseLoraConfig,
        model=None,
        optimizer=None,
        scheduler=None,
        callbacks=None,
        **kwargs,
    ):
        super().setup_train(model, optimizer, scheduler, callbacks)
        self.model, self.lora_adapter = lora_utils.setup_lora(model, lora_config=lora_config)

    def _get_batch_logps(self, all_logits: torch.Tensor, all_labels: torch.Tensor, ignore_index: int = -100):
        bs = all_logits.shape[0]

        logits = all_logits[:, :-1]
        labels = all_labels[:, 1:]

        # labels[labels == -100] = 0
        mask = labels != ignore_index

        # calculate and then mask
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=-1,
            index=labels.unsqueeze(2),
        )

        per_token_logps = per_token_logps.squeeze(-1)
        per_token_logps = per_token_logps[mask]
        return per_token_logps

    def _dpo_loss(self, policy_logps: torch.Tensor, ref_logps: torch.Tensor):
        logits = policy_logps - ref_logps
        loss = -torch.nn.functional.logsigmoid(logits)

    def train_step(self, model: torch.nn.Module, batch: Batch):
        batch.to(model.device)

        # compute
        model.enable_adapters()
        policy_outputs = model(**batch)

        policy_logps = self._get_batch_logps(all_logits=policy_outputs.logits, all_labels=batch.labels)

        with torch.no_grad():
            ref_outputs = model(**batch)
            ref_logps = self._get_batch_logps(all_logits=ref_outputs.logits, all_labels=batch.labels)

        dpo_loss = self._dpo_loss(policy_logps=policy_logps, ref_logps=ref_logps)
        breakpoint()

        loss.backward()

        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)

        return loss.item()

    def compare_policy(self, output1, output2, *args, **kwargs):
        output1_embeds = get_embeddings(
            model,
            output1,
        )

    def _generate_step(self, model):
        pass

    def _example_approval_method(self, model, batch):
        """
        thinking through how i would do the approval/rejection for policy generation
        """

        generated_output1 = self.generate_output(model, batch)
        generated_output2 = self.generate_output(model, batch)

        # compare outputs via embedding or something
        approved, rejected = self.compare_policy(generated_output1, generated_output2)


class DDPTrainer(Trainer):
    def __init__(self, config, callbacks: dict = {}):
        super().__init__(config, callbacks)
        self.rank = int(os.environ.get("LOCAL_RANK", None))
        self.world_size = int(os.environ.get("WORLD_SIZE", None))

    def setup_train(self, model=None, optimizer=None, scheduler=None, callbacks=None):
        super().setup_train(model, optimizer, scheduler, callbacks)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters=False)

        # TODO turn on gradient checkpointing
        model._set_static_graph()
