from __future__ import annotations

from typing import Iterable

import torch

from pretrain_mm import logger
from pretrain_mm.datasets.dataloader import Batch
from pretrain_mm.trainer.trainer_events import CallbackHandler, Emit, EventsEnum
from pretrain_mm.utils.config_utils import BaseTrainConfig


def save_helper(model, epoch: int, config: BaseTrainConfig):
    pass


class Trainer(object):
    Events = EventsEnum
    CallbackHandler = CallbackHandler

    def __init__(
        self,
        config: BaseTrainConfig = BaseTrainConfig(),
        callbacks: dict | CallbackHandler = None,
        config_kwargs: dict = {},
    ):
        self.config = self._parse_config(config, **config_kwargs)

        if callbacks:
            self.setup_callbacks(callbacks)
        # handle events

    @property
    def last_lr(self):
        return self.scheduler.get_last_lr()[0]

    def _parse_config(self, config: BaseTrainConfig, **config_kwargs):
        """
        if there are vals to set from the config?
        """

        self.output_dir = config.output_dir
        self.save_every = config.save_every
        self.num_iters = config.num_iters
        self.epochs = config.epochs
        self.grad_accum_steps = config.grad_accum_steps
        self.gradient_clipping = config.gradient_clipping

        for k, v in config_kwargs.items():
            setattr(config, k, v)

        return config

    def _do_grad_accum(self, batch_idx: int, dataloader=None) -> bool:
        dataloader = dataloader or self.train_dataloader

        if batch_idx == 0:  # dont do it for batch 0
            return False
        if (batch_idx % self.config.grad_accum_steps) == 0:
            return True
        if batch_idx == len(dataloader):
            return True
        return False

    def setup_callbacks(self, callbacks: CallbackHandler | dict):
        if isinstance(callbacks, dict):
            callbacks = Trainer.CallbackHandler(callbacks)

        self.callbacks = callbacks
        self.callbacks.trainer = self
        self._emit: EventsEnum = Emit(callback_handler=self.callbacks)

    def setup_helpers(self, callbacks: CallbackHandler = None, **kwargs):
        if callbacks:
            self.setup_callbacks(callbacks)
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def _save_helper(self, epoch: int, **kwargs):
        if self.config.output_dir is None:
            return

        output_path = f"{self.config.output_dir}"
        if self.config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"

        self.model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    def clip_grad(self, model):
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
        self._emit.gradient_clipping_post

    def train_epochs(
        self,
        model: torch.nn.Module = None,
        train_dataloader: Iterable = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        epochs: int = None,
    ):
        model = self.model = model or self.model
        optimizer = self.optimizer = optimizer or self.optimizer
        scheduler = self.scheduler = scheduler or self.scheduler
        train_dataloader = self.train_dataloader = train_dataloader or self.train_dataloader

        epochs = epochs or self.config.epochs

        # possibly any setup/pre testing for run
        self._emit.train_pre

        def reset_epoch(*args):
            model.train()
            epoch_loss, batch_loss, eval_metric = 0, 0, 0
            return epoch_loss, batch_loss, eval_metric

        for epoch in range(epochs):
            self._emit.epoch_pre
            epoch_loss, batch_loss, eval_metric = reset_epoch()

            for batch_idx, batch in enumerate(train_dataloader):
                self._emit.batch_pre(batch_idx=batch_idx)

                if not batch.okay:
                    # rather than resample the batch just skipping
                    continue

                batch.to(model.device)
                outputs = model(**batch)

                loss = outputs.loss / self.config.grad_accum_steps
                loss.backward()

                batch_loss += loss.item()

                self.clip_grad(model)

                if self._do_grad_accum(batch_idx):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    self._emit.grad_accum_post(batch_idx=batch_idx, batch_loss=batch_loss, trainer=self)

                    epoch_loss += batch_loss
                    batch_loss = 0

                self._emit.batch_post(batch_idx=batch_idx, batch_loss=batch_loss, epoch=epoch, trainer=self)

            self._save_helper(epoch, epoch_loss=epoch_loss)

        logger.info("Training Done")

    def train_num_iters(
        self,
        model: torch.nn.Module = None,
        train_dataloader: Iterable = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        num_iters: int = None,
    ):
        model = self.model = model or self.model
        optimizer = self.optimizer = optimizer or self.optimizer
        scheduler = self.scheduler = scheduler or self.scheduler
        train_dataloader = self.train_dataloader = train_dataloader or self.train_dataloader
        num_iters = num_iters or self.config.num_iters

        assert num_iters < len(train_dataloader), "num_iters must be less than the length of the train_dataloader"

        # dont do epochs
        def batch_iter():
            data_iter = iter(train_dataloader)
            for idx in range(num_iters):
                batch = next(data_iter)
                if batch.okay:
                    batch.to(model.device)
                    yield idx, batch

        def do_grad_accum_step(batch_idx: int):
            if batch_idx == 0:  # dont do it for batch 0
                return False
            if (batch_idx % self.config.grad_accum_steps) == 0:
                return True
            if batch_idx == len(train_dataloader):
                return True
            return False

        self._emit.train_pre

        running_loss = 0
        grad_accum_loss = 0

        model.train()
        for batch_idx, batch in batch_iter():
            self._emit.batch_pre(batch_idx=batch_idx)

            outputs = model(**batch)
            breakpoint()
            loss = outputs.loss / self.config.grad_accum_steps
            loss.backward()

            grad_accum_loss += loss.item()

            self.clip_grad(model)

            if self._do_grad_accum(batch_idx):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                self._emit.grad_accum_post(batch_idx=batch_idx, batch_loss=grad_accum_loss)

                running_loss += grad_accum_loss
                grad_accum_loss = 0

            self._emit.batch_post(batch_idx=batch_idx, batch_loss=grad_accum_loss)

        self._emit.train_post
