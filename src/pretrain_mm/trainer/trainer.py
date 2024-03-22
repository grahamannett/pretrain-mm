from enum import StrEnum, auto
from typing import Iterable

import torch

from pretrain_mm import logger
from pretrain_mm.datasets.dataloader import Batch
from pretrain_mm.utils.config_utils import BaseTrainConfig


class CallbackHandler:
    def __init__(self, callbacks: dict):
        self.cb = callbacks

    def __call__(self, name: str):
        if _cbs := self.cb.get(name, []):
            if not isinstance(_cbs, list):
                _cbs = [_cbs]

            for cb in self.cb[name]:
                try:
                    cb()
                except Exception as e:
                    logger.error(f"Callback {cb} failed with error: {e}")
                    self.callback_error()


class EventsEnum(StrEnum):
    epoch_start = auto()
    epoch_complete = auto()
    train_start = auto()
    train_complete = auto()
    batch_start = auto()
    batch_complete = auto()
    eval_start = auto()
    eval_complete = auto()
    gradient_clipping = auto()
    callback_error = auto()


class Emit:
    _now: EventsEnum = None

    def __init__(self, callback_handler: CallbackHandler):
        self.callback_handler = callback_handler

    @property
    def now(self):
        return self._now

    @now.setter
    def now(self, now: EventsEnum):
        self._now = now
        self.callback_handler(now)

    def __getattr__(self, name: str, **kwargs) -> torch.Any:
        self.now = EventsEnum[name]
        return self.now


class Trainer(object):
    Events = EventsEnum

    def __init__(
        self,
        config: BaseTrainConfig = BaseTrainConfig(),
        callbacks: dict | CallbackHandler = {},
        config_kwargs: dict = {},
    ):
        self.config = self._parse_config(config, **config_kwargs)

        self.callbacks = CallbackHandler(callbacks) if isinstance(callbacks, dict) else callbacks
        # handle events
        self._EMIT: Emit = Emit(callback_handler=self.callbacks)

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

    def setup_helpers(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def setup_train(self, model=None, optimizer=None, scheduler=None, callbacks=None, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # kwargs are extra items to set, likely for debugging/callbacks
        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f"Attribute {k} already exists on Trainer")
            setattr(self, k, v)

    def _do_callbacks(self, cbs: list[callable], **kwargs):
        for cb in cbs:
            cb(**kwargs)

    def _save_helper(self, epoch: int):
        if self.config.output_dir is None:
            return

        output_path = f"{self.config.output_dir}"
        if self.config.save_every == "epoch":
            output_path += f"/checkpoint_{epoch}"

        self.model.save_pretrained(output_path)
        logger.info(f"model for epoch: {epoch} saved to: {output_path}")

    def eval_batch(self, **kwargs):
        pass

    def train(
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

        Event: EventsEnum = self._EMIT
        Event.train_start(val1="asdf")

        def clip_grad():
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)

        def reset_epoch():
            model.train()
            epoch_loss, batch_loss, eval_metric = 0, 0, 0
            return epoch_loss, batch_loss, eval_metric

        def do_grad_accum_step(batch_idx: int):
            if batch_idx == 0:  # dont do it for batch 0
                return False
            if (batch_idx % self.config.grad_accum_steps) == 0:
                return True
            if batch_idx == len(train_dataloader):
                return True
            return False

        def _do_batch_eval(batch_idx: int):
            if self.config.do_batch_eval_every and ((batch_idx % self.config.do_batch_eval_every) == 0):
                return True
            return False

        def _batch_okay(batch):
            if not batch.is_valid:
                logger.warn("invalid batch")
                return False

        for epoch in range(epochs):
            Event.epoch_start
            epoch_loss, batch_loss, eval_metric = reset_epoch()

            for batch_idx, batch in enumerate(train_dataloader):
                Event.batch_start

                if not _batch_okay(batch):
                    continue

                batch.to(model.device)

                outputs = model(**batch)

                loss = outputs.loss / self.config.grad_accum_steps
                loss.backward()

                batch_loss += loss.item()

                clip_grad()

                if do_grad_accum_step(batch_idx):
                    optimizer.step()
                    scheduler.step()
                    logger.log_data({"train/batch_loss": batch_loss, "learning_rate": self.last_lr})

                    epoch_loss += batch_loss
                    batch_loss = 0

                if _do_batch_eval(batch_idx) and (eval_metric := self.eval_batch(model)):
                    eval_metric = self.eval_batch(model)
                    logger.log_data({"train/batch_eval_metric": eval_metric})

                self._do_callbacks(self.callbacks.post_batch, batch=batch)

            self._save_helper(epoch)
            if self.config.do_eval:
                eval_info = self.eval_epoch(model)
                _eval_info = {k: v for k, v in eval_info.items() if k.startswith("eval/")}
                logger.log_data({"train/epoch_loss": epoch_loss, **_eval_info})

                logger.log(f"E[{epoch}][L:{epoch_loss:.2f}][LR:{self.last_lr:.4f}][Eval:{_eval_info}")

        logger.info("Training Done")
