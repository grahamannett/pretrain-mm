from __future__ import annotations

import functools
import inspect
from enum import StrEnum, auto
from typing import Iterable

import torch

from pretrain_mm import logger
from pretrain_mm.datasets.dataloader import Batch
from pretrain_mm.utils.config_utils import BaseTrainConfig
from pretrain_mm.datasets.utils.transforms import dummy_func


def bad_batch(batch: Batch):
    if not batch.is_valid:
        return True
    return False


def save_helper(model, epoch: int, config: BaseTrainConfig):
    pass


class EventsEnum(StrEnum):
    epoch_pre = auto()
    epoch_post = auto()
    #
    train_pre = auto()
    train_post = auto()
    #
    batch_pre = auto()
    batch_post = auto()
    #
    eval_pre = auto()
    eval_post = auto()
    #
    gradient_clipping_pre = auto()
    gradient_clipping_post = auto()

    # gradient_accumulation
    grad_accum_pre = auto()
    grad_accum_post = auto()

    # if error occurs, not sure how i can integrate this best though
    callback_error = auto()


class CallbackHandler:
    """

    the way this is used is like
    def _show_train_pre():
        logger.log(f"show that we started training with `{len(train_dl)}` batches")


    def _show_train_post_needs_args(val1: str, optional_val: int = 10):
        logger.log(f"showing how you would need to do this one! {val1} and {optional_val}")

    callbacks = Trainer.CallbackHandler(
        {
            Trainer.Events.train_pre: (_show_train_pre, _show_train_post_needs_args),
        }
    )

    """

    def __init__(self, callbacks: dict):
        self.cb = callbacks
        self.trainer = None

    @functools.lru_cache
    def _get_spec(self, cb: callable):
        return inspect.getfullargspec(cb)

    def __call__(self, name: str):
        call_after = []

        if _cbs := self.cb.get(name, []):
            if not isinstance(_cbs, (list, tuple)):
                _cbs = [_cbs]

            for cb in _cbs:
                cb_spec = self._get_spec(cb)

                # if the callback has args, means use it in the closure afterwords
                if cb_spec.args:
                    call_after.append((cb, cb_spec))
                    continue

                try:
                    cb()
                except Exception as e:
                    logger.error(f"Callback {cb} failed with error: {e}. Will try calling after")
                    call_after.append((cb, cb_spec))

        def _ret_fn(**kwargs):
            # this is called after the argless callbacks
            for cb, cb_spec in call_after:
                _cb_kwargs = {}
                for arg_name in cb_spec.args:
                    if arg_name in kwargs:
                        _cb_kwargs[arg_name] = kwargs[arg_name]
                cb(**_cb_kwargs)

        return _ret_fn


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
        # self.callback_handler(now)

    def __getattr__(self, name: str, **kwargs) -> torch.Any:
        if name not in EventsEnum.__members__:
            logger.warning_once(f"We dont have {name} in EventsEnum.  Fix This NOW")
            return dummy_func

        self.now = EventsEnum[name]
        return self.callback_handler(self.now)


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

        # self.callbacks = Trainer.CallbackHandler(callbacks) if isinstance(callbacks, dict) else callbacks
        # self.callbacks.trainer = self
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

        # possibly any setup/pre testing for run
        self._emit.train_pre

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

        for epoch in range(epochs):
            self._emit.epoch_pre
            epoch_loss, batch_loss, eval_metric = reset_epoch()

            for batch_idx, batch in enumerate(train_dataloader):
                self._emit.batch_pre(batch_idx=batch_idx)

                if bad_batch(batch):
                    # rather than resample the batch just skipping
                    continue

                batch.to(model.device)
                outputs = model(**batch)

                loss = outputs.loss / self.config.grad_accum_steps
                loss.backward()

                batch_loss += loss.item()

                self.clip_grad(model)

                if do_grad_accum_step(batch_idx):
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
            # data_iter = iter(train_dataloader)
            data_iter = iter(train_dataloader)
            for idx in range(num_iters):
                batch = next(data_iter)
                breakpoint()
                if bad_batch(batch):
                    # rather than resample the batch just skipping
                    # if bad_batch(batch := next(data_iter)):
                    continue
                else:
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
        # batch_iter = _batch_iter()
        # for batch_idx, batch in enumerate(batch_iter):
        for batch_idx, batch in batch_iter():
            self._emit.batch_pre(batch_idx=batch_idx)
            breakpoint()
            outputs = model(**batch)

            loss = outputs.loss / self.config.grad_accum_steps
            loss.backward()

            grad_accum_loss += loss.item()

            self.clip_grad(model)

            if do_grad_accum_step(batch_idx):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                self._emit.grad_accum_post(batch_idx=batch_idx, batch_loss=grad_accum_loss)

                running_loss += grad_accum_loss
                grad_accum_loss = 0

            self._emit.batch_post(batch_idx=batch_idx, batch_loss=grad_accum_loss)
