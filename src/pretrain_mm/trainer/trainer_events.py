import inspect
from enum import StrEnum, auto
from functools import lru_cache
from typing import Any

from pretrain_mm import logger
from pretrain_mm.utils.transforms import dummy_func


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

    @lru_cache
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
                cb()
                # try:
                #     cb()
                # except Exception as e:
                #     logger.error(f"Callback {cb} failed with error: {e}. Will try calling after")
                #     call_after.append((cb, cb_spec))

        def _ret_fn(**kwargs):
            # this is called after the argless callbacks
            for cb, cb_spec in call_after:
                _cb_kwargs = {}
                for arg_name in cb_spec.args:
                    if arg_name in kwargs:
                        _cb_kwargs[arg_name] = kwargs[arg_name]
                cb(**_cb_kwargs)

        return _ret_fn

    def add(self, event: EventsEnum | str, fn):
        if event not in EventsEnum.__members__:
            logger.warning_once(f"We dont have {event} in EventsEnum.  Fix This NOW")
            return

        if isinstance(event, str):
            event = EventsEnum[event]

        if event not in self.cb:
            self.cb[event] = []

        self.cb[event].append(fn)

    def __repr__(self):
        str_out = ""
        for key, val in self.cb.items():
            str_out += f"\n\t{key}:"
            for v in val:
                str_out += f"\n\t\t{getattr(v, '__name__', v)}"
        return str_out


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
        # cant decide if this is good or bad to call the events if we change the current event
        # self.callback_handler(now)

    def __getattr__(self, name: str, **kwargs) -> Any:
        if name not in EventsEnum.__members__:
            logger.warning_once(f"We dont have {name} in EventsEnum.  Fix This NOW")
            raise AttributeError(f"{name} not in EventsEnum")
            # return dummy_func

        self.now = EventsEnum[name]
        return self.callback_handler(self.now)
