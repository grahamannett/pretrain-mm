from typing import Any


# placeholder func, returns nothing
def dummy_func(*args, **kwargs):
    pass


def make_dummy_func(default: Any = None, func: callable = None):
    def dummy_func(*args, **kwargs):
        return default

    if callable(func):

        def dummy_func(*args, **kwargs):
            return func(*args, **kwargs)

    return dummy_func
