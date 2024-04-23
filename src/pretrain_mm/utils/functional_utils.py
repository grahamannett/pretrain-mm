from functools import wraps


def wpartial(func, /, *args, **keywords):
    # this is a partial implementation of functools.partial
    # but added wraps as its helpful for printing info about callbacks
    @wraps(func)
    def wrapped_fn(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)

    wrapped_fn.func = func
    wrapped_fn.args = args
    wrapped_fn.keywords = keywords
    return wrapped_fn
