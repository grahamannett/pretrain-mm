import json
from functools import lru_cache


@lru_cache(maxsize=1024)
def _read_json(filename: str) -> dict:
    with open(filename) as f_in:
        return json.load(f_in)


def read_json(filename: str, use_cache: bool = True) -> dict:
    # if use_cache:
    func = _read_json if use_cache else _read_json.__wrapped__
    return func(filename)
