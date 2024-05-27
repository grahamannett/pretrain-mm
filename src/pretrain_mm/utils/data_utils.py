from dataclasses import dataclass


def get_fields(d: dict) -> tuple:
    """
    Returns a tuple of (key, value_type) pairs for each key-value pair in the input dictionary.

    Args:
        d (dict): The input dictionary.

    Returns:
        tuple: A tuple of (key, value_type) pairs.

    """
    return tuple((key, type(val)) for key, val in d.items())


class DTObject:
    """
    Data Transfer Object (DTObject) class.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(cls)
