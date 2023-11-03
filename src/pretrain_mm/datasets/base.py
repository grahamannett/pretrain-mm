from dataclasses import asdict, is_dataclass

IGNORE_INDEX = -100


class Sample:
    def to_dict(self):
        if is_dataclass(self):
            return asdict(self)
        raise NotImplementedError(f"to_dict not implemented for {self.__class__.__name__}")
