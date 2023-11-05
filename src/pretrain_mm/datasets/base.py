from dataclasses import dataclass, asdict, is_dataclass

IGNORE_INDEX = -100


@dataclass
class Sample:
    def __getitem__(self, item: str):
        return self.__dict__[item]

    def to_dict(self):
        if is_dataclass(self):
            return asdict(self)
        raise NotImplementedError(f"to_dict not implemented for {self.__class__.__name__}")
