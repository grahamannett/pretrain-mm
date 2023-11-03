from dataclasses import asdict, is_dataclass


class Sample:
    def to_dict(self):
        if is_dataclass(self):
            return asdict(self)
