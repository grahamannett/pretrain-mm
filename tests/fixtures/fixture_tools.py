from dataclasses import dataclass, field


@dataclass
class DataFixture:
    image_urls: dict[str, str | int] = field(default_factory=dict)

    def __getattr__(self, name, *args, **kwargs):
        breakpoint()
