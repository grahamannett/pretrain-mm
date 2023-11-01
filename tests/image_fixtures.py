from dataclasses import dataclass


@dataclass
class Screenshot:
    c: int = None
    h: int = None
    w: int = None

    def __iter__(self):
        return iter([self.c, self.h, self.w])


#
mac_screenshot = Screenshot(c=3, w=1542, h=1372)  # png mac is 4 with alpha channel
