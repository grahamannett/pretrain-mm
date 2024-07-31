IGNORE_INDEX: int = -100

# helpful for tensors
NEG_INF = -float("inf")

# default fuyu seemed to be 1920x1080
# (width, height)
# VIEWPORT_SIZE: tuple[int, int] =
VIEWPORT_SIZE: tuple[int, int] = (1280, 1080)  # alternatively:  (1920, 1080), (1290, 1080)

VIEWPORT_SIZE_DICT: dict[str, int] = {
    "width": VIEWPORT_SIZE[0],
    "height": VIEWPORT_SIZE[1],
}
