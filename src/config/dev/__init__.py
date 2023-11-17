# # get hostname
# import os

# dev_config = os.environ.get("dev_config", "eng")


# def load_eng():
#     from .dev_eng import Mind2WebDatasetInfo

# def load_local():
#     from .dev_local import Mind2WebDatasetInfo

# {
#     "eng": load_eng,
# }
from ._dev_utils import get_dev_config
