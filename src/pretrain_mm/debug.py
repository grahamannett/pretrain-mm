from os import environ
from distutils.util import strtobool

from pretrain_mm import logger

DEBUG = bool(strtobool(environ.get("DEBUG", "False")))

if DEBUG:
    logger.warn("DEBUG MODE ON")
