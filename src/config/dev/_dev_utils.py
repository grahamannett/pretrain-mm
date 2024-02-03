import os

from pretrain_mm import logger


def _load_dev_eng():
    from .dev_eng import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


def _load_dev_local():
    from .dev_local import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


def _load_dev_borah():
    from .dev_borah import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


_dev_configs = {
    "eng": _load_dev_eng,
    "local": _load_dev_local,
    "borah": _load_dev_borah,
    "gpu1": _load_dev_borah,
}


def _get_dev_config_from_hostname() -> str:

    _hostname = os.uname().nodename.lower()
    for key in list(_dev_configs.keys()):
        if key in _hostname:
            logger.warn(
                f"GOT `dev_config`: {key} FROM HOSTNAME: `{_hostname}`.\n"
                + f"\tPREFER TO SET `dev_config` AS ENVVAR OR INIT CONFIG DIRECTLY."
            )
            return key
    else:
        raise ValueError(
            f"Cannot get `dev_config` from hostname: {_hostname}. Either set `dev_config` or use a valid hostname"
        )


def get_dev_config(*args):
    dev_config = os.environ.get("dev_config")

    if dev_config is None:
        dev_config = _get_dev_config_from_hostname()

    try:
        config = _dev_configs[dev_config]()
    except KeyError:
        raise ValueError(f"Unknown `dev_config`: {dev_config}, if using get_dev_config, must be set")

    config = [config[arg] for arg in args] if args else config

    if len(config) == 1 and isinstance(config, list):
        config = config[0]

    return config
