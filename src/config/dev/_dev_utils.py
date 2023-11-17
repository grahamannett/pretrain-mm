import os


def _load_dev_eng():
    from .dev_eng import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


def _load_dev_local():
    from .dev_local import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


def _load_dev_borah():
    from .dev_borah import Mind2WebDatasetInfo

    return {"mind2web": Mind2WebDatasetInfo}


def get_dev_config(*args):
    dev_config = os.environ.get("dev_config")
    try:
        config = {
            "eng": _load_dev_eng,
            "local": _load_dev_local,
            "borah": _load_dev_borah,
        }[dev_config]()
    except KeyError:
        raise ValueError(f"Unknown `dev_config`: {dev_config}, if using get_dev_config, must be set")

    config = [config[arg] for arg in args] if args else config

    if len(config) == 1 and isinstance(config, list):
        config = config[0]

    return config
