import shutil
from pathlib import Path

from pretrain_mm import logger


def _rm(file: Path, dry_run: bool = False) -> None:
    def _rm_dir():
        shutil.rmtree(file)

    def _rm_file():
        file.unlink()

    def _log_line(start: str):
        return logger.warn(f"{start}<{filetype}>: {file}")

    filetype = {True: "dir", False: "file"}[file.is_dir()]
    rm_func = {"dir": _rm_dir, "file": _rm_file}[filetype]

    if dry_run:
        logger.warn(f"Would delete<{filetype}>: {file}")

    else:
        _log_line("Deleting")
        rm_func()

    logger.warn(f"Deleting<{filetype}>: {file}")
    if file.is_dir():
        shutil.rmtree(file)
    else:
        file.unlink()


def clean_output_dir_folder(
    output_dir: str | Path,
    glob_pattern: str = "checkpoint*",
    dry_run: bool = False,
):
    """
    Clean the output directory by removing all folders that match the glob pattern.

    Args:
        output_dir (str): The path to the output directory.
        glob_pattern (str, optional): The glob pattern to match files to remove. Defaults to "*".
    """
    if (glob_pattern == "*") or (".." in glob_pattern):
        raise ValueError("Not allowed to use '*' or '..' in glob_pattern. Specify checkpoint* or similar")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    fdir_list = list(output_dir.glob(glob_pattern))
    logger.warn(f"Deleting from: {output_dir} the following folders: {'\n'.join(map(str, fdir_list))}")

    for fdir in fdir_list:
        _rm(fdir, dry_run=dry_run)


def save_setup_config_processor(config, processor):
    config.save_pretrained(f"{config.output_dir}/model_config")
    processor.save_pretrained(f"{config.output_dir}/processor")


def save_checkpoint(checkpoint_idx, config, model, optimizer=None, scheduler=None, save_template="checkpoint_{}"):
    save_dir = f"{config.output_dir}/{save_template.format(checkpoint_idx)}"
    model.save_pretrained(save_dir)

    if optimizer:
        optimizer.save_pretrained(save_dir)
    if scheduler:
        scheduler.save_pretrained(save_dir)
