import atexit
import functools
from enum import StrEnum, auto

import tinydb
from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

import wandb


# Log levels
class LogLevel(StrEnum):
    """The log levels."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __le__(self, other: "LogLevel") -> bool:
        """Compare log levels.

        Args:
            other: The other log level.

        Returns:
            True if the log level is less than or equal to the other log level.
        """
        levels = list(LogLevel)
        return levels.index(self) <= levels.index(other)


# Console for pretty printing.
_console = Console()

# The current log level.
LEVEL = LogLevel.INFO
__RICH_TABLES = {}


def get_console():
    """helper but not sure if it is useful as can just use _console

    Returns:
        _type_: _description_
    """
    return _console


def use_table(**kwargs):
    """Get a table object

    Returns:
        rich.table.Table: rich table
    """

    if _title := kwargs.get("title", None):
        if _title in __RICH_TABLES:
            return __RICH_TABLES[_title]

    _table = Table(**kwargs)
    __RICH_TABLES[_title] = _table
    return _table


def setLEVEL(log_level: LogLevel):
    """Set the log level.

    Args:
        log_level: The log level to set.
    """
    global LEVEL
    LEVEL = log_level


def print(msg: str, _stack_offset: int = 2, **kwargs):
    """Print a message.

    Args:
        msg: The message to print.
        kwargs: Keyword arguments to pass to the print function.
    """
    _console.log(msg, _stack_offset=_stack_offset, **kwargs)


def debug(msg: str, **kwargs):
    """Print a debug message.

    Args:
        msg: The debug message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.DEBUG:
        print(f"[blue]Debug: {msg}[/blue]", **kwargs)


def info(msg: str, _stack_offset: int = 3, **kwargs):
    """Print an info message.

    Args:
        msg: The info message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        print(f"[cyan]Info: {msg}[/cyan]", _stack_offset=_stack_offset, **kwargs)


def success(msg: str, **kwargs):
    """Print a success message.

    Args:
        msg: The success message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        print(f"[green]Success: {msg}[/green]", **kwargs)


def log(msg: str, _stack_offset: int = 2, **kwargs):
    """Takes a string and logs it to the console.

    Args:
        msg: The message to log.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.INFO:
        _console.log(msg, _stack_offset=_stack_offset, **kwargs)


def rule(**kwargs):
    """Prints a horizontal rule with a title.

    Args:
        title: The title of the rule.
        kwargs: Keyword arguments to pass to the print function.
    """
    _console.rule(**kwargs)


def warn(msg: str, _stack_offset: int = 3, **kwargs):
    """Print a warning message.

    Args:
        msg: The warning message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.WARNING:
        print(f"[orange1]Warning: {msg}[/orange1]", _stack_offset=_stack_offset, **kwargs)


@functools.cache
def warning_once(msg: str):
    """Print a warning message once.

    from hf - transformers.
    seems like a useful function to have.

    Args:
        msg: The warning message.
        kwargs: Keyword arguments to pass to the print function.
    """
    warn(msg)


def deprecate(
    feature_name: str,
    reason: str,
    deprecation_version: str,
    removal_version: str,
    **kwargs,
):
    """Print a deprecation warning.

    Args:
        feature_name: The feature to deprecate.
        reason: The reason for deprecation.
        deprecation_version: The version the feature was deprecated
        removal_version: The version the deprecated feature will be removed.
        kwargs: Keyword arguments to pass to the print function.
    """
    msg = (
        f"{feature_name} has been deprecated in version {deprecation_version} {reason.rstrip('.')}. It will be completely "
        f"removed in {removal_version}"
    )
    if LEVEL <= LogLevel.WARNING:
        print(f"[yellow]DeprecationWarning: {msg}[/yellow]", **kwargs)


def error(msg: str, _stack_offset: int = 3, **kwargs):
    """Print an error message.

    Args:
        msg: The error message.
        kwargs: Keyword arguments to pass to the print function.
    """
    if LEVEL <= LogLevel.ERROR:
        print(f"[red]{msg}[/red]", _stack_offset=_stack_offset, **kwargs)


def check_or_fail(condition: bool, msg: str = "Check failed", _stack_offset: int = 3, **kwargs):
    """Check a condition and print an error message if it is False.

    Args:
        condition: The condition to check.
        msg: The error message.
        _stack_offset: The stack offset. 4 is probably the right value.
        kwargs: Keyword arguments to pass to the print function.
    """
    if not condition:
        error(msg, _stack_offset=_stack_offset, **kwargs)
        rule()
        raise AssertionError(msg)


def ask(prompt: str, choices: list[str] = None, default: str = None) -> str:
    """Takes a prompt prompt and optionally a list of choices
     and returns the user input.

    Args:
        prompt: The prompt to ask the user.
        choices: A list of choices to select from.
        default: The default option selected.

    Returns:
        A string with the user input.
    """
    return Prompt.ask(prompt, choices=choices, default=default)  # type: ignore


def progress(ensure_exit: bool = False, start: bool = False, time_remaining: bool = False, **kwargs):
    """Create a new progress bar.

    ensure_exit allows for CTRL+C to clean exit and not mess up the terminal cursor

    Default Columns are: TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

    Returns:
        A new progress bar.
    """
    _default_columns = Progress.get_default_columns()

    if not time_remaining:
        _default_columns = _default_columns[:-1]

    pbar = Progress(
        *_default_columns,
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        **kwargs,
    )

    if ensure_exit:
        atexit.register(lambda: ensure_progress_exit(pbar))

    if start:
        pbar.start()

    return pbar


def status(*args, **kwargs):
    """Create a status with a spinner.

    Args:
        *args: Args to pass to the status.
        **kwargs: Kwargs to pass to the status.

    Returns:
        A new status.
    """
    return _console.status(*args, **kwargs)


def ensure_progress_exit(progress: Progress) -> None:
    try:
        progress.stop()
    except Exception as err:
        warn(f"Error ensuring progress exits cleanly. Shell cursor may not display. Error: {err}")


class LogTool:
    """
    # tools related to
    save and log data via
    - wandb (external)
    and
    - tinydb (internal)
    """

    _wandb_run = None
    _tinydb = None

    def __init__(self, disable: bool = False):
        self.disable = disable

    def setup_wandb(self, wandb_config=None, config=None) -> wandb.sdk.wandb_run.Run:
        self._wandb_run = wandb.init(
            config=config,
            project=wandb_config.project,
            group=wandb_config.group,
            job_type=wandb_config.job_type,
            mode=wandb_config.mode,
        )

        return self._wandb_run

    def setup_local_data(
        self,
        local_data_config=None,
        config=None,
        create_dirs: bool = True,
        enforce_json_path: bool = True,
    ) -> tinydb.TinyDB:
        if local_data_config is None or local_data_config.enabled is False:
            return

        if not (path := local_data_config.path).endswith("json") and enforce_json_path:
            path = f"{path}.json"
            info(f"Enforcing local data path to end with .json: {path}")

        self._tinydb = tinydb.TinyDB(path=path, create_dirs=create_dirs)
        self._tinydb.table("config").insert(config.to_dict())
        return self._tinydb

    def check_train_config(self, train_config):
        info(f"Running Train. Config:\n{train_config.dumps_yaml()}")

        info(f"Model Config:\n{train_config.model_config.dumps_yaml()}")

        if train_config.output_dir is None:
            output_dir_warn = "`train_config.output_dir` is None"
            output_dir_warn += "\nthis will not save model and if you are doing real train you should exit now"
            warn(output_dir_warn)

    def log_data(self, *args, **kwargs):
        if self.disable:
            return

        if self._wandb_run != None:
            self._wandb_run.log(*args, **kwargs)

        if self._tinydb != None:
            self._tinydb.table("data").insert(*args, **kwargs)


tools = LogTool()


# so it is slightly easier to log data, allow for direct access to the log_data function
def log_data(*args, **kwargs):
    tools.log_data(*args, **kwargs)
