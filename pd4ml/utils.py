import re
import functools
import hashlib
from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
)
from rich.console import Console
from importlib import import_module


download_progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)

console = Console()


def _check_md5(fpath):
    """ returns md5 checksum for file in fpath """
    hash_md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class verbose_import:
    def __init__(self, package):
        self.package = package

    def __call__(self, cls, *args, **kwargs):
        try:
            import_module(self.package)
            console.print(f":tada: Successfully imported: [bold]{self.package}")

            @functools.wraps(cls)
            def req(*args, **kwargs):
                return cls(*args, **kwargs)

        except ImportError as e:
            raise ImportError(
                f"`{self.package}` is not installed! Try: `pip install '{self.package}'`"
            ) from e
        return req
