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


_VERSION_TMPL = r"^(?P<major>{v})" r"\.(?P<minor>{v})" r"\.(?P<patch>{v})$"
_VERSION_WILDCARD_REG = re.compile(_VERSION_TMPL.format(v=r"\d+|\*"))
_VERSION_RESOLVED_REG = re.compile(_VERSION_TMPL.format(v=r"\d+"))


def _str_to_version(version_str: str, allow_wildcard: bool = False):
    """
    Return the tuple (major, minor, patch) version extracted from the str.
    """
    reg = _VERSION_WILDCARD_REG if allow_wildcard else _VERSION_RESOLVED_REG
    res = reg.match(version_str)
    if not res:
        msg = f"Invalid version '{version_str}'. Format should be x.y.z"
        if allow_wildcard:
            msg += " with {x,y,z} being digits or wildcard."
        else:
            msg += " with {x,y,z} being digits."
        raise ValueError(msg)
    return tuple(
        v if v == "*" else int(v)
        for v in [res.group("major"), res.group("minor"), res.group("patch")]
    )


class Version:
    def __init__(self, version: str):
        self.major, self.minor, self.patch = _str_to_version(version)

    @property
    def tuple(self):
        return self.major, self.minor, self.patch

    def __str__(self) -> str:
        return "{}.{}.{}".format(*self.tuple)

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{str(self)}')"

    def _validate(self, other):
        if isinstance(other, str):
            return Version(other)
        elif isinstance(other, Version):
            return other
        raise AssertionError(f"{other} (type {type(other)}) cannot be compared to version.")

    def __eq__(self, other):
        other = self._validate(other)
        return self.tuple == other.tuple

    def __ne__(self, other):
        other = self._validate(other)
        return self.tuple != other.tuple

    def __lt__(self, other):
        other = self._validate(other)
        return self.tuple < other.tuple

    def __le__(self, other):
        other = self._validate(other)
        return self.tuple <= other.tuple

    def __gt__(self, other):
        other = self._validate(other)
        return self.tuple > other.tuple

    def __ge__(self, other):
        other = self._validate(other)
        return self.tuple >= other.tuple


def require_software(**kwargs):
    """
    @require_software(
        notexisting="1.1.0",
        tensorflow="2.15.0",
        numpy="1.1.0",
        matplotlib="2.0.0",
        force_env=True,
    )
    class CNN:
        pass
    """

    def decorator(cls):
        force_env = kwargs.pop("force_env", False)
        corrupted_env = []
        console.print(f"[bold green]Checking software requirements for {cls.__name__}...\n")
        for package, version in kwargs.items():
            assert isinstance(package, str)
            assert isinstance(version, str)
            required = Version(version)
            try:
                pkg = import_module(package)
            except ImportError:
                console.print(
                    f"\t:cross_mark: {package} - {required} is not installed! "
                    f"Install package: `pip install '{package}=={version}'`"
                )
                corrupted_env.append(package)
                continue
            current = Version(pkg.__version__)
            if current < required:
                console.print(
                    f"\t:cross_mark: {package} - {current} (required: {required}) is outdated! "
                    f"Try upgrading: `pip install '{package}>={version}'`"
                )
                corrupted_env.append(package)
            else:
                console.print(
                    f"\t:white_heavy_check_mark: {package} - {current} (required: {required}) is installed"
                )
        if corrupted_env and force_env:
            raise Exception(
                f"Environment is not properly set up ({len(corrupted_env)} packages are not installed or outdated). "
                "Please install the required packages!"
            )

        @functools.wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return wrapper

    return decorator
