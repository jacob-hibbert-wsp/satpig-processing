# -*- coding: utf-8 -*-
"""
    Misc utility functions / classes for carbon_vkms.
"""

##### IMPORTS #####

import collections.abc
import logging
import os
import pathlib
import re
import sys
import time
from typing import Optional

import pydantic
from pydantic import dataclasses

import caf.toolkit as ctk

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


@dataclasses.dataclass
class ScenarioPaths:
    """Parameters for running VKMs on single SATURN scenario year folder.

    Attributes
    ----------
    folder : DirectoryPath
        Folder contain SATPIG H5 files.
    link_data : FilePath
        CSV contain link A, B lookup with zone, speed and distance.
    """

    folder: pydantic.DirectoryPath
    link_data: pydantic.FilePath


class CarbonVKMConfig(ctk.BaseConfig):
    """Parameters for running carbon_vkms package."""

    output_folder: pydantic.DirectoryPath
    scenario_paths: list[ScenarioPaths]
    through_zones_lookup: Optional[pydantic.FilePath] = None
    zone_filter_path: Optional[pydantic.FilePath] = None
    chunk_size: int = 100
    output_folder_name_format: str = "VKMs-{datetime:%Y%m%d}"


def getenv_bool(key: str, default: bool) -> bool:
    """Get environment variable and convert to boolean.

    Raises
    ------
    ValueError
        If value of variable isn't an expected bool value.
    """
    value = os.getenv(key)
    if value is None:
        return default

    value = value.strip()

    true_pattern = re.compile(r"^\s*(yes|true|[yt1])\s*$", re.I)
    false_pattern = re.compile(r"^\s*(no|false|[nf0])\s*$", re.I)

    print(f"Environment variable {key} = '{value}'")

    matched = true_pattern.match(value)
    if matched is not None:
        return True
    matched = false_pattern.match(value)
    if matched is not None:
        return False

    raise ValueError(f'non-boolean value given for environment variable {key} = "{value}"')


def simple_warning_format(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    *args,
    **kwargs,
) -> str:
    del args, kwargs
    path = pathlib.Path(filename)
    return f"{path.parent.name}/{path.name}:{lineno}: {category.__name__}: {message}\n"


def shorten_list(values: collections.abc.Sequence, length: int, fmt_str: str = "{}") -> str:
    """Convert list to comma-separated string for display purposes.

    Any lists longer than `length` will show the start and end values
    but skip over the middle ones. The number of start and end values
    shown is equal to half the `length` (rounded down).

    Examples
    --------
    >>> shorten_list(range(100), 10)
    '0, 1, 2, 3, 4 ... 95, 96, 97, 98, 99'

    >>> shorten_list(range(5), 5)
    '0, 1, 2, 3, 4'

    >>> shorten_list(range(5), 3)
    '0 ... 4'
    """
    if length <= 1:
        raise ValueError(f"length should be a integer > 1 not {length}")

    if len(values) <= length:
        return ", ".join(fmt_str.format(i) for i in values)

    half = length // 2

    if half == 0:
        half = 1

    return (
        ", ".join(fmt_str.format(i) for i in values[:half])
        + " ... "
        + ", ".join(fmt_str.format(i) for i in values[-half:])
    )


def readable_size(size: int | float) -> str:
    """Convert `size` in bytes into string with units.

    Will convert the size to largest units where the
    value is <= 1000 e.g. 1,000,001 bytes -> "1.0MB".
    """
    for prefix in ["", "K", "M", "G", "T", "P"]:
        if size <= 1000:
            return f"{size:.1f}{prefix}B"
        size /= 1000

    return "> 1000PB"


def display_memory_usage(**kwargs) -> str:
    """Format memory usage of all kwargs into str table.

    Output format is sorted by size and looks like:
    keyword_one   : 5.0GB
    keyword_two   : 163.4MB
    keyword_three : 256.1KB
    """
    messages = {}

    length = max(len(i) for i in kwargs)

    for nm, value in kwargs.items():
        try:
            size = sys.getsizeof(value)
            size_str = readable_size(size)
        except Exception as exc:  # pylint: disable=broad-except
            LOG.error("Error getting size of %s - %s", nm, f"{exc.__class__.__name__}: {exc}")
            size = sys.maxsize
            size_str = "error"

        msg = f"{nm:<{length}.{length}} : {size_str}"
        messages[msg] = size

    messages = sorted(messages, key=messages.get, reverse=True)

    return "\n".join(messages)


class Timer:
    """Timer object with human-readable output.

    Keeps track of a start time when the class
    is instantiated.
    """

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def reset(self) -> None:
        """Reset the internal start time."""
        self.start = time.perf_counter()

    def time_taken(self, reset: bool = False) -> str:
        """Produce human-readable string of time taken since start.

        Parameters
        ----------
        reset : bool, default False
            If True reset the start time after calculating time taken.

        Returns
        -------
        str
            Time taken in the following formats:
            - < 60 seconds: "{time_taken} secs" e.g. "57.4 secs"
            - < 60 minutes: "{minutes} mins {seconds} secs" e.g. "5 mins 37 secs"
            - anything else: "{hours}:{minutes}:{seconds}" e.g. "3:30:09"
        """
        time_taken = time.perf_counter() - self.start
        if reset:
            self.reset()

        if time_taken < 60:
            return f"{time_taken:.1f} secs"

        mins, secs = divmod(time_taken, 60)
        if mins < 60:
            return f"{mins:.0f} mins {secs:.0f} secs"

        hours, mins = divmod(mins, 60)
        hours, mins, secs = round(hours), round(mins), round(secs)
        return f"{hours}:{mins!s:>02.2}:{secs!s:>02.2}"
