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

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


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
    return f"{path.parent.name}/{path.name}:{lineno}: {category.__name__}: {message}"


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
