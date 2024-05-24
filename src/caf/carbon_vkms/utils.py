# -*- coding: utf-8 -*-
"""
    Misc utility functions / classes for carbon_vkms.
"""

##### IMPORTS #####

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

    raise ValueError(
        f'non-boolean value given for environment variable {key} = "{value}"'
    )


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
