#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Sequence, TypeAlias
from os import PathLike
import numpy as np


NumericArrayLike: TypeAlias = Union[
    int, float, Sequence[Union[int, float]], np.ndarray
]

PathType: TypeAlias = str | Path | PathLike[str]
