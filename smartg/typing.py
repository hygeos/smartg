#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Sequence, TypeAlias
from os import PathLike
import numpy as np
from numpy.typing import NDArray


RealNumber: TypeAlias = int | float | np.integer | np.floating

NumericArrayLike: TypeAlias = Union[
    RealNumber, Sequence[Union[int, float]], NDArray[np.number]
]

PathType: TypeAlias = str | Path | PathLike[str]
