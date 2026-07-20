

"""Shared type aliases and structural protocols for SMART-G.

This module centralizes common numeric, path, and spectral-band typing
used throughout SMART-G. Runtime imports are kept minimal to avoid
circular dependencies between the typing definitions and atmospheric
profile implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Union,
    Sequence,
    TypeAlias,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)
from os import PathLike
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports
    # (atmosphere.py imports from this module at runtime).
    from smartg.atmosphere import ProfileBase


RealNumber: TypeAlias = int | float | np.integer | np.floating

NumericArrayLike: TypeAlias = Union[
    RealNumber, Sequence[Union[int, float]], NDArray[np.number]
]

PathType: TypeAlias = str | Path | PathLike[str]


@runtime_checkable
class BandLike(Protocol):
    """Structural type for a KDIS or RepTran spectral band object.

    Any object exposing a ``w`` wavelength attribute and a
    ``calc_profile`` method is considered a ``BandLike``. This covers
    both :class:`smartg.kdis.KdisIband` and
    :class:`smartg.reptran.ReptranIband` without importing them (which
    would create a circular import via ``atmosphere.py``).

    The ``@runtime_checkable`` decorator enables ``isinstance`` checks
    against this protocol at runtime.
    """

    w: float

    def calc_profile(self, prof: ProfileBase) -> np.ndarray:
        """Compute the absorption optical depth profile over ``prof.z``.
        """
        ...
