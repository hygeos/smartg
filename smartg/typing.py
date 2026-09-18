"""Shared type aliases and structural protocols for SMART-G.

This module centralizes common numeric, path, and spectral-band typing
used throughout SMART-G. Runtime imports are kept minimal to avoid
circular dependencies between the typing definitions and atmospheric
profile implementations.

Key Classes
-----------
BandLike
    Structural type for a KDIS or REPTRAN spectral band object.

Key Aliases
-----------
RealNumber, NumericArrayLike, PathType
    Common scalar, array-like, and filesystem-path type aliases.
ThetaLike
    The ``n_theta`` argument of the phase methods: a count of equally
    spaced angles, the angles themselves, or ``'native'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports
    # (atmosphere.py imports from this module at runtime).
    from smartg.atmosphere import ProfileBase


RealNumber: TypeAlias = int | float | np.integer | np.floating

NumericArrayLike: TypeAlias = RealNumber | Sequence[int | float] | NDArray[np.number]

PathType: TypeAlias = str | Path | PathLike[str]

# The ``n_theta`` argument of the phase methods: a number of equally
# spaced scattering angles, the angles themselves in degrees, or the
# string ``'native'`` for the angles the source tables carry.
ThetaLike: TypeAlias = int | str | NumericArrayLike


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
        """Compute the absorption optical depth over ``prof.z``."""
        ...
