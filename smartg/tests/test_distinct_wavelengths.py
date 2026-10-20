"""GPU-free tests of the phase matrices computed once per wavelength.

The internal bands of a KDIS channel share its wavelength. By default
(`wavelength_phase=None`) the builders now compute the phase matrices
once at each distinct wavelength, instead of once per band: the phase
table gets shorter and its indices are renumbered, but every layer at
every wavelength keeps exactly the same matrix, and every other
variable of the profile is unchanged. The reference is the former
default, which an explicit `wavelength_phase` holding every
wavelength still gives.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg.atmosphere import (
    AerOPAC,
    AerUser,
    Atm1D,
    Cloud,
    _follows_calc_wavelengths,
    extract_split,
)
from smartg.bandset import BandSet, _distinct_values
from smartg.truncation import GTTrunc

# internal bands of three channels, as KDIS orders them: not sorted
# here, so that the order of first occurrence matters
WAV = [550.0, 443.0, 550.0, 670.0, 443.0, 550.0]
N_DISTINCT = 3
PFGRID = [100.0, 3.0, 2.0, 0.0]


def _full_axis() -> np.ndarray:
    """Return the calc wavelengths as the former default took them."""
    return np.atleast_1d(BandSet(WAV)[:])


def _assert_same_profile(new: xr.Dataset, old: xr.Dataset) -> None:
    """Check that two profiles agree bitwise, but for their numbering.

    Every variable but the phase table and its indices is compared
    bytewise; the phase matrices are compared through the matrix each
    (wavelength, layer) is given, once per pair of indices.
    """
    layout = ["phase_atm", "iphase_atm", "iphase"]
    assert sorted(map(str, new.variables)) == sorted(map(str, old.variables))
    for name in new.variables:
        if name in layout:
            continue
        a, b = new[name].values, old[name].values
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.tobytes() == b.tobytes(), name
    assert new.attrs == old.attrs
    i_new = new["iphase_atm"].values
    i_old = old["iphase_atm"].values
    assert i_new.dtype == i_old.dtype and i_new.shape == i_old.shape
    pairs = np.unique(np.stack([i_new.ravel(), i_old.ravel()], 1), axis=0)
    p_new, p_old = new["phase_atm"].values, old["phase_atm"].values
    assert p_new.dtype == p_old.dtype
    for i, j in pairs:
        assert p_new[i].tobytes() == p_old[j].tobytes(), (i, j)


def _henyey_greenstein(theta: np.ndarray, g: float = 0.7) -> np.ndarray:
    mu = np.cos(np.radians(theta))
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * mu) ** 1.5


def _aer_user() -> AerUser:
    """Build a user aerosol tabulated at 400 and 700 nm."""
    theta = np.linspace(0.0, 180.0, 181)
    pfn = _henyey_greenstein(theta)
    zeros = np.zeros_like(pfn)
    pha = np.stack([pfn, zeros, pfn, zeros])
    phase = np.stack([pha, 0.9 * pha + 0.1])[None]
    aod = np.array([[0.3, 0.2]])
    ssa = np.array([[0.95, 0.9]])
    return AerUser(aod, ssa, phase, np.array([0.0]),
                   np.array([400.0, 700.0]), theta)


def _cloud(**kwargs: Any) -> Cloud:
    return Cloud("wc", 10.0, 2.0, 3.0, 5.0, 550.0, **kwargs)


def _cloud_matrix() -> xr.DataArray:
    """Return the cloud's own matrix at 550 nm, as a 2-D user matrix."""
    ds = _cloud().ds_mix
    da = ds["phase"].sel(reff=10.0, method="nearest").sel(
        wav=0.55, method="nearest"
    )
    return xr.DataArray(
        da.values,
        dims=["nphamat", "theta_atm"],
        coords={"theta_atm": ds.coords["theta"].values.astype(float)},
    )


Components = tuple[list[AerOPAC], dict[str, Any]]


def _desert() -> Components:
    return [AerOPAC("desert", 0.1, 550.0)], {}


def _truncated_cloud() -> Components:
    comp = [AerOPAC("desert", 0.1, 550.0),
            _cloud(truncation=GTTrunc(None, theta_tr=5.0))]
    return comp, {"pfgrid": PFGRID}


def _user_aerosol() -> Components:
    return [_aer_user()], {}


def _user_matrix() -> Components:
    return [_cloud(phase=_cloud_matrix())], {"pfgrid": PFGRID}


COMPONENTS = {
    "desert": _desert,
    "truncated cloud": _truncated_cloud,
    "user aerosol": _user_aerosol,
    "2-D user matrix": _user_matrix,
}


def test_distinct_values_in_order_of_appearance() -> None:
    """The distinct values come in order of first occurrence."""
    values = np.array(WAV, dtype=np.float32)
    distinct, inverse = _distinct_values(values)
    assert distinct.dtype == np.float32
    np.testing.assert_array_equal(distinct, [550.0, 443.0, 670.0])
    np.testing.assert_array_equal(inverse, [0, 1, 0, 2, 1, 0])
    np.testing.assert_array_equal(distinct[inverse], values)


def test_distinct_values_without_repeats() -> None:
    """Values that do not repeat come back as they are, as a copy."""
    values = np.array([700.0, 400.0, 550.0])
    distinct, inverse = _distinct_values(values)
    np.testing.assert_array_equal(distinct, values)
    np.testing.assert_array_equal(inverse, np.arange(3))
    distinct[0] = 0.0
    assert values[0] == 700.0


def test_distinct_values_merge_signed_zeros() -> None:
    """0.0 and -0.0 compare equal, and are merged as by numpy.unique."""
    distinct, inverse = _distinct_values(np.array([0.0, -0.0, 1.0]))
    assert distinct.size == 2
    np.testing.assert_array_equal(inverse, [0, 0, 1])


@pytest.mark.parametrize("components", COMPONENTS)
def test_atm1d_phase_once_per_distinct_wavelength(components: str) -> None:
    """Atm1D computes its phase matrices once per distinct wavelength.

    Every variable but the numbering of the phase table is the one of
    the former default, all the wavelengths of `calc`.
    """
    comp, kwargs = COMPONENTS[components]()
    new = Atm1D("afglt", comp=comp, **kwargs).calc(WAV)
    comp, kwargs = COMPONENTS[components]()
    old = Atm1D(
        "afglt", comp=comp, wavelength_phase=_full_axis(), **kwargs
    ).calc(WAV)
    n_pf = len(kwargs.get("pfgrid", [100.0, 0.0])) - 1
    assert new.sizes["iphase"] == N_DISTINCT * n_pf
    assert old.sizes["iphase"] == len(WAV) * n_pf
    _assert_same_profile(new, old)


def _ssa_cloud() -> Cloud:
    return _cloud(ssa=np.array([0.9, 0.95, 0.9, 0.8, 0.95, 0.9]))


def _matrix_per_wavelength() -> Cloud:
    user = _cloud_matrix().expand_dims(
        {"wavelength_phase": _full_axis(), "z_phase": PFGRID[1:]}
    ).copy()
    return _cloud(phase=user)


class _OwnAerosol(AerOPAC):
    """A subclass, whose inputs Atm1D cannot know."""


@pytest.mark.parametrize(
    "make",
    [_ssa_cloud, _matrix_per_wavelength,
     lambda: _OwnAerosol("desert", 0.1, 550.0)],
    ids=["ssa per wavelength", "4-D user matrix", "subclass"],
)
def test_atm1d_full_axis_kept_when_inputs_follow_calc(make: Any) -> None:
    """A component paired with the calc wavelengths keeps them all."""
    kwargs: dict[str, Any] = {"pfgrid": PFGRID}
    new = Atm1D("afglt", comp=[make()], **kwargs).calc(WAV)
    old = Atm1D(
        "afglt", comp=[make()], wavelength_phase=_full_axis(), **kwargs
    ).calc(WAV)
    assert new.sizes["iphase"] == old.sizes["iphase"] == len(WAV) * 3
    _assert_same_profile(new, old)


def test_follows_calc_wavelengths() -> None:
    """The components whose inputs follow the calc wavelengths."""
    assert not _follows_calc_wavelengths(AerOPAC("desert", 0.1, 550.0))
    assert not _follows_calc_wavelengths(_cloud(ssa=0.9))
    assert not _follows_calc_wavelengths(_cloud(ssa=np.array(0.9)))
    assert not _follows_calc_wavelengths(_cloud(phase=_cloud_matrix()))
    assert not _follows_calc_wavelengths(_aer_user())
    assert _follows_calc_wavelengths(_ssa_cloud())
    assert _follows_calc_wavelengths(_cloud(ssa=np.full((6, 2), 0.9)))
    assert _follows_calc_wavelengths(_matrix_per_wavelength())
    assert _follows_calc_wavelengths(_OwnAerosol("desert", 0.1, 550.0))


def test_explicit_wavelength_phase_with_repeats_kept() -> None:
    """An explicit wavelength_phase is used as given, with repeats."""
    pro = Atm1D(
        "afglt", comp=[AerOPAC("desert", 0.1, 550.0)],
        wavelength_phase=_full_axis(),
    ).calc(WAV)
    assert pro.sizes["iphase"] == len(WAV)


def test_calc_split_round_trip_on_distinct_layout() -> None:
    """calc_split gives the shorter list, and rebuilds the profile."""
    comp, kwargs = _truncated_cloud()
    atm = Atm1D("afglt", comp=comp, **kwargs)
    pro = atm.calc(WAV)
    prof_abs, prof_ray, prof_aer, prof_phases = atm.calc_split(WAV)
    assert prof_phases is not None
    assert len(prof_phases[1]) == N_DISTINCT * 3
    rebuilt = Atm1D(
        "afglt", prof_abs=prof_abs, prof_ray=prof_ray, prof_aer=prof_aer,
        prof_phases=prof_phases,
    ).calc(WAV, phase=False)
    np.testing.assert_array_equal(
        rebuilt["iphase_atm"].values, pro["iphase_atm"].values
    )
    np.testing.assert_array_equal(
        rebuilt["phase_atm"].values, pro["phase_atm"].values
    )


def test_extract_split_of_a_run_output_on_distinct_layout() -> None:
    """extract_split reads the shorter phase table of a run back."""
    comp, kwargs = _truncated_cloud()
    pro = Atm1D("afglt", comp=comp, **kwargs).calc(WAV)
    run = pro[["OD_p", "OD_r", "OD_g", "ssa_p_atm", "iphase_atm"]]
    run["phase_atm"] = (
        ["phase_index_atm", "nphamat_atm", "theta_atm"],
        pro["phase_atm"].values,
    )
    prof_abs, prof_ray, prof_aer, prof_phases = extract_split(run)
    assert prof_phases is not None
    assert len(prof_phases[1]) == N_DISTINCT * 3
    rebuilt = Atm1D(
        "afglt", prof_abs=prof_abs, prof_ray=prof_ray, prof_aer=prof_aer,
        prof_phases=prof_phases,
    ).calc(WAV, phase=False)
    np.testing.assert_array_equal(
        rebuilt["phase_atm"].values, pro["phase_atm"].values
    )
