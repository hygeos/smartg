"""Tests of the SMART-G runs themselves, from the kernel up.

They compile each variant of the kernel, run an atmosphere, a
surface and water in the combinations the code allows, and check
the outputs and the arguments the run accepts.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

import smartg.smartg as smartg_mod
from smartg import conftest
from smartg.albedo import AlbedoCst, AlbedoMap
from smartg.atmosphere import AerOPAC, Atm1D, Atm3D, Cloud
from smartg.grid3d import Grid3D
from smartg.reptran import Reptran, reduce_reptran
from smartg.sensor import Sensor, get_sensors_grid
from smartg.smartg import (
    Alis,
    LocalEstimate,
    Smartg,
    StdevLim,
    _alis_n_low,
    _check_alis_kernel,
    _check_alis_layers,
    _finalize,
)
from smartg.surface import Environment, LambSurface, RoughSurface
from smartg.view import smartg_view
from smartg.water import HydrosolPR, Water1D
from smartg.xarray import dataset_to_mlut

Wavelength = float | list[float] | NDArray[np.float64]

N_PHOTONS = 1e4

wavelength_list = [500.0, [500.0], np.array([400.0, 600.0])]


@pytest.fixture(params=[True, False])
def sg(request: pytest.FixtureRequest) -> Smartg:
    """Build a Smartg, plane parallel and spherical in turn."""
    return Smartg(pp=request.param)


@pytest.mark.parametrize("pp", [True, False])
@pytest.mark.parametrize("back", [True, False])
def test_compile(pp: bool, back: bool) -> None:
    """Check that each compilation of the kernel goes through."""
    Smartg(pp=pp, back=back)


def test_basic(request: pytest.FixtureRequest) -> None:
    """Run the simplest case there is."""
    m = Smartg(autoinit=True).run(
        500.0, atmosphere=Atm1D("afglms"), n_photons=N_PHOTONS
    )
    smartg_view(m)
    conftest.savefig(request)


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_atm(sg: Smartg, wavelength: Wavelength) -> None:
    """Check that a run keeps a wavelength axis only for a list."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    m = sg.run(wavelength, atmosphere=atmosphere, n_photons=N_PHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
def test_cloud(sg: Smartg, wavelength: Wavelength) -> None:
    """Check the same for a run carrying a water cloud."""
    atmosphere = Atm1D(
        "afglt",
        comp=[
            AerOPAC("desert", 0.1, 550.0),
            Cloud("wc", 12.68, 2, 3, 10.0, 550.0),
        ],
        grid=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0],
        pfgrid=[100.0, 10.0, 0.0],
    )
    m = sg.run(wavelength, atmosphere=atmosphere, n_photons=N_PHOTONS)
    assert ("wavelength" in m.coords) == ("__getitem__" in dir(wavelength))


@pytest.mark.parametrize("wavelength", wavelength_list)
@pytest.mark.parametrize("thv", [0.0, 40.0])
@pytest.mark.parametrize(
    "surface", [RoughSurface(wind=2.0), LambSurface(alb=AlbedoCst(0.2))]
)
def test_atm_surf(
    sg: Smartg,
    wavelength: Wavelength,
    surface: LambSurface | RoughSurface,
    thv: float,
) -> None:
    """Run an aerosol atmosphere over each kind of surface."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])

    sg.run(wavelength, atmosphere=atmosphere, surface=surface,
           th_deg=thv, n_photons=N_PHOTONS)


def test_surf_iop1_1() -> None:
    """Run a rough surface over water, without atmosphere."""
    surface = RoughSurface(wind=10.0)
    water = Water1D(comp=[HydrosolPR(chl=1.0)])
    Smartg().run([400.0, 500.0], surface=surface, water=water,
                 n_photons=N_PHOTONS)


def test_atm_surf_iop1() -> None:
    """Run an atmosphere, a rough surface and water together."""
    atmosphere = Atm1D(
        "afglt",
        comp=[AerOPAC("desert", 0.1, 550.0)],
        wavelength_phase=[500.0, 600.0],
        pfgrid=[100.0, 5.0, 0.0],
    )
    surface = RoughSurface(wind=10.0)
    water = Water1D(
        comp=[HydrosolPR(
            chl=1.0, wavelength_phase=np.array([450, 550, 650, 750])
        )]
    )
    wavelength = np.linspace(400, 800, 12)
    Smartg().run(wavelength, atmosphere=atmosphere, surface=surface,
                 water=water, n_photons=N_PHOTONS)


def test_reptran(sg: Smartg) -> None:
    """Run the REPTRAN bands of an MSG channel."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    surface = RoughSurface(wind=2.0)

    ibands = Reptran("reptran_solar_msg").to_smartg("msg1")

    res = sg.run(ibands.l, atmosphere=atmosphere, surface=surface,
                 water=None, n_photons=N_PHOTONS)
    reduce_reptran(res, ibands)


def test_locale_estimate(sg: Smartg) -> None:
    """Check that the local estimate gives a positive radiance."""
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = 400.0
    res = sg.run(
        wavelength,
        atmosphere=atmosphere,
        surface=surface,
        th_deg=10.0,
        le=LocalEstimate(
            th_deg=np.array([40.0], dtype="float32"),
            phi_deg=np.array([30.0], dtype="float32"),
        ),
        n_photons=N_PHOTONS,
    )
    assert (res["I_up (TOA)"].values > 0).all()


def test_local_estimate_angles() -> None:
    """Check that degrees and radians build the same estimate."""
    deg = LocalEstimate(th_deg=[0.0, 30.0], phi_deg=[0.0, 90.0])
    rad = LocalEstimate(
        th=np.array([0.0, 30.0]) * np.pi / 180.0,
        phi=np.array([0.0, 90.0]) * np.pi / 180.0,
    )
    np.testing.assert_allclose(deg.th, rad.th, rtol=1e-6)
    np.testing.assert_allclose(deg.phi, rad.phi, rtol=1e-6)
    assert deg.th.dtype == np.float32
    assert deg.zip is False
    assert deg.count_level is None


def test_local_estimate_count_level() -> None:
    """Check that the angles and levels are ravelled and typed."""
    le = LocalEstimate(
        th_deg=np.array([[0.0, 30.0]]),
        phi_deg=np.array([[0.0, 90.0]]),
        zip=True,
        count_level=[0, 4],
    )
    assert le.zip is True
    assert le.th.shape == (2,)
    assert le.count_level is not None
    assert le.count_level.dtype == np.int32
    np.testing.assert_array_equal(le.count_level, [0, 4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"th_deg": [0.0]},
        {"th": [0.0], "th_deg": [0.0], "phi_deg": [0.0]},
        {"th_deg": [0.0, 30.0], "phi_deg": [0.0], "zip": True},
        {"th_deg": [0.0, 30.0], "phi_deg": [0.0, 90.0],
         "count_level": [0]},
    ],
)
def test_local_estimate_invalid(kwargs: dict[str, Any]) -> None:
    """Check that an inconsistent local estimate is rejected."""
    with pytest.raises(ValueError):
        LocalEstimate(**kwargs)


def test_alis_defaults() -> None:
    """Check that the ALIS options keep their documented values."""
    alis = Alis(n_low=10)
    assert alis.n_low == 10
    assert alis.hist is False
    assert alis.max_hist == 8000000
    assert alis.n_jac == 0
    assert alis.n_jac_abs is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_low": 0},
        {"n_low": 1},
        {"n_low": 10, "n_jac_abs": True},
        {"n_low": 10, "n_jac": 0, "n_jac_abs": True},
    ],
)
def test_alis_invalid(kwargs: dict[str, Any]) -> None:
    """Check that the kernel refuses what it cannot honour."""
    with pytest.raises(ValueError):
        Alis(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alis": True}, "alt_pp"),
        ({"sif": True}, "sif=True"),
        ({"sif": True, "alis": True}, "alt_pp"),
        ({"sif": True, "alt_pp": True}, "sif=True"),
        ({"sif": True, "pp": False}, "sif=True"),
    ],
)
def test_alis_kernel_invalid(kwargs: dict[str, Any], match: str) -> None:
    """Check that ALIS and SIF refuse the fast plane parallel mode."""
    options = {
        "alis": False, "pp": True, "alt_pp": False, "opt3d": False,
        "sif": False, **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        _check_alis_kernel(**options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"alis": True, "alt_pp": True},
        {"alis": True, "pp": False},
        {"alis": True, "opt3d": True},
        {"alis": True, "alt_pp": True, "sif": True},
        {"alis": True, "pp": False, "sif": True},
    ],
)
def test_alis_kernel_valid(kwargs: dict[str, Any]) -> None:
    """Check that ALIS and SIF accept the layer by layer modes."""
    options = {
        "alis": False, "pp": True, "alt_pp": False, "opt3d": False,
        "sif": False, **kwargs,
    }
    _check_alis_kernel(**options)


def test_alis_fast_pp_refused() -> None:
    """Check that Smartg refuses ALIS before compiling the kernel."""
    with pytest.raises(ValueError, match="alt_pp"):
        Smartg(alis=True)


@pytest.mark.parametrize(
    ("alis", "n_lam", "n_low"),
    [
        (Alis(n_low=-1), 5, 5),
        (Alis(n_low=3), 5, 3),
        (Alis(n_low=5), 5, 5),
        (Alis(n_low=-1, n_jac=3), 40, 40),
        (Alis(n_low=-1, n_jac=3, n_jac_abs=True), 40, 10),
        (Alis(n_low=4, n_jac=3, n_jac_abs=True), 40, 4),
    ],
)
def test_alis_n_low(alis: Alis, n_lam: int, n_low: int) -> None:
    """Check that n_low=-1 stands for the wavelengths it may use."""
    assert _alis_n_low(alis, n_lam) == n_low


@pytest.mark.parametrize(
    ("alis", "n_lam", "match"),
    [
        (Alis(n_low=10), 5, "exceeds the 5 wavelengths"),
        (Alis(n_low=-1, n_jac=3, n_jac_abs=True), 41, "groups"),
        (Alis(n_low=20, n_jac=3, n_jac_abs=True), 40, "exceeds the 10"),
        (Alis(n_low=900), 1000, "exceeds the 801"),
    ],
)
def test_alis_n_low_invalid(alis: Alis, n_lam: int, match: str) -> None:
    """Check that an n_low without a kernel wavelength step fails."""
    with pytest.raises(ValueError, match=match):
        _alis_n_low(alis, n_lam)


def test_alis_n_low_uneven_warns() -> None:
    """Check that an n_low missing the last wavelength warns."""
    with pytest.warns(UserWarning, match="index 9, and the 2 wavelengths"):
        assert _alis_n_low(Alis(n_low=4), 12) == 4


def test_alis_uneven_n_low(sg_alis: Smartg) -> None:
    """Check the corrections past the last low resolution point.

    With 12 wavelengths and n_low=4, the low resolution points are
    the wavelengths 0, 3, 6 and 9: the wavelengths 10 and 11 take the
    correction of the wavelength 9. Wavelength 10 used to be
    interpolated towards an uninitialised correction. Without gas
    absorption, the three wavelengths get the same radiance.
    """
    atmosphere = Atm1D(
        "afglt", grid=np.linspace(50.0, 0.0, 11), prof_abs=np.zeros((12, 11))
    )
    with pytest.warns(UserWarning, match="does not divide"):
        m = sg_alis.run(
            np.linspace(500.0, 511.0, 12),
            atmosphere=atmosphere,
            surface=LambSurface(alb=AlbedoCst(0.2)),
            le=LocalEstimate(th_deg=[30.0], phi_deg=[0.0]),
            alis_options=Alis(n_low=4),
            n_photons=N_PHOTONS,
            seed=1,
        )
    i_up = np.squeeze(m["I_up (TOA)"].values)
    assert np.all(np.isfinite(i_up))
    np.testing.assert_allclose(i_up[10], i_up[9], rtol=1e-5)
    np.testing.assert_allclose(i_up[11], i_up[9], rtol=1e-5)


def test_alis_cdist_ocean(sg_alis: Smartg) -> None:
    """Check the cdist moments of the ocean layers.

    The ocean layers, first on the 'cdist_layer' axis, used to be
    written with the stride of a single moment and without the photon
    weights. Every photon adds its weight to every layer, so the sum
    of the weights (iAMF=0) is the same for all the layers.
    """
    m = sg_alis.run(
        np.array([500.0, 510.0]),
        atmosphere=Atm1D("afglt", grid=np.linspace(50.0, 0.0, 6)),
        surface=RoughSurface(),
        water=Water1D(grid=[0.0, -5.0, -10.0], comp=[HydrosolPR(chl=1.0)]),
        le=LocalEstimate(th_deg=[0.0, 30.0], phi_deg=[0.0]),
        alis_options=Alis(n_low=2),
        n_photons=N_PHOTONS,
        seed=1,
    )
    # (cdist_layer, Azimuth angles, Zenith angles, iAMF)
    cdist = m["cdist_up (TOA)"].values
    assert cdist.shape[0] == 2 + 5
    weights = cdist[..., 0]
    assert np.all(weights[0] > 0.0)
    np.testing.assert_allclose(
        weights, np.broadcast_to(weights[:1], weights.shape), rtol=1e-9
    )
    # the photons reaching the TOA from the ocean travel in it
    assert np.all(cdist[:2, ..., 1] > 0.0)


@pytest.fixture(scope="module")
def sg_alis_datomicadd() -> Iterator[Smartg]:
    """Build an ALIS Smartg on the DatomicAdd fallback of old GPUs."""
    source_module = smartg_mod.SourceModule

    def forced(*args: Any, **kwargs: Any) -> Any:
        options = list(kwargs.pop("options", []))
        return source_module(
            *args, options=[*options, "-DFORCE_DATOMICADD"], **kwargs
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smartg_mod, "SourceModule", forced)
        yield Smartg(alis=True, alt_pp=True, amf_variance=True)


def test_alis_datomicadd(sg_alis_datomicadd: Smartg) -> None:
    """Check the ALIS counts of the GPUs without a double atomicAdd.

    Their fallback added the no-aerosol counts through an unset
    pointer, and only the raw path lengths to the cdist moments. The
    radiances and the mean path lengths must match those of the
    native atomicAdd, within the Monte Carlo noise.
    """
    kwargs = {
        "wavelength": np.array([500.0, 510.0, 520.0]),
        "atmosphere": Atm1D("afglt", grid=np.linspace(50.0, 0.0, 6)),
        "surface": LambSurface(alb=AlbedoCst(0.2)),
        "le": LocalEstimate(th_deg=[0.0, 30.0], phi_deg=[0.0]),
        "alis_options": Alis(n_low=3),
        "n_photons": 1e5,
        "seed": 1,
    }
    forced = sg_alis_datomicadd.run(**kwargs)
    native = Smartg(alis=True, alt_pp=True, amf_variance=True).run(**kwargs)
    np.testing.assert_allclose(
        forced["I_up (TOA)"].values, native["I_up (TOA)"].values, rtol=0.03
    )
    # (cdist_layer, Azimuth angles, Zenith angles, iAMF)
    for m in (forced, native):
        cdist = m["cdist_up (TOA)"].values
        assert np.all(cdist[..., 2] > 0.0)
    mean_path = [
        m["cdist_up (TOA)"].values[..., 1] / m["cdist_up (TOA)"].values[..., 0]
        for m in (forced, native)
    ]
    np.testing.assert_allclose(mean_path[0], mean_path[1], rtol=0.03)


class _ErrorCount:
    """Stand for the GPU error counters of a run."""

    def get(self) -> NDArray[np.uint64]:
        """Return no error."""
        return np.zeros(32, dtype=np.uint64)


@pytest.mark.parametrize(
    ("n_sensor", "n_th", "zip_le", "n_layer", "n_scl", "dims", "shape"),
    [
        (1, 3, True, 3, 1, ("Zenith angles",), (3, 3)),
        (2, 3, True, 3, 1, ("sensor index", "Zenith angles"), (3, 2, 3)),
        (1, 1, True, 3, 1, ("Zenith angles",), (3, 1)),
        (1, 3, True, 1, 1, ("Zenith angles",), (1, 3)),
        (
            2, 3, True, 3, 2,
            ("sensor index", "Zenith angles", "iSCL"), (3, 2, 3, 2),
        ),
        (
            2, 3, False, 3, 1,
            ("sensor index", "Azimuth angles", "Zenith angles"),
            (3, 2, 2, 3),
        ),
    ],
    ids=["zip", "zip-sensors", "zip-one-dir", "zip-one-layer",
         "zip-sensors-scl", "sensors"],
)
def test_alis_finalize_cdist(
    n_sensor: int,
    n_th: int,
    zip_le: bool,
    n_layer: int,
    n_scl: int,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> None:
    """Check the cdist output of the zipped directions.

    With several sensors, a single direction or a single layer, the
    squeezed cdist of the zipped directions no longer matched its
    dimension names, and the output stage raised a ValueError.
    """
    le = LocalEstimate(
        th_deg=np.linspace(0.0, 60.0, n_th),
        phi_deg=np.linspace(0.0, 90.0, n_th) if zip_le else [0.0, 90.0],
        zip=zip_le,
    )
    n_lam = 2
    n_phi = 1 if zip_le else 2
    photons = np.ones((6, 4, n_sensor, n_lam, n_th, n_phi))
    n_out = np.ones((6, n_sensor, n_lam, n_th, n_phi), dtype=np.uint64)
    ds = _finalize(
        tab_photons_tot=photons,
        tab_photons_tot_no_aer=photons,
        tab_dist_tot=np.ones(
            (6, n_layer, n_sensor, n_th, n_phi, n_scl, 2)
        ),
        tab_hist_tot=None,
        wavelength=np.array([500.0, 600.0]),
        n_photons_in_tot=np.full((n_sensor, n_lam), 10, dtype=np.uint64),
        errorcount=_ErrorCount(),  # pyright: ignore[reportArgumentType]
        n_photons_out_tot=n_out,
        n_photons_out_tot_no_aer=n_out,
        output_layers=0,
        tab_trans_dir=np.zeros((n_sensor, n_lam)),
        tab_trans_dir_analytic=None,
        attrs={},
        prof_atm=None,
        prof_oc=None,
        sigma=None,
        horiz=0,
        le=le,
    )
    cdist = ds["cdist_up (TOA)"]
    assert cdist.dims == ("cdist_layer", *dims, "iAMF")
    assert cdist.shape == (*shape, 2)


def test_alis_layers() -> None:
    """Check the layer count an ALIS photon can hold."""
    _check_alis_layers(199, 199)
    with pytest.raises(ValueError, match="the atmosphere has 200"):
        _check_alis_layers(200, 0)
    with pytest.raises(ValueError, match="the ocean has 250"):
        _check_alis_layers(50, 250)


@pytest.fixture(scope="module")
def sg_alis() -> Smartg:
    """Build an ALIS Smartg, in the alternative plane parallel mode."""
    return Smartg(alis=True, alt_pp=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "alis_options"),
        ({"alis_options": Alis(n_low=10)}, "exceeds the 5 wavelengths"),
        (
            {
                "alis_options": Alis(n_low=3),
                "atmosphere": Atm1D("afglt", grid=np.linspace(100, 0, 251)),
            },
            "the atmosphere has 250",
        ),
        (
            {
                "alis_options": Alis(n_low=3, hist=True),
                "surface": RoughSurface(),
                "water": Water1D(comp=[HydrosolPR(chl=1.0)]),
            },
            "water body",
        ),
    ],
    ids=["no-options", "n_low", "layers", "hist-water"],
)
def test_alis_run_invalid(
    sg_alis: Smartg, kwargs: dict[str, Any], match: str
) -> None:
    """Check that an ALIS run refuses what the kernel cannot honour."""
    kwargs = {"atmosphere": Atm1D("afglt"), **kwargs}
    with pytest.raises(ValueError, match=match):
        sg_alis.run(
            np.linspace(500.0, 520.0, 5), n_photons=N_PHOTONS, **kwargs
        )


def test_le_dict_deprecated(sg: Smartg) -> None:
    """Check that the legacy le dict runs, warns and is kept."""
    le = {
        "th_deg": np.array([40.0], dtype="float32"),
        "phi_deg": np.array([30.0], dtype="float32"),
    }
    with pytest.warns(DeprecationWarning, match="LocalEstimate"):
        res = sg.run(
            400.0,
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            le=le,
            n_photons=N_PHOTONS,
        )
    assert (res["I_up (TOA)"].values > 0).all()
    assert sorted(le) == ["phi_deg", "th_deg"]


def test_alis_options_dict_deprecated() -> None:
    """Check that the legacy alis_options map onto Alis."""
    with pytest.warns(DeprecationWarning, match="Alis"):
        Smartg(alis=True, alt_pp=True).run(
            np.array([400.0, 600.0]),
            atmosphere=Atm1D("afglt"),
            surface=RoughSurface(),
            alis_options={"nlow": -1, "njac": 0},
            n_photons=N_PHOTONS,
        )


@pytest.fixture(scope="module")
def sg_forward() -> Smartg:
    """Build the default Smartg, forward and without ALIS."""
    return Smartg()


INVALID_RUNS = [
    ({"flux": "planer"}, ValueError, "unknown flux"),
    (
        {"flux": "planar", "le": LocalEstimate(th_deg=[0.0], phi_deg=[0.0])},
        ValueError,
        "flux and le",
    ),
    ({"alis_options": Alis(n_low=-1)}, ValueError, "alis=True"),
    ({"environment": Environment(env=1)}, ValueError, "needs a surface"),
    (
        {"wavelength_proba": np.zeros(4, dtype=np.int32)},
        TypeError,
        "int64",
    ),
    ({"sensor_proba": np.zeros(4, dtype=np.int32)}, TypeError, "int64"),
    ({"cell_proba": "automatic"}, ValueError, "unknown cell_proba"),
    ({"cell_proba": "auto"}, ValueError, "forward thermal"),
    ({"cell_proba": np.zeros((3, 5))}, ValueError, "one column"),
    ({"depol": -1.0}, ValueError, "depol must be positive"),
    ({"depol_water": -1.0}, ValueError, "depol_water must be positive"),
]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    INVALID_RUNS,
    ids=[
        "flux", "flux-le", "alis", "environment", "wavelength_proba",
        "sensor_proba", "cell_proba-name", "cell_proba-auto",
        "cell_proba-shape", "depol", "depol_water",
    ],
)
def test_run_invalid(
    sg_forward: Smartg,
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    """Check that the run refuses the arguments it would ignore."""
    with pytest.raises(error, match=match):
        sg_forward.run(
            500.0, atmosphere=Atm1D("afglt"), n_photons=N_PHOTONS, **kwargs
        )


def test_dataset_to_mlut_roundtrip() -> None:
    """Check that the output converts to an MLUT losslessly."""
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 0.1, 550.0)])
    res = Smartg(autoinit=True).run(
        np.array([400.0, 600.0]),
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.1)),
        n_photons=N_PHOTONS,
    )
    assert isinstance(res, xr.Dataset)

    mlut = dataset_to_mlut(res)
    assert mlut.datasets() == list(res.data_vars)
    for name in res.data_vars:
        lut = mlut[name]
        assert lut.names == list(res[name].dims)
        np.testing.assert_array_equal(lut.data, res[name].values)
        assert dict(lut.attrs) == dict(res[name].attrs)
    assert dict(mlut.attrs) == dict(res.attrs)


@pytest.mark.parametrize("rng", ["PHILOX", "CURAND_PHILOX"])
def test_rng(rng: str) -> None:
    """Run each of the random number generators."""
    atmosphere = Atm1D("afglt")
    surface = RoughSurface()
    wavelength = np.linspace(400, 800, 5)
    Smartg(rng=rng).run(wavelength, atmosphere=atmosphere,
                        surface=surface, n_photons=N_PHOTONS)


def test_adjacency() -> None:
    """Skipped, the adjacency effect is still in progress."""
    pytest.skip("Cannot test this, it is still in progress.")


def test_no_aer_output() -> None:
    """Check that the no_aer outputs match the ones with aerosols."""
    atm1 = Atm1D("afglt")
    water = Water1D(grid=[0, -5.0], comp=[HydrosolPR(chl=0.5)])
    surface = RoughSurface(wind=5.0, nh2o=1.34)
    sg = Smartg()
    le = LocalEstimate(
        th_deg=np.array([0.0, 45.0, 89.9]),
        phi_deg=np.array([0.0, 15.6, 289.0]),
    )
    m1 = sg.run(
        550.0,
        atmosphere=atm1,
        surface=surface,
        water=water,
        output_layers=3,
        le=le,
        n_photons=1e6,
        n_loop=1e6,
        no_aer_output=True,
    )
    assert np.allclose(
        m1["I_up (TOA)"].values,
        m1["I_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (TOA)"].values,
        m1["Q_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (TOA)"].values,
        m1["U_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (TOA)"].values,
        m1["V_up (TOA), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (0+)"].values,
        m1["I_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (0+)"].values,
        m1["Q_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (0+)"].values,
        m1["U_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (0+)"].values,
        m1["V_down (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (0-)"].values,
        m1["I_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (0-)"].values,
        m1["Q_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (0-)"].values,
        m1["U_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (0-)"].values,
        m1["V_down (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_up (0+)"].values,
        m1["I_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (0+)"].values,
        m1["Q_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (0+)"].values,
        m1["U_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (0+)"].values,
        m1["V_up (0+), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_up (0-)"].values,
        m1["I_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_up (0-)"].values,
        m1["Q_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_up (0-)"].values,
        m1["U_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_up (0-)"].values,
        m1["V_up (0-), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["I_down (B)"].values,
        m1["I_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["Q_down (B)"].values,
        m1["Q_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["U_down (B)"].values,
        m1["U_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )
    assert np.allclose(
        m1["V_down (B)"].values,
        m1["V_down (B), no_aer"].values,
        0,
        1e-12,
        equal_nan=True,
    )


def _z_scores(a: xr.Dataset, b: xr.Dataset, level: str) -> NDArray:
    """Return the differences of two runs over their joint stdev."""
    return (a[f"I_{level}"].values - b[f"I_{level}"].values) / np.sqrt(
        a[f"I_stdev_{level}"].values ** 2 + b[f"I_stdev_{level}"].values ** 2
    )


def _adjacency_run(alt_pp: bool, back: bool, seed: int) -> xr.Dataset:
    """Run a black disc of 2 km in a white environment, under dust."""
    # a single thick aerosol layer, where the height of a scattering
    # inside the layer sets how far from it the photon reaches the
    # ground
    atmosphere = Atm1D("afglt", comp=[AerOPAC("desert", 1.0, 550.0)],
                       grid=[100.0, 10.0, 0.0])
    kwargs: dict[str, Any] = {}
    if back:
        kwargs["sensor"] = Sensor(pos_z=100.0, th_deg=180.0, loc="ATMOS")
        th_deg = 30.0
    else:
        kwargs["th_deg"] = 30.0
        th_deg = 0.0
    le = LocalEstimate(th_deg=[th_deg], phi_deg=[0.0], count_level=[0])
    return Smartg(alt_pp=alt_pp, back=back).run(
        550.0,
        atmosphere=atmosphere,
        surface=LambSurface(alb=AlbedoCst(0.0)),
        environment=Environment(env=1, env_size=2.0, alb=AlbedoCst(1.0)),
        le=le,
        n_photons=1e7,
        stdev=True,
        seed=seed,
        progress=False,
        **kwargs,
    )


@pytest.mark.parametrize("back", [True, False])
def test_adjacency_fast_and_alt_pp_moves_agree(back: bool) -> None:
    """Check that the fast PP move places the photons at their height.

    The adjacency effect depends on where the photons reach the
    ground, so on the altitude of their scatterings: the fast move,
    which follows the optical depth, must agree with the alternative
    one, which follows the geometry.
    """
    fast = _adjacency_run(alt_pp=False, back=back, seed=11)
    alt = _adjacency_run(alt_pp=True, back=back, seed=12)
    assert np.all(np.abs(_z_scores(fast, alt, "up (TOA)")) < 5)


def _empty_atmosphere() -> Atm1D:
    """Return an atmosphere without scattering nor absorption."""
    return Atm1D("afglt", tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)


def test_brdf_surface_in_3d_atmosphere() -> None:
    """Check that a 3D atmosphere reflects on a BRDF surface as in 1D.

    Without anything in the atmosphere the radiance is the one of the
    Cox and Munk BRDF alone, the same for every sensor of the 3D
    domain and for the 1D run.
    """
    surface = RoughSurface(wind=5.0, brdf=True)
    le = LocalEstimate(th_deg=[40.0], phi_deg=[0.0], count_level=[0])
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    grid3 = Grid3D(edges, edges, np.array([0.0, 1.0, 2.0]), periodic=True)
    sensors = get_sensors_grid(
        edges, edges, pos_z=2.0, th_deg=150.0, ph_deg=0.0, loc="ATMOS",
        cell_size=1.0, grid_3d=grid3,
    )
    m3 = Smartg(opt3d=True, alt_pp=True, back=True).run(
        550.0,
        atmosphere=Atm3D(atm_1d=_empty_atmosphere(), grid_3d=grid3),
        surface=surface,
        sensor=sensors,
        le=le,
        n_photons=1e6,
        seed=41,
        progress=False,
    )
    m1 = Smartg(alt_pp=True, back=True).run(
        550.0,
        atmosphere=_empty_atmosphere(),
        surface=surface,
        sensor=Sensor(pos_z=120.0, th_deg=150.0, ph_deg=0.0, loc="ATMOS"),
        le=le,
        n_photons=1e6,
        seed=42,
        progress=False,
    )
    np.testing.assert_allclose(
        m3["I_up (TOA)"].values.ravel(),
        float(m1["I_up (TOA)"].values.ravel()[0]),
        rtol=1e-3,
    )


def _sza_max_run(
    sg: Smartg, n_theta: int, sza_max: float, seed: int
) -> xr.Dataset:
    """Run the cone sampling of a dusty atmosphere on a zenith grid."""
    return sg.run(
        550.0,
        atmosphere=Atm1D("afglt", comp=[AerOPAC("desert", 0.3, 550.0)]),
        surface=LambSurface(alb=AlbedoCst(0.2)),
        th_deg=30.0,
        n_theta=n_theta,
        n_phi=2,
        sza_max=sza_max,
        output_layers=1,
        n_photons=1e7,
        stdev=True,
        seed=seed,
        progress=False,
    )


def test_sza_max_keeps_the_radiances(sg: Smartg) -> None:
    """Check that sza_max only cuts the zenith grid of the output.

    The 2 degree boxes up to 60 degrees are the same with 45 boxes up
    to 90 degrees and with 30 boxes up to 60 degrees, and so must be
    their radiances, at the top of the atmosphere as at the surface.
    """
    full = _sza_max_run(sg, 45, 90.0, 21)
    part = _sza_max_run(sg, 30, 60.0, 22)
    common = {"Zenith angles": slice(0, 30)}
    np.testing.assert_allclose(
        part["Zenith angles"].values, full["Zenith angles"][common].values
    )
    for level in ("up (TOA)", "down (0+)"):
        assert np.all(np.abs(_z_scores(full[common], part, level)) < 6)


def test_sun_disc_boxes_of_every_level() -> None:
    """Check that sun_disc counts the downward photons too.

    With sun_disc, a box counts only the photons within sun_disc of
    its centre, and its radiance is normalised by the solid angle of
    the disc instead of the one of the box, which divides it by 2 pi.
    Where the disc lies inside the box, 2 pi times the radiance must
    match the one of the whole box, at the top of the atmosphere as at
    the surface. The molecular atmosphere has no aureole, whose peak
    would make the radiance at the centre of a box differ from its
    mean over the box.
    """
    sg = Smartg()
    kwargs: dict[str, Any] = {
        "atmosphere": Atm1D("afglt"),
        "surface": LambSurface(alb=AlbedoCst(0.2)),
        "th_deg": 30.0,
        "n_theta": 9,
        "n_phi": 36,
        "output_layers": 1,
        "n_photons": 2e7,
        "progress": False,
    }
    boxes = sg.run(550.0, seed=31, **kwargs)
    disc = sg.run(550.0, seed=32, sun_disc=2.0, **kwargs)
    for level in ("up (TOA)", "down (0+)"):
        # mean over the azimuth, where the 2 degree disc lies inside
        # the 10 x 10 degree boxes: 30 to 80 degrees of zenith angle
        ratio = (
            2 * np.pi * disc[f"I_{level}"].values.mean(axis=0)
            / boxes[f"I_{level}"].values.mean(axis=0)
        )
        np.testing.assert_allclose(ratio[3:8], 1.0, atol=0.05)


def test_albedo_map_near_the_origin_in_spherical_mode() -> None:
    """Check that a spherical run reads the albedo map at 1 km.

    Two nadir sensors look at the ground 1 km on each side of the
    origin, through an empty atmosphere, one onto a white strip of the
    map and the other onto black ground.
    """
    albedo_map = AlbedoMap(
        np.array([[0], [1], [0]]),
        np.array([0.5, 1.5, 1e8]),
        np.array([1e8]),
        [AlbedoCst(0.0), AlbedoCst(1.0)],
    )
    sensors = [
        Sensor(pos_x=x, pos_z=6371.0 + 120.0, th_deg=180.0, loc="ATMOS")
        for x in (1.0, -1.0)
    ]
    m = Smartg(pp=False, back=True).run(
        550.0,
        atmosphere=_empty_atmosphere(),
        surface=LambSurface(alb=AlbedoCst(0.0)),
        environment=Environment(env=5, alb=albedo_map),
        sensor=sensors,
        le=LocalEstimate(th_deg=[30.0], phi_deg=[0.0], count_level=[0]),
        n_photons=1e5,
        seed=51,
        progress=False,
    )
    white, black = m["I_up (TOA)"].values.ravel()
    assert white > 0.5
    assert black < 1e-6


def test_albedo_map_seafloor_below_the_land() -> None:
    """Check the seafloor below the land cells of an albedo map.

    The sensor looks at the sea 2 m from the coast, obliquely, so that
    the photons refracted into the water reach the seafloor, 5 m below,
    under the land cell. There the seafloor keeps the albedo of the
    water profile, the same as the one the map gives below the sea
    cells here, so the radiance must be the one of a map of sea only.
    """
    sg = Smartg(back=True)
    water = Water1D(
        grid=[0.0, -5.0], comp=[HydrosolPR(chl=0.01)], alb=AlbedoCst(0.5)
    )
    alist = [AlbedoCst(0.2), AlbedoCst(0.5)]
    runs = []
    for seed, codes in ((91, [[1], [-1]]), (92, [[-1], [-1]])):
        albedo_map = AlbedoMap(
            np.array(codes), np.array([0.0, 1e8]), np.array([1e8]), alist
        )
        runs.append(sg.run(
            550.0,
            surface=RoughSurface(),
            water=water,
            environment=Environment(env=5, alb=albedo_map),
            sensor=Sensor(
                pos_x=0.002, th_deg=120.0, ph_deg=180.0, loc="SURF0P"
            ),
            le=LocalEstimate(th_deg=[30.0], phi_deg=[0.0], count_level=[0]),
            n_photons=1e6,
            stdev=True,
            seed=seed,
            progress=False,
        ))
    assert np.all(np.abs(_z_scores(runs[0], runs[1], "up (TOA)")) < 5)


@pytest.mark.parametrize("alt_pp", [False, True])
def test_environment_without_atmosphere_reflects_to_space(
    alt_pp: bool,
) -> None:
    """Check the environment of a run with water and no atmosphere.

    The light reaches the ground on the environment, a Lambertian
    albedo of 0.3 around a disc of sea 10 km away, and leaves to space:
    the radiance is the albedo, above the surface as at the top of the
    atmosphere. The environment sent the photon into the empty
    atmosphere, from where the alternative PP move returned it to the
    ground at once, forever, and counted nothing at the top of the
    atmosphere.
    """
    m = Smartg(alt_pp=alt_pp).run(
        450.0,
        surface=RoughSurface(),
        water=Water1D(grid=[0.0, -10.0], comp=[]),
        environment=Environment(
            env=1, env_size=1.0, x0=10.0, alb=AlbedoCst(0.3)
        ),
        th_deg=30.0,
        le=LocalEstimate(th_deg=[30.0], phi_deg=[90.0]),
        output_layers=2,
        n_photons=1e4,
        progress=False,
    )
    for level in ("up (TOA)", "up (0+)"):
        np.testing.assert_allclose(m[f"I_{level}"].values, 0.3, rtol=1e-5)


def test_toa_sphere_start_without_objects() -> None:
    """Check the cell_size=-2 sensors of a run without 3D objects.

    They start on the top of atmosphere sphere: a far sensor whose line
    of sight grazes the atmosphere 30 km above the ground, missing the
    Earth, sees the light the atmosphere scatters, as does one looking
    at the ground.
    """
    distance = 6371.0 + 1e4
    alpha = np.degrees(np.arcsin((6371.0 + 30.0) / distance))
    sensors = [
        Sensor(pos_z=distance, th_deg=180.0 - a, ph_deg=0.0, loc="ATMOS",
               cell_size=-2)
        for a in (alpha, 10.0)
    ]
    m = Smartg(pp=False, obj3d=True, back=True).run(
        550.0,
        atmosphere=Atm1D("afglt"),
        sensor=sensors,
        n_photons=1e5,
        seed=61,
        progress=False,
    )
    assert np.all(m["N_up (TOA)"].values.reshape(2, -1).sum(axis=1) > 0)


def test_default_sensor_aims_at_the_origin_in_any_azimuth() -> None:
    """Check that the sun of the default sensor lights the origin.

    A white disc of 5 km around the origin in a black environment
    reflects the direct sunlight only if the sun, of any azimuth,
    aims at the origin, and the nadir radiance does not depend on the
    azimuth of the sun.
    """
    sg = Smartg()
    runs = [
        sg.run(
            550.0,
            atmosphere=Atm1D("afglt"),
            surface=LambSurface(alb=AlbedoCst(1.0)),
            environment=Environment(env=1, env_size=5.0,
                                    alb=AlbedoCst(0.0)),
            th_deg=30.0,
            ph_deg=ph_deg,
            le=LocalEstimate(th_deg=[0.0], phi_deg=[0.0], count_level=[0]),
            n_photons=1e6,
            stdev=True,
            seed=seed,
            progress=False,
        )
        for seed, ph_deg in ((71, 0.0), (72, 90.0))
    ]
    assert np.all(np.abs(_z_scores(runs[0], runs[1], "up (TOA)")) < 5)


def test_device_with_cuda_device_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that device and CUDA_DEVICE cannot be given together."""
    monkeypatch.setenv("CUDA_DEVICE", "0")
    with pytest.raises(ValueError, match="CUDA_DEVICE"):
        Smartg(device=0)


def test_few_photons(sg_forward: Smartg) -> None:
    """Check that a run of fewer photons than 30 ends."""
    m = sg_forward.run(
        500.0, atmosphere=Atm1D("afglt"), n_photons=10, progress=False
    )
    assert m.attrs["NPhotonIn_sum"] >= 10


@pytest.mark.parametrize("double", [True, False])
def test_direct_transmission_dev_precision(double: bool) -> None:
    """Check the direct transmission of the kernel in any precision.

    In spherical mode, the local estimate records the optical depth of
    the path to the top of the atmosphere, below 1 for a scattering
    inside the atmosphere.
    """
    m = Smartg(pp=False, double=double).run(
        400.0,
        atmosphere=Atm1D("afglt"),
        th_deg=30.0,
        le=LocalEstimate(th_deg=[0.0], phi_deg=[0.0], count_level=[0]),
        n_photons=1e5,
        seed=81,
        progress=False,
    )
    assert float(m["direct transmission (dev)"].values) < 0.9999


def test_spherical_3d_atmosphere_refused() -> None:
    """Check that the 3D atmosphere refuses the spherical geometry."""
    with pytest.raises(ValueError, match="pp=True"):
        Smartg(pp=False, opt3d=True)


def test_stdev_lim_on_a_level_not_counted(sg_forward: Smartg) -> None:
    """Check that a StdevLim on a level left out does not stop a run.

    With output_layers=4, only the levels 0+ down and 0- up are
    counted, not the default level of StdevLim, the top of the
    atmosphere, whose zero error must not stop the run.
    """
    m = sg_forward.run(
        550.0,
        atmosphere=Atm1D("afglt"),
        surface=LambSurface(alb=AlbedoCst(0.1)),
        output_layers=4,
        stdev=True,
        stdev_lim=StdevLim(err_rel_min=1.0),
        n_photons=1e6,
        n_loop=1e4,
        seed=101,
        progress=False,
    )
    assert m.attrs["NPhotonIn_sum"] >= 1e6


def test_stdev_of_sensors_missed_by_some_loops(sg_forward: Smartg) -> None:
    """Check the stdev of the sensors that some kernel loops miss.

    100 sensors share about 200 photons per loop, so that each of them
    gets none in some loops.
    """
    sensors = [
        Sensor(pos_z=120.0, th_deg=150.0, ph_deg=180.0, loc="ATMOS")
        for _ in range(100)
    ]
    m = sg_forward.run(
        550.0,
        atmosphere=Atm1D("afglt"),
        sensor=sensors,
        stdev=True,
        n_photons=1e4,
        n_loop=200,
        xblock=32,
        xgrid=4,
        seed=111,
        progress=False,
    )
    assert not np.any(np.isnan(m["I_stdev_up (TOA)"].values))
