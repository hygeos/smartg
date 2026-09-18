"""K-distribution absorption parameterization.

KDIS provides a compact representation of molecular absorption using
correlated-k coefficients tabulated over pressure, temperature, and,
for selected species, concentration. This module reads KDIS definition
files, represents sensor channels and internal absorption bands, and
calculates gaseous absorption and thermal emission for SMART-G profiles.

The public reduction and emission functions use xarray datasets and data
arrays. Legacy LUT and MLUT inputs remain accepted at compatibility
boundaries required by existing SMART-G workflows.

Key Classes
-----------
Kdis
    Read and expose a KDIS correlated-k definition.
KdisBand
    Represent a KDIS sensor channel.
KdisIband
    Represent one internal KDIS absorption band.
KdisIbandList
    Store a selected list of internal KDIS bands.

Key Functions
-------------
reduce_kdis
    Reduce spectral results to KDIS channel values.
kdis_emission
    Calculate spectrally resolved thermal emission.
kdis_avg_emission
    Calculate thermal emission integrated over atmospheric
    altitude.
"""

from __future__ import annotations

import glob
import sys
import warnings
from collections.abc import Iterator, Sequence
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

import h5py
import numpy as np
import xarray as xr
from luts.luts import LUT, MLUT
from numpy.typing import NDArray
from scipy.integrate import quad, simpson
from scipy.interpolate import interpn

from smartg.atmosphere import blackbody_radiance, od2k
from smartg.config import DIR_AUXDATA
from smartg.interp import interp2
from smartg.typing import NumericArrayLike, PathType

dir_kdis = DIR_AUXDATA / "kdis"

if TYPE_CHECKING:
    from smartg.atmosphere import ProfileBase


def reduce_kdis(
    ds: xr.Dataset | MLUT,
    ibands: KdisIbandList,
    use_solar: bool = False,
    integrated: bool = False,
    extern_weights: LUT | xr.DataArray | None = None,
) -> xr.Dataset:
    """Reduce spectral results to KDIS channel values.

    The spectral variables selected from ``ds`` are weighted by the
    internal-band weights and grouped by their central channel
    wavelength.

    Parameters
    ----------
    ds : Dataset or MLUT
        Spectral SMART-G results containing a ``wavelength``
        coordinate.
    ibands : KdisIbandList
        KDIS internal bands providing weights, channel wavelengths,
        and bandwidths.
    use_solar : bool, optional
        Include extraterrestrial solar irradiance in the weighting
        factor. Default is False.
    integrated : bool, optional
        Normalize by the sum of weights instead of the
        bandwidth-weighted sum. Default is False.
    extern_weights : DataArray or LUT, optional
        Additional wavelength-dependent weights. LUT input is
        deprecated. Default is None.

    Returns
    -------
    Dataset
        Channel-reduced variables whose names contain an accepted
        output prefix, with source variable attributes preserved.
    """
    if isinstance(ds, MLUT):
        warnings.warn(
            "Passing an MLUT to reduce_kdis is deprecated; pass an "
            "xarray Dataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        ds = ds.to_xarray()
    elif not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset or an MLUT")

    we, wb, ex, dl, _, _ = ibands.get_weights()
    wavelength = ds.coords["wavelength"]
    grouping = xr.DataArray(
        wb.to_numpy(),
        dims=("wavelength",),
        coords={"wavelength": wavelength},
        name="wavelength",
    )

    if extern_weights is not None:
        if isinstance(extern_weights, LUT):
            warnings.warn(
                "Passing a LUT as extern_weights is deprecated; pass an "
                "xarray DataArray instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            extern_weights = extern_weights.to_xarray()
        elif not isinstance(extern_weights, xr.DataArray):
            raise TypeError(
                "extern_weights must be an xarray DataArray or LUT"
            )

    factor = we * ex * dl if use_solar else we * dl
    norm = we.groupby(grouping).sum(dim="wavelength")
    norm_dl = (we * dl).groupby(grouping).sum(dim="wavelength")

    result = xr.Dataset(attrs=ds.attrs)
    prefixes = ("I_", "Q_", "U_", "V_", "N_", "transmission", "flux")
    for name, data_array in ds.data_vars.items():
        description = data_array.attrs.get("desc", name)
        if not any(prefix in description for prefix in prefixes):
            continue

        attrs = dict(data_array.attrs)
        weighted = data_array * factor
        if extern_weights is not None:
            weighted = weighted * extern_weights

        reduced = weighted.groupby(grouping).sum(dim="wavelength")
        reduced = reduced / (norm if integrated else norm_dl)
        reduced.attrs = attrs
        result[name] = reduced

    return result


def kdis_emission(
    ds: xr.Dataset | MLUT, ibands: KdisIbandList
) -> xr.DataArray:
    """Calculate spectrally resolved thermal emission.

    The absorption coefficient is multiplied by the Planck radiance
    averaged over each KDIS channel and returned at every atmospheric
    altitude.

    Parameters
    ----------
    ds : Dataset or MLUT
        Atmospheric optical properties containing ``OD_abs_atm``,
        ``T_atm``, ``wavelength``, and ``z_atm``.
    ibands : KdisIbandList
        KDIS internal bands used to determine channel limits and groups.

    Returns
    -------
    DataArray
        Thermal emission with dimensions ``("wavelength", "z_atm")``.
    """
    if isinstance(ds, MLUT):
        warnings.warn(
            "Passing an MLUT to kdis_emission is deprecated; pass an "
            "xarray Dataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        ds = ds.to_xarray()
    elif not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset or an MLUT")

    z_axis = ds.coords["z_atm"].to_numpy()
    wavelength_axis = ds.coords["wavelength"].to_numpy()
    t_atm = ds["T_atm"].to_numpy()

    bsgroup = ibands.get_groups()
    kabs = np.asarray(od2k(ds, "OD_abs_atm")) * 1e-3  # m-1
    z = -z_axis * 1e3  # m
    band_wmin = np.unique([ib.band.wmin for ib in ibands.l])
    band_wmax = np.unique([ib.band.wmax for ib in ibands.l])
    avg_b = np.zeros((len(band_wmin), len(z)))
    for i, (current_wmin, current_wmax) in enumerate(
        zip(band_wmin, band_wmax, strict=True)
    ):
        for j, temperature in enumerate(t_atm):
            wavelength_min, wavelength_max = (
                current_wmin * 1e-9,
                current_wmax * 1e-9,
            )  # m
            bandwidth = current_wmax - current_wmin  # nm
            avg_b[i, j] = (
                quad(
                    blackbody_radiance,
                    wavelength_min,
                    wavelength_max,
                    args=temperature,
                )[0]
                / bandwidth
            )
    emission = xr.DataArray(
        kabs * avg_b[bsgroup, :],
        dims=("wavelength", "z_atm"),
        coords={"wavelength": wavelength_axis, "z_atm": z},
        name="emission",
    )
    return emission


def kdis_avg_emission(
    ds: xr.Dataset | MLUT, ibands: KdisIbandList
) -> xr.DataArray:
    """Calculate thermal emission integrated over atmospheric altitude.

    Parameters
    ----------
    ds : Dataset or MLUT
        Atmospheric optical properties accepted by
        :func:`kdis_emission`.
    ibands : KdisIbandList
        KDIS internal bands used to determine channel groups.

    Returns
    -------
    DataArray
        Vertically integrated emission with a ``wavelength`` dimension.
    """
    if isinstance(ds, MLUT):
        ds = ds.to_xarray()
    elif not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset or an MLUT")

    emission = kdis_emission(ds, ibands)
    z_axis = emission.coords["z_atm"].to_numpy()
    emission_values = simpson(
        emission.to_numpy(),
        x=z_axis,
        axis=emission.get_axis_num("z_atm"),
    )

    return xr.DataArray(
        4 * np.pi * emission_values,
        dims=("wavelength",),
        coords={"wavelength": emission.coords["wavelength"]},
        name="emission",
    )


class KdisIband(object):
    """Represent one internal KDIS absorption band.

    Parameters
    ----------
    band : KdisBand
        Parent KDIS channel containing this internal band.
    index : int
        Zero-based index of the internal band within ``band``.

    Attributes
    ----------
    band : KdisBand
        Parent KDIS channel containing this internal band.
    index : int
        Zero-based index of the internal band within ``band``.
    w : float
        Representative wavelength of the internal band.
    weight : float
        Internal-band quadrature weight.
    ex : float
        Extraterrestrial solar irradiance associated with the band.
    dl : float
        Bandwidth of the parent KDIS channel.
    """

    def __init__(self, band: KdisBand, index: int) -> None:
        self.band = band
        self.index = index
        self.w = band.awvl[index]
        self.ex = band.solarflux
        self.dl = band.dl
        self.weight = band.awvl_weight[index]

    def calc_profile(self, prof: ProfileBase) -> NDArray[np.float64]:
        """Calculate gaseous absorption for the atmospheric profile.

        Parameters
        ----------
        prof : ProfileBase
            Atmospheric profile containing pressure, temperature, air
            density, and molecular densities for the absorbing gases.

        Returns
        -------
        numpy.ndarray
            Absorption coefficient profile in inverse kilometres,
            with one value for each altitude in ``prof``.
        """
        species = [
            "h2o",
            "co2",
            "o3",
            "no2",
            "co",
            "ch4",
            "o2",
            "n2",
            "n2o",
            "so2",
        ]
        temperature = prof.t.copy()
        pressure = prof.p.copy()
        n_molecules = 10
        profile_length = len(temperature)
        data_molecules = np.zeros(profile_length, np.float64)

        density_molecules = np.zeros((profile_length, n_molecules), np.float64)
        density_molecules[:, 0] = prof.dens_h2o[:]
        density_molecules[:, 1] = prof.dens_co2[:]
        density_molecules[:, 2] = prof.dens_o3[:]
        density_molecules[:, 3] = prof.dens_no2[:]
        density_molecules[:, 4] = prof.dens_co[:]
        density_molecules[:, 5] = prof.dens_ch4[:]
        density_molecules[:, 6] = prof.dens_o2[:]
        density_molecules[:, 7] = prof.dens_n2[:]
        density_molecules[:, 8] = prof.dens_n2o[:]
        density_molecules[:, 9] = prof.dens_so2[:]

        for species_index in range(self.band.kdis.nsp_c):
            species_name = self.band.kdis.species_c[species_index]
            molecular_index = species.index(species_name)
            coefficient_index = self.band.kdis.iki_eff_c[
                species_index, self.band.band, self.index
            ]
            coefficient_table = self.band.kdis.ki_c[
                species_index, self.band.band, coefficient_index, :, :, :
            ]
            interpolation_points = (
                self.band.kdis.p,
                self.band.kdis.t,
                self.band.kdis.c,
            )
            if self.band.kdis.c_desc == "density":
                concentration = density_molecules[:, molecular_index]
            elif self.band.kdis.c_desc == "molar_fraction":
                concentration = density_molecules[:, molecular_index] / (
                    prof.dens_air.copy()
                )
            else:
                raise ValueError(
                    "Unsupported KDIS concentration description: "
                    f"{self.band.kdis.c_desc}"
                )
            concentration[concentration > np.max(self.band.kdis.c)] = (
                np.max(self.band.kdis.c) * 0.99
            )
            concentration[concentration < np.min(self.band.kdis.c)] = (
                np.min(self.band.kdis.c) * 1.01
            )
            pressure[pressure > np.max(self.band.kdis.p)] = (
                np.max(self.band.kdis.p) * 0.99
            )
            pressure[pressure < np.min(self.band.kdis.p)] = (
                np.min(self.band.kdis.p) * 1.01
            )
            temperature[temperature > np.max(self.band.kdis.t)] = (
                np.max(self.band.kdis.t) * 0.99
            )
            temperature[temperature < np.min(self.band.kdis.t)] = (
                np.min(self.band.kdis.t) * 1.01
            )
            interpolation_values = np.concatenate(
                (
                    np.array([pressure]),
                    np.array([temperature]),
                    np.array([concentration]),
                ),
                axis=0,
            ).T
            data_molecules += (
                interpn(
                    interpolation_points,
                    coefficient_table,
                    interpolation_values,
                )
                * density_molecules[:, molecular_index]
            )

        for species_index in range(self.band.kdis.nsp):
            species_name = self.band.kdis.species[species_index]
            molecular_index = species.index(species_name)
            coefficient_index = self.band.kdis.iki_eff[
                species_index, self.band.band, self.index
            ]
            coefficient_table = self.band.kdis.ki[
                species_index, self.band.band, coefficient_index, :, :
            ]
            data_molecules += (
                interp2(
                    self.band.kdis.p,
                    self.band.kdis.t,
                    np.squeeze(coefficient_table),
                    pressure,
                    temperature,
                )
                * density_molecules[:, molecular_index]
            )

        return data_molecules * 1e5


class KdisBand(object):
    """Represent a KDIS sensor channel.

    Parameters
    ----------
    kdis : Kdis
        Parent KDIS dataset.
    band_index : int
        Zero-based index of the sensor channel.

    Attributes
    ----------
    kdis : Kdis
        Parent KDIS dataset containing this channel.
    band : int
        Zero-based index of this channel in the parent KDIS dataset.
    nband : int
        Number of internal bands in this channel.
    awvl : list of float
        Representative wavelengths of the internal bands in
        nanometres.
    awvl_weight : numpy.ndarray
        Quadrature weights of the internal bands.
    solarflux : float
        Extraterrestrial solar irradiance per nanometre for this
        channel.
    dl : float
        Wavelength bandwidth of this channel in nanometres.
    w : float
        Central wavelength of this channel in nanometres.
    wmin, wmax : float
        Lower and upper wavelength limits of this channel in nanometres.
    """

    def __init__(self, kdis: Kdis, band: int) -> None:
        self.kdis = kdis
        self.band = band
        self.w = kdis.wvlband[0, self.band]
        self.wmin = kdis.wvlband[1, self.band]
        self.wmax = kdis.wvlband[2, self.band]
        self.nband = kdis.nai_eff[self.band]
        self.awvl = [self.w] * self.nband
        self.awvl_weight = kdis.ai_eff[self.band, : self.nband]
        self.dl = kdis.wvlband[2, self.band] - kdis.wvlband[1, self.band]
        self.solarflux = kdis.solarflux[self.band] / self.dl

    def iband(self, index: int) -> KdisIband:
        """Return an internal band by its zero-based index.

        Parameters
        ----------
        index : int
            Zero-based index within this sensor channel.

        Returns
        -------
        KdisIband
            The selected internal band.
        """
        return KdisIband(self, index)

    def ibands(self) -> Iterator[KdisIband]:
        """Iterate over the internal bands in this sensor channel.

        Yields
        ------
        KdisIband
            Each internal band in increasing index order.
        """
        for index in range(self.nband):
            yield self.iband(index)


class Kdis(object):
    """Read and expose a KDIS correlated-k definition.

    Parameters
    ----------
    model : str
        KDIS model name.
    dir_data : path-like, optional
        Directory containing the model files. The auxiliary KDIS
        directory is used when omitted.
    format : str, optional
        Input format, either ``"ascii"`` or ``"h5"``. When omitted,
        the format is detected from the available files.

    Attributes
    ----------
    model : str
        KDIS model name.
    wvlband : numpy.ndarray
        Central, lower, and upper wavelengths for each channel in nm.
    solarflux : numpy.ndarray
        Solar flux integrated over each channel.
    nsp, nsp_c : int
        Number of ordinary and concentration-dependent absorbing
        species.
    species, species_c : list of str
        Names of ordinary and concentration-dependent species.
    p, t, c : numpy.ndarray
        Pressure, temperature, and concentration lookup axes.
    nai_eff : numpy.ndarray
        Number of effective quadrature coefficients for each channel.
    """

    def __init__(
        self,
        model: str,
        dir_data: PathType = "",
        format: str | None = None,
    ) -> None:

        # Read the entire K-distribution definition from files.
        #
        # Selection of the desired KDIS band or absorbing gases must be
        # done later while setting up the artdeco variables.
        # If dir_data is not specified, the standard directory is
        # assumed to be auxdata/kdis.

        self.model = model

        dir_data = Path(dir_data)
        if dir_data == Path("."):
            dir_data = dir_kdis / model

        def is_sorted(values: NDArray[np.floating]) -> bool:
            """Return whether values are monotonically non-decreasing."""
            return bool(np.all(values[:-1] <= values[1:]))

        if format is None:
            if len(glob.glob(str(dir_data) + "/*.h5")) > 0:
                format = "h5"
            else:
                format = "ascii"
        else:
            warnings.simplefilter("always", DeprecationWarning)
            warn_message = (
                "\nThe key argument 'format' is now useless and deprecated "
                "as of SMART-G 1.0.0,\n"
                "and will be removed in one of the next release."
            )
            warnings.warn(warn_message, DeprecationWarning, stacklevel=2)

        if format == "ascii":
            fname = dir_data / f"kdis_{model}_def.dat"
            if not fname.is_file():
                print("(kdis_coef) ERROR")
                print("            Missing file:", fname)
                sys.exit()
            definition_file = open(fname, "r")
            skip_comment(definition_file)
            line = definition_file.readline()
            self.nmaxai = int(line.split()[0])
            skip_comment(definition_file)
            line = definition_file.readline()
            self.nsp_tot = int(line.split()[0])
            self.nsp = 0
            self.fcont = []
            self.species = []
            self.nsp_c = 0
            self.fcont_c = []
            self.species_c = []
            skip_comment(definition_file)
            for _species_index in range(self.nsp_tot):
                line = definition_file.readline()
                if int(line.split()[1]) == 0:
                    self.nsp = self.nsp + 1
                    self.species.append(line.split()[0])
                    self.fcont.append(float(line.split()[2]))
                elif int(line.split()[1]) == 1:
                    self.nsp_c = self.nsp_c + 1
                    self.species_c.append(line.split()[0])
                    self.fcont_c.append(float(line.split()[2]))
            self.fcont = np.array(self.fcont)
            self.fcont_c = np.array(self.fcont_c)
            skip_comment(definition_file)
            line = definition_file.readline()
            self.nwvl = int(line.split()[0])
            self.wvlband = np.zeros((3, self.nwvl))
            skip_comment(definition_file)
            for wavelength_index in range(self.nwvl):
                line = definition_file.readline()
                self.wvlband[0, wavelength_index] = (
                    float(line.split()[1]) * 1e3
                )
                self.wvlband[1, wavelength_index] = (
                    float(line.split()[2]) * 1e3
                )
                self.wvlband[2, wavelength_index] = (
                    float(line.split()[3]) * 1e3
                )
                if wavelength_index > 0:
                    if (
                        self.wvlband[0, wavelength_index]
                        < self.wvlband[0, wavelength_index - 1]
                    ):
                        print(" kdis_coeff ERROR")
                        print(
                            "            wavelengths must be sorted in increasing order"
                        )
                        sys.exit()
            skip_comment(definition_file)
            line = definition_file.readline()
            skip_comment(definition_file)
            self.np = int(line.split()[0])
            self.p = np.zeros(self.np)
            for pressure_index in range(self.np):
                line = definition_file.readline()
                self.p[pressure_index] = float(line.split()[0])
                if pressure_index > 0:
                    if self.p[pressure_index] < self.p[pressure_index - 1]:
                        print(" kdis_coeff ERROR")
                        print(
                            "            pressure must be sorted in increasing order"
                        )
                        sys.exit()
            skip_comment(definition_file)
            line = definition_file.readline()
            skip_comment(definition_file)
            self.nt = int(line.split()[0])
            self.t = np.zeros(self.nt)
            for temperature_index in range(self.nt):
                line = definition_file.readline()
                self.t[temperature_index] = float(line.split()[0])
                if temperature_index > 0:
                    if (
                        self.t[temperature_index]
                        < self.t[temperature_index - 1]
                    ):
                        print(" kdis_coeff ERROR")
                        print(
                            "            temperature must be sorted in increasing order"
                        )
                        sys.exit()
            if self.nsp_c > 0:
                skip_comment(definition_file)
                line = definition_file.readline()
                skip_comment(definition_file)
                self.nc = int(line.split()[0])
                self.c = np.zeros(self.nc)
                for concentration_index in range(self.nc):
                    line = definition_file.readline()
                    self.c[concentration_index] = float(line.split()[0])
                    if concentration_index > 0:
                        if (
                            self.c[concentration_index]
                            < self.c[concentration_index - 1]
                        ):
                            print(" kdis_coeff ERROR")
                            print(
                                "            concentration must be sorted in increasing order"
                            )
                            sys.exit()
            definition_file.close()
            if self.nsp > 0:
                self.nai = np.zeros((self.nsp, self.nwvl), dtype="int")
                self.ki = np.zeros(
                    (self.nsp, self.nwvl, self.nmaxai, self.np, self.nt)
                )
                self.ai = np.zeros((self.nsp, self.nwvl, self.nmaxai))
            if self.nsp_c > 0:
                self.nai_c = np.zeros((self.nsp_c, self.nwvl), dtype="int")
                self.ki_c = np.zeros(
                    (
                        self.nsp_c,
                        self.nwvl,
                        self.nmaxai,
                        self.np,
                        self.nt,
                        self.nc,
                    )
                )
                self.ai_c = np.zeros((self.nsp_c, self.nwvl, self.nmaxai))
            for species_index in range(self.nsp):
                fname = (
                    dir_data
                    / f"kdis_{model}_{self.species[species_index]}.dat"
                )
                if not fname.is_file():
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", fname)
                    sys.exit()
                species_file = open(fname, "r")
                skip_comment(species_file)
                for wavelength_index in range(self.nwvl):
                    line = species_file.readline()
                    self.nai[species_index, wavelength_index] = int(
                        line.split()[1]
                    )
                for wavelength_index in range(self.nwvl):
                    if self.nai[species_index, wavelength_index] > 1:
                        skip_comment(species_file)
                        line = species_file.readline()
                        for coefficient_index in range(
                            self.nai[species_index, wavelength_index]
                        ):
                            self.ai[
                                species_index,
                                wavelength_index,
                                coefficient_index,
                            ] = float(line.split()[coefficient_index])
                        for temperature_index in range(self.nt):
                            for pressure_index in range(self.np):
                                line = species_file.readline()
                                for coefficient_index in range(
                                    self.nai[species_index, wavelength_index]
                                ):
                                    self.ki[
                                        species_index,
                                        wavelength_index,
                                        coefficient_index,
                                        pressure_index,
                                        temperature_index,
                                    ] = float(line.split()[coefficient_index])
                species_file.close()
            if self.nsp_c > 0:
                self.c_desc = "density"
            else:
                self.c_desc = "none"
            for species_index in range(self.nsp_c):
                fname = (
                    dir_data
                    / f"kdis_{model}_{self.species_c[species_index]}.dat"
                )
                if not fname.is_file():
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", fname)
                    sys.exit()
                species_file = open(fname, "r")
                skip_comment(species_file)
                for wavelength_index in range(self.nwvl):
                    line = species_file.readline()
                    self.nai_c[species_index, wavelength_index] = int(
                        line.split()[1]
                    )
                for wavelength_index in range(self.nwvl):
                    if self.nai_c[species_index, wavelength_index] > 1:
                        skip_comment(species_file)
                        line = species_file.readline()
                        for coefficient_index in range(
                            self.nai_c[species_index, wavelength_index]
                        ):
                            self.ai_c[
                                species_index,
                                wavelength_index,
                                coefficient_index,
                            ] = float(line.split()[coefficient_index])
                        for concentration_index in range(self.nc):
                            for temperature_index in range(self.nt):
                                for pressure_index in range(self.np):
                                    line = species_file.readline()
                                    for coefficient_index in range(
                                        self.nai_c[
                                            species_index, wavelength_index
                                        ]
                                    ):
                                        self.ki_c[
                                            species_index,
                                            wavelength_index,
                                            coefficient_index,
                                            pressure_index,
                                            temperature_index,
                                            concentration_index,
                                        ] = float(
                                            line.split()[coefficient_index]
                                        )
                species_file.close()

            fname = dir_data / f"kdis_{model}_solarflux.dat"
            if not fname.is_file():
                fname = dir_data / f"solrad_kdis_{model}_thuillier2003.dat"
                if not fname.is_file():
                    print("(kdis_coef) ERROR")
                    print("            Missing file:", fname)
                    sys.exit()
            solar_file = open(fname, "r")
            skip_comment(solar_file)
            line = solar_file.readline()
            skip_comment(solar_file)
            line = solar_file.readline()
            band_count = float(line.split()[0])
            if band_count != self.nwvl:
                print(" solar flux and kdis have uncompatible band number")
                sys.exit()
            skip_comment(solar_file)
            self.solarflux = np.zeros(self.nwvl)
            skip_comment(solar_file)
            for wavelength_index in range(self.nwvl):
                line = solar_file.readline()
                self.solarflux[wavelength_index] = float(line.split()[0])
            solar_file.close()

        elif format in ["h5", "hdf5"]:
            fname = dir_data / f"kdis_{model}.h5"
            hdf5_file = h5py.File(fname, "r")

            def h5_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
                return cast(h5py.Dataset, group[name])

            definition_group = cast(h5py.Group, hdf5_file["def"])
            coefficient_group = cast(h5py.Group, hdf5_file["coeff"])
            self.nmaxai = int(
                np.asarray(h5_dataset(definition_group, "maxnai")[()])
            )
            species_names = list(coefficient_group.keys())
            self.nsp_tot = len(species_names)
            self.nsp = 0
            self.fcont = []
            self.species = []
            self.nsp_c = 0
            self.fcont_c = []
            self.species_c = []
            for species_name in species_names:
                species_group = cast(
                    h5py.Group, coefficient_group[species_name]
                )
                if "rho_dep" in species_group.attrs:
                    rho_dependent = bool(species_group.attrs["rho_dep"])
                else:
                    rho_dependent = False
                if not rho_dependent:
                    self.nsp = self.nsp + 1
                    self.species.append(species_name)
                    self.fcont.append(species_group.attrs["add_continuum"])
                else:
                    self.nsp_c = self.nsp_c + 1
                    self.species_c.append(species_name)
                    self.fcont_c.append(species_group.attrs["add_continuum"])
            self.fcont = np.array(self.fcont)
            self.fcont_c = np.array(self.fcont_c)
            self.nwvl = len(h5_dataset(definition_group, "central_wvl"))
            self.wvlband = np.zeros((3, self.nwvl))
            self.wvlband[0, :] = (
                np.asarray(h5_dataset(definition_group, "central_wvl")[()])
                * 1e3
            )
            self.wvlband[1, :] = (
                np.asarray(h5_dataset(definition_group, "min_wvl")[()]) * 1e3
            )
            self.wvlband[2, :] = (
                np.asarray(h5_dataset(definition_group, "max_wvl")[()]) * 1e3
            )
            self.p = np.asarray(h5_dataset(definition_group, "pressure")[()])
            self.t = np.asarray(
                h5_dataset(definition_group, "temperature")[()]
            )
            self.np = len(self.p)
            self.nt = len(self.t)
            if self.nsp_c > 0:
                rho_dataset = h5_dataset(definition_group, "rho")
                self.c = np.asarray(rho_dataset[()])
                concentration_description = rho_dataset.attrs["desc"]
                if isinstance(concentration_description, bytes):
                    self.c_desc = concentration_description.decode()
                else:
                    self.c_desc = str(concentration_description)
                self.nc = len(self.c)
                if not is_sorted(self.c):
                    print(" kdis_coeff ERROR")
                    print(
                        "            concentration must be sorted in increasing order"
                    )
                    sys.exit()
            else:
                self.c_desc = "none"
            if not is_sorted(self.wvlband[0, :]):
                print(" kdis_coeff ERROR")
                print(
                    "            (h5 format) read NOT implemented for concentration dependent species"
                )
                sys.exit()
            if not is_sorted(self.p):
                print(" kdis_coeff ERROR")
                print(
                    "            pressure must be sorted in increasing order"
                )
                sys.exit()
            if not is_sorted(self.t):
                print(" kdis_coeff ERROR")
                print(
                    "            temperature must be sorted in increasing order"
                )
                sys.exit()
            if self.nsp > 0:
                self.nai = np.zeros((self.nsp, self.nwvl), dtype="int")
                self.ki = np.zeros(
                    (self.nsp, self.nwvl, self.nmaxai, self.np, self.nt)
                )
                self.ai = np.zeros((self.nsp, self.nwvl, self.nmaxai))
                for species_index, species_name in enumerate(self.species):
                    species_group = cast(
                        h5py.Group, coefficient_group[species_name]
                    )
                    nai_dataset = h5_dataset(species_group, "nai")
                    ki_dataset = h5_dataset(species_group, "ki")
                    ai_dataset = h5_dataset(species_group, "ai")
                    self.nai[species_index, :] = np.asarray(nai_dataset[()])
                    coefficient_count = int(
                        np.nanmax(self.nai[species_index, :])
                    )
                    self.ki[species_index, :, 0:coefficient_count, :, :] = (
                        np.copy(ki_dataset[:, 0:coefficient_count, :, :])
                    )
                    self.ai[species_index, :, 0:coefficient_count] = np.copy(
                        ai_dataset[..., 0:coefficient_count]
                    )
            if self.nsp_c > 0:
                self.nai_c = np.zeros((self.nsp_c, self.nwvl), dtype="int")
                self.ki_c = np.zeros(
                    (
                        self.nsp_c,
                        self.nwvl,
                        self.nmaxai,
                        self.np,
                        self.nt,
                        self.nc,
                    )
                )
                self.ai_c = np.zeros((self.nsp_c, self.nwvl, self.nmaxai))
                for species_index, species_name in enumerate(self.species_c):
                    species_group = cast(
                        h5py.Group, coefficient_group[species_name]
                    )
                    nai_dataset = h5_dataset(species_group, "nai")
                    ki_dataset = h5_dataset(species_group, "ki")
                    ai_dataset = h5_dataset(species_group, "ai")
                    self.nai_c[species_index, :] = np.asarray(nai_dataset[()])
                    coefficient_count = int(
                        np.nanmax(self.nai_c[species_index, :])
                    )
                    self.ki_c[species_index, :, 0:coefficient_count, :, :] = (
                        np.copy(ki_dataset[:, 0:coefficient_count, :, :])
                    )
                    self.ai_c[species_index, :, 0:coefficient_count] = np.copy(
                        ai_dataset[..., 0:coefficient_count]
                    )
            # solar flux
            solar_group = hdf5_file.require_group("solrad")
            solar_dataset = h5_dataset(solar_group, "solrad")
            self.solarflux = solar_dataset[:]
            hdf5_file.close()

            for species_index, species_name in enumerate(self.species):
                self.species[species_index] = species_name.lower()
            for species_index, species_name in enumerate(self.species_c):
                self.species_c[species_index] = species_name.lower()

        # support for multi species
        if (self.nsp > 0) and (self.nsp_c > 0):
            self.nai_eff = np.prod(
                np.append(self.nai, self.nai_c, axis=0), axis=0
            )
        elif self.nsp > 0:
            self.nai_eff = np.prod(self.nai, axis=0)
        elif self.nsp_c > 0:
            self.nai_eff = np.prod(self.nai_c, axis=0)
        self.nmaxai_eff = np.max(self.nai_eff)
        self.ai_eff = np.zeros((self.nwvl, self.nmaxai_eff))
        if self.nsp > 0:
            self.iki_eff = np.zeros(
                (self.nsp, self.nwvl, self.nmaxai_eff), dtype="int"
            )
        if self.nsp_c > 0:
            self.iki_eff_c = np.zeros(
                (self.nsp_c, self.nwvl, self.nmaxai_eff), dtype="int"
            )
        for wavelength_index in range(self.nwvl):
            effective_index = 0
            coefficient_index_ranges = []
            for species_index in range(self.nsp):
                coefficient_index_ranges.append(
                    range(self.nai[species_index, wavelength_index])
                )
            for species_index in range(self.nsp_c):
                coefficient_index_ranges.append(
                    range(self.nai_c[species_index, wavelength_index])
                )
            for coefficient_indices in product(*coefficient_index_ranges):
                if self.nsp > 0:
                    self.iki_eff[:, wavelength_index, effective_index] = (
                        coefficient_indices[0 : self.nsp]
                    )
                if self.nsp_c > 0:
                    self.iki_eff_c[:, wavelength_index, effective_index] = (
                        coefficient_indices[self.nsp : self.nsp + self.nsp_c]
                    )
                self.ai_eff[wavelength_index, effective_index] = 1.0
                for species_index in range(self.nsp):
                    coefficient_index = self.iki_eff[
                        species_index, wavelength_index, effective_index
                    ]
                    if (
                        self.nai[species_index, wavelength_index] >= 1
                        and self.ai[
                            species_index, wavelength_index, coefficient_index
                        ]
                        != 0.0
                    ):
                        self.ai_eff[wavelength_index, effective_index] *= (
                            self.ai[
                                species_index,
                                wavelength_index,
                                coefficient_index,
                            ]
                        )
                for species_index in range(self.nsp_c):
                    coefficient_index = self.iki_eff_c[
                        species_index, wavelength_index, effective_index
                    ]
                    if (
                        self.nai_c[species_index, wavelength_index] >= 1
                        and self.ai_c[
                            species_index, wavelength_index, coefficient_index
                        ]
                        != 0.0
                    ):
                        self.ai_eff[wavelength_index, effective_index] *= (
                            self.ai_c[
                                species_index,
                                wavelength_index,
                                coefficient_index,
                            ]
                        )
                effective_index += 1

    def nbands(self) -> int:
        """Return the number of KDIS channels.

        Returns
        -------
        int
            Number of channels in the KDIS definition.
        """
        return self.nwvl

    def band(self, band: int) -> KdisBand:
        """Return a KDIS channel by its zero-based index.

        Parameters
        ----------
        band : int
            Zero-based channel index.

        Returns
        -------
        KdisBand
            The selected KDIS channel.
        """
        return KdisBand(self, band)

    def bands(self) -> Iterator[KdisBand]:
        """Iterate over all KDIS channels in file order.

        Yields
        ------
        KdisBand
            Each available KDIS channel.
        """
        for band_index in range(self.nbands()):
            yield self.band(band_index)

    def to_smartg(
        self,
        include: str = "",
        lmin: NumericArrayLike = -np.inf,
        lmax: NumericArrayLike = np.inf,
        band_indices: Sequence[int] | None = None,
    ) -> KdisIbandList:
        """Select internal KDIS bands for use with ``Smartg.run``.

        Parameters
        ----------
        include : str, optional
            Retained for API compatibility. KDIS channels have no names,
            so this value does not filter the selection.
        lmin, lmax : scalar or array-like, optional
            Lower and upper wavelength limits in nanometres. Multiple
            intervals can be supplied as matching sequences. Defaults
            are negative and positive infinity, respectively.
        band_indices : sequence of int, optional
            Explicit channel indices to consider. If None, all channels
            are considered.

        Returns
        -------
        KdisIbandList
                Selected internal bands sorted by representative
                wavelength.

        Raises
        ------
        ValueError
            If ``lmin`` and ``lmax`` contain different numbers of
            intervals.
        AssertionError
            If no internal bands match the selection.
        """
        internal_bands = []
        if band_indices is None:
            bands = self.bands()
        else:
            bands = [self.band(band_index) for band_index in band_indices]

        lmin_values = np.atleast_1d(lmin)
        lmax_values = np.atleast_1d(lmax)
        if len(lmin_values) != len(lmax_values):
            raise ValueError("lmin and lmax must contain matching intervals")
        for band in bands:
            for interval_index in range(len(lmin_values)):
                if (band.wmin >= lmin_values[interval_index]) and (
                    band.wmax <= lmax_values[interval_index]
                ):
                    for internal_band in band.ibands():
                        internal_bands.append(internal_band)

        assert len(internal_bands) != 0

        return KdisIbandList(
            sorted(internal_bands, key=lambda internal_band: internal_band.w)
        )

    def get_weights(
        self,
    ) -> tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ]:
        """Return all internal-band weights and channel metadata.

        Returns
        -------
        tuple of DataArray
            Internal-band weights, channel wavelengths, solar
            irradiance, bandwidth, weight sums, and bandwidth-weighted
            sums.
        """
        return KdisIbandList(
            [
                internal_band
                for band in self.bands()
                for internal_band in band.ibands()
            ]
        ).get_weights()


class KdisIbandList(object):
    """Store a selected list of internal KDIS bands.

    Parameters
    ----------
    ibands : sequence of KdisIband
        Internal bands included in the list.

    Attributes
    ----------
    l : sequence of KdisIband
        Internal bands included in the list, in wavelength order when
        returned by :meth:`Kdis.to_smartg`.
    """

    def __init__(self, ibands: Sequence[KdisIband]) -> None:
        self.l = ibands

    def get_weights(
        self,
    ) -> tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ]:
        """Return channel weights and metadata as xarray DataArrays.

        Returns
        -------
        tuple of DataArray
            Internal-band weights, channel wavelengths, solar
            irradiance, bandwidth, weight sums, and bandwidth-weighted
            sums.
        """
        wi_l = [internal_band.w for internal_band in self.l]
        we_l = [internal_band.weight for internal_band in self.l]
        ex_l = []
        dl_l = []
        for internal_band in self.l:
            ex_l.append(internal_band.ex)
            dl_l.append(internal_band.dl)

        wi_arr = np.array(wi_l, dtype=np.float32)
        wb = xr.DataArray(
            np.array(wi_l, dtype=np.float32),
            dims=["wavelength"],
            coords={"wavelength": wi_arr},
            name="wavelength",
            attrs={"desc": "wavelength central band"},
        )
        we = xr.DataArray(
            np.array(we_l),
            dims=["wavelength"],
            coords={"wavelength": wi_arr},
            name="weight",
            attrs={"desc": "Weight"},
        )
        ex = xr.DataArray(
            np.array(ex_l),
            dims=["wavelength"],
            coords={"wavelength": wi_arr},
            name="solarflux",
            attrs={"desc": "E0"},
        )
        dl = xr.DataArray(
            np.array(dl_l),
            dims=["wavelength"],
            coords={"wavelength": wi_arr},
            name="bandwidth",
            attrs={"desc": "Dlambda"},
        )
        norm_dl = (we * dl).groupby("wavelength").sum(dim="wavelength")
        norm = we.groupby("wavelength").sum(dim="wavelength")
        return we, wb, ex, dl, norm, norm_dl

    def get_groups(self) -> np.ndarray:
        """Return the zero-based channel group for each internal band.

        Returns
        -------
        numpy.ndarray
            Channel indices shifted so that the first selected channel
            is group zero.
        """
        bsgroup = []
        for iband in self.l:
            bsgroup.append(iband.band.band)
        bsgroup = np.array(bsgroup)
        return bsgroup - bsgroup[0]


def skip_comment(file_handle: TextIO) -> None:
    """Skip consecutive comment lines and rewind to the first data line."""
    while True:
        position = file_handle.tell()
        if not file_handle.readline().strip().startswith("#"):
            break
    file_handle.seek(position, 0)
