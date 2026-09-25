"""REPTRAN representative-wavelength absorption parameterization.

REPTRAN (representative wavelengths absorption parameterization)
provides a compact representation of molecular absorption for
satellite channels and spectral bands. High-resolution absorption
information is reduced to a set of representative wavelengths and
associated weights for each channel or band, allowing radiative-transfer
calculations to retain channel-integrated absorption while using fewer
spectral points.

This module loads the resulting REPTRAN correlated-k data and
molecular cross-section lookup tables for SMART-G simulations. It
represents sensor channels and internal absorption bands, calculates
gaseous absorption and thermal emission for atmospheric profiles, and
reduces spectrally resolved results to sensor-channel quantities.

References
----------
.. [1] J. Gasteiger, C. Emde, B. Mayer, R. Buras, S. A. Buehler,
    and O. Lemke, "Representative wavelengths absorption
    parameterization applied to satellite channels and spectral bands,"
    *Journal of Quantitative Spectroscopy and Radiative Transfer*, vol.
    148, pp. 99-115, 2014.
    https://doi.org/10.1016/j.jqsrt.2014.06.024

Key Classes
-----------
Reptran
    Read and expose a REPTRAN correlated-k file.
ReptranBand
    Represent a REPTRAN sensor channel.
ReptranIband
    Represent one internal REPTRAN absorption band.
ReptranIbandList
    Store a selected list of internal REPTRAN bands.
ReadCrs
    Read a REPTRAN molecular cross-section lookup table.

Key Functions
-------------
reduce_reptran
    Reduce spectral results to REPTRAN channel values.
reptran_emission
    Calculate spectrally resolved thermal emission.
reptran_avg_emission
    Calculate thermal emission integrated over atmospheric
    altitude.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from luts.luts import LUT, MLUT
from scipy.integrate import quad, simpson
from scipy.interpolate import make_interp_spline

from smartg.atmosphere import blackbody_radiance, od2k
from smartg.config import DIR_AUXDATA
from smartg.interp import interp2, interp3
from smartg.typing import NumericArrayLike, PathType

if TYPE_CHECKING:
    from smartg.atmosphere import ProfileBase

dir_reptran = DIR_AUXDATA / "reptran"


def _channel_name(name: str | bytes) -> str:
    """Return a REPTRAN channel name as text without its spaces."""
    if isinstance(name, bytes):
        name = name.decode()
    return name.replace(" ", "")


def reduce_reptran(
    ds: xr.Dataset | MLUT,
    ibands: ReptranIbandList,
    use_solar: bool = False,
    integrated: bool = False,
    extern_weights: LUT | xr.DataArray | None = None,
) -> xr.Dataset:
    """Reduce spectral results to REPTRAN channel values.

    The spectral variables selected from ``ds`` are weighted by the
    internal-band weights and grouped by their central channel
    wavelength.

    Parameters
    ----------
    ds : Dataset or MLUT
        Spectral SMART-G results containing a ``wavelength`` coordinate.
    ibands : ReptranIbandList
        REPTRAN internal bands providing weights, channel wavelengths,
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
        Channel-reduced variables whose names contain an accepted output
        prefix, with the source variable attributes preserved.
    """
    if isinstance(ds, MLUT):
        warnings.warn(
            "Passing an MLUT to reduce_reptran is deprecated; pass an "
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
    prefixes = ("I_", "Q_", "U_", "V_", "transmission", "flux")
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


def reptran_emission(
    ds: xr.Dataset | MLUT, ibands: ReptranIbandList
) -> xr.DataArray:
    """Calculate spectrally resolved thermal emission.

    The absorption coefficient is multiplied by the Planck radiance
    averaged over each REPTRAN channel and returned at every atmospheric
    altitude.

    Parameters
    ----------
    ds : Dataset or MLUT
        Atmospheric optical properties containing ``OD_abs_atm``,
        ``T_atm``, ``wavelength``, and ``z_atm``.
    ibands : ReptranIbandList
        REPTRAN internal bands used to determine channel limits and
        groups.

    Returns
    -------
    DataArray
        Thermal emission with dimensions ``("wavelength", "z_atm")``.
    """
    if isinstance(ds, MLUT):
        warnings.warn(
            "Passing an MLUT to reduce_reptran is deprecated; pass an "
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


def reptran_avg_emission(
    ds: xr.Dataset | MLUT, ibands: ReptranIbandList
) -> xr.DataArray:
    """Calculate thermal emission integrated over atmospheric altitude.

    Parameters
    ----------
    ds : Dataset or MLUT
        Atmospheric optical properties accepted by
        :func:`reptran_emission`.
    ibands : ReptranIbandList
        REPTRAN internal bands used to determine channel groups.

    Returns
    -------
    DataArray
        Vertically integrated emission with a ``wavelength`` dimension.
    """
    if isinstance(ds, MLUT):
        ds = ds.to_xarray()
    elif not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray Dataset or an MLUT")

    emission = reptran_emission(ds, ibands)
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


class ReptranIband:
    """Represent one internal REPTRAN absorption band.

    Parameters
    ----------
    band : ReptranBand
        Parent sensor band containing this internal band.
    index : int
        Zero-based index of the internal band within ``band``.

    Attributes
    ----------
    band : ReptranBand
        Parent sensor channel containing this internal band.
    index : int
        Zero-based index of this internal band within ``band``.
    w : float
        Representative wavelength of the internal band.
    weight : float
        Internal-band quadrature weight.
    extra : float
        Extraterrestrial solar irradiance at ``w``.
    crs_source : numpy.ndarray
        Flags indicating which molecular species contribute to
        absorption.
    species : list of str
        Molecular species corresponding to the entries in
        ``crs_source``.
    fname : pathlib.Path
        REPTRAN file associated with the parent sensor channel.
    """

    def __init__(self, band: ReptranBand, index: int) -> None:

        self.band = band  # parent ReptranBand
        self.index = index  # internal band index
        self.w = band.awvl[index]  # band wavelength
        self._iband = band._iband[index]
        self.weight = band.awvl_weight[index]  # weight
        self.extra = band.aextra[index]  # solar irradiance
        self.crs_source = band.across_section_source[
            index, :
        ]  # table of absorbing gases
        self.species = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2", "N2"]
        self.fname = Path(band.fname)

    def calc_profile(self, prof: ProfileBase) -> np.ndarray:
        """Calculate gaseous absorption for the atmospheric profile.

        Parameters
        ----------
        prof : ProfileBase
            Atmospheric profile containing pressure, temperature, air
            density, and the densities of the absorbing gases.

        Returns
        -------
        numpy.ndarray
            Absorption coefficient profile in inverse kilometres, with
            one value for each altitude in ``prof``.
        """
        n_molecules = 8
        temperature = prof.t
        pressure = prof.p
        profile_length = len(temperature)

        density_molecules = np.zeros((profile_length, n_molecules), np.float64)
        density_molecules[:, 0] = prof.dens_h2o
        density_molecules[:, 1] = prof.dens_co2
        density_molecules[:, 2] = prof.dens_o3
        density_molecules[:, 3] = prof.dens_n2o
        density_molecules[:, 4] = prof.dens_co
        density_molecules[:, 5] = prof.dens_ch4
        density_molecules[:, 6] = prof.dens_o2
        density_molecules[:, 7] = prof.dens_n2

        x_h2o = prof.dens_h2o / prof.dens_air

        data_molecules = np.zeros(profile_length, np.float64)

        assert len(temperature) == len(pressure)

        # for each gas
        for molecule_index in np.arange(n_molecules):
            # if the gas absorbs at this wavelength
            if self.crs_source[molecule_index] == 1:
                # fetch the absorption LUT
                crs_filename = self.fname.with_suffix(
                    ""
                )  # supprime l'extension
                crs_filename = crs_filename.with_name(
                    f"{crs_filename.name}.lookup."
                    f"{self.species[molecule_index]}"
                )
                crs_mol = ReadCrs(crs_filename, self._iband)

                # interpolate the reference vertical temperature
                # profile of the LUT
                # k=1: linear interpolation; BSpline extrapolates
                # linearly
                # beyond the data range by default.
                pressure_order = np.argsort(crs_mol.pressure)
                reference_temperature = make_interp_spline(
                    crs_mol.pressure[pressure_order],
                    crs_mol.t_ref[pressure_order],
                    k=1,
                )

                # temperature departure from the reference profile
                # (the reference P is in Pa, the AFGL P in hPa)
                delta_temperature = temperature - reference_temperature(
                    pressure * 100
                )

                if molecule_index == 0:  # si h2o
                    # interpolate the absorption LUT in pressure,
                    # temperature departure and H2O vmr, multiply by
                    # the density; REPTRAN computes with LUT in
                    # 10^(-20) m2, converted to km-1
                    data_molecules += (
                        interp3(
                            crs_mol.t_pert,
                            crs_mol.vmrs,
                            crs_mol.pressure,
                            crs_mol.xsec,
                            delta_temperature,
                            x_h2o,
                            pressure * 100,
                        )
                        * density_molecules[:, molecule_index]
                        * 1e-11
                    )
                else:
                    tab = crs_mol.xsec
                    # interpolate the absorption LUT in pressure and
                    # temperature departure, multiply by the density;
                    # REPTRAN computes with LUT in 10^(-20) m2,
                    # converted to km-1
                    data_molecules += (
                        interp2(
                            crs_mol.t_pert,
                            crs_mol.pressure,
                            np.squeeze(tab),
                            delta_temperature,
                            pressure * 100,
                        )
                        * density_molecules[:, molecule_index]
                        * 1e-11
                    )

        return data_molecules


class ReptranBand:
    """Represent a REPTRAN sensor channel.

    Parameters
    ----------
    reptran : Reptran
        Parent REPTRAN dataset.
    band : int
        Zero-based index of the sensor channel.

    Attributes
    ----------
    band : int
        Zero-based index of this sensor channel in the parent REPTRAN
        file.
    nband : int
        Number of internal bands in this sensor channel.
    awvl : numpy.ndarray
        Representative wavelengths of the internal bands in nanometres.
    awvl_weight : numpy.ndarray
        Quadrature weights of the internal bands.
    aextra : numpy.ndarray
        Extraterrestrial solar irradiance at the internal-band
        wavelengths.
    across_section_source : numpy.ndarray
        Molecular absorption-source flags for each internal band.
    name : str
        Sensor channel name.
    fname : pathlib.Path
        REPTRAN file associated with this sensor channel.
    w : float
        Mean internal-band wavelength, used when channel limits cannot
        be parsed from ``name``.
    r_int : float
        Integral of the channel response function, over wavelength in
        nanometres for a solar file, over wavenumber in inverse
        centimetres for a thermal file.
    dl : float
        Channel bandwidth in nanometres. It is ``r_int`` for a solar
        file. For a thermal file it is ``wmax - wmin`` when the limits
        are parsed from ``name``, and otherwise ``r_int`` converted
        with the weighted mean of the squared internal-band
        wavelengths, since dlambda = lambda**2 dnu / 1e7 (nm, cm-1).
    wmin, wmax : float
        Lower and upper wavelength limits of the channel in nanometres.
        When they cannot be parsed from ``name``, they are ``w -+
        dl / 2``.
    """

    def __init__(self, reptran: Reptran, band: int) -> None:

        self.band = band
        # the number of internal bands (representative bands) in this
        # channel
        self.nband = reptran.nwvl_in_band[self.band]
        # the indices of the internal bands within the wavelength grid
        # for this channel
        self._iband = reptran.iwvl[: self.nband, self.band]
        # the corresponsing wavelenghts of the internal bands
        self.awvl = reptran.wvl[self._iband - 1]
        # the weights of the internal bands for this channel
        self.awvl_weight = reptran.iwvl_weight[: self.nband, self.band]
        # the extra terrestrial solar irradiance of the internal bands
        # for this channel
        self.aextra = reptran.extra[self._iband - 1]
        # the source of absorption by species of the internal bands
        # for this channel
        self.across_section_source = reptran.cross_section_source[
            self._iband - 1
        ]
        self.name = reptran.band_names[band]
        self.fname = Path(reptran.fname)
        # the integral of the response function of this channel, in nm
        # for a solar file and in cm-1 for a thermal one
        self.r_int = reptran.wvl_integral[self.band]

        try:
            self.wmin = float(
                self.name.split("to")[0].rstrip().split("bandfrom")[1]
            )
            self.wmax = float(self.name.split("to")[1].rstrip().split("nm")[0])
            parsed = True
        except (IndexError, ValueError):
            parsed = False

        # the bandwidth in nm
        if not reptran.thermal:
            self.dl = float(self.r_int)
        elif parsed:
            self.dl = self.wmax - self.wmin
        else:
            mean_square = np.average(self.awvl**2, weights=self.awvl_weight)
            self.dl = float(self.r_int * mean_square * 1e-7)

        if not parsed:
            self.w = np.mean(self.awvl)
            self.wmin = self.w - self.dl / 2.0
            self.wmax = self.w + self.dl / 2.0

    def iband(self, index: int) -> ReptranIband:
        """Return an internal band by its zero-based index.

        Parameters
        ----------
        index : int
            Zero-based index within the sensor channel.

        Returns
        -------
        ReptranIband
            The selected internal band.
        """
        return ReptranIband(self, index)

    def ibands(self) -> Iterator[ReptranIband]:
        """Iterate over the internal bands in this sensor channel.

        Yields
        ------
        ReptranIband
            Each internal band in increasing index order.
        """
        for i in range(self.nband):
            yield self.iband(i)


class Reptran:
    """Read and expose a REPTRAN correlated-k file.

    Parameters
    ----------
    fname : path-like
        REPTRAN file path. If no directory is provided, the file is
        looked up in the auxiliary REPTRAN directory. The ``.cdf``
        suffix is appended when it is absent.

    Attributes
    ----------
    fname : pathlib.Path
        Path to the REPTRAN correlated-k file.
    wvl : numpy.ndarray
        Internal REPTRAN wavelength grid in nanometres.
    extra : numpy.ndarray
        Extraterrestrial solar irradiance at the internal wavelengths.
    wvl_integral : numpy.ndarray
        Integral of the response function of each sensor channel, over
        wavelength in nanometres for a solar file, over wavenumber in
        inverse centimetres for a thermal file.
    thermal : bool
        Whether the file is a thermal one, which is told by the absence
        of the extraterrestrial solar irradiance ``extra``.
    nwvl_in_band : numpy.ndarray
        Number of internal bands in each sensor channel.
    iwvl : numpy.ndarray
        Indices of internal bands in the wavelength grid for each
        channel.
    iwvl_weight : numpy.ndarray
        Quadrature weights associated with the internal bands.
    cross_section_source : numpy.ndarray
        Molecular absorption-source flags for each internal band.
    band_names : list of str
        Names of the available sensor channels, without their spaces.
    """

    def __init__(self, fname: PathType) -> None:
        fname = Path(fname)
        if fname.parent == Path("."):
            self.fname = dir_reptran / fname
        else:
            self.fname = fname

        if fname.suffix != ".cdf":
            self.fname = self.fname.with_name(
                self.fname.name + ".cdf"
            )

        self._read_file_general()

    def _read_file_general(self) -> None:
        with xr.open_dataset(self.fname) as dataset:
            self.wvl = dataset["wvl"].values  # the wavelength grid
            # only the solar files hold the extra terrestrial solar
            # irradiance for the wavelength grid
            self.thermal = "extra" not in dataset.variables
            if self.thermal:
                self.extra = np.ones_like(self.wvl)
            else:
                self.extra = dataset["extra"].values
            # the integral of the response function of each sensor
            # channel, over wavelength in nm (solar) or over wavenumber
            # in cm-1 (thermal)
            self.wvl_integral = dataset["wvl_integral"].values
            # the number of internal bands in each sensor channel
            self.nwvl_in_band = dataset["nwvl_in_band"].values
            # the indices of internal bands within the wavelength grid
            self.iwvl = dataset["iwvl"].values
            # the weight associated with each internal band
            self.iwvl_weight = dataset["iwvl_weight"].values
            # the species contributing to absorption for each internal
            # band
            self.cross_section_source = dataset["cross_section_source"].values

            # the names of the sensor channels, without their spaces
            # ('band from  119.9976 to  120.0192 nm' becomes
            # 'bandfrom119.9976to120.0192nm')
            self.band_names = [
                _channel_name(bname) for bname in dataset["band_name"].values
            ]

    def nbands(self) -> int:
        """Return the number of sensor channels in the file.

        Returns
        -------
        int
            Number of sensor channels.
        """
        return len(self.wvl_integral)

    def band(self, band: int | str) -> ReptranBand:
        """Return a sensor channel by index or name.

        Parameters
        ----------
        band : int or str
            Zero-based channel index or channel name. The spaces of a
            name are ignored, as in :attr:`band_names`.

        Returns
        -------
        ReptranBand
            The selected sensor channel.
        """
        if isinstance(band, str):
            return self.band(self.band_names.index(_channel_name(band)))
        else:
            return ReptranBand(self, band)

    def bands(self) -> Iterator[ReptranBand]:
        """Iterate over all sensor channels in file order.

        Yields
        ------
        ReptranBand
            Each available sensor channel.
        """
        for i in range(self.nbands()):
            yield self.band(i)

    def to_smartg(
        self,
        include: str = "",
        lmin: NumericArrayLike = -np.inf,
        lmax: NumericArrayLike = np.inf,
        band_indices: Sequence[int] | None = None,
    ) -> ReptranIbandList:
        """Select internal bands for use with ``Smartg.run``.

        Parameters
        ----------
        include : str, optional
            Substring that must occur in a channel name. The empty
            string selects all channels. Default is ``''``.
        lmin, lmax : scalar or array-like, optional
            Lower and upper wavelength limits in nanometres. Multiple
            intervals can be supplied as matching sequences. Defaults
            are negative and positive infinity, respectively.
        band_indices : sequence of int, optional
            Explicit channel indices to consider. If None, all channels
            are considered.

        Returns
        -------
        ReptranIbandList
            Selected internal bands sorted by representative wavelength.
        """
        ik_l = []
        if band_indices is None:
            bl = self.bands()
        else:
            bl = [self.band(i) for i in band_indices]

        lmin_values = np.atleast_1d(lmin)
        lmax_values = np.atleast_1d(lmax)
        for k in bl:
            if include in k.name:
                for ii in range(len(lmin_values)):
                    if (k.wmin >= lmin_values[ii]) and (
                        k.wmax <= lmax_values[ii]
                    ):
                        ik_l.extend(k.ibands())

        assert len(ik_l) != 0

        return ReptranIbandList(sorted(ik_l, key=lambda x: x.w))


class ReptranIbandList:
    """Store a selected list of internal REPTRAN bands.

    Parameters
    ----------
    ibands : sequence of ReptranIband
        Internal bands included in the list.
    """

    def __init__(self, ibands: Sequence[ReptranIband]) -> None:
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
            Six arrays containing, in order, internal-band weights,
            channel central wavelengths, solar irradiance, bandwidth,
            weight sums, and bandwidth-weighted sums.
        """
        we_l = []
        ex_l = []
        dl_l = []
        wb_l = []
        wi_l = []
        for iband in self.l:
            # for iband in band.ibands():
            wi = iband.band.awvl[iband.index]  # wvl of internal band
            wi_l.append(wi)
            # weight of internal band
            we = iband.band.awvl_weight[iband.index]
            we_l.append(we)
            ex = iband.band.aextra[iband.index]  # E0 of internal band
            ex_l.append(ex)
            dl = iband.band.dl  # bandwidth in nm
            dl_l.append(dl)
            wb = np.mean(iband.band.awvl[:])
            wb_l.append(wb)

        wi_arr = np.array(wi_l, dtype=np.float32)
        wb = xr.DataArray(
            np.array(wb_l, dtype=np.float32),
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

    def get_names(self) -> list[str]:
        """Return the unique channel names represented by this list.

        Returns
        -------
        list of str
            Unique sensor channel names, sorted by channel central
            wavelength, the order of the ``wavelength`` axis of
            :func:`reduce_reptran`. Channels with the same central
            wavelength keep their order in the list.
        """
        centres: dict[str, np.float32] = {}
        for iband in self.l:
            centres.setdefault(
                iband.band.name, np.float32(np.mean(iband.band.awvl))
            )

        return sorted(centres, key=centres.__getitem__)


class ReadCrs:
    """Read a REPTRAN molecular cross-section lookup table.

    Parameters
    ----------
    fname : path-like
        Lookup-table path without its final ``.cdf`` suffix. If no
        directory is provided, the table is looked up in the auxiliary
        REPTRAN directory.
    iband : int
        REPTRAN internal-band index to select from the lookup table.

    Attributes
    ----------
    fname : pathlib.Path
        Lookup-table path without its final ``.cdf`` suffix.
    xsec : ndarray
        Cross-section values for the selected internal band.
    pressure, t_ref, t_pert, vmrs : ndarray
        Pressure, reference-temperature, temperature-perturbation, and
        water-vapour-mixing-ratio lookup axes.
    """

    def __init__(self, fname: PathType, iband: int) -> None:
        fname = Path(fname)
        if fname.parent == Path("."):
            fname = dir_reptran / fname
        self.fname = fname
        self._read_file_general(iband)

    def _read_file_general(self, iband: int) -> None:
        fname = self.fname.with_name(f"{self.fname.name}.cdf")
        with xr.open_dataset(fname) as dataset:
            self.wvl_index = dataset["wvl_index"].values
            ii = list(self.wvl_index).index(iband)
            # read only the slice of this internal band from the file
            self.xsec = dataset["xsec"][:, :, ii, :].to_numpy()
            self.pressure = dataset["pressure"].values
            self.t_ref = dataset["t_ref"].values
            self.t_pert = dataset["t_pert"].values
            self.vmrs = dataset["vmrs"].values
