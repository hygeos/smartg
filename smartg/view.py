"""Visualization utilities for SMART-G outputs.

This module provides plotting helpers for polar maps, transects,
spectra, phase functions, profiles, and receiver/category diagnostics
stored in SMART-G xarray datasets.

Key Functions
-------------
smartg_view
    Visualization of SMART-G output in polar coordinates.
transect_view
    Transect visualization of SMART-G output.
spectrum_view
    Visualization of wavelength-dependent Stokes parameters.
phase_view
    Visualization of SMART-G phase functions.
profile_view
    Visualization of SMART-G vertical profiles.
input_view
    Visualization of SMART-G input profile and phase functions.
receiver_view
    Plot receiver irradiance from a SMART-G simulation output.
satellite_view
    'Satellite' 2D image of SMART-G 3D atmosphere results.
visualize_entity
    3D visualization of the created scene objects.
"""

import math
import warnings
from pylab import (
    figure,
    subplot2grid,
    tight_layout,
    setp,
    subplots,
    xlabel,
    ylabel,
    FormatStrFormatter,
)
import numpy as np

# ignore division by zero errors
np.seterr(invalid="ignore", divide="ignore")
import xarray as xr
from xarray import Dataset
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import floating_axes
from matplotlib.projections import PolarAxes
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Any, Literal, Sequence, cast
import geoclide as gc
from luts.luts import Idx_base, MLUT, LUT
from smartg.diff import diff1, diff1_end
from smartg.grid3d import is_same_cell_size
from smartg.objects3d import (
    Entity,
    GroupE,
    Plane,
    Spheric,
    convert_lg_to_le,
    ref_fresnel,
)


def mdesc(desc: str, log_i: bool = False) -> str:
    """
    Format Stokes parameter description for display with LaTeX notation.

    Parses a description string to extract Stokes parameter, direction,
    and other components, then formats them with proper LaTeX notation
    including directional arrows (up/down).

    Parameters
    ----------
    desc : str
        Description string in format 'Stokes_direction(component)_info'
        (e.g., 'I_up(TOA)', 'Q_down(0+)').
    log_i : bool, optional
        If True and Stokes parameter is 'I', prepends 'log10' to the
        output. Default is False.

    Returns
    -------
    str
        Formatted LaTeX string with Stokes parameter, directional arrow,
        component subscripts, and optional log scale notation.

    Examples
    --------
    >>> mdesc('I_up(TOA)')
    '$I^{\\uparrow}_{TOA}$'
    >>> mdesc('I_up(TOA)', log_i=True)
    '$log_{10} I^{\\uparrow}_{TOA}$'
    >>> mdesc('Q_down(0+)')
    '$Q^{\\downarrow}_{0+}$'
    """
    sep1 = desc.find("_")
    sep2 = desc.find("(")
    sep3 = desc.find(")")
    if sep1 == 1:
        stokes = desc[0:1]
    elif sep1 == 2:
        stokes = desc[sep1 - 2 : sep1]
    elif sep1 == 4:
        stokes = desc[sep1 - 4 : sep1]
    else:
        stokes = desc[0:sep1]
    direction = desc[sep1 + 1 : sep2 - 1]

    if log_i and stokes == "I":
        pref = r"$log_{10} "
    else:
        pref = r"$"

    if direction == "up":
        return (
            pref
            + stokes
            + r"^{\uparrow}"
            + "_{"
            + desc[sep2 + 1 : sep3]
            + "}"
            + desc[sep3 + 1 :]
            + "$"
        )
    else:
        return (
            pref
            + stokes
            + r"^{\downarrow}"
            + "_{"
            + desc[sep2 + 1 : sep3]
            + "}"
            + desc[sep3 + 1 :]
            + "$"
        )


def _interp_and_squeeze_scalar_dims(
    da: xr.DataArray, interp_dict: dict[str, Any]
) -> xr.DataArray:
    """Interpolate then squeeze only existing singleton dimensions."""
    valid_interp = {
        dim: value for dim, value in interp_dict.items() if dim in da.dims
    }
    if valid_interp:
        da_interp = da.interp(valid_interp)
    else:
        da_interp = da
    dims_to_squeeze = [
        dim
        for dim, value in valid_interp.items()
        if np.atleast_1d(value).size <= 1
        and dim in da_interp.dims
        and da_interp.sizes[dim] <= 1
    ]
    if dims_to_squeeze:
        da_interp = da_interp.squeeze(dim=dims_to_squeeze, drop=True)
    return da_interp


def smartg_view(
    ds_sg: xr.Dataset | MLUT,
    log_i: bool = False,
    qu: bool = False,
    circ: bool = False,
    full: bool = False,
    field: str = "up (TOA)",
    prefix: str = "",
    ind: int | list[int] | np.ndarray | Idx_base | None = None,
    cmap: str | mcolors.Colormap | None = None,
    fig: Figure | None = None,
    subdict: dict[str, Any] | None = None,
    interp_dict: dict[str, Any] | None = None,
    i_min: float | None = None,
    i_max: float | None = None,
    p_min: float = 0,
    p_max: float = 100,
) -> Figure:
    """
    Visualization of SMART-G output in polar coordinates.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G simulation.
    log_i : bool, optional
        If True, display Intensity (I) in log10 scale. Default is False.
    qu : bool, optional
        If True, show Q, U, and DoLP. If False, show only I and
        polarization metrics. Default is False.
    circ : bool, optional
        If True, display circular polarization metrics (V and DoCP -
        Degree of Circular Polarization). If False, display linear
        polarization metrics (Q, U, and DoLP - Degree of Linear
        Polarization). Effective with both ``qu=True`` and
        ``qu=False``. When ``full=True``, both circular and linear
        polarization metrics are displayed. Default is False.
    full : bool, optional
        If True, display everything. Default is False.
    field : str, optional
        Name of the output level to visualize. Default is 'up (TOA)'.
    prefix : str, optional
        Prefix for field variable names. Default is empty string.
    ind : int or list of int, optional
        Azimuthal plane indices to display. Default is [0].
    cmap : str, optional
        Colormap name for polar plots. Default is None (uses default
        colormap).
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates a new figure.
        Default is None.
    subdict : dict, optional
        **Deprecated**. Use `interp_dict` instead. Dictionary of
        coordinate values for interpolation. This parameter corresponds
        to the input dictionary of the `sub()` method of deprecated LUT
        and MLUT objects, for backward compatibility. Default is None.
    interp_dict : dict, optional
        Dictionary of coordinate values for interpolation. Keys are
        dimension names, values are the coordinate values to interpolate
        to. Uses xarray's `interp()` method. Mutually exclusive with
        `subdict`. Default is None.
    i_min : float, optional
        Minimum value for Intensity display. If None, determined from
        data. Default is None.
    i_max : float, optional
        Maximum value for Intensity display. If None, determined from
        data. Default is None.
    p_min : float, optional
        Minimum value for polarization display. Default is 0.
    p_max : float, optional
        Maximum value for polarization display. Default is 100.

    Returns
    -------
    fig : Figure or tuple of Figure

    Notes
    -----
    Polarization metrics are computed from Stokes parameters
    (I, Q, U, V):

    - **Degree of Linear Polarization (dolp)**:

      dolp = 100 * sqrt(Q² + U²) / I

    - **Degree of Polarization (dop)**:

      dop = 100 * sqrt(Q² + U² + V²) / I

    - **Degree of Circular Polarization (docp)**:

      docp = 100 * |V| / I
    """
    if ind is None:
        ind = [0]

    if isinstance(ds_sg, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_sg is deprecated, use an "
            + "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    if isinstance(ind, Idx_base):
        warn_message = (
            "\nUsing luts.Idx_base objects for the 'ind' parameter is "
            "deprecated and will result in an error in future versions."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        indexer = getattr(ind, "index", None)
        if not callable(indexer):
            raise TypeError(
                "Expected an index-capable Idx_base object for 'ind'."
            )
        index_values = cast(Any, indexer)(
            ds_sg.coords["Azimuth angles"].values
        )
        ind = np.atleast_1d(np.round(index_values).astype(np.int32))

    stk_i = ds_sg[prefix + "I_" + field]
    stk_u = ds_sg[prefix + "Q_" + field]
    stk_q = ds_sg[prefix + "U_" + field]
    stk_v = ds_sg[prefix + "V_" + field]

    # Handle deprecated subdict parameter
    if subdict is not None and interp_dict is not None:
        raise ValueError(
            "Cannot specify both 'subdict' and 'interp_dict'. "
            + "Use 'interp_dict' instead."
        )

    if subdict is not None:
        warn_message = (
            "\nThe 'subdict' parameter is deprecated. "
            + "Use 'interp_dict' instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        # Convert Idx_base objects to values before converting
        # to interp_dict
        for dic_name in list(subdict.keys()):
            if isinstance(subdict[dic_name], Idx_base):
                subdict[dic_name] = subdict[dic_name].value
            else:
                subdict[dic_name] = ds_sg[dic_name][subdict[dic_name]]
        interp_dict = subdict

    if interp_dict is not None:
        # Identify dimensions to drop (those with scalar values)
        stk_i = _interp_and_squeeze_scalar_dims(stk_i, interp_dict)
        stk_u = _interp_and_squeeze_scalar_dims(stk_u, interp_dict)
        stk_q = _interp_and_squeeze_scalar_dims(stk_q, interp_dict)
        stk_v = _interp_and_squeeze_scalar_dims(stk_v, interp_dict)

    # Linearly polarized reflectance
    ipl = cast(xr.DataArray, np.sqrt(stk_u * stk_u + stk_q * stk_q))

    # Polarized reflectance
    ip = cast(
        xr.DataArray, np.sqrt(stk_u * stk_u + stk_q * stk_q + stk_v * stk_v)
    )

    # Degree of Linear Polarization (%)
    dolp = cast(xr.DataArray, 100 * ipl / stk_i)

    # Angle of Linear Polarization (deg)
    # aolp = np.arctan(stk_q / stk_u) * 90 / np.pi

    # Degree of Circular Polarization (%)
    docp = cast(xr.DataArray, 100 * np.abs(stk_v) / stk_i)

    # Degree of Polarization (%)
    dop = cast(xr.DataArray, 100 * ip / stk_i)

    if not full:
        if qu:
            if fig is None:
                fig = figure(figsize=(9, 14))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = mdesc(
                    str(stk_i.name or "I"), log_i=True
                )
                plot_polar(
                    li.assign_coords(li.coords),
                    index=ind,
                    rect=421,
                    sub=423,
                    fig=fig,
                    cmap=cmap,
                    vmin=i_min,
                    vmax=i_max,
                )
            else:
                plot_polar(
                    stk_i.assign_coords(stk_i.coords),
                    index=ind,
                    rect=421,
                    sub=423,
                    fig=fig,
                    cmap=cmap,
                    vmin=i_min,
                    vmax=i_max,
                )
            plot_polar(
                stk_u.assign_coords(stk_u.coords),
                index=ind,
                rect=422,
                sub=424,
                fig=fig,
                cmap=cmap,
            )
            if ind is not None:
                rect_u = 425
            else:
                rect_u = 423
            plot_polar(
                stk_q.assign_coords(stk_q.coords),
                index=ind,
                rect=rect_u,
                sub=427,
                fig=fig,
                cmap=cmap,
            )
            if circ:
                if ind is not None:
                    rect_v = 426
                else:
                    rect_v = 424
                plot_polar(
                    stk_v.assign_coords(stk_v.coords),
                    index=ind,
                    rect=rect_v,
                    sub=428,
                    fig=fig,
                    cmap=cmap,
                )
            else:
                if ind is not None:
                    rect_dop = 426
                else:
                    rect_dop = 424
                dop.attrs["latex_name"] = r"$DoP$"
                plot_polar(
                    dop.assign_coords(dop.coords),
                    index=ind,
                    rect=rect_dop,
                    sub=428,
                    fig=fig,
                    vmin=p_min,
                    vmax=p_max,
                    cmap=cmap,
                )
        else:
            # show only I and PR
            if fig is None:
                fig = figure(figsize=(9, 6))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = mdesc(
                    str(stk_i.name or "I"), log_i=True
                )
                plot_polar(
                    li.assign_coords(li.coords),
                    index=ind,
                    rect=221,
                    sub=223,
                    fig=fig,
                    cmap=cmap,
                    vmin=i_min,
                    vmax=i_max,
                )
            else:
                plot_polar(
                    stk_i.assign_coords(stk_i.coords),
                    index=ind,
                    rect=221,
                    sub=223,
                    fig=fig,
                    cmap=cmap,
                    vmin=i_min,
                    vmax=i_max,
                )

            if circ:
                docp.attrs["latex_name"] = r"$DoCP$"
                plot_polar(
                    docp.assign_coords(docp.coords),
                    index=ind,
                    rect=222,
                    sub=224,
                    fig=fig,
                    vmin=0,
                    vmax=p_max,
                    cmap=cmap,
                )
            else:
                dop.attrs["latex_name"] = r"$DoP$"
                plot_polar(
                    dop.assign_coords(dop.coords),
                    index=ind,
                    rect=222,
                    sub=224,
                    fig=fig,
                    vmin=p_min,
                    vmax=p_max,
                    cmap=cmap,
                )
    else:
        # full plots
        li = cast(xr.DataArray, np.log10(stk_i))
        li.attrs["latex_name"] = mdesc(str(stk_i.name or "I"), log_i=True)
        dolp.attrs["latex_name"] = r"$DoLP$"
        docp.attrs["latex_name"] = r"$DoCP$"
        dop.attrs["latex_name"] = r"$DoP$"

        if fig is None:
            fig = figure(figsize=(18, 14))

        plot_polar(
            stk_i.assign_coords(stk_i.coords),
            index=ind,
            rect=441,
            sub=445,
            fig=fig,
            cmap=cmap,
            vmin=i_min,
            vmax=i_max,
        )
        plot_polar(
            stk_u.assign_coords(stk_u.coords),
            index=ind,
            rect=442,
            sub=446,
            fig=fig,
            cmap=cmap,
        )
        plot_polar(
            stk_q.assign_coords(stk_q.coords),
            index=ind,
            rect=443,
            sub=447,
            fig=fig,
            cmap=cmap,
        )
        plot_polar(
            stk_v.assign_coords(stk_v.coords),
            index=ind,
            rect=444,
            sub=448,
            fig=fig,
            cmap=cmap,
        )

        plot_polar(
            li.assign_coords(li.coords),
            index=ind,
            rect=449,
            sub=(4, 4, 13),
            fig=fig,
            cmap=cmap,
        )
        plot_polar(
            dolp.assign_coords(dolp.coords),
            index=ind,
            rect=(4, 4, 10),
            sub=(4, 4, 14),
            fig=fig,
            vmin=p_min,
            vmax=p_max,
            cmap=cmap,
        )
        plot_polar(
            docp.assign_coords(docp.coords),
            index=ind,
            rect=(4, 4, 11),
            sub=(4, 4, 15),
            fig=fig,
            vmin=p_min,
            vmax=p_max,
            cmap=cmap,
        )
        plot_polar(
            dop.assign_coords(dop.coords),
            index=ind,
            rect=(4, 4, 12),
            sub=(4, 4, 16),
            fig=fig,
            vmin=p_min,
            vmax=p_max,
            cmap=cmap,
        )

    fig.subplots_adjust(hspace=0.3)
    return fig


def transect_view(
    ds_sg: xr.Dataset | MLUT,
    log_i: bool = False,
    qu: bool = False,
    circ: bool = False,
    full: bool = False,
    field: str = "up (TOA)",
    prefix: str = "",
    ind: int | list[int] | np.ndarray | Idx_base | None = None,
    fig: Figure | tuple[Figure, Figure] | None = None,
    color: str = "k",
    subdict: dict[str, Any] | None = None,
    interp_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Figure | tuple[Figure, Figure]:
    """
    Transect visualization of SMART-G output.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G simulation.
    log_i : bool, optional
        If True, display Intensity (I) in log10 scale. Default is False.
    qu : bool, optional
        If True, show Q, U, and DoLP. If False, show only I and
        polarization metrics. Default is False.
    circ : bool, optional
        If True, show circular polarization metrics. If False, show
        linear polarization. Default is False.
    full : bool, optional
        If True, return two figures with full and reduced polarization
        info. If False, return one figure. Default is False.
    field : str, optional
        Name of the output level to visualize. Default is 'up (TOA)'.
    prefix : str, optional
        Prefix for field variable names. Default is empty string.
    ind : int or list of int, optional
        Azimuthal plane indices to display. Default is [0].
    fig : Figure or tuple of Figure, optional
        Existing figure container used for plotting, depending on
        `full`:

        - If `full` is False: pass a single Figure.
        - If `full` is True: pass a tuple `(fig1, fig2)`.

        If None, new figure(s) are created automatically. Default is
        None.
    color : str, optional
        Color for the transect line. Default is 'k' (black).
    subdict : dict, optional
        **Deprecated**. Use `interp_dict` instead. Dictionary of
        coordinate values for interpolation. This parameter corresponds
        to the input dictionary of the `sub()` method of deprecated LUT
        and MLUT objects, for backward compatibility. Default is None.
    interp_dict : dict, optional
        Dictionary of coordinate values for interpolation. Keys are
        dimension names, values are the coordinate values to interpolate
        to. Uses xarray's `interp()` method. Mutually exclusive with
        `subdict`. Default is None.
    **kwargs
        Additional keyword arguments passed to transect_2d, including:
        - vmin, vmax : float, optional.
            Minimum and maximum values for data range display. If None,
            determined from data.
        - sym : bool, optional.
            If True, use symmetrical axis for the transect. Default is
            True.
        - swap : bool or 'auto', optional.
            If True or 'auto', swap the order of the 2 axes. If
            'auto', searches for 'azi' in dimension names. Default is
            'auto'.
        - fmt : str, optional.
            Plot format string (e.g., '-', '--', '.', etc.). Default is
            '-'.

    Returns
    -------
    fig : Figure or tuple of Figure
        If full is False: single figure containing transect slices of
        Stokes parameters. If full is True: tuple of (fig1, fig2) with
        raw and processed Stokes parameters.
    """

    if ind is None:
        ind = [0]

    if isinstance(ds_sg, MLUT):
        warn_message = "\nUsing an MLUT for ds_sg is deprecated, use an xarray.Dataset instead."
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    if isinstance(ind, Idx_base):
        warn_message = (
            "\nUsing luts.Idx_base objects for the 'ind' parameter is "
            "deprecated and will result in an error in future versions."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        indexer = getattr(ind, "index", None)
        if not callable(indexer):
            raise TypeError(
                "Expected an index-capable Idx_base object for 'ind'."
            )
        index_values = cast(Any, indexer)(
            ds_sg.coords["Azimuth angles"].values
        )
        ind = np.atleast_1d(np.round(index_values).astype(np.int32))

    stk_i = ds_sg[prefix + "I_" + field]
    stk_u = ds_sg[prefix + "Q_" + field]
    stk_q = ds_sg[prefix + "U_" + field]
    stk_v = ds_sg[prefix + "V_" + field]

    # Handle deprecated subdict parameter
    if subdict is not None and interp_dict is not None:
        raise ValueError(
            "Cannot specify both 'subdict' and 'interp_dict'. Use 'interp_dict' instead."
        )

    if subdict is not None:
        warn_message = "\nThe 'subdict' parameter is deprecated. Use 'interp_dict' instead."
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        # Convert Idx_base objects to values before converting
        # to interp_dict
        for dic_name in list(subdict.keys()):
            if isinstance(subdict[dic_name], Idx_base):
                subdict[dic_name] = subdict[dic_name].value
            else:
                subdict[dic_name] = ds_sg[dic_name][subdict[dic_name]]
        interp_dict = subdict

    if interp_dict is not None:
        # Identify dimensions to drop (those with scalar values)
        stk_i = _interp_and_squeeze_scalar_dims(stk_i, interp_dict)
        stk_u = _interp_and_squeeze_scalar_dims(stk_u, interp_dict)
        stk_q = _interp_and_squeeze_scalar_dims(stk_q, interp_dict)
        stk_v = _interp_and_squeeze_scalar_dims(stk_v, interp_dict)

    # Linearly polarized reflectance
    ipl = cast(xr.DataArray, np.sqrt(stk_u * stk_u + stk_q * stk_q))

    # Polarized reflectance
    ip = cast(
        xr.DataArray,
        np.sqrt(stk_u * stk_u + stk_q * stk_q + stk_v * stk_v),
    )

    # Degree of Linear Polarization (%)
    dolp = cast(xr.DataArray, 100 * ipl / stk_i)
    dolp.attrs["latex_name"] = prefix + r"$DoLP$"

    # Angle of Linear Polarization (deg)
    aolp = cast(xr.DataArray, np.arctan(stk_q / stk_u) * 90 / np.pi)
    aolp.attrs["latex_name"] = prefix + r"$AoLP$"

    # Degree of Circular Polarization (%)
    docp = cast(xr.DataArray, 100 * np.abs(stk_v) / stk_i)
    docp.attrs["latex_name"] = prefix + r"$DoCP$"

    # Degree of Polarization (%)
    dop = cast(xr.DataArray, 100 * ip / stk_i)
    dop.attrs["latex_name"] = prefix + r"$DoP$"

    if not full:
        if fig is None:
            plot_fig: Figure | None = None
        elif isinstance(fig, Figure):
            plot_fig = fig
        else:
            raise ValueError(
                "If 'full' is False, 'fig' must be None or a Figure."
            )

        if qu:
            if plot_fig is None:
                plot_fig = figure(figsize=(8, 8))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = "log$_{10}$ " + stk_i.attrs.get(
                    "latex_name", "I"
                )
                transect_2d(
                    li,
                    index=ind,
                    sub=221,
                    fig=plot_fig,
                    color=color,
                    **kwargs,
                )
            else:
                transect_2d(
                    stk_i,
                    index=ind,
                    sub=221,
                    fig=plot_fig,
                    color=color,
                    **kwargs,
                )
            transect_2d(
                stk_u,
                index=ind,
                sub=222,
                fig=plot_fig,
                color=color,
                **kwargs,
            )
            transect_2d(
                stk_q,
                index=ind,
                sub=223,
                fig=plot_fig,
                color=color,
                **kwargs,
            )
            if circ:
                transect_2d(
                    stk_v,
                    index=ind,
                    sub=224,
                    fig=plot_fig,
                    color=color,
                    **kwargs,
                )
            else:
                transect_2d(
                    dop,
                    index=ind,
                    sub=224,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )
        else:
            # show only I and PR
            if plot_fig is None:
                plot_fig = figure(figsize=(8, 4))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = "log$_{10}$ " + stk_i.attrs.get(
                    "latex_name", "I"
                )
                transect_2d(
                    li,
                    index=ind,
                    sub=121,
                    fig=plot_fig,
                    color=color,
                    **kwargs,
                )
            else:
                transect_2d(
                    stk_i,
                    index=ind,
                    sub=121,
                    fig=plot_fig,
                    color=color,
                    **kwargs,
                )

            if circ:
                transect_2d(
                    docp,
                    index=ind,
                    sub=122,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )
            else:
                transect_2d(
                    dop,
                    index=ind,
                    sub=122,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )

        if plot_fig is None:
            raise RuntimeError("Failed to initialize transect figure.")
        return plot_fig

    else:
        # full plots
        if fig is None:
            fig1 = figure(figsize=(16, 4))
            fig2 = figure(figsize=(16, 4))
        elif (
            (isinstance(fig, tuple) and len(fig) == 2)
            and isinstance(fig[0], Figure)
            and isinstance(fig[1], Figure)
        ):
            fig1, fig2 = fig
        else:
            raise ValueError(
                "If 'full' is True, 'fig' must be None or a tuple of "
                "two Figure objects."
            )

        li = cast(xr.DataArray, np.log10(stk_i))
        li.attrs["latex_name"] = "log$_{10}$ " + stk_i.attrs.get(
            "latex_name", "I"
        )

        transect_2d(stk_i, index=ind, sub=141, fig=fig1, color=color, **kwargs)
        transect_2d(stk_u, index=ind, sub=142, fig=fig1, color=color, **kwargs)
        transect_2d(stk_q, index=ind, sub=143, fig=fig1, color=color, **kwargs)
        transect_2d(stk_v, index=ind, sub=144, fig=fig1, color=color, **kwargs)

        transect_2d(li, index=ind, sub=141, fig=fig2, color=color, **kwargs)
        transect_2d(
            dolp,
            index=ind,
            sub=142,
            fig=fig2,
            color=color,
            percent=True,
            **kwargs,
        )
        transect_2d(
            docp,
            index=ind,
            sub=143,
            fig=fig2,
            color=color,
            percent=True,
            **kwargs,
        )
        transect_2d(
            dop,
            index=ind,
            sub=144,
            fig=fig2,
            color=color,
            percent=True,
            **kwargs,
        )

        return fig1, fig2


def spectrum(
    da: xr.DataArray | LUT,
    vmin: float | None = None,
    vmax: float | None = None,
    sub: int | str | tuple[int, int, int] = "111",
    fig: Figure | None = None,
    color: str = "k",
    percent: bool = False,
    fmt: str = "-",
) -> Figure:
    """
    Plot spectrum of a 1D DataArray.

    Parameters
    ----------
    da : DataArray
        One-dimensional xarray DataArray with 'wavelength' dimension.
    vmin, vmax : float, optional
        Range of values. If None (default), determined from data
        min/max.
    sub : str, optional
        Subplot specification. Default is '111'.
    fig : matplotlib.figure.Figure, optional
        Destination figure. If None, creates a new figure.
    color : str, optional
        Color of the plot line. Default is 'k' (black).
    percent : bool, optional
        If True, scale y-axis to 0-100%. Default is False.
    fmt : str, optional
        Line format. Default is '-'.

    Returns
    -------
    fig : Figure
        Figure object containing the spectrum plot.
    """
    from pylab import figure

    if isinstance(da, LUT):
        warn_message = (
            "\nUsing an LUT for da is deprecated, use "
            + "an xarray.DataArray instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        da = da.to_xarray()

    assert "wavelength" in da.dims, (
        "DataArray must have 'wavelength' dimension"
    )

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    ax1 = da.coords["wavelength"].values
    data = da.values

    if vmin is None:
        vmin = float(np.amin(data[~np.isnan(data)]))
    if vmax is None:
        vmax = float(np.amax(data[~np.isnan(data)]))
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if percent:
        vmin = 0.0
        vmax = 100.0

    ax1_min = np.amin(ax1)
    ax1_max = np.amax(ax1)

    # Parse subplot specification and build a stable marker key
    sub = _parse_subplot_position(sub)
    if isinstance(sub, tuple):
        marker_key = "_".join(map(str, sub))
    else:
        marker_key = str(sub)

    # Check if subplot already exists by using a marker attribute
    marker_name = f"_spectrum_sub_{marker_key}"
    ax_cart: Any | None = None
    is_new_axes = True
    if hasattr(fig, marker_name):
        ax_cart = getattr(fig, marker_name)
        is_new_axes = False

    if is_new_axes:
        if isinstance(sub, tuple):
            ax_cart = fig.add_subplot(*sub)
        else:
            ax_cart = fig.add_subplot(sub)
        setattr(fig, marker_name, ax_cart)  # Store reference
        ax_cart.grid(True)
        ax_cart.set_xlim(ax1_min, ax1_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax_cart.set_xlabel(r"$\lambda$ (nm)")
    else:
        if ax_cart is None:
            raise RuntimeError("Failed to retrieve existing spectrum axes.")
        # Extend ylimits if needed
        current_ylim = ax_cart.get_ylim()
        new_vmin = min(current_ylim[0], vmin)
        new_vmax = max(current_ylim[1], vmax)
        ax_cart.set_ylim(new_vmin, new_vmax)

    if ax_cart is None:
        raise RuntimeError("Failed to initialize spectrum axes.")

    # Plot
    ax_cart.plot(ax1, data[:], fmt, color=color)

    # Add title
    title = da.attrs.get("latex_name", da.name)
    if title is not None:
        ax_cart.set_title(title)

    return fig


def spectrum_view(
    ds_sg: xr.Dataset | MLUT,
    log_i: bool = False,
    qu: bool = False,
    circ: bool = False,
    full: bool = False,
    field: str = "up (TOA)",
    prefix: str = "",
    fig: Figure | tuple[Figure, Figure] | None = None,
    color: str = "k",
    subdict: dict[str, Any] | None = None,
    interp_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Figure | tuple[Figure, Figure]:
    """
    Visualization of SMART-G spectrum (wavelength-dependent Stokes
    parameters).

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G simulation.
    log_i : bool, optional
        If True, display Intensity (I) in log10 scale. Default is False.
    qu : bool, optional
        If True, show Q, U, and DoLP. If False, show only I and
        polarization metrics. Default is False.
    circ : bool, optional
        If True, show circular polarization metrics. If False, show
        linear polarization. Default is False.
    full : bool, optional
        If True, return two figures with full and reduced polarization
        info. If False, return one figure. Default is False.
    field : str, optional
        Name of the output level to visualize. Default is 'up (TOA)'.
    prefix : str, optional
        Prefix for field variable names. Default is empty string.
    fig : Figure or tuple of Figure, optional
        Existing figure container used for plotting, depending on
        `full`:

        - If `full` is False: pass a single Figure.
        - If `full` is True: pass a tuple `(fig1, fig2)`.

        If None, new figure(s) are created automatically. Default is
        None.
    color : str, optional
        Color for the spectrum lines. Default is 'k' (black).
    subdict : dict, optional
        **Deprecated**. Use `interp_dict` instead. Dictionary of
        coordinate values for interpolation. This parameter corresponds
        to the input dictionary of the `sub()` method of deprecated LUT
        and MLUT objects, for backward compatibility. Default is None.
    interp_dict : dict, optional
        Dictionary of coordinate values for interpolation. Keys are
        dimension names, values are the coordinate values to interpolate
        to. Uses xarray's `interp()` method. Mutually exclusive with
        `subdict`. Default is None.
    **kwargs
        Additional keyword arguments passed to the spectrum plotting
        function (vmin, vmax, fmt, etc.).

    Returns
    -------
    fig : Figure or tuple of Figure
        If full is False: single figure containing spectrum plots. If
        full is True: tuple of (fig1, fig2) with raw Stokes parameters
        and processed metrics.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_sg is deprecated, use an "
            + "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    # Handle deprecated subdict parameter
    if subdict is not None and interp_dict is not None:
        raise ValueError(
            "Cannot specify both 'subdict' and 'interp_dict'. Use 'interp_dict' instead."
        )

    if subdict is not None:
        warn_message = (
            "\nThe 'subdict' parameter is deprecated. "
            + "Use 'interp_dict' instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        # Convert Idx_base objects to values before converting
        # to interp_dict
        for dic_name in list(subdict.keys()):
            if isinstance(subdict[dic_name], Idx_base):
                subdict[dic_name] = subdict[dic_name].value
            else:
                subdict[dic_name] = ds_sg[dic_name][subdict[dic_name]]
        interp_dict = subdict

    stk_i = ds_sg[prefix + "I_" + field]
    stk_u = ds_sg[prefix + "Q_" + field]
    stk_q = ds_sg[prefix + "U_" + field]
    stk_v = ds_sg[prefix + "V_" + field]

    # Handle interpolation for multi-dimensional data
    if interp_dict is not None:
        # Identify dimensions to drop (those with scalar values)
        stk_i = _interp_and_squeeze_scalar_dims(stk_i, interp_dict)
        stk_u = _interp_and_squeeze_scalar_dims(stk_u, interp_dict)
        stk_q = _interp_and_squeeze_scalar_dims(stk_q, interp_dict)
        stk_v = _interp_and_squeeze_scalar_dims(stk_v, interp_dict)

    # Linearly polarized reflectance
    ipl = cast(xr.DataArray, np.sqrt(stk_u * stk_u + stk_q * stk_q))
    ipl.attrs["latex_name"] = prefix + r"$Lin. Pol. ref.$"

    # Polarized reflectance
    ip = cast(
        xr.DataArray,
        np.sqrt(stk_u * stk_u + stk_q * stk_q + stk_v * stk_v),
    )
    ip.attrs["latex_name"] = prefix + r"$Pol. ref.$"

    # Degree of Linear Polarization (%)
    dolp = cast(xr.DataArray, 100 * ipl / stk_i)
    dolp.attrs["latex_name"] = prefix + r"$DoLP$"

    # Angle of Linear Polarization (deg)
    aolp = cast(xr.DataArray, np.arctan(stk_q / stk_u) * 90 / np.pi)
    aolp.attrs["latex_name"] = prefix + r"$AoLP$"

    # Degree of Circular Polarization (%)
    docp = cast(xr.DataArray, 100 * np.abs(stk_v) / stk_i)
    docp.attrs["latex_name"] = prefix + r"$DoCP$"

    # Degree of Polarization (%)
    dop = cast(xr.DataArray, 100 * ip / stk_i)
    dop.attrs["latex_name"] = prefix + r"$DoP$"

    if not full:
        if fig is None:
            plot_fig: Figure | None = None
        elif isinstance(fig, Figure):
            plot_fig = fig
        else:
            raise ValueError(
                "If 'full' is False, 'fig' must be None or a Figure."
            )

        if qu:
            if plot_fig is None:
                plot_fig = figure(figsize=(8, 8))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = mdesc(
                    str(stk_i.name or "I"), log_i=True
                )
                spectrum(li, sub=221, fig=plot_fig, color=color, **kwargs)
            else:
                stk_i.attrs["latex_name"] = mdesc(str(stk_i.name or "I"))
                spectrum(stk_i, sub=221, fig=plot_fig, color=color, **kwargs)
            stk_u.attrs["latex_name"] = mdesc(str(stk_u.name or "Q"))
            stk_q.attrs["latex_name"] = mdesc(str(stk_q.name or "U"))
            spectrum(stk_u, sub=222, fig=plot_fig, color=color, **kwargs)
            spectrum(stk_q, sub=223, fig=plot_fig, color=color, **kwargs)
            if circ:
                stk_v.attrs["latex_name"] = mdesc(str(stk_v.name or "V"))
                spectrum(stk_v, sub=224, fig=plot_fig, color=color, **kwargs)
            else:
                spectrum(
                    dop,
                    sub=224,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )
        else:
            # show only I and polarization
            if plot_fig is None:
                plot_fig = figure(figsize=(8, 4))
            if log_i:
                li = cast(xr.DataArray, np.log10(stk_i))
                li.attrs["latex_name"] = mdesc(
                    str(stk_i.name or "I"), log_i=True
                )
                spectrum(li, sub=121, fig=plot_fig, color=color, **kwargs)
            else:
                stk_i.attrs["latex_name"] = mdesc(str(stk_i.name or "I"))
                spectrum(stk_i, sub=121, fig=plot_fig, color=color, **kwargs)

            if circ:
                spectrum(
                    docp,
                    sub=122,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )
            else:
                spectrum(
                    dop,
                    sub=122,
                    fig=plot_fig,
                    color=color,
                    percent=True,
                    **kwargs,
                )

        if plot_fig is None:
            raise RuntimeError("Failed to initialize spectrum figure.")
        return plot_fig

    else:
        # full plots
        if fig is None:
            fig1 = figure(figsize=(16, 4))
            fig2 = figure(figsize=(16, 4))
        elif (
            (isinstance(fig, tuple) and len(fig) == 2)
            and isinstance(fig[0], Figure)
            and isinstance(fig[1], Figure)
        ):
            fig1, fig2 = fig
        else:
            raise ValueError(
                "If 'full' is True, 'fig' must be None or a tuple of "
                "two Figure objects."
            )

        li = cast(xr.DataArray, np.log10(stk_i))
        li.attrs["latex_name"] = mdesc(str(stk_i.name or "I"), log_i=True)
        stk_i.attrs["latex_name"] = mdesc(str(stk_i.name or "I"))
        stk_u.attrs["latex_name"] = mdesc(str(stk_u.name or "Q"))
        stk_q.attrs["latex_name"] = mdesc(str(stk_q.name or "U"))
        stk_v.attrs["latex_name"] = mdesc(str(stk_v.name or "V"))

        spectrum(stk_i, sub=141, fig=fig1, color=color, **kwargs)
        spectrum(stk_u, sub=142, fig=fig1, color=color, **kwargs)
        spectrum(stk_q, sub=143, fig=fig1, color=color, **kwargs)
        spectrum(stk_v, sub=144, fig=fig1, color=color, **kwargs)

        spectrum(li, sub=141, fig=fig2, color=color, **kwargs)
        spectrum(dolp, sub=142, fig=fig2, color=color, percent=True, **kwargs)
        spectrum(docp, sub=143, fig=fig2, color=color, percent=True, **kwargs)
        spectrum(dop, sub=144, fig=fig2, color=color, percent=True, **kwargs)

        return fig1, fig2


def phase_view(
    ds_sg: xr.Dataset | MLUT,
    ipha: int
    | Sequence[int]
    | np.ndarray[Any, Any]
    | xr.DataArray
    | None = None,
    fig: Figure | None = None,
    axarr: np.ndarray[Any, Any] | None = None,
    iw: int = 0,
    kind: str = "atm",
    show_trunc: bool = False,
    force_4stk: bool = False,
) -> tuple[Figure, np.ndarray[Any, Any]]:
    """
    Visualization of SMART-G phase function.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G, can be from simulation results
        or smartg input profile, containing phase function data with
        variables 'phase_atm' or 'phase_oc', and 'OD_atm' or 'OD_oc'.
    ipha : int or ndarray, optional
        Absolute index (or indices) of the phase function(s) coming from
        Profile. Can be an int (single index) or a 1D ndarray of int
        indices. If None, uses all unique indices from iphase_kind.
    fig : matplotlib.figure.Figure, optional
        Figure object. If None, creates a new figure.
    axarr : numpy.ndarray, optional
        2D array of matplotlib axes. If None, creates appropriate
        subplot grid.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Phase function type: 'atm' for atmospheric, 'oc' for oceanic.
        Default is 'atm'.
    show_trunc : bool, optional
        If True, also plots truncated phase function. Default is False.
    force_4stk : bool, optional
        If True, forces 2x2 subplot layout even for 6-stokes. Default is
        False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the phase function plots.
    axarr : numpy.ndarray
        Array of matplotlib axes.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_sg is deprecated, use an "
            + "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    od_key = "OD_" + kind
    phase_key = "phase_" + kind
    theta_key = "theta_" + kind

    # Handle multi-wavelength case
    od_data = ds_sg[od_key]
    nd = len(od_data.dims)

    if nd > 1:
        # Find wavelength dimension index
        if "wavelength" in od_data.dims:
            wavelength = ds_sg.coords["wavelength"].values
            labw = r" at $%.1f nm$" % wavelength[iw]
        else:
            labw = ""
    else:
        labw = ""

    phase = ds_sg[phase_key].values
    if show_trunc:
        phase_tr: np.ndarray[Any, Any] | None = ds_sg[
            "phase_" + kind + "_tr"
        ].values
    else:
        phase_tr = None

    ang = ds_sg.coords[theta_key].values
    nstk = phase.shape[1]

    if axarr is None:
        if nstk == 4 or force_4stk:
            fig, axarr = subplots(2, 2)
            fig.set_size_inches(10, 6)
        elif nstk == 6:
            fig, axarr = subplots(nrows=3, ncols=2)
            fig.set_size_inches(10, 9)

    if axarr is None:
        raise ValueError("Unable to create axes for phase_view.")

    if fig is None:
        fig = cast(Figure, np.asarray(axarr).flat[0].figure)

    if ipha is None:
        iphase_key = "iphase_" + kind
        if iphase_key in ds_sg:
            iphase_data = ds_sg[iphase_key].values
            if nd > 1:
                # Multi-wavelength case: get unique phases at the
                # specific wavelength iw
                ni = np.unique(iphase_data[iw, :])
            else:
                # Single wavelength case
                ni = np.unique(iphase_data)
        else:
            ni = [0]
    else:
        # Handle ipha as int-like scalar, DataArray scalar,
        # or 1-D iterable.
        if isinstance(ipha, xr.DataArray):
            ipha_arr = np.asarray(ipha.values)
        else:
            ipha_arr = np.asarray(ipha)

        if ipha_arr.ndim == 0:
            ni = [int(ipha_arr.item())]
        elif ipha_arr.ndim == 1:
            ni = [int(x) for x in ipha_arr.tolist()]
        else:
            raise ValueError(
                "ipha must be an int-like scalar or a 1-D "
                "array of int-like values"
            )

        # Validate that all given ipha values exist in iphase_data
        # at wavelength iw
        iphase_key = "iphase_" + kind
        if iphase_key in ds_sg:
            iphase_data = ds_sg[iphase_key].values
            if nd > 1:
                valid_phases = np.unique(iphase_data[iw, :])
            else:
                valid_phases = np.unique(iphase_data)
            valid_phases_set = set(
                np.asarray(valid_phases).astype(int).tolist()
            )
            for phase_idx in ni:
                if phase_idx not in valid_phases_set:
                    raise ValueError(
                        f"Phase index {phase_idx} not found in iphase_{kind} "
                        f"at wavelength index {iw}. "
                        f"Valid indices: {sorted(valid_phases.tolist())}"
                    )

    for i in ni:
        if nstk == 4:
            p_11 = phase[i, 0, :]  # P11
            p_12 = phase[i, 1, :]  # P12 = P21
            p_33 = phase[i, 2, :]
            p_43 = phase[i, 3, :]

            if np.max(p_11[:]) > 0.0:
                axarr[0, 0].semilogy(ang, p_11, label="%3i" % i)
                if show_trunc and phase_tr is not None:
                    axarr[0, 0].semilogy(ang, phase_tr[i, 0, :], "k--")
            axarr[0, 0].set_title(r"$P_{11}$" + labw)
            axarr[0, 0].grid()
            axarr[0, 0].set_xlim([0, 180])
            axarr[0, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])

            if np.max(p_11[:]) > 0.0:
                axarr[0, 1].plot(ang, -p_12 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[0, 1].plot(ang, -phase_tr[i, 1, :] / p_11, "k--")
            axarr[0, 1].set_title(r"-$P_{12}/P_{11}$")
            axarr[0, 1].grid()
            axarr[0, 1].set_xlim([0, 180])
            axarr[0, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])

            if np.max(p_11[:]) > 0.0:
                axarr[1, 0].plot(ang, p_33 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[1, 0].plot(ang, phase_tr[i, 2, :] / p_11, "k--")
            axarr[1, 0].set_title(r"$P_{33}/P_{11}$")
            axarr[1, 0].grid()
            axarr[1, 0].set_xlim([0, 180])
            axarr[1, 0].set_xlabel(r"$\theta$")
            axarr[1, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])

            if np.max(p_11[:]) > 0.0:
                axarr[1, 1].plot(ang, p_43 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[1, 1].plot(ang, phase_tr[i, 3, :] / p_11, "k--")
            axarr[1, 1].set_title(r"$P_{43}/P_{11}$")
            axarr[1, 1].grid()
            axarr[1, 1].set_xlim([0, 180])
            axarr[1, 1].set_xlabel(r"$\theta$")
            axarr[1, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])
        elif nstk == 6:
            p_11 = phase[i, 0, :]  # P11
            p_12 = phase[i, 1, :]  # P12 = P21
            p_22 = phase[i, 4, :]  # P22
            p_33 = phase[i, 2, :]  # P33
            p_34 = phase[i, 3, :]  # P34 = -P43
            p_44 = phase[i, 5, :]  # P44

            if np.max(p_11[:]) > 0.0:
                axarr[0, 0].semilogy(ang, p_11, label="%3i" % i)
                if show_trunc and phase_tr is not None:
                    axarr[0, 0].semilogy(ang, phase_tr[i, 0, :], "k--")
            axarr[0, 0].set_title(r"$P_{11}$" + labw)
            axarr[0, 0].grid()
            axarr[0, 0].set_xlim([0, 180])
            axarr[0, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])

            if np.max(p_11[:]) > 0.0:
                axarr[0, 1].plot(ang, -p_12 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[0, 1].plot(ang, -phase_tr[i, 1, :] / p_11, "k--")
            axarr[0, 1].set_title(r"-$P_{12}/P_{11}$")
            axarr[0, 1].grid()
            axarr[0, 1].set_xlim([0, 180])
            axarr[0, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])

            if np.max(p_11[:]) > 0.0:
                axarr[1, 0].plot(ang, p_33 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[1, 0].plot(ang, phase_tr[i, 2, :] / p_11, "k--")
            axarr[1, 0].set_title(r"$P_{33}/P_{11}$")
            axarr[1, 0].grid()
            axarr[1, 0].set_xlim([0, 180])
            axarr[1, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])
            if force_4stk:
                axarr[1, 0].set_xlabel(r"$\theta$")

            if np.max(p_11[:]) > 0.0:
                axarr[1, 1].plot(ang, p_34 / p_11)
                if show_trunc and phase_tr is not None:
                    axarr[1, 1].plot(ang, phase_tr[i, 3, :] / p_11, "k--")
            axarr[1, 1].set_title(r"$P_{34}/P_{11}$")
            axarr[1, 1].grid()
            axarr[1, 1].set_xlim([0, 180])
            axarr[1, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])
            if force_4stk:
                axarr[1, 1].set_xlabel(r"$\theta$")

            if not force_4stk:
                if np.max(p_11[:]) > 0.0:
                    axarr[2, 0].plot(ang, p_22 / p_11)
                    if show_trunc and phase_tr is not None:
                        axarr[2, 0].plot(ang, phase_tr[i, 4, :] / p_11, "k--")
                axarr[2, 0].set_title(r"$P_{22}/P_{11}$")
                axarr[2, 0].grid()
                axarr[2, 0].set_xlim([0, 180])
                axarr[2, 0].set_xlabel(r"$\theta$")
                axarr[2, 0].set_xticks([0, 30, 60, 90, 120, 150, 180])

                if np.max(p_11[:]) > 0.0:
                    axarr[2, 1].plot(ang, p_44 / p_11)
                    if show_trunc and phase_tr is not None:
                        axarr[2, 1].plot(ang, phase_tr[i, 5, :] / p_11, "k--")
                axarr[2, 1].set_title(r"$P_{44}/P_{11}$")
                axarr[2, 1].grid()
                axarr[2, 1].set_xlim([0, 180])
                axarr[2, 1].set_xlabel(r"$\theta$")
                axarr[2, 1].set_xticks([0, 30, 60, 90, 120, 150, 180])
                setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)

    setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    axarr[0, 0].legend(
        loc="upper center", fontsize="medium", labelspacing=0.01
    )

    return fig, axarr


def profile_view(
    ds_sg: xr.Dataset | MLUT,
    fig: Figure | None = None,
    ax: Axes | None = None,
    iw: int = 0,
    kind: str = "atm",
    zmax: float | None = None,
) -> tuple[Figure, Axes]:
    """
    Visualization of SMART-G vertical profile.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G, can be from simulation results
        or smartg input profile, containing optical depth and other
        profile data with variables 'OD_atm' or 'OD_oc', and related
        optical properties.
    fig : matplotlib.figure.Figure, optional
        Figure object. If None, creates a new figure.
    ax : matplotlib.axes.Axes, optional
        Axes object. If None, creates a new axes.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Profile type: 'atm' for atmospheric, 'oc' for oceanic. Default
        is 'atm'.
    zmax : float, optional
        Maximum altitude (for 'atm') or depth (for 'oc') to plot. If
        None, automatically determined from data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the profile plot.
    ax : matplotlib.axes.Axes
        Axes object containing the profile plot.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_sg is deprecated, use an "
            + "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    if ax is None:
        fig, ax = subplots(1, 1)
        fig.set_size_inches(5, 5)

    if fig is None:
        fig = cast(Figure, ax.figure)

    od_key = "OD_" + kind
    z_key = "z_" + kind

    od_data = ds_sg[od_key]
    nd = len(od_data.dims)

    # Handle multi-wavelength case
    labw = ""
    if nd > 1 and "wavelength" in od_data.dims:
        wavelength = ds_sg.coords["wavelength"].values
        labw = r" at $%.1f nm$" % wavelength[iw]

    z = ds_sg.coords[z_key].values
    if kind == "oc":
        sign = -1.0
        func = diff1_end
    else:
        sign = 1.0
        func = diff1

    d_z = np.abs(func(z))

    # Select wavelength index if multi-wavelength data
    if nd > 1 and "wavelength" in od_data.dims:
        od_data_sel = od_data.isel(wavelength=iw)
        sca_data = ds_sg["OD_sca_" + kind].isel(wavelength=iw)
        abs_data = ds_sg["OD_abs_" + kind].isel(wavelength=iw)
    else:
        od_data_sel = od_data
        sca_data = ds_sg["OD_sca_" + kind]
        abs_data = ds_sg["OD_abs_" + kind]

    # Extract and compute optical depths
    d_tau = sign * func(od_data_sel.values)
    d_tau_sca = sign * func(sca_data.values)
    d_tau_abs = sign * func(abs_data.values)
    if kind == "atm":
        if nd > 1 and "wavelength" in od_data.dims:
            d_tau_ext_a = sign * func(ds_sg["OD_p"].isel(wavelength=iw).values)
            d_tau_sca_r = sign * func(ds_sg["OD_r"].isel(wavelength=iw).values)
            d_tau_abs_g = sign * func(ds_sg["OD_g"].isel(wavelength=iw).values)
        else:
            d_tau_ext_a = sign * func(ds_sg["OD_p"].values)
            d_tau_sca_r = sign * func(ds_sg["OD_r"].values)
            d_tau_abs_g = sign * func(ds_sg["OD_g"].values)
        if nd > 1 and "wavelength" in od_data.dims:
            ssa_p = ds_sg["ssa_p_" + kind].isel(wavelength=iw).values
        else:
            ssa_p = ds_sg["ssa_p_" + kind].values
        d_tau_sca_a = d_tau_ext_a * ssa_p
        d_tau_abs_a = d_tau_ext_a * (1.0 - ssa_p)
        if np.max(d_tau_abs_a) > 0.0:
            ax.semilogx(
                (d_tau_abs_a / d_z), z, "r--", label=r"$\sigma_{abs}^{a+c}$"
            )
        if np.max(d_tau_sca_a) > 0.0:
            ax.semilogx(
                (d_tau_sca_a / d_z), z, "r", label=r"$\sigma_{sca}^{a+c}$"
            )
        if np.max(d_tau_abs_g) > 0.0:
            ax.semilogx(
                (d_tau_abs_g / d_z), z, "g--", label=r"$\sigma_{abs}^{gas}$"
            )
        ax.semilogx((d_tau_sca_r / d_z), z, "b", label=r"$\sigma_{sca}^{R}$")
        ax.set_xlim(1e-6, 10)
        xlabel("Vertical profile" + labw + r" $(km^{-1})$")
        ylabel(r"$z (km)$")
        if zmax is None:
            zmax = max(100.0, z.max())
        ax.set_ylim(0, zmax)
    else:
        if nd > 1 and "wavelength" in od_data.dims:
            d_tau_ext_p = sign * func(
                ds_sg["OD_p_oc"].isel(wavelength=iw).values
            )
            d_tau_ext_w = sign * func(ds_sg["OD_w"].isel(wavelength=iw).values)
            d_tau_abs_y = sign * func(ds_sg["OD_y"].isel(wavelength=iw).values)
            ssa_p = ds_sg["ssa_p_" + kind].isel(wavelength=iw).values
            ssa_w = ds_sg["ssa_w"].isel(wavelength=iw).values
            pine = ds_sg["pine_oc"].isel(wavelength=iw).values
        else:
            d_tau_ext_p = sign * func(ds_sg["OD_p_oc"].values)
            d_tau_ext_w = sign * func(ds_sg["OD_w"].values)
            d_tau_abs_y = sign * func(ds_sg["OD_y"].values)
            ssa_p = ds_sg["ssa_p_" + kind].values
            ssa_w = ds_sg["ssa_w"].values
            pine = ds_sg["pine_oc"].values
        d_tau_sca_p = d_tau_ext_p * ssa_p
        d_tau_abs_p = d_tau_ext_p * (1.0 - ssa_p)
        d_tau_sca_w = d_tau_ext_w * ssa_w
        d_tau_abs_w = d_tau_ext_w * (1.0 - ssa_w)
        d_tau_ine = d_tau_sca * pine
        if np.max(d_tau_abs_p) > 0.0:
            ax.semilogx(
                (d_tau_abs_p / d_z), z, "r--", label=r"$\sigma_{abs}^{p}$"
            )
        if np.max(d_tau_sca_p) > 0.0:
            ax.semilogx((d_tau_sca_p / d_z), z, "r", label=r"$\sigma_{sca}^{p}$")
        if np.max(d_tau_abs_w) > 0.0:
            ax.semilogx(
                (d_tau_abs_w / d_z), z, "b--", label=r"$\sigma_{abs}^{w}$"
            )
        if np.max(d_tau_sca_w) > 0.0:
            ax.semilogx((d_tau_sca_w / d_z), z, "b", label=r"$\sigma_{sca}^{w}$")
        if np.max(d_tau_abs_y) > 0.0:
            ax.semilogx(
                (d_tau_abs_y / d_z), z, "y--", label=r"$\sigma_{abs}^{y}$"
            )
        if np.max(d_tau_ine) > 0.0:
            ax.semilogx((d_tau_ine / d_z), z, "m:", label=r"$\sigma_{ine}^{}$")
        ax.set_xlim(1e-4, 10)
        xlabel("Vertical profile" + labw + r" $(m^{-1})$")
        ylabel(r"$z (m)$")
        if zmax is None:
            zmax = min(-100.0, z.min())
        ax.set_ylim(zmax, 0)
    ax.semilogx((d_tau / d_z), z, "k.-", label=r"$\sigma_{ext}^{tot}$")
    ax.semilogx((d_tau_abs / d_z), z, "k.--", label=r"$\sigma_{abs}^{tot}$")
    # ax.set_title('Vertical profile'+labw)
    ax.grid()
    ax.legend()

    try:
        ax2 = ax.twiny()
        nf = ds_sg["iphase_" + kind].values
        z_vals = ds_sg.coords[z_key].values
        ax2.plot(nf[1:], z_vals[1:], "m-", drawstyle="steps-post", label="i")
        ax2.set_xlabel("Phase Matrix index", color="m")
        ax2.tick_params("x", colors="m")
        ax2.xaxis.set_major_formatter(FormatStrFormatter("%i"))
        return fig, ax

    except Exception:
        return fig, ax


def input_view(
    ds_sg: xr.Dataset | MLUT,
    iw: int = 0,
    kind: str = "atm",
    zmax: float | None = None,
    ipha: int
    | Sequence[int]
    | np.ndarray[Any, Any]
    | xr.DataArray
    | None = None,
) -> None:
    """
    Visualization of SMART-G input profile and phase functions.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G, can be from simulation results
        or smartg input profile, containing phase function data and
        optical depth profiles.
    iw : int, optional
        Wavelength index for multi-wavelength simulations. Default is 0.
    kind : {'atm', 'oc'}, optional
        Profile type: 'atm' for atmospheric, 'oc' for oceanic. Default
        is 'atm'.
    zmax : float, optional
        Maximum altitude (for 'atm') or depth (for 'oc') to plot. If
        None, automatically determined from data.
    ipha : int, optional
        Absolute index of the phase function coming from Profile. If
        None, uses all unique indices.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = "\nUsing an MLUT for ds_sg is deprecated, use an xarray.Dataset instead."
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    if "phase_" + kind in ds_sg:
        fig = figure()
        phase_data = ds_sg["phase_" + kind].values
        nstk = phase_data.shape[1]
        if nstk == 4:
            fig.set_size_inches(12, 6)
            ax1 = subplot2grid((2, 3), (0, 0))
            ax2 = subplot2grid((2, 3), (0, 1))
            ax3 = subplot2grid((2, 3), (1, 0))
            ax4 = subplot2grid((2, 3), (1, 1))

            axarr: np.ndarray[Any, Any] = np.array([[ax1, ax2], [ax3, ax4]])

            _, _ = phase_view(ds_sg, iw=iw, axarr=axarr, kind=kind, ipha=ipha)

            ax5 = subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)

            profile_view(ds_sg, iw=iw, ax=ax5, kind=kind, zmax=zmax)
        else:
            fig.set_size_inches(12, 9)
            ax1 = subplot2grid((3, 3), (0, 0))
            ax2 = subplot2grid((3, 3), (0, 1))
            ax3 = subplot2grid((3, 3), (1, 0))
            ax4 = subplot2grid((3, 3), (1, 1))
            ax5 = subplot2grid((3, 3), (2, 0))
            ax6 = subplot2grid((3, 3), (2, 1))

            axarr = np.array([[ax1, ax2], [ax3, ax4], [ax5, ax6]])

            _, _ = phase_view(ds_sg, iw=iw, axarr=axarr, kind=kind, ipha=ipha)

            ax7 = subplot2grid((3, 3), (0, 2), rowspan=2, colspan=1)

            profile_view(ds_sg, iw=iw, ax=ax7, kind=kind, zmax=zmax)
    else:
        fig, _ = profile_view(ds_sg, iw=iw, kind=kind, zmax=zmax)

    tight_layout()


def compare(
    ds_sg: xr.Dataset | MLUT,
    ds_ref: xr.Dataset | MLUT,
    field: str = "up (TOA)",
    errb: bool = False,
    log_i: bool = False,
    u_sign: int = 1,
    same_u_conv: bool = True,
    u_symetry: bool = True,
    nparam: int = 4,
    vmax: Sequence[float] | None = None,
    vmin: Sequence[float] | None = None,
    emax: Sequence[float] | None = None,
    ermax: Sequence[float] | None = None,
    same_azi_conv: bool = True,
    azimuth: Sequence[float] | None = None,
    title: str = "",
    sza_max: float = 89.0,
    zenith_title: str = r"$SZA (°)$",
    errref: np.ndarray[Any, Any] | Sequence[float] | None = None,
) -> Figure:
    """
    Compare results of two SMART-G simulations in two different azimuth
    planes.

    Parameters
    ----------
    ds_sg : Dataset
        An xarray Dataset from SMART-G simulation.
    ds_ref : Dataset
        Reference Dataset for comparison.
    field : str, optional
        Name of the output level to compare. Default is 'up (TOA)'.
    errb : bool, optional
        If True, show error bars for ds_sg (requires stdev data).
        Default is False.
    log_i : bool, optional
        If True, plot Intensity (I) in log10 scale. Default is False.
    u_sign : int, optional
        Sign convention for U parameter. Default is 1.
    same_u_conv : bool, optional
        If True, ds_sg and ds_ref have the same U convention. Default is
        True.
    u_symetry : bool, optional
        If True, U changes sign convention for the two halves of the
        plane. Default is True.
    nparam : int, optional
        Number of parameters to plot: 4 for I,Q,U,DoLP (default); 5 adds
        V; 2 keeps only I,DoLP.
    vmin, vmax : list, optional
        List of min/max values for each parameter. If None, use
        defaults.
    emax : list, optional
        List of max absolute error scales for each parameter. If None,
        use defaults.
    ermax : list, optional
        List of max relative error scales (in %) for each parameter. If
        None, use defaults.
    same_azi_conv : bool, optional
        If True, ds_sg and ds_ref have the same azimuth convention.
        Default is True.
    azimuth : list, optional
        List of two azimuth angles to display. Default is [0., 90.].
    title : str, optional
        Title for the figure. Default is empty string.
    sza_max : float, optional
        Maximum SZA (Solar Zenith Angle) for x-axis limits. Default is
        89.
    zenith_title : str, optional
        Label for zenith angle axis. Default is '$SZA (°)$'.
    errref : array-like, optional
        Reference intensity absolute error. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the comparison plots.
    """

    if isinstance(ds_sg, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_sg is deprecated, "
            + "use an xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_sg = ds_sg.to_xarray()

    if isinstance(ds_ref, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds_ref is deprecated, "
            + "use an xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds_ref = ds_ref.to_xarray()

    from pylab import subplots

    if vmax is None:
        vmax_values = [0.1] * nparam
    else:
        vmax_values = list(vmax)
    if vmin is None:
        vmin_values = [-0.1] * nparam
    else:
        vmin_values = list(vmin)
    if emax is None:
        emax_values = [0.1] * nparam
    else:
        emax_values = list(emax)
    if ermax is None:
        ermax_values = [0.1] * nparam
    else:
        ermax_values = list(ermax)
    if azimuth is None:
        azimuth_values = [0.0, 90.0]
    else:
        azimuth_values = list(azimuth)
    if len(azimuth_values) < 2:
        raise ValueError("azimuth must contain at least two angles")
    stokes_t = ["I", "Q", "U", "V"]
    stokes = stokes_t[: nparam - 1]
    sign_t = [1, 1, u_sign * 1, 1, 1]  # sign convention for both datasets
    sign = sign_t[: nparam - 1] + [1]
    if same_u_conv:
        diffsign_t = [1, 1, 1, 1, 1]  # sign convention difference
    else:
        diffsign_t = [1, 1, -1, 1, 1]
    diffsign = diffsign_t[: nparam - 1] + [1]
    if u_symetry:
        symetry_t = [1, 1, 1, 1, 1]
    else:
        symetry_t = [1, 1, -1, 1, 1]
    symetry = symetry_t[: nparam - 1] + [1]
    fig, ax = subplots(
        3,
        nparam,
        sharey=False,
        sharex=True,
        gridspec_kw=dict(hspace=0.2, wspace=0.3),
    )
    fig.set_size_inches(nparam * 3, 8)
    fig.set_dpi(600)
    fig.suptitle(title)

    def _isel_values(
        da: xr.DataArray, dim_name: str, idx: int
    ) -> np.ndarray[Any, Any]:
        return np.asarray(da.isel({dim_name: idx}).values)

    for i in range(nparam):
        s: xr.DataArray
        sref: xr.DataArray
        e: xr.DataArray | None = None
        if i != nparam - 1:
            s = cast(xr.DataArray, ds_sg[stokes[i] + "_" + field])
            sref = cast(xr.DataArray, ds_ref[stokes[i] + "_" + field])

            # Determine which dimension is azimuth angle and get
            # coordinate values
            if "Azimuth angles" in s.dims:
                az_idx = s.dims.index("Azimuth angles")
                if az_idx == 0:
                    th = s.coords[list(s.dims)[1]].values
                else:
                    th = s.coords[list(s.dims)[0]].values
            else:
                # Fallback: use first dimension coordinate
                th = s.coords[list(s.dims)[0]].values

            # Extract description from attributes
            desc = s.attrs.get("latex_name", stokes[i])
            desc = mdesc(str(desc))

            if errb:
                e = cast(
                    xr.DataArray,
                    ds_sg[stokes[i] + "_" + "stdev" + "_" + field],
                )

            if log_i and stokes[i] == "I":
                s = cast(xr.DataArray, np.log10(s))
                sref = cast(xr.DataArray, np.log10(sref))
                desc = r"$log_{10}$ " + desc
        else:
            stk_i = ds_sg["I" + "_" + field]
            stk_u = ds_sg["Q" + "_" + field]
            stk_q = ds_sg["U" + "_" + field]

            ip = np.sqrt(stk_u * stk_u + stk_q * stk_q)
            s = cast(xr.DataArray, (ip / stk_i) * 100)

            iref = ds_ref["I" + "_" + field]
            qref = ds_ref["Q" + "_" + field]
            uref = ds_ref["U" + "_" + field]
            sref = cast(
                xr.DataArray,
                (np.sqrt(qref * qref + uref * uref) / iref) * 100,
            )

            # Get description
            i_desc = stk_i.attrs.get("latex_name", "I")
            desc = "DoLP" + i_desc[1:]
            desc = mdesc(str(desc))

            # Determine azimuth coordinate
            if "Azimuth angles" in s.dims:
                az_idx = s.dims.index("Azimuth angles")
                if az_idx == 0:
                    th = s.coords[list(s.dims)[1]].values
                else:
                    th = s.coords[list(s.dims)[0]].values
            else:
                th = s.coords[list(s.dims)[0]].values

            if errb:
                d_i = ds_sg["I" + "_" + "stdev" + "_" + field]
                d_q = ds_sg["Q" + "_" + "stdev" + "_" + field]
                d_u = ds_sg["U" + "_" + "stdev" + "_" + field]
                d_ip = np.sqrt(d_q * d_q + d_u * d_u)
                e = cast(xr.DataArray, (d_i / stk_i + d_ip / ip) * s)

        vmi = vmin_values[i]
        vma = vmax_values[i]
        ema = emax_values[i]
        erma = ermax_values[i]

        for phi0, sym1, sym2, labref in [
            (azimuth_values[0], "r", "-", "ref."),
            (azimuth_values[1], "g", "-", ""),
        ]:
            # for phi0,sym1,sym2,labref in
            # [(azimuth[0],'r','.','ref.'),(azimuth[1],'g','.','')]:

            # both points at their own abscissas
            if same_azi_conv:
                # For xarray, use .sel() to select by azimuth
                # angle value
                if "Azimuth angles" in s.dims:
                    az_dim = "Azimuth angles"

                    # Find closest azimuth angle values
                    az_vals = s.coords["Azimuth angles"].values
                    phi0_idx = int(np.argmin(np.abs(az_vals - phi0)))
                    phi180_idx = int(
                        np.argmin(np.abs(az_vals - (180.0 - phi0)))
                    )

                    refp = sign[i] * _isel_values(sref, az_dim, phi0_idx)
                    refm = sign[i] * _isel_values(sref, az_dim, phi180_idx)
                    sp = (
                        diffsign[i]
                        * sign[i]
                        * _isel_values(s, az_dim, phi0_idx)
                    )
                    sm = (
                        symetry[i]
                        * diffsign[i]
                        * sign[i]
                        * _isel_values(s, az_dim, phi180_idx)
                    )

                    if errb:
                        if e is None:
                            raise RuntimeError(
                                "Error data must be initialized when errb "
                                + "is True"
                            )
                        dsp = _isel_values(e, az_dim, phi0_idx)
                        dsm = _isel_values(e, az_dim, phi180_idx)
                    else:
                        (dsp, dsm) = (0, 0)
                else:
                    # Fallback if dimension naming differs
                    refp = sign[i] * sref.values.ravel()
                    refm = sign[i] * sref.values.ravel()
                    sp = diffsign[i] * sign[i] * s.values.ravel()
                    sm = symetry[i] * diffsign[i] * sign[i] * s.values.ravel()
                    if errb:
                        if e is None:
                            raise RuntimeError(
                                "Error data must be initialized when errb "
                                + "is True"
                            )
                        dsp = e.values.ravel()
                        dsm = e.values.ravel()
                    else:
                        (dsp, dsm) = (0, 0)
            else:
                # Different azimuth convention - swap angle selection
                if "Azimuth angles" in s.dims:
                    az_dim = "Azimuth angles"
                    az_vals = s.coords["Azimuth angles"].values
                    phi0_idx = int(np.argmin(np.abs(az_vals - phi0)))
                    phi180_idx = int(
                        np.argmin(np.abs(az_vals - (180.0 - phi0)))
                    )

                    refp = sign[i] * _isel_values(sref, az_dim, phi180_idx)
                    refm = sign[i] * _isel_values(sref, az_dim, phi0_idx)
                    sp = (
                        diffsign[i]
                        * sign[i]
                        * _isel_values(s, az_dim, phi0_idx)
                    )
                    sm = (
                        symetry[i]
                        * diffsign[i]
                        * sign[i]
                        * _isel_values(s, az_dim, phi180_idx)
                    )

                    if errb:
                        if e is None:
                            raise RuntimeError(
                                "Error data must be initialized when errb "
                                + "is True"
                            )
                        dsp = _isel_values(e, az_dim, phi0_idx)
                        dsm = _isel_values(e, az_dim, phi180_idx)
                    else:
                        (dsp, dsm) = (0, 0)
                else:
                    refp = sign[i] * sref.values.ravel()
                    refm = sign[i] * sref.values.ravel()
                    sp = diffsign[i] * sign[i] * s.values.ravel()
                    sm = symetry[i] * diffsign[i] * sign[i] * s.values.ravel()
                    if errb:
                        if e is None:
                            raise RuntimeError(
                                "Error data must be initialized when errb "
                                + "is True"
                            )
                        dsp = e.values.ravel()
                        dsm = e.values.ravel()
                    else:
                        (dsp, dsm) = (0, 0)

            ax[0, i].plot(th, refp, "k" + ".")
            ax[0, i].plot(-th, refm, "k" + ".", label=labref)
            ax[0, i].errorbar(th, sp, fmt=sym1 + "")
            ax[0, i].errorbar(
                -th,
                sm,
                fmt=sym1 + "",
                label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
            )
            ax[0, i].set_ylim([vmi, vma])
            ax[0, i].set_xlim([-sza_max, sza_max])
            ax[0, i].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

            if log_i and i == 0:
                if errb:
                    ax[1, i].errorbar(
                        th,
                        10**sp - 10**refp,
                        yerr=dsp,
                        fmt=sym1 + sym2,
                        label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                        ecolor="k",
                        capsize=2,
                    )
                    ax[1, i].errorbar(
                        -th,
                        10**sm - 10**refm,
                        yerr=dsm,
                        fmt=sym1 + sym2,
                        ecolor="k",
                        capsize=2,
                    )
                else:
                    ax[1, i].errorbar(
                        th,
                        10**sp - 10**refp,
                        fmt=sym1 + sym2,
                        label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                        ecolor="k",
                        capsize=2,
                    )
                    ax[1, i].errorbar(
                        -th,
                        10**sm - 10**refm,
                        fmt=sym1 + sym2,
                        ecolor="k",
                        capsize=2,
                    )

            else:
                if errb:
                    ax[1, i].errorbar(
                        th,
                        sp - refp,
                        yerr=dsp,
                        fmt=sym1 + sym2,
                        label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                        ecolor=sym1,
                        capsize=2,
                    )
                    ax[1, i].errorbar(
                        -th,
                        sm - refm,
                        yerr=dsm,
                        fmt=sym1 + sym2,
                        ecolor=sym1,
                        capsize=2,
                    )
                else:
                    ax[1, i].errorbar(
                        th,
                        sp - refp,
                        fmt=sym1 + sym2,
                        label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                        ecolor=sym1,
                        capsize=2,
                    )
                    ax[1, i].errorbar(
                        -th, sm - refm, fmt=sym1 + sym2, ecolor=sym1, capsize=2
                    )
            ax[1, i].set_ylim([-1 * ema, ema])
            ax[1, i].set_xlim([-sza_max, sza_max])

            if errb:
                ax[2, i].errorbar(
                    th,
                    (sp - refp) / refp * 100,
                    yerr=dsp / abs(refp) * 100,
                    fmt=sym1 + sym2,
                    label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                    ecolor=sym1,
                    capsize=2,
                )
                ax[2, i].errorbar(
                    -th,
                    (sm - refm) / refm * 100,
                    yerr=dsm / abs(refm) * 100,
                    fmt=sym1 + sym2,
                    ecolor=sym1,
                    capsize=2,
                )
                if i == 0 and errref is not None:
                    errref_arr = np.asarray(errref)
                    ax[2, 0].plot(th, errref_arr / refp * 100, sym1 + "-.")
                    ax[2, 0].plot(th, -errref_arr / refp * 100, sym1 + "-.")
                    ax[2, 0].plot(-th, errref_arr / refm * 100, sym1 + "-.")
                    ax[2, 0].plot(-th, -errref_arr / refm * 100, sym1 + "-.")
            else:
                ax[2, i].errorbar(
                    th,
                    (sp - refp) / refp * 100,
                    fmt=sym1 + sym2,
                    label=r"$\Phi=%.0f-%.0f$" % (phi0, 180.0 - phi0),
                    ecolor="k",
                    capsize=2,
                )
                ax[2, i].errorbar(
                    -th,
                    (sm - refm) / refm * 100,
                    fmt=sym1 + sym2,
                    ecolor="k",
                    capsize=2,
                )

            if i != nparam - 1:
                ax[2, i].set_ylim([-1 * erma, erma])
            else:
                ax[2, i].set_ylim([-1 * erma, erma])

            ax[2, i].set_xlim([-sza_max, sza_max])
            ax[1, i].plot([-sza_max, sza_max], [0.0, 0.0], "k--")
            ax[2, i].plot([-sza_max, sza_max], [0.0, 0.0], "k--")
            ax[1, i].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

            ax[0, i].set_title(desc)
            if i == 0:
                ax[0, i].legend(
                    loc="upper center", fontsize=8, labelspacing=0.0
                )
                # ax[1,i].text(
                # -50.,ema*0.75,r'$N_{\Phi}$:%i, $N_{\theta}$:%i'%\
                #         (S.axes[0].shape[0],S.axes[1].shape[0]))
                ax[1, i].set_ylabel(r"$\Delta$")
                ax[2, i].set_ylabel(r"$\Delta (\%)$")
            ax[2, i].set_xlabel(zenith_title)
    return fig


def _bin_edges(
    x: np.ndarray[Any, Any],
    min: float | None = None,
    max: float | None = None,
) -> np.ndarray[Any, Any]:
    """Helper function to compute bin edges from bin centers"""
    edges = np.zeros(len(x) + 1)
    edges[1:-1] = (x[1:] + x[:-1]) / 2.0
    edges[0] = 2 * x[0] - edges[1]
    edges[-1] = 2 * x[-1] - edges[-2]
    if min is not None:
        edges[0] = min
    if max is not None:
        edges[-1] = max
    return edges


def _parse_subplot_position(
    position: int | str | tuple[int, int, int],
) -> int | tuple[int, int, int]:
    """
    Convert subplot position to format for add_subplot.

    Parameters
    ----------
    position : int, str, or tuple
        - int : 3-digit integer (e.g., 211)
        - str : converted to int (e.g., '211')
        - tuple : 3-value tuple (rows, cols, position)
          for positions >= 10

    Returns
    -------
    int or tuple
        Format compatible with fig.add_subplot()
    """
    if isinstance(position, int):
        return position
    elif isinstance(position, str):
        return int(position)
    elif isinstance(position, tuple) and len(position) == 3:
        return position
    else:
        raise ValueError(
            "position must be int, str, or 3-element tuple, "
            + f"got {type(position)}: {position}"
        )


def plot_polar(
    da: xr.DataArray,
    index: int | np.ndarray | list[int] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    rect: int | str | tuple[int, int, int] = 211,
    sub: int | str | tuple[int, int, int] = 212,
    sym: bool = True,
    swap: bool | Literal["auto"] = "auto",
    fig: Figure | None = None,
    cmap: str | mcolors.Colormap | None = None,
    semi: bool = False,
) -> Figure:
    """
    Contour and optionally transect of 2D DataArray on a semi-polar
    plot.

    xarray version of luts.plot_polar, compatible with DataArray
    objects.

    Parameters
    ----------
    da : DataArray
        2D data array with dimensions (angle, radius) or similar.
        Angle is assumed to be in degrees and is not scaled.
    index : int, ndarray, or list, optional
        Index or indices to transect in the first dimension. If None
        (default), no transect is drawn.
    vmin, vmax : float, optional
        Range of values. If None, determined from data
    rect : int, str, or tuple, optional
        Subplot position of the main plot.
        - int: 3-digit integer (e.g., 211)
        - str: string converted to int (e.g., '211')
        - tuple: (rows, cols, position) for positions >= 10
          (e.g., (4, 4, 13))
    sub : int, str, or tuple, optional
        Subplot position of the transect (same format options as rect).
    sym : bool, optional
        If True, the transect uses symmetrical axis
    swap : bool or str, optional
        If True, swap the order of the 2 axes. If 'auto', searches for
        'azi' in both dimension names
    fig : Figure, optional
        Destination figure. If None, create a new figure.
    cmap : Colormap, optional
        Colormap to use.
    semi : bool, optional
        If True, use semi-polar (180 deg), otherwise polar (360 deg).

    Returns
    -------
    fig : Figure
        The figure containing the plot.
    """

    # Convert subplot positions
    rect = _parse_subplot_position(rect)
    sub = _parse_subplot_position(sub)

    # Initialization
    Phimax = 360.0
    if semi:
        Phimax = 180.0

    assert da.ndim == 2, "DataArray must be 2D"

    show_sub = index is not None
    if fig is None:
        if show_sub:
            fig = cast(Figure, figure(figsize=(4.5, 4.5)))
        else:
            fig = cast(Figure, figure(figsize=(4.5, 6)))

    if fig is None:
        raise RuntimeError("Failed to initialize figure in plot_polar.")

    # Get dimension names
    dim_names = list(da.dims)
    dim0_name, dim1_name = str(dim_names[0]), str(dim_names[1])

    # Determine if we need to swap axes
    if swap == "auto":
        if ("azi" in dim1_name.lower()) and ("azi" not in dim0_name.lower()):
            swap = True
        else:
            swap = False

    # Get axes values and data
    if swap:
        ax1_name, ax2_name = dim1_name, dim0_name
        ax1 = da.coords[dim1_name].values
        ax2 = da.coords[dim0_name].values
        data = da.values.T  # Transpose to get (angle, radius)
    else:
        ax1_name, ax2_name = dim0_name, dim1_name
        ax1 = da.coords[dim0_name].values
        ax2 = da.coords[dim1_name].values
        data = da.values

    # Determine axis labels
    label1 = da.coords[ax1_name].attrs.get("long_name", ax1_name)
    label2 = da.coords[ax2_name].attrs.get("long_name", ax2_name)

    # Determine min/max values
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    vmin_val = float(vmin)
    vmax_val = float(vmax)
    if vmin_val == vmax_val:
        vmin_val -= 0.001
        vmax_val += 0.001
    if vmin_val > vmax_val:
        vmin_val, vmax_val = vmax_val, vmin_val

    # Semi-polar axis setup
    ax1_scaled = ax1
    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    ax2_scaled = (ax2 - ax2_min) / (ax2_max - ax2_min) * 90.0

    # Setup angle and radius axis locators/formatters
    grid_locator1 = angle_helper.LocatorDMS(
        {True: 4, False: 8}[semi], include_last=False
    )
    tick_formatter1 = angle_helper.FormatterDMS()

    class Locator(object):
        def __call__(self, *args):
            return [np.array([0, 30, 60, 90]), 4, 1.0]

    class Formatter(object):
        def __call__(self, *args):
            return list(
                map(
                    lambda x: "{:.3g}".format(x),
                    np.linspace(ax2_min, ax2_max, 4),
                )
            )

    # Radius axis locator/formatter
    if (
        (ax2_min < 10.0)
        and (ax2_min >= 0)
        and (ax2_max <= 90)
        and (ax2_max > 80)
    ):
        grid_locator2 = angle_helper.LocatorDMS(4)
        tick_formatter2 = angle_helper.FormatterDMS()
    else:
        grid_locator2 = Locator()
        tick_formatter2 = Formatter()

    # Setup transform
    tr_translate = Affine2D().translate(0, 0)
    tr_scale = Affine2D().scale(np.pi / 180.0, 1.0)
    tr = (
        tr_translate
        + tr_scale
        + PolarAxes.PolarTransform(apply_theta_transforms=False)
    )

    # Create grid helper and floating subplot
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(0.0, Phimax, 0.0, 90.0),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=tick_formatter2,
    )

    # Unpack rect if it's a tuple
    if isinstance(rect, tuple):
        ax_polar = floating_axes.FloatingSubplot(
            fig, *rect, grid_helper=grid_helper
        )
    else:
        ax_polar = floating_axes.FloatingSubplot(
            fig, rect, grid_helper=grid_helper
        )
    fig.add_subplot(ax_polar)

    # Adjust polar axis
    ax_polar.grid(True)
    ax_polar.axis["left"].set_axis_direction("bottom")
    ax_polar.axis["right"].set_axis_direction("top")
    ax_polar.axis["bottom"].set_visible(False)
    ax_polar.axis["top"].set_axis_direction("bottom")
    ax_polar.axis["top"].toggle(ticklabels=True, label=True)
    ax_polar.axis["top"].major_ticklabels.set_axis_direction("top")
    ax_polar.axis["top"].label.set_axis_direction("top")

    ax_polar.axis["top"].axes.text(
        0.72,
        0.98,
        label1,
        transform=ax_polar.transAxes,
        ha="left",
        va="bottom",
    )
    ax_polar.axis["left"].axes.text(
        0.10,
        -0.03,
        label2,
        transform=ax_polar.transAxes,
        ha="center",
        va="top",
    )

    # Create auxiliary polar axes
    aux_ax_polar = ax_polar.get_aux_axes(tr)
    aux_ax_polar.patch = ax_polar.patch
    ax_polar.patch.zorder = 0.9

    # Initialize cartesian axis for transect
    ax_cart: Axes | None = None
    if show_sub:
        # Unpack sub if it's a tuple
        if isinstance(sub, tuple):
            ax_cart = fig.add_subplot(*sub)
        else:
            ax_cart = fig.add_subplot(sub)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin_val, vmax_val)
        ax_cart.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax_cart.grid(True)

    # Setup colormap
    cmap_obj: mcolors.Colormap
    if cmap is None:
        cmap_obj = plt.get_cmap("rainbow").copy()
    elif isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap).copy()
    else:
        cmap_obj = cmap
    cmap_obj.set_under("black")
    cmap_obj.set_over("white")
    cmap_obj.set_bad("0.5")

    # Draw colormesh
    r, t = np.meshgrid(
        _bin_edges(ax2_scaled, min=0, max=90), _bin_edges(ax1_scaled)
    )
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    im = aux_ax_polar.pcolormesh(
        t, r, masked_data, cmap=cmap_obj, vmin=vmin_val, vmax=vmax_val
    )

    # Draw transects if requested
    if show_sub:
        if ax_cart is None:
            raise RuntimeError("Failed to initialize transect axis.")
        # Ensure index is array-like
        if isinstance(index, (int, np.integer)):
            indexes = [index]
        elif isinstance(index, (list, tuple)):
            indexes = list(index)
        else:
            indexes = np.atleast_1d(index).astype(int)

        for ii, idx in enumerate(indexes):
            if semi:
                mirror_index = -1 - idx
            else:
                mirror_index = (
                    ax1_scaled.shape[0] // 2 + idx
                ) % ax1_scaled.shape[0]

            # Draw line over colormesh
            vertex0 = np.array([[0, 0], [ax1_scaled[idx], ax2_max]])
            vertex1 = np.array([[0, 0], [ax1_scaled[mirror_index], ax2_max]])
            aux_ax_polar.plot(vertex0[:, 0], vertex0[:, 1], "w")
            if sym:
                aux_ax_polar.plot(
                    vertex1[:, 0], vertex1[:, 1], "w--", linewidth=2
                )

            # Plot transects
            color = ["k", "r", "g", "b", "m", "y"][ii % 6]
            ax_cart.plot(ax2, data[idx, :], "-" + color)
            if sym:
                ax_cart.plot(-ax2, data[mirror_index, :], "--" + color)

    # Add colorbar
    cbar = fig.colorbar(
        im,
        ax=ax_polar,
        orientation="horizontal",
        extend="both",
        ticks=np.linspace(vmin_val, vmax_val, 5),
        shrink=1.0,
        pad=0.15,
        fraction=0.06,
        aspect=20,
    )

    # Format colorbar tick labels with scientific notation when needed
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits(
        (-2, 5)
    )  # Use scientific notation for numbers < 10^-2 or >= 10^5
    cbar.ax.xaxis.set_major_formatter(formatter)

    # Add title
    if "latex_name" not in da.attrs and da.name is not None:
        da.attrs["latex_name"] = mdesc(str(da.name))
    title = da.attrs["latex_name"]

    if title is not None:
        ax_polar.set_title(title, weight="bold", position=(0.05, 0.97))

    return fig


def transect_2d(
    da: xr.DataArray,
    index: int | Sequence[int] | np.ndarray[Any, Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    sym: bool = True,
    swap: bool | Literal["auto"] = "auto",
    fig: Figure | None = None,
    sub: int | str | tuple[int, int, int] = 121,
    color: str = "k",
    percent: bool = False,
    fmt: str = "-",
) -> Figure:
    """
    Transect of a 2D DataArray.

    Parameters
    ----------
    da : DataArray
        2D data array to display.
    index : int or array-like, optional
        Index or indices to transect. If None, the first index (0) is
        used.
    vmin, vmax : float, optional
        Value range.
    sym : bool
        Use a symmetrical x-axis.
    swap : bool or 'auto'
        Swap axes if needed.
    fig : Figure, optional
        Destination figure.
    sub : int, str, or tuple
        Subplot position.
        - int: 3-digit integer (e.g., 121)
        - str: string converted to int (e.g., '121')
        - tuple: (rows, cols, position) for positions >= 10
          (e.g., (4, 4, 13))
    color : str
        Color for the plot.
    percent : bool
        If True, set scale to 0-100%.
    fmt : str
        Plot format string.

    Returns
    -------
    fig : Figure
        Figure containing the transect plot.
    """

    assert da.ndim == 2, "DataArray must be 2D"

    if fig is None:
        fig = cast(Figure, figure(figsize=(4.5, 2.5)))

    # Get dimension names
    dim_names = [str(dim) for dim in da.dims]
    dim0_name, dim1_name = dim_names[0], dim_names[1]

    if swap == "auto":
        if ("azi" in dim1_name.lower()) and ("azi" not in dim0_name.lower()):
            swap = True
        else:
            swap = False

    # Get axes and data
    if swap:
        ax1 = da.coords[dim1_name].values
        ax2 = da.coords[dim0_name].values
        name2 = dim0_name
        data = da.values.T
    else:
        ax1 = da.coords[dim0_name].values
        ax2 = da.coords[dim1_name].values
        name2 = dim1_name
        data = da.values

    # Determine value range
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    vmin_val = float(vmin)
    vmax_val = float(vmax)
    if vmin_val == vmax_val:
        vmin_val -= 0.001
        vmax_val += 0.001
    if vmin_val > vmax_val:
        vmin_val, vmax_val = vmax_val, vmin_val
    if percent:
        vmin_val = 0.0
        vmax_val = 100.0

    ax1_scaled = ax1
    label2 = da.coords[name2].attrs.get("latex_name", name2)

    # Ensure index is an integer
    if index is not None:
        if isinstance(index, (list, tuple)):
            index_val = int(index[0])
        else:
            index_arr = np.asarray(index)
            if index_arr.ndim == 0:
                index_val = int(index_arr.item())
            else:
                index_val = int(index_arr.reshape(-1)[0])
    else:
        index_val = 0

    mirror_index = (ax1_scaled.shape[0] // 2 + index_val) % ax1_scaled.shape[0]

    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)

    # Parse subplot specification
    sub = _parse_subplot_position(sub)

    # Create a valid marker name from sub (handle both int and tuple)
    if isinstance(sub, tuple):
        marker_key = "_".join(map(str, sub))
    else:
        marker_key = str(sub)
    marker_name = f"_transect_2d_sub_{marker_key}"

    # Check if subplot already exists
    ax_cart: Axes | None = None
    if hasattr(fig, marker_name):
        existing_ax = getattr(fig, marker_name)
        if isinstance(existing_ax, Axes):
            ax_cart = existing_ax

    is_new_axes = ax_cart is None
    if is_new_axes:
        # Unpack sub if it's a tuple
        if isinstance(sub, tuple):
            ax_cart = fig.add_subplot(*sub)
        else:
            ax_cart = fig.add_subplot(sub)
        setattr(fig, marker_name, ax_cart)
        ax_cart.grid(True)
        ax_cart.set_xlabel(label2)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin_val, vmax_val)
    else:
        if ax_cart is None:
            raise RuntimeError("Failed to retrieve existing transect axes.")
        # Expand ylim to accommodate new data
        current_ylim = ax_cart.get_ylim()
        new_vmin = min(current_ylim[0], vmin_val)
        new_vmax = max(current_ylim[1], vmax_val)
        ax_cart.set_ylim(new_vmin, new_vmax)

    if ax_cart is None:
        raise RuntimeError("Failed to initialize transect axes.")

    ax_cart.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    # Plot transects
    ax_cart.plot(ax2, data[index_val, :], fmt, color=color)
    if sym:
        ax_cart.plot(-ax2, data[mirror_index, :], fmt, color=color)

    # Add title
    if "latex_name" not in da.attrs and da.name is not None:
        da.attrs["latex_name"] = mdesc(str(da.name))
    title = da.attrs.get("latex_name")

    if title is not None:
        ax_cart.set_title(str(title))

    return fig


def receiver_view(
    ds_sg_out: xr.Dataset,
    cat: int | Sequence[int] = 0,
    log_color_scale: bool = False,
    save_path: str | None = None,
    mtoa: float = 1320,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "none",
    flux_unit: Literal["W", "kW", "MW"] = "W",
) -> None:
    """
    Plot receiver irradiance from a SMART-G simulation output.

    The function reads receiver weights from
    ``ds_sg_out['C_Receiver']``, optionally selecting and summing one or
    more categories, converts the cell size from km to m using
    ``ds_sg_out.attrs['S_Cell']``, normalizes by cell area, multiplies
    by ``mtoa``, applies the selected power ``flux_unit``, and displays
    the 2-D map with :func:`matplotlib.pyplot.imshow`.

    The displayed axes are labeled as relative receiver coordinates (m):
    ``x`` points upward and ``y`` points to the left.

    Parameters
    ----------
    ds_sg_out : Dataset
        SMART-G output Dataset containing simulation results.
    cat : int or sequence of int, default=0
        Receiver category index as defined in [1]_.

        - ``0``: sum of all categories (scalar only).
        - ``1``-``8``: a single specific category.
        - A list / tuple / array of ints in ``1``-``8``: the selected
          categories are summed together. ``0`` is not allowed in this
          case.
    log_color_scale : bool, optional
        If ``True``, use a logarithmic color normalization. Default:
        False
    save_path : str, optional
        Output filename (without extension). If provided, the figure is
        saved as ``<save_path>.pdf``. Default: None
    mtoa : float, optional
        Solar flux at TOA (W/m²). Multiplicative factor applied to the
        receiver weights before display. Typically the TOA solar
        irradiance for physical units, but can be set to any value to
        rescale monochromatic simulation outputs. Default: 1320
    vmin : float, optional
        Lower color limit for linear scale. Ignored when
        ``log_color_scale=True``. Default: None
    vmax : float, optional
        Upper color limit for linear scale. Ignored when
        ``log_color_scale=True``. Default: None
    interpolation : str, optional
        Default: 'none'
    flux_unit : str, optional
        Power unit used for displayed irradiance values. Choices are 'W'
        (Watt), 'kW' (kiloWatt), 'MW' (MegaWatt). Default: 'W'.


    Returns
    -------
    None
        This function creates a matplotlib figure and colorbar, and
        optionally saves the figure to disk.

    References
    ----------
    .. [1] Moulana, M., Elias, T., Cornet, C., & Ramon, D. (2019).
           First results to evaluate losses and gains in solar radiation
           collected by solar tower plants. *SOLARPACES 2018:
           International Conference on Concentrating Solar Power and
           Chemical Energy Systems*. https://doi.org/10.1063/1.5117709
    """

    if np.isscalar(cat):
        cat_index = int(cast(Any, cat))
        m = ds_sg_out["C_Receiver"].isel(Categories=cat_index).values
    else:
        cat_seq = cast(Sequence[int], cat)
        cat_list = [int(c) for c in cat_seq]
        if 0 in cat_list:
            raise ValueError(
                "Category index 0 (sum of all) is not allowed when specifying "
                "multiple categories. Use individual indices 1-8."
            )
        if any(c < 1 or c > 8 for c in cat_list):
            raise ValueError("Category indices must be in the range 1-8.")
        m = (
            ds_sg_out["C_Receiver"]
            .isel(Categories=cat_list)
            .sum(dim="Categories")
            .values
        )
    # Cell size: S_Cell attribute is in km, convert to m
    cell_size = float(ds_sg_out.attrs["S_Cell"]) * 1e3
    half_x = (ds_sg_out.dims["X_Cell_Index"] * cell_size) / 2.0
    half_y = (ds_sg_out.dims["Y_Cell_Index"] * cell_size) / 2.0
    cell_area = cell_size * cell_size
    extent: tuple[float, float, float, float] = (
        half_y,
        -half_y,
        -half_x,
        half_x,
    )

    if flux_unit == "W":
        unit_scale = 1.0
        unit_label = "W"
    elif flux_unit == "kW":
        unit_scale = 1e-3
        unit_label = "kW"
    elif flux_unit == "MW":
        unit_scale = 1e-6
        unit_label = "MW"
    else:
        raise NameError("Unknown argument for unit!")

    plt.figure()

    if not log_color_scale:
        im = plt.imshow(
            (unit_scale * m * mtoa) / cell_area,
            cmap=plt.get_cmap("jet"),
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
    else:
        log_vmin = 0.00001 if np.amin(m) < 0.00001 else np.amin(m)
        im = plt.imshow(
            (unit_scale * m * mtoa) / cell_area,
            cmap=plt.get_cmap("jet"),
            norm=mcolors.LogNorm(vmin=log_vmin * mtoa, vmax=np.amax(m * mtoa)),
            interpolation=interpolation,
            extent=extent,
        )

    cbar = plt.colorbar()
    cbar.remove()
    cbar = plt.colorbar(im)
    cbar.set_label(r"Irradiance (" + unit_label + r".m$^{-2}$)", fontsize=12)
    plt.xlabel(r"Position (m) in relative y axis")
    plt.ylabel(r"Position (m) in relative x axis")
    plt.title("Receiver surface")
    if save_path is not None:
        plt.savefig(save_path + ".pdf")


_STOKES_LABELS = {
    "I": "I_up (TOA)",
    "Q": "Q_up (TOA)",
    "U": "U_up (TOA)",
    "V": "V_up (TOA)",
}


class _FixedOrderFormatter(ScalarFormatter):
    """
    Scalar formatter with a fixed order of magnitude and format.

    Parameters
    ----------
    order : int, optional
        The fixed order of magnitude. Default: 0.
    fformat : str, optional
        The fixed tick format. Default: "%2.2f".
    offset : bool, optional
        Whether to use offset notation. Default: True.
    math_text : bool, optional
        Whether to use fancy math formatting. Default: True.
    """

    def __init__(
        self,
        order: int = 0,
        fformat: str = "%2.2f",
        offset: bool = True,
        math_text: bool = True,
    ):
        self._fixed_order = order
        self._fixed_format = fformat
        super().__init__(useOffset=offset, useMathText=math_text)

    def _set_order_of_magnitude(self) -> None:
        self.orderOfMagnitude = self._fixed_order

    def _set_format(self) -> None:
        self.format = self._fixed_format
        if self.get_useMathText():
            self.format = r"$\mathdefault{%s}$" % self.format


def _order_of_magnitude(values: np.ndarray) -> int:
    """
    Return the order of magnitude of the largest absolute value.
    """
    return math.floor(math.log(np.max(np.abs(values)), 10))


def _colorbar_formatter(
    values: np.ndarray, sci_format: bool
) -> ScalarFormatter | None:
    """
    Return the colorbar tick formatter, or None for the default one.
    """
    if sci_format:
        return _FixedOrderFormatter(_order_of_magnitude(values))
    return None


def _colorbar_ticks(
    vmin: float | None, vmax: float | None, values: np.ndarray
) -> np.ndarray:
    """
    Return 9 evenly spaced colorbar tick values.

    The bounds default to the extrema of ``values`` when ``vmin`` or
    ``vmax`` is None.
    """
    lo = np.min(values) if vmin is None else vmin
    hi = np.max(values) if vmax is None else vmax
    return np.linspace(lo, hi, 9, endpoint=True)


def _as_list(value: Any, size: int) -> list:
    """
    Broadcast a scalar or a 1-element list to a list of given size.
    """
    if not isinstance(value, list):
        value = [value]
    if len(value) == 1:
        value = value * size
    return list(value)


def _get_cmaps(
    cmap: str | mcolors.Colormap | list,
    cmap_reverse: bool | list[bool],
    size: int,
) -> list[mcolors.Colormap]:
    """
    Build one colormap per panel, with NaN values shown in white.
    """
    cmaps = []
    for cm, reverse in zip(_as_list(cmap, size),
                           _as_list(cmap_reverse, size)):
        if isinstance(cm, str):
            cm = plt.get_cmap(cm)
        if reverse:
            cm = cm.reversed()
        cm.set_bad("white", 1.0)
        cmaps.append(cm)
    return cmaps


def _extract_matrices(
    ds_sg: xr.Dataset,
    stokes_labels: list[str],
    wavelength: float | None,
    factor: float,
    n_x: int,
    n_y: int,
) -> list[np.ndarray]:
    """
    Extract the (n_y, n_x) Stokes matrices from a SMART-G Dataset.
    """
    dims = ds_sg[stokes_labels[0]].dims
    if not all(d in dims for d in ("Azimuth angles", "Zenith angles")):
        raise ValueError(
            "The Azimuth angles and/or Zenith angles dimension(s) "
            "are/is missing"
        )

    if (ds_sg.sizes["Azimuth angles"] > 1
            or ds_sg.sizes["Zenith angles"] > 1):
        raise ValueError(
            "Dimension size > 1 is not authorized for both Azimuth "
            "and Zenith angles"
        )

    if "sensor index" in dims:
        n_sensor = ds_sg.sizes["sensor index"]
    else:
        n_sensor = 1
    if n_x * n_y != n_sensor:
        raise ValueError(
            "The product of Nx and Ny must be equal to the number "
            "of sensors!"
        )

    # The order of n_y and n_x below is very important! The matrices
    # are in the following form:
    # - x0 ... xn
    # y0
    #  :
    # yn
    matrices = []
    for label in stokes_labels:
        # The Azimuth and Zenith angles are forced to their first
        # value (we consider only one sun position)
        # TODO Consider also the case with several sun positions
        da = ds_sg[label].isel(
            {"Azimuth angles": 0, "Zenith angles": 0}
        )
        if "wavelength" in da.dims:
            # TODO Enable a default value, for example for the
            # monochromatic case
            da = da.interp(wavelength=wavelength)
        matrices.append(np.asarray(da).reshape(n_y, n_x) * factor)
    return matrices


def _draw_map(
    ax: Axes,
    mat: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    cmap: mcolors.Colormap,
    interpolation: str,
) -> ScalarMappable:
    """
    Draw one 2D map on an axes.

    Use :func:`matplotlib.pyplot.imshow` when both grids are regular,
    :func:`matplotlib.pyplot.pcolormesh` otherwise.
    """
    if is_same_cell_size(xgrid) and is_same_cell_size(ygrid):
        # By default in the imshow function the origin
        # (origin='upper'), i.e. mat[0, 0], is at the upper left, and
        # we want the origin at the bottom left (origin='lower')
        return ax.imshow(
            mat, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap,
            interpolation=interpolation,
            extent=(float(xgrid.min()), float(xgrid.max()),
                    float(ygrid.min()), float(ygrid.max())),
        )
    if interpolation != "none":
        warnings.warn(
            "the interpolation variable cannot be used (and then "
            "ignored) when using pcolormesh! i.e. when we have a "
            "cell size varying along the x or y axis."
        )
    img = ax.pcolormesh(xgrid, ygrid, mat, vmin=vmin, vmax=vmax,
                        cmap=cmap)
    ax.axis("scaled")  # x and y axes with the same scaling
    return img


def _add_colorbar(
    ax: Axes,
    img: ScalarMappable,
    mat: np.ndarray,
    label: str,
    vmin: float | None,
    vmax: float | None,
    cbar_shrink: float,
    cbar_sci_format: bool,
    fontsize: int,
) -> None:
    """
    Add a vertical colorbar next to one panel.

    The tick values and format are computed on the non-NaN values of
    the matrix.
    """
    fig = cast(Figure, ax.figure)
    values = mat[~np.isnan(mat)]
    cbar = fig.colorbar(
        img, ax=ax, shrink=cbar_shrink, orientation="vertical",
        format=_colorbar_formatter(values, cbar_sci_format),
        ticks=_colorbar_ticks(vmin, vmax, values),
    )
    cbar.set_label(label, fontsize=fontsize)


def satellite_view(
    ds_sg: xr.Dataset | MLUT | None,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    wavelength: float | None = None,
    interpolation: str = "none",
    cmap: str | mcolors.Colormap | list = "Blues_r",
    cmap_reverse: bool | list[bool] = False,
    figsize: tuple[float, float] | None = (8, 8),
    fontsize: int = 18,
    vmin: float | list | None = None,
    vmax: float | list | None = None,
    scale: bool = False,
    save_path: str | None = None,
    stokes: str | list[str] = "I",
    factor: float = 1.0,
    matrices: np.ndarray | list[np.ndarray] | None = None,
    cbar_shrink: float = 0.9,
    cbar_sci_format: bool = True,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """
    Give a 'satellite' 2D image of SMART-G 3D atmosphere results.

    The image shows one panel per requested Stokes parameter (up to
    4), each with its own colorbar.

    Parameters
    ----------
    ds_sg : xr.Dataset or MLUT or None
        SMART-G output Dataset (MLUT input is deprecated). Can be
        None when ``matrices`` is given.
    xgrid : np.ndarray
        Numpy array with the grid profile in the x axis.
    ygrid : np.ndarray
        Numpy array with the grid profile in the y axis.
    wavelength : float, optional
        The wavelength (nm). Required when the Dataset has a
        wavelength dimension.
    interpolation : str, optional
        Interpolation for :func:`matplotlib.pyplot.imshow`, e.g.
        'nearest', 'bilinear', 'bicubic', ... Default: 'none'.
    cmap : str or Colormap or list, optional
        The colormap(s) of the panels, e.g. 'Blues_r', 'jet', ...
        Default: 'Blues_r'.
    cmap_reverse : bool or list of bool, optional
        Whether to reverse the colormap(s). Default: False.
    figsize : tuple of float, optional
        The width and height of the figure in inches. If None, a
        default depending on the panel number is used.
    fontsize : int, optional
        The font size of the figure. Default: 18.
    vmin : float or list, optional
        The lower bound(s) of the colorbar interval(s).
    vmax : float or list, optional
        The upper bound(s) of the colorbar interval(s).
    scale : bool, optional
        If True, scale each matrix between 0 and 1 (or between
        ``vmin`` and ``vmax`` if given). Default: False.
    save_path : str, optional
        If given, save the figure at this path. The '.pdf' extension
        is appended when neither '.pdf' nor '.png' is specified.
    stokes : str or list of str, optional
        The Stokes parameter(s) to show, among 'I', 'Q', 'U' and 'V'
        (max 4). Default: 'I'.
    factor : float, optional
        Multiplication factor applied to the matrices extracted from
        the Dataset. Default: 1.
    matrices : np.ndarray or list of np.ndarray, optional
        Force the shown matrix(ces) instead of extracting them from
        the Dataset (max 4).
    cbar_shrink : float, optional
        The colorbar shrink value. Default: 0.9.
    cbar_sci_format : bool, optional
        Use the scientific form for the colorbar values.
        Default: True.
    title : str, optional
        The figure title.
    xlim : tuple of float, optional
        The x limits of the panels.
    ylim : tuple of float, optional
        The y limits of the panels.

    Returns
    -------
    Figure
        The created matplotlib figure.
    """
    if not isinstance(stokes, list):
        stokes = [stokes]
    stokes_labels = []
    for stk in stokes:
        if stk not in _STOKES_LABELS:
            raise ValueError(f"Unknown stokes '{stk}'!")
        stokes_labels.append(_STOKES_LABELS[stk])

    # Number of sensors in the x and y axes
    n_x = xgrid.size - 1
    n_y = ygrid.size - 1

    if matrices is None:
        if ds_sg is None:
            raise ValueError(
                "The ds_sg argument is required when matrices is not "
                "given!"
            )
        if isinstance(ds_sg, MLUT):
            warn_message = (
                "\nUsing an MLUT for ds_sg is deprecated, use an "
                + "xarray.Dataset instead."
            )
            warnings.warn(warn_message, DeprecationWarning,
                          stacklevel=2)
            ds_sg = ds_sg.to_xarray()
        matrix = _extract_matrices(
            ds_sg, stokes_labels, wavelength, factor, n_x, n_y)
    else:
        matrix = _as_list(matrices, 1)
        if n_x * n_y != matrix[0].shape[0] * matrix[0].shape[1]:
            raise ValueError(
                "The product of Nx and Ny must be equal to the "
                "number of sensors!"
            )
    n_panel = len(matrix)
    if n_panel > 4:
        raise ValueError("Give more than 4 stokes is not authorized!")

    vmin = _as_list(vmin, n_panel)
    vmax = _as_list(vmax, n_panel)

    # Deal with all the possibilities where vmin, vmax and scale are
    # used
    if scale:
        for idm, mat in enumerate(matrix):
            vmin_scale = vmin[idm] if vmin[idm] is not None else 0.0
            vmax_scale = vmax[idm] if vmax[idm] is not None else 1.0
            matrix[idm] = np.interp(mat, (mat.min(), mat.max()),
                                    (vmin_scale, vmax_scale))

    plt.rcParams.update({"font.size": fontsize})
    cmaps = _get_cmaps(cmap, cmap_reverse, n_panel)

    def draw(ax: Axes, idm: int) -> None:
        img = _draw_map(ax, matrix[idm], xgrid, ygrid, vmin[idm],
                        vmax[idm], cmaps[idm], interpolation)
        _add_colorbar(ax, img, matrix[idm], stokes_labels[idm],
                      vmin[idm], vmax[idm], cbar_shrink,
                      cbar_sci_format, fontsize)

    if n_panel == 1:
        if figsize is None:
            figsize = (6, 4)
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        if title is not None:
            plt.title(title)
        draw(plt.gca(), 0)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.xlabel(r"X (km)")
        plt.ylabel(r"Y (km)")

    elif n_panel == 2:
        if figsize is None:
            figsize = (12, 4)
        fig, axs = plt.subplots(1, 2, figsize=figsize,
                                constrained_layout=True,
                                sharex=True, sharey=True)
        if title is not None:
            fig.suptitle(title)
        for idm in range(2):
            draw(axs[idm], idm)
        axs[0].set_xlim(xgrid[0], xgrid[-1])
        axs[1].set_ylim(ygrid[0], ygrid[-1])
        axs[0].set_ylabel(r"Y (km)")
        fig.supxlabel(r"X (km)")

    elif n_panel == 3:
        if figsize is None:
            figsize = (12, 8)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(4, 4, figure=fig)
        if title is not None:
            fig.suptitle(title)
        ax1 = plt.subplot(gs[:2, :2])
        ax2 = plt.subplot(gs[:2, 2:], sharey=ax1)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax3 = plt.subplot(gs[2:4, 1:3])
        for idm, ax in enumerate((ax1, ax2, ax3)):
            draw(ax, idm)
            ax.set_xlim(xgrid[0], xgrid[-1])
        ax1.set_ylabel(r"Y (km)")
        ax3.set_ylabel(r"Y (km)")
        ax3.set_xlabel(r"X (km)")
        gs.tight_layout(fig)

    else:
        if figsize is None:
            figsize = (12, 8)
        fig, axs = plt.subplots(2, 2, figsize=figsize,
                                constrained_layout=True,
                                sharex=True, sharey=True)
        if title is not None:
            fig.suptitle(title)
        plt.rcParams.update({"font.size": fontsize})
        for idm in range(4):
            ax = axs[idm // 2, idm % 2]
            draw(ax, idm)
            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
        fig.supxlabel(r"X (km)")
        fig.supylabel(r"Y (km)")

    if save_path is not None:
        # Deal with the case where the extension is not specified
        if (not save_path.endswith(".pdf")
                and not save_path.endswith(".png")):
            save_path += ".pdf"
        plt.savefig(save_path)

    return fig


def cat_view(
    ds: xr.Dataset | MLUT,
    mtoa: float | np.ndarray | xr.DataArray | LUT = 1320,
    ncl: Literal["68%", "87%", "95%", "99%", "99.99%"] = "68%",
    output_unit: Literal["FLUX", "FLUX_DENSITY", "RADIANCE"] = "FLUX_DENSITY",
    flux_unit: Literal["uW", "mW", "W", "kW", "MW"] = "W",
    length_unit: Literal["mm", "cm", "dm", "m", "km"] = "m",
    print_results: bool = True,
    accuracy: int = 6,
    kdis_rep_bands: object | None = None,
) -> xr.Dataset:
    """
    Normalize photon weights from a SMART-G simulation output to flux,
    flux density, or radiance with error estimates.

    Processes receiver weights from ``ds['wPhCats']`` and
    ``ds['wPhCats2']``, applies the specified ``output_unit``,
    multiplies by ``mtoa``, applies the selected ``flux_unit``, and
    returns a new Dataset with normalized intensity and error estimates
    for category 0 (sum of all) and categories 1-8.

    Parameters
    ----------
    ds : Dataset or MLUT
        SMART-G output Dataset containing simulation results. An MLUT is
        converted to a Dataset and emits a deprecation warning.
    mtoa : float, ndarray, DataArray, or LUT, optional
        Solar flux at TOA (W/m²). If there is a wavelength dimension,
        provide a 1D array with the flux as a function of wavelength. A
        legacy LUT is converted to a DataArray and emits a deprecation
        warning.
        Default: 1320
    ncl : str, optional
        Nominal Confidence Limit for the error estimation. Default:
        "68%"
    output_unit : str, optional
        Output unit type. Choices are:
        - 'FLUX' (Watt)
        - 'FLUX_DENSITY' (Watt/meter²)
        - 'RADIANCE' (Watt/meter²/sr)
        Default: "FLUX_DENSITY"
    flux_unit : str, optional
        Power unit used for displayed irradiance values. Choices are
        'uW' (microWatt), 'mW' (milliWatt), 'W' (Watt), 'kW'
        (kiloWatt), and 'MW' (MegaWatt). Default: 'W'.
    length_unit : str, optional
        Length unit for display. Choices are "cm" (centimeter), "m"
        (meter), "km" (kilometer), etc. Default: "m"
    print_results : bool, optional
        If True, print results. If there is a wavelength dimension,
        prints the spectrally integrated results. Default: True
    accuracy : int, optional
        Accuracy: number of decimal points to display when printing.
        Default: 6
    kdis_rep_bands : KdisIbandList or ReptranIbandList, optional
        Band information object. Used for spectral processing. Default:
        None

    Returns
    -------
    output : Dataset
        Dataset containing intensity (flux, flux density, or radiance)
        with associated error estimates for each category.
    """

    if isinstance(ds, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds is deprecated, use an "
            "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds = ds.to_xarray()

    if isinstance(mtoa, LUT):
        warn_message = (
            "\nUsing a LUT for mtoa is deprecated, use an "
            "xarray.DataArray instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        mtoa = mtoa.to_xarray()

    # Initialize the output Dataset
    output = xr.Dataset()

    # Add the Categories dimension
    # (See Moulana et al. 2019 for 8 Categories)
    categories = np.arange(9, dtype=np.float64)
    output = output.assign_coords(Categories=categories)

    # Parameters not dependant on the wavelength
    aldeg = float(ds.attrs["ALDEG"])

    # Parameters needed in case kdis or reptran is used
    norm: Any | None = None
    norm_dl: Any | None = None
    if kdis_rep_bands is not None:
        _, _, _, _, norm, norm_dl = cast(Any, kdis_rep_bands).get_weights()

    # Check if there is a dimension wavelength
    is_wave_axis = "wavelength" in ds["wPhCats"].dims

    # Fill needed parameters considering the case with and without
    # the wavelength
    # dimension
    nph_int: float | None = None
    if is_wave_axis:
        nph = ds["norm_npho"].values
        nph_int = float(ds.attrs["NPHOTONS"])
    else:
        nph = float(ds.attrs["NPHOTONS"])

    # DataArrays with sum of photon weight (and squared weight)
    # as function of
    # Categories and (if there is wavelength dim) wavelength
    mf = ds["wPhCats"]
    mf2 = ds["wPhCats2"]

    # The desired unit of measurement between Watt, kiloWatt,
    # MegaWatt...
    if flux_unit == "uW":
        k = 1e6
        flux_unit_long = "microWatt"
    elif flux_unit == "mW":
        k = 1e3
        flux_unit_long = "milliWatt"
    elif flux_unit == "W":
        k = 1.0
        flux_unit_long = "Watt"
    elif flux_unit == "kW":
        k = 1e-3
        flux_unit_long = "kiloWatt"
    elif flux_unit == "MW":
        k = 1e-6
        flux_unit_long = "MegaWatt"
    else:
        raise NameError("Unknown argument for flux_unit!")

    # The desired unit of measurement of length (centimeter, meter, ...)
    if length_unit == "mm":
        kl = 1e-3 * 1e-3
        length_unit_long = "millimeter"
    elif length_unit == "cm":
        kl = 1e-2 * 1e-2
        length_unit_long = "centimeter"
    elif length_unit == "dm":
        kl = 1e-1 * 1e-1
        length_unit_long = "decimeter"
    elif length_unit == "m":
        kl = 1.0
        length_unit_long = "meter"
    elif length_unit == "km":
        kl = 1e3 * 1e3
        length_unit_long = "kilometer"
    else:
        raise NameError("Unknown argument for length_unit!")

    if output_unit == "FLUX":
        cst = 1.0 * k
        str_print = f"Flux in {flux_unit_long} for each categories"
        str_type = "flux"
    elif output_unit == "FLUX_DENSITY":
        cst = (1.0 * k * kl) / (float(ds.attrs["S_Receiver"]) * 1e6)
        str_print = (
            f"Irradiance in {flux_unit_long}/"
            f"{length_unit_long}² for each categories"
        )
        str_type = "irradiance"
    elif output_unit == "RADIANCE":
        cst = (1.0 * k * kl) / (float(ds.attrs["S_Receiver"]) * 1e6)
        cst *= 2.0 / (np.pi * (1 - np.cos(np.radians(2 * aldeg))))
        str_print = (
            f"Radiance in {flux_unit_long}/"
            f"{length_unit_long}²/sr for each categories"
        )
        str_type = "radiance"
    else:
        raise NameError("Unknown argument for output_unit!")

    if is_wave_axis:
        cst *= float(ds.attrs["n_cte"])
        cst *= np.sum(nph) / nph
    else:
        cst *= float(ds.attrs["n_cte"])

    # Normalized intensity
    mf_n_int: xr.DataArray | None = None
    mf_2_n_int: xr.DataArray | None = None
    mf_2_n: xr.DataArray | None = None
    abs_err_da_n_int: xr.DataArray | None = None
    if is_wave_axis:
        if kdis_rep_bands is not None:
            if norm is None or norm_dl is None:
                raise RuntimeError(
                    "kdis/reptran normalization weights are required."
                )
            # Group wavelengths by band structure and sum within
            # each band
            mf_n = (
                (mf * cst * mtoa).groupby("wavelength").sum(dim="wavelength")
            )
            mf_n_int = mf_n / norm

            mf_2_n_int = (
                (mf2 * (cst * mtoa) * (cst * mtoa))
                .groupby("wavelength")
                .sum(dim="wavelength")
            )
            mf_2_n_int /= norm

            if mf_n_int is None:
                raise RuntimeError("mf_n_int must be initialized.")

            # Convert to DataArray with proper coordinates
            mf_n_int = xr.DataArray(
                mf_n_int.values,
                dims=["Categories", "wavelength"],
                coords={
                    "Categories": np.arange(9, dtype=np.float64),
                    "wavelength": mf_n_int.wavelength,
                },
            )
            mf_n /= norm_dl
        else:
            mf_n = mf * cst * mtoa
            mf_2_n = mf2 * (cst * mtoa) * (cst * mtoa)

        # For non-grouped case, wrap as DataArray if needed
        if not isinstance(mf_n, xr.DataArray):
            mf_n = xr.DataArray(
                mf_n.values if hasattr(mf_n, "values") else mf_n,
                dims=["Categories", "wavelength"],
                coords={
                    "Categories": np.arange(9, dtype=np.float64),
                    "wavelength": ds.wavelength,
                },
            )

        # Add the wavelength dimension in the output Dataset
        if "wavelength" not in output.coords:
            output = output.assign_coords(wavelength=mf_n.wavelength)
    else:
        mf_n = mf * cst * mtoa

    # Nominal confidence limit factor for error calculation
    if ncl == "68%":
        ld = 1
    elif ncl == "87%":
        ld = 1.5
    elif ncl == "95%":
        ld = 2
    elif ncl == "99%":
        ld = 3
    elif ncl == "99.99%":
        ld = 4

    # Absolute error calculation and normalization
    if is_wave_axis:
        s_wavelength = len(ds.wavelength)
        abs_err = np.zeros((9, s_wavelength), dtype="float64")
        sum_2_z = np.zeros((9, s_wavelength), dtype="float64")
        sum_z_2 = np.zeros((9, s_wavelength), dtype="float64")

        n_bis = nph / (nph - 1)

        sum_2_z[:, :] = (mf.values[:, :] * mf.values[:, :]) / nph
        sum_z_2 = mf2.values[:, :]
        abs_err[:, :] = (n_bis * np.abs(sum_z_2 - sum_2_z)) ** 0.5
        abs_err_da = xr.DataArray(
            abs_err[:, :],
            dims=["Categories", "wavelength"],
            coords={
                "Categories": np.arange(9, dtype=np.float64),
                "wavelength": ds.wavelength,
            },
        )
        if kdis_rep_bands is not None:
            # Group by bands and sum within each band
            abs_err_da_n = (
                (abs_err_da * cst * mtoa * ld)
                .groupby("wavelength")
                .sum(dim="wavelength")
            )
            if norm_dl is None:
                raise RuntimeError(
                    "kdis/reptran differential normalization is required."
                )
            abs_err_da_n /= norm_dl
        else:
            abs_err_da_n = abs_err_da.values[:, :] * cst * mtoa * ld

        abs_err_values: np.ndarray[Any, Any]
        abs_err_wavelength: Any
        if isinstance(abs_err_da_n, xr.DataArray):
            abs_err_values = np.asarray(abs_err_da_n.values)
            abs_err_wavelength = abs_err_da_n.wavelength
        else:
            abs_err_values = np.asarray(abs_err_da_n)
            abs_err_wavelength = ds.wavelength

        abs_err_da_n = xr.DataArray(
            abs_err_values,
            dims=["Categories", "wavelength"],
            coords={
                "Categories": np.arange(9, dtype=np.float64),
                "wavelength": abs_err_wavelength,
            },
        )

        abs_err_int = np.zeros(9, dtype="float64")
        sum_2_z_int = np.zeros(9, dtype="float64")
        sum_z_2_int = np.zeros(9, dtype="float64")

        if nph_int is None:
            raise RuntimeError("nph_int must be defined for wave-axis mode.")
        n_bis_int = nph_int / (nph_int - 1)

        if kdis_rep_bands is not None:
            if mf_n_int is None or mf_2_n_int is None:
                raise RuntimeError(
                    "Integrated band arrays must be initialized."
                )
            mf_int = np.sum(mf_n_int.values[:, :], axis=1)
            mf_2_int = np.sum(mf_2_n_int.values[:, :], axis=1)
        else:
            if mf_2_n is None:
                raise RuntimeError("mf_2_n must be initialized.")
            mf_int = np.sum(mf_n.values[:, :], axis=1)
            mf_2_int = np.sum(mf_2_n.values[:, :], axis=1)

        sum_2_z_int[:] = (mf_int[:] * mf_int[:]) / nph_int
        sum_z_2_int = mf_2_int[:]
        abs_err_int[:] = (n_bis_int * np.abs(sum_z_2_int - sum_2_z_int)) ** 0.5
        abs_err_da_int = xr.DataArray(
            abs_err_int[:],
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        abs_err_da_n_int = abs_err_da_int

    else:
        abs_err = np.zeros(9, dtype="float64")
        sum_2_z = np.zeros(9, dtype="float64")
        sum_z_2 = np.zeros(9, dtype="float64")

        n_bis = nph / (nph - 1)

        sum_2_z[:] = (mf.values[:] * mf.values[:]) / nph
        sum_z_2 = mf2.values[:]
        abs_err[:] = (n_bis * np.abs(sum_z_2 - sum_2_z)) ** 0.5
        abs_err_da = xr.DataArray(
            abs_err[:],
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        abs_err_da_n = abs_err_da * cst * mtoa * ld
    # Relative error calculation
    rel_err_da_n = (abs_err_da_n / mf_n) * 100

    # Create DataArray for the number of photons as function
    # of Categories
    nb_ph_da = xr.DataArray(
        ds["cat_PhNb"].values,
        dims=["Categories"],
        coords={"Categories": np.arange(9, dtype=np.float64)},
    )

    # Add descriptions and DataArrays to output Dataset
    mf_n.attrs["description"] = str_print
    nb_ph_da.attrs["description"] = (
        "Number of photons as function of Categories"
    )
    abs_err_da_n.attrs["description"] = f"Absolute error of {output_unit}"
    rel_err_da_n.attrs["description"] = (
        f"Relative error in percentage of {output_unit}"
    )

    output[output_unit] = mf_n
    output["NbPhotons"] = nb_ph_da
    output["AbsoluteErr"] = abs_err_da_n
    output["RelativeErr"] = rel_err_da_n

    if kdis_rep_bands is not None and is_wave_axis:
        if mf_n_int is None or abs_err_da_n_int is None:
            raise RuntimeError("Integrated outputs are not initialized.")
        output[output_unit + "_int"] = mf_n_int
        mf_n_tot = xr.DataArray(
            np.sum(mf_n_int.values[:, :], axis=1),
            dims=["Categories"],
            coords={"Categories": np.arange(9, dtype=np.float64)},
        )
        output[output_unit + "_tot"] = mf_n_tot
        output["AbsoluteErr_tot"] = abs_err_da_n_int

    # Print results if requested
    if print_results:
        l_p = [
            "(  D  )",
            "(  H  )",
            "(  E  )",
            "(  A  )",
            "( H+A )",
            "( H+E )",
            "( E+A )",
            "(H+E+A)",
        ]
        int_acc = int(accuracy)
        str_acc = str(int_acc)
        str_acc = "%." + str_acc + "f"

        mat = np.zeros((9, 4), dtype="float64")
        if is_wave_axis:
            if kdis_rep_bands is not None:
                if mf_n_int is None or abs_err_da_n_int is None:
                    raise RuntimeError(
                        "Integrated outputs are not initialized."
                    )
                mat[:, 0] = np.sum(mf_n_int.values[:, :], axis=1)
            else:
                mat[:, 0] = np.sum(mf_n.values[:, :], axis=1)
            mat[:, 1] = ds["cat_PhNb"].values
            if abs_err_da_n_int is None:
                raise RuntimeError("Absolute integrated errors are missing.")
            mat[:, 2] = abs_err_da_n_int.values
            mat[:, 3] = (mat[:, 2] / mat[:, 0]) * 100
        else:
            mat[:, 0] = mf_n.values
            mat[:, 1] = ds["cat_PhNb"].values
            mat[:, 2] = abs_err_da_n.values
            mat[:, 3] = rel_err_da_n.values

        print("**********************************************************")
        print(str_print)
        print("**********************************************************")
        print(
            "SUM_CATS      " + ": " + str_type + "=",
            str_acc % (mat[0, 0]),
            " number_ph=",
            np.uint64(mat[0, 1]),
            " errAbs=",
            str_acc % (mat[0, 2]),
            " err(%)=",
            str_acc % (mat[0, 3] * ld),
        )
        for i in range(0, 8):
            print(
                "CAT",
                i + 1,
                l_p[i],
                ": " + str_type + "=",
                str_acc % (mat[i + 1, 0]),
                " number_ph=",
                np.uint64(mat[i + 1, 1]),
                " errAbs=",
                str_acc % (mat[i + 1, 2]),
                " err(%)=",
                str_acc % (mat[i + 1, 3] * ld),
            )
    return output


def nopt_view(
    ds: xr.Dataset | MLUT,
    back: bool = False,
    acc: int = 6,
    ncl: Literal["68%", "87%", "95%", "99%", "99.99%"] = "68%",
    mtoa: None | np.ndarray | xr.DataArray | LUT = None,
    natm_approx: bool = False,
) -> None:
    """
    Calculate and display the detailed optical efficiencies with
    associated error estimates of a Solar Tower Power simulated with
    SMART-G.

    Parameters
    ----------
    ds : Dataset or MLUT
        SMART-G output Dataset containing simulation results. An MLUT is
        converted to a Dataset and emits a deprecation warning.
    back : bool, optional
        False for forward mode (default), True for backward mode.
        Determines which efficiency metrics are calculated and
        displayed. Default: False
    acc : int, optional
        Accuracy: number of decimal points to display in the output.
        Default: 6
    ncl : str, optional
        Nominal Confidence Limit for error estimation. Options are:
        - "68%" (1 sigma)
        - "87%" (1.5 sigma)
        - "95%" (2 sigma)
        - "99%" (3 sigma)
        - "99.99%" (4 sigma)
        Default: "68%"
    mtoa : None, ndarray, DataArray, or LUT, optional
        Solar flux at TOA for each wavelength band. If None, uses the
        total power. If provided, pass a 1D array and the computation
        is weighted by flux per band. A legacy LUT is converted to a
        DataArray and emits a deprecation warning. Default: None
    natm_approx : bool, optional
        If True, calculate and display the analytic approximation of
        atmospheric transmission (natm_approx) in backward mode. Ignored
        in forward mode. Default: False


    Notes
    -----
    In forward mode, displays:
    - nopt: Total optical efficiency
    - ncos: Cosine efficiency
    - nsha: Shading efficiency
    - nref: Reflection efficiency
    - nblo: Blocking efficiency
    - nspi: Spillage efficiency
    - natm: Atmospheric transmission

    In backward mode, displays:
    - nopt: Total optical efficiency
    - ncos: Cosine efficiency
    - nref: Reflection efficiency
    - nsbsa: Product of blocking, shading, and atmospheric efficiencies

    Each metric includes an estimate of absolute error and relative
    error.
    """
    if isinstance(ds, MLUT):
        warn_message = (
            "\nUsing an MLUT for ds is deprecated, use an "
            "xarray.Dataset instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        ds = ds.to_xarray()

    if isinstance(mtoa, LUT):
        warn_message = (
            "\nUsing a LUT for mtoa is deprecated, use an "
            "xarray.DataArray instead."
        )
        warnings.warn(warn_message, DeprecationWarning, stacklevel=2)
        mtoa = mtoa.to_xarray()

    # Number of photons launched
    nph = float(ds.attrs["NPHOTONS"])
    # n/(n-1)
    nbis = nph / (nph - 1)

    if mtoa is None:
        powc_h_values = np.asarray(ds["powc_H"].values)
        powc_h = float(powc_h_values.reshape(-1)[0])
    else:
        powc_h = 0.0
        for i in range(0, len(mtoa)):
            powc_h += ds["powc_H"].values[i] * mtoa[i]
        powc_h /= np.sum(mtoa)
        powc_h = float(powc_h)

    k = float(ds.attrs["n_cte"]) / powc_h

    int_acc = int(acc)
    str_acc = str(int_acc)
    str_acc = "%." + str_acc + "f"
    if ncl == "68%":
        ld = 1
    elif ncl == "87%":
        ld = 1.5
    elif ncl == "95%":
        ld = 2
    elif ncl == "99%":
        ld = 3
    elif ncl == "99.99%":
        ld = 4

    print("**********************************************")
    print(" Optical Efficiencies")
    print("**********************************************")

    if not back:  # Forward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wrhoP, w3=wBM, w4=wBP, w5=wSM, w6=wSP
        # w7=wREC
        w0 = ds["wLoss"].values[0]
        w1 = ds["wLoss"].values[1]
        w2 = ds["wLoss"].values[2]
        w3 = ds["wLoss"].values[3]
        w4 = ds["wLoss"].values[4]
        w5 = ds["wLoss"].values[5]
        w6 = ds["wLoss"].values[6]
        w7 = ds["cat_w"].values[2]
        # Sum of (weights²)
        w0_2 = ds["wLoss2"].values[0]
        w1_2 = ds["wLoss2"].values[1]
        w2_2 = ds["wLoss2"].values[2]
        w3_2 = ds["wLoss2"].values[3]
        w4_2 = ds["wLoss2"].values[4]
        w5_2 = ds["wLoss2"].values[5]
        w6_2 = ds["wLoss2"].values[6]
        w7_2 = ds["cat_w2"].values[2]
        # (Sum of weights)² divided by the number of photons
        sum_z_bar2 = [
            (w0 * w0) / nph,
            (w1 * w1) / nph,
            (w2 * w2) / nph,
            (w3 * w3) / nph,
            (w4 * w4) / nph,
            (w5 * w5) / nph,
            (w6 * w6) / nph,
            (w7 * w7) / nph,
        ]
        # Sum of (weights²)
        sum_z2_bar = [w0_2, w1_2, w2_2, w3_2, w4_2, w5_2, w6_2, w7_2]
        dw = []
        for i in range(0, len(sum_z_bar2)):
            dw_temp = ld * nbis * (sum_z2_bar[i] - sum_z_bar2[i]) ** 0.5
            dw.append(dw_temp)

        nopt = gc.clamp(k * w7, 0, 1)
        k_s = k / float(ds.attrs["n_cos"])
        ncos = float(ds.attrs["n_cos"])
        nsha = gc.clamp(k_s * w0, 0, 1)
        nref = gc.clamp(1 - (w1 / w0), 0, 1)
        nblo = gc.clamp(1 - (w3 / w2), 0, 1)
        nspi = gc.clamp(1 - (w5 / w4), 0, 1)
        natm = gc.clamp(w7 / w6, 0, 1)

        d_nopt = abs(k) * dw[7]
        d_ncos = 0.0
        d_nsha = abs(k_s) * dw[0]
        d_nref = abs(-1.0 / w0) * dw[1] + abs(w1 / w0**2) * dw[0]
        d_nblo = abs(-1.0 / w2) * dw[3] + abs(w3 / w2**2) * dw[2]
        d_nspi = abs(-1.0 / w4) * dw[5] + abs(w5 / w4**2) * dw[4]
        d_natm = abs(1.0 / w6) * dw[7] + abs(w7 / w6**2) * dw[6]

        print(
            "nopt =",
            str_acc % nopt,
            ", errAbs =",
            str_acc % d_nopt,
            ", err% =",
            str_acc % ((d_nopt / nopt) * 100),
        )
        print(
            "ncos =",
            str_acc % ncos,
            ", errAbs =",
            str_acc % d_ncos,
            ", err% =",
            str_acc % ((d_ncos / ncos) * 100),
        )
        print(
            "nsha =",
            str_acc % nsha,
            ", errAbs =",
            str_acc % d_nsha,
            ", err% =",
            str_acc % ((d_nsha / nsha) * 100),
        )
        print(
            "nref =",
            str_acc % nref,
            ", errAbs =",
            str_acc % d_nref,
            ", err% =",
            str_acc % ((d_nref / nref) * 100),
        )
        print(
            "nblo =",
            str_acc % nblo,
            ", errAbs =",
            str_acc % d_nblo,
            ", err% =",
            str_acc % ((d_nblo / nblo) * 100),
        )
        print(
            "nspi =",
            str_acc % nspi,
            ", errAbs =",
            str_acc % d_nspi,
            ", err% =",
            str_acc % ((d_nspi / nspi) * 100),
        )
        print(
            "natm =",
            str_acc % natm,
            ", errAbs =",
            str_acc % d_natm,
            ", err% =",
            str_acc % ((d_natm / natm) * 100),
        )
    else:  # Backward mode ->
        # Sum of weights
        # w0=wI, w1=wrhoM, w2=wREC
        w0 = float(ds["wLoss"].values[0])
        w1 = float(ds["wLoss"].values[1])
        w2 = float(ds["cat_w"].values[2])
        # Sum of (weights²)
        w0_2 = ds["wLoss2"].values[0]
        w1_2 = ds["wLoss2"].values[1]
        w2_2 = ds["cat_w2"].values[2]
        # (Sum of weights)² divided by the number of photons
        sum_z_bar2 = [(w0 * w0) / nph, (w1 * w1) / nph, (w2 * w2) / nph]
        # Sum of (weights²)
        sum_z2_bar = [w0_2, w1_2, w2_2]
        dw = []
        for i in range(0, len(sum_z_bar2)):
            dw_temp = ld * nbis * (sum_z2_bar[i] - sum_z_bar2[i]) ** 0.5
            dw.append(dw_temp)
        nopt = gc.clamp(k * w2, 0, 1)
        ncos = float(ds.attrs["n_cos"])
        nref_raw = 1.0 - (w1 / w0)
        nref_safe = max(0.0, min(1.0, nref_raw))
        nref = gc.clamp(nref_raw, 0, 1)
        nsbsa = gc.clamp((k * w2) / (ncos * nref_safe), 0, 1)

        d_nopt = abs(k) * dw[2]
        d_ncos = 0.0
        d_nref = abs(-1.0 / w0) * dw[1] + abs(w1 / w0**2) * dw[0]

        d_nsbsa = (
            abs(k / (ncos * (1 - (w1 / w0)))) * dw[2]
            + abs((k * w2) / (ncos * w0 * (1 - (w1 / w0)) ** 2)) * dw[1]
            + abs((-k * w2 * w1) / (ncos * w0 * w0 * (1 - (w1 / w0)) ** 2))
        )

        print(
            "nopt =",
            str_acc % nopt,
            ", errAbs =",
            str_acc % d_nopt,
            ", err% =",
            str_acc % ((d_nopt / nopt) * 100),
        )
        print(
            "ncos =",
            str_acc % ncos,
            ", errAbs =",
            str_acc % d_ncos,
            ", err% =",
            str_acc % ((d_ncos / ncos) * 100),
        )
        print(
            "nref =",
            str_acc % nref,
            ", errAbs =",
            str_acc % d_nref,
            ", err% =",
            str_acc % ((d_nref / nref) * 100),
        )
        print(
            "nsbsa =",
            str_acc % nsbsa,
            ", errAbs =",
            str_acc % d_nsbsa,
            ", err% =",
            str_acc % ((d_nsbsa / nsbsa) * 100),
        )

        if natm_approx:
            if mtoa is None:
                naatm = ds["n_aatm"].values
            else:
                naatm = 0.0
                for i in range(0, len(mtoa)):
                    naatm += ds["n_aatm"].values[i] * mtoa[i]
                naatm /= np.sum(mtoa)
            print("naatm =", str_acc % naatm, " -> analytic approx of natm")


def visualize_entity(
    entities: list[Entity | GroupE] | Entity | GroupE,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    draw_method: str = "SM",
    ray_color: str = "r",
    sr_view: int = 1,
    xyz_limit: dict | None = None,
    show_rays: bool = True,
    rs_fac: float = 1,
) -> Figure:
    """Enable a 3D visualization of created objects.

    Parameters
    ----------
    entities : list | Entity
        A list of Entity objects to visualize.
    theta_deg : float, optional
        The zenith angle of the sun in degrees. Default is 0.
    phi_deg : float, optional
        The azimuth angle of the sun in degrees. Default is 0.
    draw_method : str, optional
        The drawing method. 'SM' (Second Method) is the default and
        recommended. 'FM' (First Method) is useful for debugging
        issues.
    ray_color : str, optional
        Sun rays color, e.g., 'r', 'b', 'g', etc. Default is 'r'.
    sr_view : int, optional
        Number of sun rays that can be seen in the figure. Default is 1.
    xyz_limit : dict, optional
        Dictionary specifying x, y, z view limits in km. If None
        (default), limits are automatically chosen. Example format:
        {'x_min': 0., 'x_max': 10., 'y_min': 0., 'y_max': 10.,
         'z_min': 0., 'z_max': 10.}
    show_rays : bool, optional
        Whether to show sun rays. Default is True.
    rs_fac : float, optional
        Ray scale factor. Default is 1.

    Returns
    -------
    out : matplotlib.figure.Figure
        A matplotlib figure object containing the 3D visualization.
    """

    if not isinstance(entities, (list)):
        entities = [entities]

    if not (all(isinstance(x, (Entity, GroupE)) for x in entities)):
        raise NameError(
            "The only objects accepted for entities parameter are: "
            "Entity or GroupE"
        )

    # ensure we have only Entity objects (converts if necessary GroupE
    # to Entity objects)
    entity_list: list[Entity] = convert_lg_to_le(entities)
    entity_tfs = []
    box = gc.BBox()
    for i in range(0, len(entity_list)):
        entity_tfs.append(entity_list[i].get_transformation())
        box = box.union((entity_list[i].bbox_pmin))
        box = box.union((entity_list[i].bbox_pmax))

    box_center = box.pmin + 0.5 * (box.pmax - box.pmin)
    box_max_size = gc.vmax(box.pmax - box.pmin)
    pmin_n = gc.Point(
        box_center.x - 0.5 * box_max_size,
        box_center.y - 0.5 * box_max_size,
        box_center.z - 0.5 * box_max_size,
    )
    pmax_n = gc.Point(
        box_center.x + 0.5 * box_max_size,
        box_center.y + 0.5 * box_max_size,
        box_center.z + 0.5 * box_max_size,
    )
    box_n = gc.BBox(pmin_n, pmax_n)

    # calculate the sun direction vector
    sun_dir = gc.ang2vec(theta_deg, phi_deg, vec_view="nadir")
    wsx = -sun_dir.x
    wsy = -sun_dir.y
    wsz = -sun_dir.z

    ltmesh = []
    n_mirror_hits = int(0)
    rec_entities = []
    ref_entities = []
    rec_tfs = []
    ref_tfs = []
    for i in range(0, len(entity_list)):
        if entity_list[i].name == "reflector":
            ref_entities.append(entity_list[i])
            ref_tfs.append(entity_tfs[i])
        if entity_list[i].name == "receiver":
            rec_entities.append(entity_list[i])
            rec_tfs.append(entity_tfs[i])

    n_ref = len(ref_entities)
    xr: list[np.ndarray | None] = [None] * n_ref
    yr: list[np.ndarray | None] = [None] * n_ref
    zr: list[np.ndarray | None] = [None] * n_ref
    has_intersection = [False] * n_ref
    reflected_photons: list[gc.Ray] = []

    for k in range(0, len(ref_entities)):
        # Get the transformation
        tt = ref_tfs[k]

        photon_pos = gc.Point(
            wsx + ref_entities[k].transformation.transx,
            wsy + ref_entities[k].transformation.transy,
            wsz + ref_entities[k].transformation.transz,
        )
        photon = gc.Ray(o=photon_pos, d=sun_dir, maxt=1200.0)

        if isinstance(ref_entities[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array(
                [
                    np.array([0, 1, 2]),  # indices or triangle 1
                    np.array([2, 3, 1]),
                ],
                dtype=np.int32,
            )  # indices of triangle 2

            # List of points of the plane
            pts = np.array(
                [
                    np.array(
                        [
                            ref_entities[k].geo.p1.x,
                            ref_entities[k].geo.p1.y,
                            ref_entities[k].geo.p1.z,
                        ]
                    ),
                    np.array(
                        [
                            ref_entities[k].geo.p2.x,
                            ref_entities[k].geo.p2.y,
                            ref_entities[k].geo.p2.z,
                        ]
                    ),
                    np.array(
                        [
                            ref_entities[k].geo.p3.x,
                            ref_entities[k].geo.p3.y,
                            ref_entities[k].geo.p3.z,
                        ]
                    ),
                    np.array(
                        [
                            ref_entities[k].geo.p4.x,
                            ref_entities[k].geo.p4.y,
                            ref_entities[k].geo.p4.z,
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            tmesh = gc.TriangleMesh(vertices=pts, faces=vi)
        elif isinstance(ref_entities[k].geo, Spheric):
            sphere = gc.Sphere(
                ref_entities[k].geo.radius,
                ref_entities[k].geo.z0,
                ref_entities[k].geo.z1,
                ref_entities[k].geo.phi,
            )
            tmesh = sphere.to_trianglemesh()
        else:
            raise NameError("This geometry is unknown or not yet accepted!")

        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        # cast: calc_intersection always forces ds_output=True, so it
        # always returns a Dataset
        ds = cast(Dataset, gc.calc_intersection(tmesh, photon))
        if ds["is_intersection"].values and ds["thit"].values < float("inf"):
            has_intersection[k] = True
            n_mirror_hits += int(1)
            p_hit = gc.Point(ds["phit"].values)
            t_hit = ds["thit"].values
            tr = np.linspace(t_hit * 0.98 * (1 / rs_fac), t_hit, 100)
            xr[k] = photon.o.x + tr * photon.d.x
            yr[k] = photon.o.y + tr * photon.d.y
            zr[k] = photon.o.z + tr * photon.d.z
            reflected_dir = ref_fresnel(dir_in=photon.d, geo_transform=tt)
            reflected_photons.append(
                gc.Ray(o=p_hit, d=reflected_dir, maxt=120)
            )

    xr2: list[np.ndarray | None] = [None] * n_mirror_hits
    yr2: list[np.ndarray | None] = [None] * n_mirror_hits
    zr2: list[np.ndarray | None] = [None] * n_mirror_hits
    rec_has_intersection = [False] * n_mirror_hits

    for k in range(0, len(rec_entities)):
        # Get the transformation
        tt = rec_entities[k].get_transformation()

        if isinstance(rec_entities[k].geo, Plane):
            # Vertex triangle indices
            vi = np.array(
                [
                    np.array([0, 1, 2]),  # indices or triangle 1
                    np.array([2, 3, 1]),
                ],
                dtype=np.int32,
            )  # indices of triangle 2

            # List of points of the plane
            pts = np.array(
                [
                    np.array(
                        [
                            rec_entities[k].geo.p1.x,
                            rec_entities[k].geo.p1.y,
                            rec_entities[k].geo.p1.z,
                        ]
                    ),
                    np.array(
                        [
                            rec_entities[k].geo.p2.x,
                            rec_entities[k].geo.p2.y,
                            rec_entities[k].geo.p2.z,
                        ]
                    ),
                    np.array(
                        [
                            rec_entities[k].geo.p3.x,
                            rec_entities[k].geo.p3.y,
                            rec_entities[k].geo.p3.z,
                        ]
                    ),
                    np.array(
                        [
                            rec_entities[k].geo.p4.x,
                            rec_entities[k].geo.p4.y,
                            rec_entities[k].geo.p4.z,
                        ]
                    ),
                ],
                dtype=np.float64,
            )

            tmesh = gc.TriangleMesh(vertices=pts, faces=vi)
        elif isinstance(rec_entities[k].geo, Spheric):
            sphere = gc.Sphere(
                rec_entities[k].geo.radius,
                rec_entities[k].geo.z0,
                rec_entities[k].geo.z1,
                rec_entities[k].geo.phi,
            )
            tmesh = sphere.to_trianglemesh()
        else:
            raise NameError("This geometry is unknown or not yet accepted!")
        tmesh.apply_tf(tt)
        ltmesh.append(tmesh)

        for i in range(0, n_mirror_hits):
            ds = cast(
                Dataset, gc.calc_intersection(tmesh, reflected_photons[i])
            )
            if ds["is_intersection"].values and ds["thit"].values < float(
                "inf"
            ):
                rec_has_intersection[i] = True
                p_hit = gc.Point(ds["phit"].values)
                t_hit = ds["thit"].values
                # cast: mint is a scalar, but its type is wrongly
                # inferred from the untyped Ray constructor of geoclide
                tr = np.linspace(
                    cast(float, reflected_photons[i].mint), t_hit, 100
                )
                xr2[i] = (
                    reflected_photons[i].o.x + tr * reflected_photons[i].d.x
                )
                yr2[i] = (
                    reflected_photons[i].o.y + tr * reflected_photons[i].d.y
                )
                zr2[i] = (
                    reflected_photons[i].o.z + tr * reflected_photons[i].d.z
                )

    # create the matplotlib figure
    fig = plt.figure()  # figsize=[128, 96])
    # cast: with a 3d projection add_subplot returns an Axes3D, but it
    # is only annotated as returning the base Axes class
    ax = cast(Axes3D, fig.add_subplot(111, projection=Axes3D.name))
    # type: ignore -> zs accepts an array-like, but being unannotated
    # its type is wrongly inferred from its default value 0, i.e. as
    # an int
    ax.scatter([-1, 1], [-1, 1], [-1, 1], alpha=0.0)  # type: ignore

    for itmesh, tmesh in enumerate(ltmesh):
        # Triangles mesh parameters for plot
        # First method (draw even if there is error with an object,
        # useful for debug):
        # ----------------------------->
        if draw_method == "FM":
            for itri in range(0, tmesh.ntriangles):
                p0 = gc.Point(tmesh.vertices[tmesh.faces[itri, 0], :])
                p1 = gc.Point(tmesh.vertices[tmesh.faces[itri, 1], :])
                p2 = gc.Point(tmesh.vertices[tmesh.faces[itri, 2], :])
                face_pts = np.array(
                    [
                        [p0.x, p0.y, p0.z],
                        [p1.x, p1.y, p1.z],
                        [p2.x, p2.y, p2.z],
                    ]
                )
                face1 = art3d.Poly3DCollection(
                    [face_pts],
                    alpha=entity_list[itmesh].alpha_color,
                    linewidths=0.2,
                )
                face1.set_facecolor(mcolors.to_rgba(entity_list[itmesh].color))
                ax.add_collection3d(face1)

        # Second method (better visual, avoid some matplotlib bugs):
        # ----------------------------->
        if draw_method == "SM":
            p0_t0 = gc.Point(tmesh.vertices[tmesh.faces[0, 0], :])
            p1_t0 = gc.Point(tmesh.vertices[tmesh.faces[0, 1], :])
            p2_t0 = gc.Point(tmesh.vertices[tmesh.faces[0, 2], :])
            p0_t1 = gc.Point(tmesh.vertices[tmesh.faces[1, 0], :])
            p1_t1 = gc.Point(tmesh.vertices[tmesh.faces[1, 1], :])
            p2_t1 = gc.Point(tmesh.vertices[tmesh.faces[1, 2], :])
            face_pts = np.array(
                [
                    [p0_t0.x, p0_t0.y, p0_t0.z],
                    [p1_t0.x, p1_t0.y, p1_t0.z],
                    [p2_t0.x, p2_t0.y, p2_t0.z],
                    [p0_t1.x, p0_t1.y, p0_t1.z],
                    [p1_t1.x, p1_t1.y, p1_t1.z],
                    [p2_t1.x, p2_t1.y, p2_t1.z],
                ]
            )

            if np.array_equal(face_pts[:, 0], np.full((6), face_pts[0, 0])):
                yy, zz = np.meshgrid(face_pts[:, 0], face_pts[:, 2])
                xx = np.full((6, 6), face_pts[0, 0])
                ax.plot_surface(
                    xx,
                    yy,
                    zz,
                    color=mcolors.to_rgba(entity_list[itmesh].color),
                    alpha=entity_list[itmesh].alpha_color,
                    linewidth=0.2,
                    antialiased=True,
                )
            elif np.array_equal(face_pts[:, 1], np.full((6), face_pts[0, 1])):
                xx, zz = np.meshgrid(face_pts[:, 0], face_pts[:, 2])
                yy = np.full((6, 6), face_pts[0, 1])
                ax.plot_surface(
                    xx,
                    yy,
                    zz,
                    color=mcolors.to_rgba(entity_list[itmesh].color),
                    alpha=entity_list[itmesh].alpha_color,
                    linewidth=0.2,
                    antialiased=True,
                )
            elif np.array_equal(
                face_pts[:, 2], np.full((6), face_pts[0, 2])
            ):  # need to be verified
                xx, yy = np.meshgrid(face_pts[:, 0], face_pts[:, 1])
                zz = np.full((6, 6), face_pts[0, 2])
                ax.plot_surface(
                    xx,
                    yy,
                    zz,
                    color=mcolors.to_rgba(entity_list[itmesh].color),
                    alpha=entity_list[itmesh].alpha_color,
                    linewidth=0.2,
                    antialiased=True,
                )
            else:
                ax.plot_trisurf(
                    face_pts[:, 0],
                    face_pts[:, 1],
                    face_pts[:, 2],
                    color=mcolors.to_rgba(entity_list[itmesh].color),
                    alpha=0.5,
                    linewidth=0.2,
                    antialiased=True,
                )

    # ==============================================
    # plot all the geometries
    if show_rays:
        for i in range(0, n_ref):
            if has_intersection[i] and i % sr_view == 0:
                ax.plot(
                    xr[i], yr[i], zr[i], color=ray_color, linewidth=1 * rs_fac
                )

        for i in range(0, n_mirror_hits):
            if rec_has_intersection[i] and i % sr_view == 0:
                ax.plot(
                    xr2[i],
                    yr2[i],
                    zr2[i],
                    color=ray_color,
                    linewidth=1 * rs_fac,
                )

    if xyz_limit is not None:
        ax.set_xlim3d(xyz_limit["x_min"], xyz_limit["x_max"])
        ax.set_ylim3d(xyz_limit["y_min"], xyz_limit["y_max"])
        ax.set_zlim3d(xyz_limit["z_min"], xyz_limit["z_max"])
    else:  # generic local visualization
        ax.set_xlim3d(box_n.pmin.x, box_n.pmax.x)
        ax.set_ylim3d(box_n.pmin.y, box_n.pmax.y)
        ax.set_zlim3d(box_n.pmin.z, box_n.pmax.z)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Show the geometries
    return fig
