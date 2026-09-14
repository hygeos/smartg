#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import os 

from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface, RoughSurface
from smartg.sensor import Sensor
from smartg.atmosphere import Atm1D, AerOPAC, Cloud
from smartg.albedo import AlbedoCst

import geoclide as gc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from smartg.config import DIR_AUXDATA

from smartg.phase import calc_iphase, is_native_theta, union_theta_grid

# may be to replace
from smartg.phase import read_phase_cdf

from smartg.iprt.iprt import plot_polar_iquv
from smartg.xarray import drop_axes

from tempfile import TemporaryDirectory
from pathlib import Path


S1DB = Smartg(back=True, double=True, bias=True, pp=False)
S1DB_PP = Smartg(back=True, double=True, bias=True, pp=True, alt_pp=True)

OPT_PROP_PATH_PHASE3 = DIR_AUXDATA / 'IPRT' / 'phase3' / 'opt_prop'


def reshape_sza_vaa_vza(m, sza, vaa, vza):
    """
    Reorganize a d1-to-e5 run output on the (sza, vaa, vza) grid.

    The run stacks the viewing directions in the sensor dimension and
    the sun positions in the Zenith angles dimension; rebuild each
    variable on the sza/vaa/vza dimensions instead.
    """
    if hasattr(m, 'to_xarray'): m = m.to_xarray()  # legacy MLUT
    m = drop_axes(m, 'Azimuth angles')
    for name in list(m.data_vars):
        dims = m[name].dims
        if 'Zenith angles' in dims and 'sensor index' in dims:
            mat_tmp = np.swapaxes(m[name].data.reshape(len(vza), len(vaa), len(sza)), 0, 2)
            attrs_tmp = m[name].attrs
            m = m.drop_vars([name])
            m[name] = xr.Variable(('sza', 'vaa', 'vza'), mat_tmp, attrs=attrs_tmp)
        elif 'Zenith angles' in dims:
            mat_tmp = m[name].data
            attrs_tmp = m[name].attrs
            m = m.drop_vars([name])
            m[name] = xr.Variable(('sza',), mat_tmp, attrs=attrs_tmp)
        elif 'sensor index' in dims:
            mat_tmp = np.swapaxes(m[name].data.reshape(len(vza), len(vaa)), 0, 1)
            attrs_tmp = m[name].attrs
            m = m.drop_vars([name])
            m[name] = xr.Variable(('vaa', 'vza'), mat_tmp, attrs=attrs_tmp)
    m = drop_axes(m, 'Zenith angles', 'sensor index')
    m = m.assign_coords(sza=sza, vaa=vaa, vza=vza)
    return m

def get_d1_to_e5_boa_sensors(vza, phi, nvza, nvaa, earth_r):
    sensors = []
    for ivza in range (0, nvza):
        for ivaa in range (0, nvaa):
            vza_boa = vza[ivza]
            phi_boa = phi[ivaa]
            # Put 89.9999 instead of 90° to avoid problems due z=0 looking at horizon
            if vza_boa == 90.: vza_boa = 89.9999
            sen_tmp = Sensor(
                            pos_x=0.,
                            pos_y=0.,
                            pos_z=earth_r,
                            th_deg=vza_boa,
                            ph_deg=phi_boa,
                            loc='ATMOS',
                            sensor_type=0
                            )
            sensors.append(sen_tmp)
    return sensors

# TOA but theta instead of theta'
def get_d1_to_e5_toa_sensors_old(vza, phi, nvza, nvaa, earth_r, z):
    sensors = []
    zeros = np.zeros((nvaa), dtype=np.float64)
    origin = gc.Point(zeros, zeros, np.full((nvaa), earth_r, dtype=np.float64))
    toa_layer = gc.Sphere(earth_r+np.max(z))
    vza_bis = vza.copy()
    vza_bis[vza_bis==90.] = 89.9999
    vza_toa = 180.-vza_bis
    phi_toa = phi + 180.
    for ivza in range (0, nvza):
        dirs_tmp = gc.ang2vec(theta=vza_bis[ivza], phi=phi)#, vec_view='nadir')
        ray_tmp = gc.Ray(o=origin, d=dirs_tmp)
        ds_geo_tmp = gc.calc_intersection(toa_layer, ray_tmp)
        for ivaa in range (0, nvaa):
            if not (ds_geo_tmp['is_intersection'].values[ivaa]):
                raise NameError(f"No intersection have been found for VZA={vza_bis[ivza]} and VAA={phi[ivaa]}." + \
                                " Please check input paramaters.")
            phit = gc.Point(ds_geo_tmp['phit'].values[ivaa,:])
            if ivaa == 9 : print('vza=', vza[ivza], '; phi=', phi[ivaa], ' ; point=', phit)
            sen_tmp = Sensor(
                            pos_x=phit.x,
                            pos_y=phit.y,
                            pos_z=phit.z,
                            th_deg=vza_toa[ivza],#180-vza[ivza],
                            ph_deg=phi_toa[ivaa],
                            loc='ATMOS',
                            sensor_type=0
                            )
            sensors.append(sen_tmp)
    return sensors

# theta'
def get_d1_to_e5_toa_sensors(vza, phi, nvza, nvaa, earth_r, z):
    sensors = []
    vza_toa = 180. - vza
    phi_toa = phi + 180.
    for ivza in range (0, nvza):
        for ivaa in range (0, nvaa):
            #vza_tmp, phi_tmp = gc.vec2ang(gc.ang2vec(theta=vza[ivza], phi=phi[ivaa]), vec_view='nadir')
            sen_tmp = Sensor(
                            pos_x=0.,
                            pos_y=0.,
                            pos_z=earth_r+np.max(z),
                            th_deg=vza_toa[ivza],
                            ph_deg=phi_toa[ivaa],
                            loc='ATMOS',
                            sensor_type=0
                            )
            sensors.append(sen_tmp)
    return sensors


def get_e6_toa_sensors(vza, phi, nvza, nvaa, earth_r, z):
    sensors = []
    zeros = np.zeros((nvaa), dtype=np.float64)
    origin = gc.Point(zeros, zeros, np.full((nvaa), 3e5, dtype=np.float64))
    toa_layer = gc.Sphere(earth_r+np.max(z))

    vza_toa = 180.-vza
    phi_toa = phi + 180.

    for ivza in range (0, nvza):
        dirs_tmp = -gc.ang2vec(theta=vza[ivza], phi=phi)
        ray_tmp = gc.Ray(o=origin, d=dirs_tmp)
        ds_geo_tmp = gc.calc_intersection(toa_layer, ray_tmp)
        for ivaa in range (0, nvaa):
            if not (ds_geo_tmp['is_intersection'].values[ivaa]):
                raise NameError(f"No intersection have been found for VZA={vza[ivza]} and VAA={phi[ivaa]}." + \
                                " Please check input paramaters.")
            phit = gc.Point(ds_geo_tmp['phit'].values[ivaa,:])
            sen_tmp = Sensor(
                            pos_x=phit.x,
                            pos_y=phit.y,
                            pos_z=phit.z,
                            th_deg=vza_toa[ivza],#180-vza[ivza],
                            ph_deg=phi_toa[ivaa],
                            loc='ATMOS',
                            sensor_type=0
                            )
            sensors.append(sen_tmp)
    return sensors


def to_iprt_output(case_name, sza, saa, vza, vaa, z,
                   overwrite=True, output_dir='./'):
    
    """
    In progress...

    Parameters
    ----------
    case_name : str
        Name of the case. example 'd1', 'd2', ...
    
    """
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    output_name = f'iprt_phase3_{case_name}.nc'
    f_path = dir_output / output_name
    f_exists_tmp = f_path.exists()
    if not overwrite and f_exists_tmp:
        print(f'File {f_path} already exists. Skipping.')
    
    else:
        if case_name != 'e6':
            fboa_name = f'iprt_phase3_{case_name}_boa.nc'
            fboa_path = dir_output/ fboa_name
            ds_boa = xr.open_dataset(fboa_path)
        ftoa_name = f'iprt_phase3_{case_name}_toa.nc'
        ftoa_path = dir_output / ftoa_name
        ds_toa = xr.open_dataset(ftoa_path)
        if case_name != 'e6':
            ds = xr.Dataset(coords={f'{case_name}_sza':sza,
                                    f'{case_name}_saa':saa,
                                    f'{case_name}_vza':vza,
                                    f'{case_name}_vaa':vaa,
                                    f'{case_name}_zout':np.array([0.,np.max(z)]),
                                    'stokes':np.arange(4, dtype=np.int32)})
        else:
            ds = xr.Dataset(coords={f'{case_name}_sza':sza,
                                    f'{case_name}_saa':saa,
                                    f'{case_name}_vza':vza,
                                    f'{case_name}_vaa':vaa,
                                    f'{case_name}_zout':np.array([3e5]),
                                    'stokes':np.arange(4, dtype=np.int32)})
        nsza = len(sza)
        nsaa = len(saa)
        nvza = len(vza)
        nvaa = len(vaa)
        if case_name != 'e6': nzout = 2
        else : nzout = 1
        nstokes = 4

        radiance = np.zeros((nzout,nsza,nsaa,nvza,nvaa,nstokes), dtype=np.float32)
        std = np.zeros_like(radiance)
        if case_name != 'e6':
            radiance[0,:,0,:,:,0] = np.swapaxes(ds_boa['I_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,1] = np.swapaxes(ds_boa['Q_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,2] = np.swapaxes(ds_boa['U_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,3] = np.swapaxes(ds_boa['V_up (TOA)'].values, 1,2)
            radiance[1,:,0,:,:,0] = np.swapaxes(ds_toa['I_up (TOA)'].values, 1,2)
            radiance[1,:,0,:,:,1] = np.swapaxes(ds_toa['Q_up (TOA)'].values, 1,2)
            radiance[1,:,0,:,:,2] = np.swapaxes(ds_toa['U_up (TOA)'].values, 1,2)
            radiance[1,:,0,:,:,3] = np.swapaxes(ds_toa['V_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,0] = np.swapaxes(ds_boa['I_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,1] = np.swapaxes(ds_boa['Q_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,2] = np.swapaxes(ds_boa['U_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,3] = np.swapaxes(ds_boa['V_stdev_up (TOA)'].values, 1,2)
            std[1,:,0,:,:,0] = np.swapaxes(ds_toa['I_stdev_up (TOA)'].values, 1,2)
            std[1,:,0,:,:,1] = np.swapaxes(ds_toa['Q_stdev_up (TOA)'].values, 1,2)
            std[1,:,0,:,:,2] = np.swapaxes(ds_toa['U_stdev_up (TOA)'].values, 1,2)
            std[1,:,0,:,:,3] = np.swapaxes(ds_toa['V_stdev_up (TOA)'].values, 1,2)
        else:

            radiance[0,:,0,:,:,0] = np.swapaxes(ds_toa['I_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,1] = np.swapaxes(ds_toa['Q_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,2] = np.swapaxes(ds_toa['U_up (TOA)'].values, 1,2)
            radiance[0,:,0,:,:,3] = np.swapaxes(ds_toa['V_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,0] = np.swapaxes(ds_toa['I_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,1] = np.swapaxes(ds_toa['Q_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,2] = np.swapaxes(ds_toa['U_stdev_up (TOA)'].values, 1,2)
            std[0,:,0,:,:,3] = np.swapaxes(ds_toa['V_stdev_up (TOA)'].values, 1,2)

        
        ds[f'radiance_{case_name}'] = xr.DataArray(radiance, dims=[f'{case_name}_zout', f'{case_name}_sza', f'{case_name}_saa',
                                                                   f'{case_name}_vza', f'{case_name}_vaa', 'stokes' ])
        ds[f'std_{case_name}'] = xr.DataArray(std, dims=[f'{case_name}_zout', f'{case_name}_sza', f'{case_name}_saa',
                                                         f'{case_name}_vza', f'{case_name}_vaa', 'stokes' ])
        ds.to_netcdf(f_path)


def to_iprt_output_e6_v1(case_name, sza, saa, nx, ny, is_sens, vecs,
                         overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    output_name = f'iprt_phase3_{case_name}.nc'
    f_path = dir_output / output_name
    f_exists_tmp = f_path.exists()
    if not overwrite and f_exists_tmp:
        print(f'File {f_path} already exists. Skipping.')
    else:
        ftoa_name = f'iprt_phase3_{case_name}_toa.nc'
        ftoa_path = dir_output / ftoa_name
        ds0 = xr.open_dataset(ftoa_path) 
        ds = xr.Dataset(coords={f'{case_name}_lat':saa,
                                f'{case_name}_lon':sza,
                                f'{case_name}_prow':np.arange(ny, dtype=np.int32),
                                f'{case_name}_pcol':np.arange(nx, dtype=np.int32),
                                f'{case_name}_zout':np.array([3e5]),
                                'stokes':np.arange(4, dtype=np.int32)})
        nzout = int(1)
        nlat = len(ds[f'{case_name}_lat'].values)
        nlon = len(ds[f'{case_name}_lon'].values)
        nprow = len(ds[f'{case_name}_prow'].values)
        npcol = len(ds[f'{case_name}_pcol'].values)
        nstokes = int(4)

        radiance = np.zeros((nzout,nlat,nlon,nprow,npcol,nstokes), dtype=np.float32)
        std = np.zeros_like(radiance)
        vza_ = np.zeros((nprow,npcol), dtype=np.float64)
        vaa_ = np.zeros_like(vza_)

        for ilon in range (0, nlon):
            ic = 0
            ic_ = 0
            for ix in range (0, nx):
                for iy in range(0, ny):
                    th, ph = gc.vec2ang(gc.Vector(vecs.x[ic], vecs.y[ic], vecs.z[ic]), vec_view='nadir')
                    if (th == 0. or th == 180.): ph = 0.
                    vza_[iy,ix] = th
                    vaa_[iy,ix] = ph
                    if is_sens[ic]:
                        radiance[0,0,ilon,iy,ix,0] = ds0['I_up (TOA)'].values[ic_,0,ilon]
                        radiance[0,0,ilon,iy,ix,1] = ds0['Q_up (TOA)'].values[ic_,0,ilon]
                        radiance[0,0,ilon,iy,ix,2] = ds0['U_up (TOA)'].values[ic_,0,ilon]
                        radiance[0,0,ilon,iy,ix,3] = ds0['V_up (TOA)'].values[ic_,0,ilon]
                        std[0,0,ilon,iy,ix,0] = ds0['I_stdev_up (TOA)'].values[ic_,0,ilon]
                        std[0,0,ilon,iy,ix,1] = ds0['Q_stdev_up (TOA)'].values[ic_,0,ilon]
                        std[0,0,ilon,iy,ix,2] = ds0['U_stdev_up (TOA)'].values[ic_,0,ilon]
                        std[0,0,ilon,iy,ix,3] = ds0['V_stdev_up (TOA)'].values[ic_,0,ilon]
                        ic_+=1
                    ic += 1

        ds[f'radiance_{case_name}'] = xr.DataArray(radiance, dims=[f'{case_name}_zout', f'{case_name}_lat', f'{case_name}_lon',
                                                                f'{case_name}_prow', f'{case_name}_pcol', 'stokes' ])
        ds[f'std_{case_name}'] = xr.DataArray(std, dims=[f'{case_name}_zout', f'{case_name}_lat', f'{case_name}_lon',
                                                        f'{case_name}_prow', f'{case_name}_pcol', 'stokes' ])
        ds[f'vza_{case_name}'] = xr.DataArray(vza_, dims=[f'{case_name}_prow', f'{case_name}_pcol'])
        ds[f'vaa_{case_name}'] = xr.DataArray(vaa_, dims=[f'{case_name}_prow', f'{case_name}_pcol'])
        ds.to_netcdf(f_path)

def to_iprt_output_e6_v2(case_name, sza, saa, nx, ny, vecs,
                         overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    output_name = f'iprt_phase3_{case_name}.nc'
    f_path = dir_output / output_name
    f_exists_tmp = f_path.exists()
    if not overwrite and f_exists_tmp:
        print(f'File {f_path} already exists. Skipping.')
    else:
        ftoa_name = f'iprt_phase3_{case_name}_toa.nc'
        ftoa_path = dir_output / ftoa_name
        ds0 = xr.open_dataset(ftoa_path) 
        ds = xr.Dataset(coords={f'{case_name}_lat':saa,
                                f'{case_name}_lon':sza,
                                f'{case_name}_prow':np.arange(ny, dtype=np.int32),
                                f'{case_name}_pcol':np.arange(nx, dtype=np.int32),
                                f'{case_name}_zout':np.array([3e5]),
                                'stokes':np.arange(4, dtype=np.int32)})
        nzout = int(1)
        nlat = len(ds[f'{case_name}_lat'].values)
        nlon = len(ds[f'{case_name}_lon'].values)
        nprow = len(ds[f'{case_name}_prow'].values)
        npcol = len(ds[f'{case_name}_pcol'].values)
        nstokes = int(4)

        radiance = np.zeros((nzout,nlat,nlon,nprow,npcol,nstokes), dtype=np.float32)
        std = np.zeros_like(radiance)
        vza_ = np.zeros((nprow,npcol), dtype=np.float64)
        vaa_ = np.zeros_like(vza_)

        for ilon in range (0, nlon):
            ic = 0
            for ix in range (0, nx):
                for iy in range(0, ny):
                    th, ph = gc.vec2ang(gc.Vector(vecs.x[ic], vecs.y[ic], vecs.z[ic]), vec_view='nadir')
                    if (th == 0. or th == 180.): ph = 0.
                    vza_[iy,ix] = th
                    vaa_[iy,ix] = ph
                    radiance[0,0,ilon,iy,ix,0] = ds0['I_up (TOA)'].values[ic,0,ilon]
                    radiance[0,0,ilon,iy,ix,1] = ds0['Q_up (TOA)'].values[ic,0,ilon]
                    radiance[0,0,ilon,iy,ix,2] = ds0['U_up (TOA)'].values[ic,0,ilon]
                    radiance[0,0,ilon,iy,ix,3] = ds0['V_up (TOA)'].values[ic,0,ilon]
                    std[0,0,ilon,iy,ix,0] = ds0['I_stdev_up (TOA)'].values[ic,0,ilon]
                    std[0,0,ilon,iy,ix,1] = ds0['Q_stdev_up (TOA)'].values[ic,0,ilon]
                    std[0,0,ilon,iy,ix,2] = ds0['U_stdev_up (TOA)'].values[ic,0,ilon]
                    std[0,0,ilon,iy,ix,3] = ds0['V_stdev_up (TOA)'].values[ic,0,ilon]
                    ic += 1

        ds[f'radiance_{case_name}'] = xr.DataArray(radiance, dims=[f'{case_name}_zout', f'{case_name}_lat', f'{case_name}_lon',
                                                                f'{case_name}_prow', f'{case_name}_pcol', 'stokes' ])
        ds[f'std_{case_name}'] = xr.DataArray(std, dims=[f'{case_name}_zout', f'{case_name}_lat', f'{case_name}_lon',
                                                        f'{case_name}_prow', f'{case_name}_pcol', 'stokes' ])
        ds[f'vza_{case_name}'] = xr.DataArray(vza_, dims=[f'{case_name}_prow', f'{case_name}_pcol'])
        ds[f'vaa_{case_name}'] = xr.DataArray(vaa_, dims=[f'{case_name}_prow', f'{case_name}_pcol'])
        ds.to_netcdf(f_path)

def run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
            sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wavelength, le,
            surface, pro, dep, z, ntheta=18001, pp=False, is_e6=False,
            theta_grid=None):

    if pp :
        sg = S1DB_PP
        earth_radius = 6371.
    else:
        sg = S1DB
        earth_radius=earth_r

    # BOA
    if (overwrite or not fboa_exist) and not is_e6:
        sensors = get_d1_to_e5_boa_sensors(vza, phi, nvza, nvaa, earth_r)
        m_boa = sg.run(wavelength=wavelength, n_photons=nvza*nvaa*nphotons,
                        n_loop=nphotons, atmosphere=pro, sensor=sensors,
                        output_layers=1,
                        le=le, surface=surface, xblock=64, xgrid=1024, beer=1, depo=dep, reflectance=False, earth_radius=earth_radius,
                        stdev=True, progress=True, n_icdf=ntheta,
                        theta_grid=theta_grid)#, seed=1e8)

        m_boa = reshape_sza_vaa_vza(m_boa, sza, vaa, vza)
        m_boa.to_netcdf(str(fboa_path))

    # TOA
    if overwrite or not ftoa_exist:
        if not is_e6: sensors = get_d1_to_e5_toa_sensors(vza, phi, nvza, nvaa, earth_r, z)
        else : sensors = get_e6_toa_sensors(vza, phi, nvza, nvaa, earth_r, z)
        m_toa = sg.run(wavelength=wavelength, n_photons=nvza*nvaa*nphotons,
                        n_loop=nphotons, atmosphere=pro, sensor=sensors,
                        output_layers=1,
                        le=le, surface=surface, xblock=64, xgrid=1024, beer=1, depo=dep, reflectance=False, earth_radius=earth_radius,
                        stdev=True, progress=True, n_icdf=ntheta,
                        theta_grid=theta_grid)#, seed=1e8)

        m_toa = reshape_sza_vaa_vza(m_toa, sza, vaa, vza)
        m_toa.to_netcdf(str(ftoa_path))


def aer2smartg(filename, n_theta: int | str = 1801, rh_or_reff=None, rh_reff=None):
    """
    
    In progress

    Parameters
    ----------
    filename : str | pathlib.Path | xr.Dataset
        The path to the file to be converted. It can be directly an xr.Dataset.
    n_theta : int | str
        The number of equally spaced angles for the phase matrix, or
        'native' for the union of the angle grids the file carries, on
        which the phase matrix is kept without resampling.

    Results
    -------
    out : xr.Dataset
        The converted dataset.
    """

    if isinstance(filename, xr.Dataset): ds = filename
    else:
        filename = Path(filename)
        ds = xr.open_dataset(filename)

    if 'hum' in ds.variables: 
        if rh_reff is None: rh_reff = ds["hum"].values
        if rh_or_reff is None: rh_or_reff = 'hum'
    elif 'reff' in ds.variables:
        if rh_reff is None: rh_reff = ds["reff"].values
        if rh_or_reff is None: rh_or_reff = 'reff'
    else:
        raise Exception('Error')
    
    phase = ds["phase"][:, :, :, :].values

    NBSTK   = ds.nphamat.size
    NBRH_OR_REFF  = rh_reff.size
    if is_native_theta(n_theta):
        # one grid per (wavelength, rh/reff, term), stored descending
        # and padded past ntheta; identical grids are merged first
        theta_all = ds["theta"].values
        ntheta_all = ds["ntheta"].values
        grids = []
        for idx in np.ndindex(ntheta_all.shape):
            nth = int(ntheta_all[idx])
            grid = np.sort(theta_all[idx][:nth].astype(np.float64))
            if not any(np.array_equal(grid, g) for g in grids):
                grids.append(grid)
        theta = union_theta_grid(grids)
    else:
        theta = np.linspace(0., 180., num=int(n_theta))
    NWAV    = max(ds["wavelen"].size, int(2))
    if NWAV > ds["wavelen"].size:
        wavelength = np.concatenate((ds["wavelen"].values*1e3, ds["wavelen"].values*1e3 + 0.1))
    else:
        wavelength = ds["wavelen"].values*1e3

    ext_out = np.zeros((NBRH_OR_REFF, NWAV), dtype=np.float64)
    ssa_out = np.zeros_like(ext_out)
    pha_out = np.zeros((NBRH_OR_REFF, NWAV, NBSTK, len(theta)), dtype=np.float64)

    for i_wavelength in range (0, ds["wavelen"].size):
        for irhreff in range(NBRH_OR_REFF):
            ext_out[irhreff,i_wavelength] = ds["ext"][i_wavelength,irhreff]
            ssa_out[irhreff,i_wavelength] = ds["ssa"][i_wavelength,irhreff]
            for istk in range (NBSTK):
                # ntheta (wavelength, reff, stk)
                nth = ds["ntheta"][i_wavelength,irhreff,istk].data
                # theta (wavelength, reff, stk, ntheta)
                th = ds["theta"][i_wavelength,irhreff,istk,:].data
                pha_out[irhreff,i_wavelength,istk,:] = np.interp(
                    theta, th[:nth],
                    phase[i_wavelength,irhreff,istk,:nth],
                    period=np.inf,
                )
    
    if NWAV > ds["wavelen"].size:
        ext_out[:,-1] = ext_out[irhreff,0]
        ssa_out[:,-1] = ssa_out[irhreff,0]
        pha_out[:,-1,:,:] = pha_out[:,0,:,:]
        
    ds_out = xr.Dataset(coords={rh_or_reff:rh_reff, 'wav':wavelength, 'theta': theta})
    ds_out['ext'] = xr.DataArray(ext_out, dims=[rh_or_reff, 'wav'])
    ds_out['ext'].attrs = {'description': 'Extinction coefficient'}
    ds_out['ssa'] = xr.DataArray(ssa_out, dims=[rh_or_reff, 'wav'])
    ds_out['ssa'].attrs = {'description': 'Single scattering albedo'}
    ds_out['phase'] = xr.DataArray(pha_out, dims=[rh_or_reff, 'wav', 'stk', 'theta'])
    ds_out['phase'].attrs = {'description': 'scattering phase matrix'}
    if isinstance(filename, Path) : name = filename.name
    else: name = 'none'
    if rh_or_reff == 'hum':
        ds_out.attrs = {'name': name,
                        'H_mix_min': '0.',
                        'H_mix_max': '2',
                        'H_free_min': '2',
                        'H_free_max': '12',
                        'H_stra_max': '12',
                        'Z_mix': '8',
                        'Z_free': '8',
                        'Z_stra': '99'}
    else:
        ds_out.attrs = {'name': name}

    return ds_out


def plot_polar_iprt(I, Q, U, V, thetas, phis, change_Q_sign=False, change_U_sign=False,
                    change_V_sign=False, maxI=None, maxQ=None, maxU=None, maxV=None,  cmapI=None, cmapQ=None, cmapU=None, cmapV=None,
                    title=None, save_fig=None, sym=False, minI=None):
    """
    Plot phase 3 I, Q, U and V matrices in polar view.

    Wrapper of smartg.iprt.iprt.plot_polar_iquv for the phase 3 layout,
    where row j holds the viewing zenith angle thetas[j] and is drawn
    at its own rescaled radius; plot_polar_iquv takes the rows in the
    reverse order.

    Parameters
    ----------
    I, Q, U, V : ndarray
        Stokes parameters, each of shape (ntheta, nphi).
    thetas : ndarray
        Viewing zenith angles of the rows, in degrees. They are
        rescaled to span the radius from 0 to 90.
    phis : ndarray
        Viewing azimuth angles of the columns, in degrees.
    change_Q_sign, change_U_sign, change_V_sign : bool
        Multiply Q, U or V by -1.
    maxI, maxQ, maxU, maxV : float, optional
        Upper bound of the colour scale of each panel. By default the
        largest absolute value of the panel is used. The Q, U and V
        panels are drawn from -max to +max, the I panel from minI.
    cmapI, cmapQ, cmapU, cmapV : str or Colormap, optional
        Colour map of each panel, 'jet' for I and 'RdBu_r' for the
        other panels by default.
    title : str, optional
        Title of the whole figure.
    save_fig : str or Path, optional
        Save the figure at this path, the extension giving the
        format, e.g. save_fig='myFigName.png'.
    sym : bool
        The IPRT azimuth angles cover 0 to 180 degrees; also plot the
        symmetrical results from 180 to 360 degrees.
    minI : float, optional
        Lower bound of the colour scale of the I panel. By default 0,
        or -maxI when maxI is given.
    """
    Q_sign = -1 if change_Q_sign else 1
    U_sign = -1 if change_U_sign else 1
    V_sign = -1 if change_V_sign else 1
    iquv = (np.asarray(I)[::-1],
            np.asarray(Q)[::-1] * Q_sign,
            np.asarray(U)[::-1] * U_sign,
            np.asarray(V)[::-1] * V_sign)
    plot_polar_iquv(iquv, np.asarray(thetas), np.asarray(phis),
                    max_i=maxI, max_q=maxQ, max_u=maxU, max_v=maxV,
                    min_i=minI, cmap_i=cmapI, cmap_q=cmapQ, cmap_u=cmapU,
                    cmap_v=cmapV, title=title, save_fig=save_fig, sym=sym)

def plot_camera_iprt(I, Q, U, V,
                     I_min=0., I_max=None, I_cmap='viridis',
                     title=None, save_fig=None):
    """
    """
    matplotlib.rcParams.update({'font.size': 16})
    fig_size = (20,4)
    fig, axs = plt.subplots(1,4, figsize=fig_size, constrained_layout=True, sharex=True, sharey=True)
    if title is not None: fig.suptitle(title, fontsize=20)
    
    caxI = axs[0].imshow(I, vmin=I_min, vmax=I_max, origin='upper', cmap=plt.get_cmap(I_cmap))
    cbarI = plt.colorbar(caxI)
    cbarI.set_label('I', fontsize=20)

    caxQ = axs[1].imshow(Q, vmin=-np.max(np.abs(Q)), vmax=np.max(np.abs(Q)),
                        origin='upper', cmap=plt.get_cmap('coolwarm'))
    cbarQ = plt.colorbar(caxQ)
    cbarQ.set_label('Q', fontsize=20)

    caxU = axs[2].imshow(U, vmin=-np.max(np.abs(U)), vmax=np.max(np.abs(U)),
                        origin='upper', cmap=plt.get_cmap('coolwarm'))
    cbarU = plt.colorbar(caxU)
    cbarU.set_label('U', fontsize=20)

    caxV = axs[3].imshow(V, vmin=-np.max(np.abs(V)), vmax=np.max(np.abs(V)),
                        origin='upper', cmap=plt.get_cmap('coolwarm'))
    cbarV = plt.colorbar(caxV)
    cbarV.set_label('V', fontsize=20)

    if save_fig is not None: plt.savefig(save_fig)


def case_D1(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d1_boa.nc'
    ftoa_name = f'iprt_phase3_d1_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):
        mol_sca = np.array([0., 0.5])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        wavelength = 550.

        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs
        ).calc(wavelength)
        surface = None
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d1', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_D2(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d2_boa.nc'
    ftoa_name = f'iprt_phase3_d2_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):
        mol_sca = np.array([0., 0.1])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        wavelength = 550.

        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs
        ).calc(wavelength)
        surface = LambSurface(alb=AlbedoCst(0.3))
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d2', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    

def case_D3(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d3_boa.nc'
    ftoa_name = f'iprt_phase3_d3_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        nz = len(z[1:])

        # aerosol extinction and single scattering albedo
        aer_tau_ext = np.full_like(mol_sca, 0.2, dtype=np.float32)
        aer_tau_ext[:,0] = 0. # dtau TOA equal to 0
        aer_ssa = np.full_like(mol_sca, 0.975683, dtype=np.float32)
        prof_aer = (aer_tau_ext, aer_ssa)

        # aerosol phase matrix
        nth = 18001
        theta = np.linspace(0, 180, nth)
        wavelength = np.array([350.])
        n_wavelength = len(wavelength)
        file_aer_phase = OPT_PROP_PATH_PHASE3 / "waso.mie.cdf"
        aer_phase = read_phase_cdf(file_aer_phase, n_theta=nth, normalize=True, output_sg_ready=False)
        nstk = aer_phase.shape[2]

        aer_pha = np.zeros((n_wavelength, nz, nstk, nth), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            aer_pha[:,iz,:,:] = (
                aer_phase.isel({aer_phase.dims[1]: 0})
                # a single wavelength in the file: interp would give NaN
                .sel(wavelength_phase=wavelength, method='nearest')
                .interp(theta_atm=theta).data
            )
        aer_phase = xr.DataArray(
            aer_pha,
            dims=['wavelength_phase', 'z_phase', 'nphamat', 'theta_atm'],
            coords={'wavelength_phase': wavelength, 'z_phase': z[1:],
                    'theta_atm': theta},
        )
        pha_atm, ipha_atm = calc_iphase(aer_phase, np.array([wavelength]), z)
        lpha = []
        for i in range (0, pha_atm.shape[0]):
            lpha.append(xr.DataArray(
                pha_atm[i,:,:],
                dims=['nphamat', 'theta_atm'],
                coords={'theta_atm': np.linspace(0, 180, nth)},
            ))

        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs,
            prof_aer=prof_aer, prof_phases=(ipha_atm, lpha),
        ).calc(wavelength, phase=False)
        surface = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa 
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d3', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_D4(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d4_boa.nc'
    ftoa_name = f'iprt_phase3_d4_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        nz = len(z[1:])

        # aerosol extinction and single scattering albedo
        aer_tau_ext = np.full_like(mol_sca, 0.2, dtype=np.float32)
        aer_tau_ext[:,0] = 0. # dtau TOA equal to 0
        aer_ssa = np.full_like(mol_sca, 0.787581, dtype=np.float32)
        prof_aer = (aer_tau_ext, aer_ssa)

        # aerosol phase matrix
        nth = 18001
        theta = np.linspace(0, 180, nth)
        wavelength = np.array([350.])
        n_wavelength = len(wavelength)

        file_aer_phase = OPT_PROP_PATH_PHASE3 / "sizedistr_spheroid.cdf"
        aer_phase = read_phase_cdf(file_aer_phase, n_theta=nth, normalize=True, output_sg_ready=False)
        nstk = aer_phase.shape[2]

        aer_pha = np.zeros((n_wavelength, nz, nstk, nth), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            aer_pha[:,iz,:,:] = (
                aer_phase.isel({aer_phase.dims[1]: 0})
                # a single wavelength in the file: interp would give NaN
                .sel(wavelength_phase=wavelength, method='nearest')
                .interp(theta_atm=theta).data
            )
        aer_phase = xr.DataArray(
            aer_pha,
            dims=['wavelength_phase', 'z_phase', 'nphamat', 'theta_atm'],
            coords={'wavelength_phase': wavelength, 'z_phase': z[1:],
                    'theta_atm': theta},
        )

        pha_atm, ipha_atm = calc_iphase(aer_phase, np.array([wavelength]), z)
        lpha = []
        for i in range (0, pha_atm.shape[0]):
            lpha.append(xr.DataArray(
                pha_atm[i,:,:],
                dims=['nphamat', 'theta_atm'],
                coords={'theta_atm': np.linspace(0, 180, nth)},
            ))

        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs,
            prof_aer=prof_aer, prof_phases=(ipha_atm, lpha),
        ).calc(wavelength, phase=False)
        surface = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d4', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_D4_bis(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d4_bis_boa.nc'
    ftoa_name = f'iprt_phase3_d4_bis_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        wavelength = np.array([350.])
        nth = 1801

        ds_spheroid = aer2smartg(OPT_PROP_PATH_PHASE3 / "sizedistr_spheroid.cdf",
                                 n_theta=nth, rh_or_reff='hum', rh_reff=np.array([0.]))

        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'spheroid_d4.nc'
            ds_spheroid.to_netcdf(file_path)
            aer = AerOPAC(str(file_path), 0.2, 350., h_min_mix=0., h_mix_max=120.,
                          h_free_min=120., h_free_max=120., h_stra_min=120., h_stra_max=120., z_mix=1e6,
                          rh_mix=0.)
        pro = Atm1D(
            'afglt', comp=[aer], grid=z, prof_ray=mol_sca,
            prof_abs=mol_abs,
        ).calc(wavelength, phase=True, n_theta=nth)
        surface = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d4_bis', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_D5(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d5_boa.nc'
    ftoa_name = f'iprt_phase3_d5_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        nz = len(z[1:])

        # aerosol extinction and single scattering albedo
        aer_tau_ext = np.full_like(mol_sca, 5, dtype=np.float32)
        aer_tau_ext[:,0] = 0. # dtau TOA equal to 0
        aer_ssa = np.full_like(mol_sca, 0.999979, dtype=np.float32)
        prof_aer = (aer_tau_ext, aer_ssa)

        # cloud phase matrix
        nth = 18001
        wavelength = np.array([800.])
        n_wavelength = len(wavelength)
        theta = np.linspace(0, 180, nth)
        file_cld_phase = OPT_PROP_PATH_PHASE3 / "watercloud.mie.cdf"
        cld_phase = read_phase_cdf(file_cld_phase, n_theta=nth, normalize=True, output_sg_ready=False)
        nstk = cld_phase.shape[2]

        cld_pha = np.zeros((n_wavelength, nz, nstk, nth), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            cld_pha[:,iz,:,:] = (
                cld_phase.isel({cld_phase.dims[1]: 0})
                # a single wavelength in the file: interp would give NaN
                .sel(wavelength_phase=wavelength, method='nearest')
                .interp(theta_atm=theta).data
            )
        cld_phase = xr.DataArray(
            cld_pha,
            dims=['wavelength_phase', 'z_phase', 'nphamat', 'theta_atm'],
            coords={'wavelength_phase': wavelength, 'z_phase': z[1:],
                    'theta_atm': theta},
        )

        pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([wavelength]), z)
        lpha = []
        for i in range (0, pha_atm.shape[0]):
            lpha.append(xr.DataArray(
                pha_atm[i,:,:], dims=['nphamat', 'theta_atm'],
                coords={'theta_atm': theta},
            ))

        # atmosphere profil
        pro = Atm1D('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer,
                    prof_phases=(ipha_atm, lpha)
                    ).calc(wavelength, phase=False)
        surface = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d5', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    

def case_D6(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d6_boa.nc'
    ftoa_name = f'iprt_phase3_d6_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.1])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.]) 
        wavelength = 550.
        
        # atmosphere profil
        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs
        ).calc(wavelength)
        surface = RoughSurface(wind=2., brdf=True, wave_shadow=True, nh2o=1.33)

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d6', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    
def case_D6_pp(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_d6_pp_boa.nc'
    ftoa_name = f'iprt_phase3_d6_pp_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.1])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.]) 
        wavelength = 550.
        
        # atmosphere profil
        pro = Atm1D(
            'afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs
        ).calc(wavelength)
        # surface = RoughSurface(wind=2., brdf=False, wave_shadow=True, nh2o=1.33,
        #                     sur=1, single=True)
        surface = RoughSurface(wind=2., brdf=True, wave_shadow=True, nh2o=1.33)

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 0. # here in pp this is the ground altitude

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le,
                surface, pro, dep, z, pp=True)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d6_pp', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    
def case_E1(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e1_boa.nc'
    ftoa_name = f'iprt_phase3_e1_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        wavelength = 450.
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        pro = Atm1D(
            'afglt', grid=z, prof_ray=sca, prof_abs=np.zeros_like(sca)
        ).calc(wavelength)
        surface = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e1', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_E2(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e2_boa.nc'
    ftoa_name = f'iprt_phase3_e2_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_320nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_320nm_usstd.dat"
        wavelength = 320.
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        pro = Atm1D(
            'afglt', grid=z, prof_ray=sca, prof_abs=abs
        ).calc(wavelength)
        surface = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e2', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    

def case_E3(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e3_boa.nc'
    ftoa_name = f'iprt_phase3_e3_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
        wavelength = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        nth = 18001

        file_aer1_phase = OPT_PROP_PATH_PHASE3 / "desert.cdf"
        ds_desert = aer2smartg(file_aer1_phase, n_theta=nth, rh_or_reff='hum', rh_reff=np.array([0.]))
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'desert_e3.nc'
            ds_desert.to_netcdf(file_path)
            aer1 = AerOPAC(str(file_path), 0.5, wavelength[0],
                           h_min_mix=0., h_mix_max=3.,
                           h_free_min=2., h_free_max=2., h_stra_min=12., h_stra_max=12., z_mix=1e6, rh_mix=0.)

        pro = Atm1D(
            'afglt', comp=[aer1], grid=z, prof_ray=sca, prof_abs=abs
        ).calc(wavelength, phase=True, n_theta=nth)
        surface = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e3', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_E4(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e4_boa.nc'
    ftoa_name = f'iprt_phase3_e4_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
        wavelength = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        nth = 18001

        file_aer1_phase = OPT_PROP_PATH_PHASE3 / "desert.cdf"
        file_aer2_phase = OPT_PROP_PATH_PHASE3 / "sulfate.cdf"
        ds_desert = aer2smartg(file_aer1_phase, n_theta=nth, rh_or_reff='hum', rh_reff=np.array([0.]))
        ds_sulfate = aer2smartg(file_aer2_phase, n_theta=nth, rh_or_reff='hum', rh_reff=np.array([0.]))
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'desert_e4.nc'
            ds_desert.to_netcdf(file_path)
            aer1 = AerOPAC(str(file_path), 0.5, wavelength[0],
                           h_min_mix=0., h_mix_max=3.,
                           h_free_min=2., h_free_max=2., h_stra_min=12., h_stra_max=12., z_mix=1e6,
                           rh_mix=0.)
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'sulfate_e4.nc'
            ds_sulfate.to_netcdf(file_path)
            aer2 = AerOPAC(str(file_path), 0.05, wavelength[0],
                           h_min_mix=20., h_mix_max=21.,
                           h_free_min=2., h_free_max=2., h_stra_min=12., h_stra_max=12., z_mix=1e6,
                           rh_mix=0.)

        pro = Atm1D('afglt', comp=[aer1, aer2], grid=z, prof_ray=sca, 
                      prof_abs=abs, pfgrid=[120., 21., 20., 3., 0.]
                      ).calc(wavelength, phase=True, n_theta=nth)
        surface = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e4', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    

def case_E5(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e5_boa.nc'
    ftoa_name = f'iprt_phase3_e5_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
        wavelength = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        nth = 18001

        file_cld1_phase = OPT_PROP_PATH_PHASE3 / "ic.ghm.baum.cdf"
        ds_ic_baum_ghm_ = xr.open_dataset(file_cld1_phase)
        wavelength_ = ds_ic_baum_ghm_.wavelen.values
        # wavelength in micrometers, here take only between 400
        # and 500nm
        nlam_ = np.squeeze(np.argwhere(
            np.logical_and(wavelength_>=0.4, wavelength_<=0.5)
        ))
        ds_ic_baum_ghm_  = ds_ic_baum_ghm_ .sel(nlam = nlam_)
        ds_ic_baum_ghm = aer2smartg(ds_ic_baum_ghm_, n_theta=nth)
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'ic_ghm_baum_e5.nc'
            ds_ic_baum_ghm.to_netcdf(file_path)
            cld1 = Cloud(str(file_path), reff=50., zmin=10.,
                         zmax=11., tau_ref=1., w_ref=wavelength[0])

        pro = Atm1D(
            'afglt', comp=[cld1], grid=z, prof_ray=sca, prof_abs=abs
        ).calc(wavelength, phase=True, n_theta=nth)
        surface = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, ntheta=nth)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e5', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_E5_bis(nphotons=1e8, overwrite=True, output_dir='./'):
    """
    Run the IPRT case E5 on the native scattering angles of the cloud.

    The same ice cloud as case_E5, whose phase matrix is resampled onto
    18001 equally spaced angles, but kept on the angle grid of the
    ic.ghm.baum.cdf file (498 angles, 0.01 degree apart in the forward
    peak), from the conversion of the file to the GPU tables.

    Parameters
    ----------
    nphotons : float
        Number of photons per viewing direction.
    overwrite : bool
        Run the simulations even when their intermediate files exist.
    output_dir : str or Path
        Folder of the intermediate files and of the IPRT output file
        iprt_phase3_e5_bis.nc.
    """

    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e5_bis_boa.nc'
    ftoa_name = f'iprt_phase3_e5_bis_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = None

    if (overwrite      or
        not fboa_exist or
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
        wavelength = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)

        file_cld1_phase = OPT_PROP_PATH_PHASE3 / "ic.ghm.baum.cdf"
        ds_ic_baum_ghm_ = xr.open_dataset(file_cld1_phase)
        wavelength_ = ds_ic_baum_ghm_.wavelen.values
        # wavelength in micrometers, here take only between 400
        # and 500nm
        nlam_ = np.squeeze(np.argwhere(
            np.logical_and(wavelength_>=0.4, wavelength_<=0.5)
        ))
        ds_ic_baum_ghm_  = ds_ic_baum_ghm_ .sel(nlam = nlam_)
        # keep the angles of the file instead of resampling them
        ds_ic_baum_ghm = aer2smartg(ds_ic_baum_ghm_, n_theta='native')
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'ic_ghm_baum_e5_bis.nc'
            ds_ic_baum_ghm.to_netcdf(file_path)
            cld1 = Cloud(str(file_path), reff=50., zmin=10.,
                         zmax=11., tau_ref=1., w_ref=wavelength[0])

        pro = Atm1D(
            'afglt', comp=[cld1], grid=z, prof_ray=sca, prof_abs=abs
        ).calc(wavelength, phase=True, n_theta='native')
        surface = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files, the GPU tables
        # adopting the angles of the phase matrices
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons,
                wavelength, le, surface, pro, dep, z, theta_grid='phase')

    # open intermediate files and convert to iprt phase3 output format
    to_iprt_output('e5_bis', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_E6_old(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    ftoa_name = f'iprt_phase3_e6_toa.nc'
    ftoa_path = dir_output / ftoa_name
    ftoa_exist = ftoa_path.exists()


    sza = np.array([50., 90., 110., 130.])
    saa = np.array([0.])
    vza = np.arange(0., 1.2+0.04, 0.04)
    vaa = np.arange(0., 360.+10, 10)
    z = None

    if (overwrite      or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
        wavelength = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        zs = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
        pro = Atm1D(
            'afglt', grid=z, prof_ray=sca, prof_abs=abs
        ).calc(wavelength)
        surface = RoughSurface(wind=5., brdf=True, wave_shadow=True, nh2o=1.33)
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                               count_level=count_lvl)

        # run simulations and create intermediate files
        run_sim(overwrite, False, ftoa_exist, 'none.nc', ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, 
                wavelength, le, surface, pro, dep, z, is_e6=True)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e6', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)
    
    
def case_E6_v1(nphotons=1e8, overwrite=True, output_dir='./'):
    """
    In this version 1, we take one direction at each pixel center
    """
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    ftoa_name = f'iprt_phase3_e6_v1_toa.nc'
    ftoa_path = dir_output / ftoa_name
    ftoa_exist = ftoa_path.exists()


    sza = np.array([50., 90., 110., 130.])
    saa = np.array([0.])
    vza = np.arange(0., 1.2+0.04, 0.04)
    z = None
    nx = 61
    ny = 61
    nsens = nx*ny
    ntheta = 18001

    # atmosphere profil
    mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
    mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
    wavelength = np.array([450.])
    z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
    zs = len(z)
    sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    pro = Atm1D('afglt', grid=z, prof_ray=sca, prof_abs=abs).calc(wavelength)
    surface = RoughSurface(wind=5., brdf=True, wave_shadow=True, nh2o=1.33)
    
    nvza = len(vza)
    dep = 0.03
    earth_r = 6371.

    count_lvl = np.zeros_like(sza, dtype=np.int32)
    phi_0 = -saa # To follow iprt anti-clockwise convention
    le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                           count_level=count_lvl)

    # run simulations and create intermediate files
    # ==== Get directions
    zeros = np.zeros((nvza), dtype=np.float64)
    sen = gc.Point(zeros, zeros,np.full((nvza), 1, dtype=np.float64))
    dirs = -gc.ang2vec(theta=vza, phi=180.)
    rays = gc.Ray(o=sen, d=dirs)
    ground = gc.BBox(p1=gc.Point(-np.inf, -np.inf, 0.), p2=gc.Point(np.inf, np.inf, 0.))
    ds = gc.calc_intersection(ground, rays)

    coord = np.concatenate((-(ds['phit'][1:,0].values)[::-1], ds['phit'][:,0].values))
    pts = np.zeros((nsens,3), dtype=np.float64)
    ic = 0
    for ix in range (0, nx):
        for iy in range(0, ny):
            pts[ic,0] = coord[ix]
            pts[ic,1] = coord[::-1][iy] # to begin from top left pixel
            ic += 1
    points = gc.Point(pts)

    zeros_ini = np.zeros((nsens), dtype=np.float64)
    points_ini = gc.Point(zeros_ini, zeros_ini, np.full((nsens), 1, dtype=np.float64))

    vecs = gc.normalize(points - points_ini)
    # ==== create toa sensors
    is_sens = np.full((nsens), False, dtype=np.bool)
    zeros = np.zeros((nsens), dtype=np.float64)
    origin = gc.Point(zeros, zeros, np.full((nsens), 3e5, dtype=np.float64))
    toa_layer = gc.Sphere(earth_r+np.max(z))
    rays = gc.Ray(o=origin, d=vecs)
    ds_toa = gc.calc_intersection(toa_layer, rays)
    sensors = []
    for isens in range (0, nsens):
        if (ds_toa['is_intersection'].values[isens]):
            is_sens[isens] = True
            phit = gc.Point(ds_toa['phit'].values[isens,:])
            th, ph = gc.vec2ang(gc.Vector(vecs.x[isens], vecs.y[isens], vecs.z[isens]))
            if (th == 0. or th == 180.): ph = 0.
            sen_tmp = Sensor(
                            pos_x=phit.x,
                            pos_y=phit.y,
                            pos_z=phit.z,
                            th_deg=th,
                            ph_deg=ph,
                            loc='ATMOS',
                            sensor_type=0,
                            )
            sensors.append(sen_tmp)
    if (overwrite      or 
        not ftoa_exist  ):
        sg = S1DB
        m_toa = sg.run(wavelength=wavelength, n_photons=nsens*nphotons,
                    n_loop=nphotons, atmosphere=pro, sensor=sensors,
                    output_layers=1,
                    le=le, surface=surface, xblock=64, xgrid=1024, beer=1, depo=dep, reflectance=False, earth_radius=earth_r,
                    stdev=True, progress=True, n_icdf=ntheta)
        if not isinstance(m_toa, xr.Dataset): m_toa = m_toa.to_xarray()
        m_toa.to_netcdf(str(ftoa_path))
        

    # open intermediate files and convert to iprt phase3 output format
    to_iprt_output_e6_v1('e6_v1', sza, saa, nx, ny, is_sens, vecs,
                         overwrite=overwrite, output_dir=output_dir)

def case_E6_v2(nphotons=1e8, overwrite=True, output_dir='./'):
    """
    In this version 1, we take one direction at each pixel center
    """
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    ftoa_name = f'iprt_phase3_e6_v2_toa.nc'
    ftoa_path = dir_output / ftoa_name
    ftoa_exist = ftoa_path.exists()


    sza = np.array([50., 90., 110., 130.])
    saa = np.array([0.])
    vza = np.arange(0., 1.2+0.04, 0.04)
    z = None
    nx = 61
    ny = 61
    nsens = nx*ny
    ntheta = 18001

    # atmosphere profil
    mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
    mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
    wavelength = np.array([450.])
    z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
    zs = len(z)
    sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    pro = Atm1D('afglt', grid=z, prof_ray=sca, prof_abs=abs).calc(wavelength)
    surface = RoughSurface(wind=5., brdf=True, wave_shadow=True, nh2o=1.33)
    
    nvza = len(vza)
    dep = 0.03
    earth_r = 6371.

    count_lvl = np.zeros_like(sza, dtype=np.int32)
    phi_0 = -saa # To follow iprt anti-clockwise convention
    le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                           count_level=count_lvl)

    # run simulations and create intermediate files
    # ==== Get directions
    zeros = np.zeros((nvza), dtype=np.float64)
    sen = gc.Point(zeros, zeros,np.full((nvza), 1, dtype=np.float64))
    dirs = -gc.ang2vec(theta=vza, phi=180.)
    rays = gc.Ray(o=sen, d=dirs)
    ground = gc.BBox(p1=gc.Point(-np.inf, -np.inf, 0.), p2=gc.Point(np.inf, np.inf, 0.))
    ds = gc.calc_intersection(ground, rays)

    coord = np.concatenate((-(ds['phit'][1:,0].values)[::-1], ds['phit'][:,0].values))
    pts = np.zeros((nsens,3), dtype=np.float64)
    ic = 0
    for ix in range (0, nx):
        for iy in range(0, ny):
            pts[ic,0] = coord[ix]
            pts[ic,1] = coord[::-1][iy] # to begin from top left pixel
            ic += 1
    points = gc.Point(pts)

    zeros_ini = np.zeros((nsens), dtype=np.float64)
    points_ini = gc.Point(zeros_ini, zeros_ini, np.full((nsens), 1, dtype=np.float64))

    vecs = gc.normalize(points - points_ini)
    # ==== create toa sensors
    sensors = []
    for isens in range (0, nsens):
        th, ph = gc.vec2ang(gc.Vector(vecs.x[isens], vecs.y[isens], vecs.z[isens]))
        if (th == 0. or th == 180.): ph = 0.
        sen_tmp = Sensor(
                        pos_x=0.,
                        pos_y=0.,
                        pos_z=3e5,
                        th_deg=th,
                        ph_deg=ph,
                        loc='ATMOS',
                        sensor_type=1,
                        fov=0.04,
                        cell_size=-2
                        )
        sensors.append(sen_tmp)
    if (overwrite      or 
        not ftoa_exist  ):
        sg = Smartg(back=True, double=True, bias=True, pp=False, obj3d=True)
        m_toa = sg.run(wavelength=wavelength, n_photons=nsens*nphotons,
                    n_loop=nphotons, atmosphere=pro, sensor=sensors,
                    output_layers=1,
                    le=le, surface=surface, xblock=64, xgrid=1024, beer=1, depo=dep, reflectance=False, earth_radius=earth_r,
                    stdev=True, progress=True, n_icdf=ntheta)
        if not isinstance(m_toa, xr.Dataset): m_toa = m_toa.to_xarray()
        m_toa.to_netcdf(str(ftoa_path))
        

    # open intermediate files and convert to iprt phase3 output format
    to_iprt_output_e6_v2('e6_v2', sza, saa, nx, ny, vecs,
                         overwrite=overwrite, output_dir=output_dir)

def case_E6_v3(nphotons=1e8, overwrite=True, output_dir='./'):
    """
    In this version 1, we take one direction at each pixel center
    """
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    ftoa_name = f'iprt_phase3_e6_v3_toa.nc'
    ftoa_path = dir_output / ftoa_name
    ftoa_exist = ftoa_path.exists()


    sza = np.array([50., 90., 110., 130.])
    saa = np.array([0.])
    vza = np.arange(0., 1.2+0.04, 0.04)
    z = None
    nx = 61
    ny = 61
    nsens = nx*ny
    ntheta = 18001

    # atmosphere profil
    mol_sca_filename  =  OPT_PROP_PATH_PHASE3 / "tau_rayleigh_450nm_usstd.dat"
    mol_abs_filename  =  OPT_PROP_PATH_PHASE3 / "tau_absorption_450nm_usstd.dat"
    wavelength = np.array([450.])
    z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
    zs = len(z)
    sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,zs)
    
    z = np.concatenate((np.array([3e5]), z))
    sca = np.concatenate((np.array([[0.]]), sca), axis=1)
    abs = np.concatenate((np.array([[0.]]), abs), axis=1)
    
    pro = Atm1D('afglt', grid=z, prof_ray=sca, prof_abs=abs).calc(wavelength)
    surface = RoughSurface(wind=5., brdf=True, wave_shadow=True, nh2o=1.33)
    
    nvza = len(vza)
    dep = 0.03
    earth_r = 6371.

    count_lvl = np.zeros_like(sza, dtype=np.int32)
    phi_0 = -saa # To follow iprt anti-clockwise convention
    le     = LocalEstimate(th_deg=sza, phi_deg=phi_0,
                           count_level=count_lvl)

    # run simulations and create intermediate files
    # ==== Get directions
    zeros = np.zeros((nvza), dtype=np.float64)
    sen = gc.Point(zeros, zeros,np.full((nvza), 1, dtype=np.float64))
    dirs = -gc.ang2vec(theta=vza, phi=180.)
    rays = gc.Ray(o=sen, d=dirs)
    ground = gc.BBox(p1=gc.Point(-np.inf, -np.inf, 0.), p2=gc.Point(np.inf, np.inf, 0.))
    ds = gc.calc_intersection(ground, rays)

    coord = np.concatenate((-(ds['phit'][1:,0].values)[::-1], ds['phit'][:,0].values))
    pts = np.zeros((nsens,3), dtype=np.float64)
    ic = 0
    for ix in range (0, nx):
        for iy in range(0, ny):
            pts[ic,0] = coord[ix]
            pts[ic,1] = coord[::-1][iy] # to begin from top left pixel
            ic += 1
    points = gc.Point(pts)

    zeros_ini = np.zeros((nsens), dtype=np.float64)
    points_ini = gc.Point(zeros_ini, zeros_ini, np.full((nsens), 1, dtype=np.float64))

    vecs = gc.normalize(points - points_ini)
    # ==== create toa sensors
    sensors = []
    for isens in range (0, nsens):
        th, ph = gc.vec2ang(gc.Vector(vecs.x[isens], vecs.y[isens], vecs.z[isens]))
        if (th == 0. or th == 180.): ph = 0.
        sen_tmp = Sensor(
                        pos_x=0.,
                        pos_y=0.,
                        pos_z=3e5,
                        th_deg=th,
                        ph_deg=ph,
                        loc='ATMOS',
                        sensor_type=1,
                        fov=0.04,
                        )
        sensors.append(sen_tmp)
    if (overwrite      or 
        not ftoa_exist  ):
        sg = S1DB
        m_toa = sg.run(wavelength=wavelength, n_photons=nsens*nphotons,
                    n_loop=nphotons, atmosphere=pro, sensor=sensors,
                    output_layers=1,
                    le=le, surface=surface, xblock=64, xgrid=1024, beer=1, depo=dep, reflectance=False, earth_radius=earth_r,
                    stdev=True, progress=True, n_icdf=ntheta)
        if not isinstance(m_toa, xr.Dataset): m_toa = m_toa.to_xarray()
        m_toa.to_netcdf(str(ftoa_path))
        

    # open intermediate files and convert to iprt phase3 output format
    # we can use v2 for v3
    to_iprt_output_e6_v2('e6_v3', sza, saa, nx, ny, vecs,
                         overwrite=overwrite, output_dir=output_dir)


if __name__ == '__main__':
    # D - Test cases for fully spherical geometry with one layer
    output_dir='./res_iprt_phase3_1e8photons_v5/'
    # case_D1(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_D2(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_D3(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_D4(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_D5(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_D6(nphotons=1e8, overwrite=False, output_dir=output_dir)


    # E - Test cases for fully spherical geometry for a vertically inhomogeneous atmosphere
    # case_E1(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E2(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E3(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E4(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E5(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E5_bis(nphotons=1e8, overwrite=False, output_dir=output_dir)
    # case_E6_old(nphotons=1e8, overwrite=False, output_dir=output_dir)
    case_E6_v1(nphotons=1e8, overwrite=False, output_dir=output_dir)
    case_E6_v2(nphotons=1e8, overwrite=False, output_dir=output_dir)
    case_E6_v3(nphotons=1e6, overwrite=False, output_dir=output_dir)