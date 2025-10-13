#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import os 

from smartg.smartg import Smartg, Sensor, LambSurface, RoughSurface
from smartg.atmosphere import AtmAFGL, AerOPAC
from smartg.albedo import Albedo_cst

import geoclide as gc
import pandas as pd
import xarray as xr
import math

from pathlib import Path
from smartg.config import DIR_AUXDATA

from smartg.tools.phase import calc_iphase

# may be to replace
from smartg.iprt import read_phase_nth_cte

from luts.luts import LUT, Idx

from tempfile import TemporaryDirectory
from pathlib import Path


S1DB = Smartg(back=True, double=True, bias=True, pp=False)

# 
OPT_PROP_PATH_PHASE3 = DIR_AUXDATA + "/IPRT/phase3/opt_prop/"

def get_d1_to_e5_boa_sensors(vza, phi, nvza, nvaa, earth_r):
    sensors = []
    for ivza in range (0, nvza):
        for ivaa in range (0, nvaa):
            vza_boa = vza[ivza]
            phi_boa = phi[ivaa]
            # Put 89.9999 instead of 90° to avoid problems due z=0 looking at horizon
            if vza_boa == 90.: vza_boa = 89.9999
            sen_tmp = Sensor(
                            POSX = 0.,
                            POSY = 0.,
                            POSZ = earth_r,
                            THDEG= vza_boa,
                            PHDEG= phi_boa,
                            LOC  = 'ATMOS',
                            TYPE = 0
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
                            POSX = phit.x,
                            POSY = phit.y,
                            POSZ = phit.z,
                            THDEG= vza_toa[ivza],#180-vza[ivza],
                            PHDEG= phi_toa[ivaa],
                            LOC  = 'ATMOS',
                            TYPE = 0
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
                            POSX = 0.,
                            POSY = 0.,
                            POSZ = earth_r+np.max(z),
                            THDEG= vza_toa[ivza],
                            PHDEG= phi_toa[ivaa],
                            LOC  = 'ATMOS',
                            TYPE = 0
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
        fboa_name = f'iprt_phase3_{case_name}_boa.nc'
        fboa_path = dir_output/ fboa_name
        ds_boa = xr.open_dataset(fboa_path)
        ftoa_name = f'iprt_phase3_{case_name}_toa.nc'
        ftoa_path = dir_output / ftoa_name
        ds_toa = xr.open_dataset(ftoa_path)

        ds = xr.Dataset(coords={f'{case_name}_sza':sza,
                                f'{case_name}_saa':saa,
                                f'{case_name}_vza':vza,
                                f'{case_name}_vaa':vaa,
                                f'{case_name}_zout':np.array([0.,np.max(z)]),
                                'stokes':np.arange(4, dtype=np.int32)})
        nsza = len(sza)
        nsaa = len(saa)
        nvza = len(vza)
        nvaa = len(vaa)
        nzout = 2
        nstokes = 4

        radiance = np.zeros((nzout,nsza,nsaa,nvza,nvaa,nstokes), dtype=np.float32)
        std = np.zeros_like(radiance)
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
        
        ds[f'radiance_{case_name}'] = xr.DataArray(radiance, dims=[f'{case_name}_zout', f'{case_name}_sza', f'{case_name}_saa',
                                                                   f'{case_name}_vza', f'{case_name}_vaa', 'stokes' ])
        ds[f'std_{case_name}'] = xr.DataArray(std, dims=[f'{case_name}_zout', f'{case_name}_sza', f'{case_name}_saa',
                                                         f'{case_name}_vza', f'{case_name}_vaa', 'stokes' ])
        ds.to_netcdf(f_path)

def run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
            sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z, ntheta=18001):
     # BOA
    if overwrite or not fboa_exist:
        sensors = get_d1_to_e5_boa_sensors(vza, phi, nvza, nvaa, earth_r)
        m_boa = S1DB.run(wl=wl, NBPHOTONS=nvza*nvaa*nphotons, NBLOOP=nphotons, atm=pro, sensor=sensors, OUTPUT_LAYERS=1,
                        le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=dep, reflectance=False,
                        stdev=True, progress=True, NF=ntheta)

        m_boa = m_boa.dropaxis('Azimuth angles')
        m_boa.add_axis('sza', sza)
        m_boa.add_axis('vaa', vaa)
        m_boa.add_axis('vza', vza)
        for name in m_boa.datasets():
            if 'Zenith angles' in m_boa[name].names and 'sensor index' in m_boa[name].names:
                mat_tmp = np.swapaxes(m_boa[name].data.reshape(len(vza), len(vaa), len(sza)), 0, 2)
                attrs_tmp = m_boa[name].attrs
                m_boa.rm_lut(name)
                m_boa.add_dataset(name, mat_tmp, ['sza', 'vaa', 'vza'], attrs=attrs_tmp)
            elif 'Zenith angles' in m_boa[name].names:
                mat_tmp = m_boa[name].data
                attrs_tmp = m_boa[name].attrs
                m_boa.rm_lut(name)
                m_boa.add_dataset(name, mat_tmp, ['sza'], attrs=attrs_tmp)
            elif 'sensor index' in m_boa[name].names:
                mat_tmp = np.swapaxes(m_boa[name].data.reshape(len(vza), len(vaa)), 0, 1)
                attrs_tmp = m_boa[name].attrs
                m_boa.rm_lut(name)
                m_boa.add_dataset(name, mat_tmp, ['vaa', 'vza'], attrs=attrs_tmp)

        m_boa = m_boa.dropaxis('Zenith angles', 'sensor index')
        m_boa.save(str(fboa_path), overwrite=overwrite)

    # TOA
    if overwrite or not ftoa_exist:
        sensors = get_d1_to_e5_toa_sensors(vza, phi, nvza, nvaa, earth_r, z)
        m_toa = S1DB.run(wl=wl, NBPHOTONS=nvza*nvaa*nphotons, NBLOOP=nphotons, atm=pro, sensor=sensors, OUTPUT_LAYERS=1,
                        le=le, surf=surf, XBLOCK = 64, XGRID = 1024, BEER=1, DEPO=dep, 
                        stdev=True, progress=True, NF=ntheta)

        m_toa = m_toa.dropaxis('Azimuth angles')
        m_toa.add_axis('sza', sza)
        m_toa.add_axis('vaa', vaa)
        m_toa.add_axis('vza', vza)
        for name in m_toa.datasets():
            if 'Zenith angles' in m_toa[name].names and 'sensor index' in m_toa[name].names:
                mat_tmp = np.swapaxes(m_toa[name].data.reshape(len(vza), len(vaa), len(sza)), 0, 2)
                attrs_tmp = m_toa[name].attrs
                m_toa.rm_lut(name)
                m_toa.add_dataset(name, mat_tmp, ['sza', 'vaa', 'vza'], attrs=attrs_tmp)
            elif 'Zenith angles' in m_toa[name].names:
                mat_tmp = m_toa[name].data
                attrs_tmp = m_toa[name].attrs
                m_toa.rm_lut(name)
                m_toa.add_dataset(name, mat_tmp, ['sza'], attrs=attrs_tmp)
            elif 'sensor index' in m_toa[name].names:
                mat_tmp = np.swapaxes(m_toa[name].data.reshape(len(vza), len(vaa)), 0, 1)
                attrs_tmp = m_toa[name].attrs
                m_toa.rm_lut(name)
                m_toa.add_dataset(name, mat_tmp, ['vaa', 'vza'], attrs=attrs_tmp)

        m_toa = m_toa.dropaxis('Zenith angles', 'sensor index')
        m_toa.save(str(ftoa_path), overwrite=overwrite)


def aer2smartg(filename, nb_theta=int(1801), rh_or_reff=None, rh_reff=None):
    """
    In progress
    """

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
    NBTHETA = nb_theta
    NBRH_OR_REFF  = rh_reff.size
    theta = np.linspace(0., 180., num=NBTHETA)
    NWAV    = max(ds["wavelen"].size, int(2))
    if NWAV > ds["wavelen"].size:
        wavelength = np.concatenate((ds["wavelen"].values*1e3, ds["wavelen"].values*1e3 + 0.1))
    else:
        wavelength = ds["wavelen"].values*1e3

    ext_out = np.zeros((NBRH_OR_REFF, NWAV), dtype=np.float64)
    ssa_out = np.zeros_like(ext_out)
    pha_out = np.zeros((NBRH_OR_REFF, NWAV, NBSTK, NBTHETA), dtype=np.float64)

    for iwav in range (0, ds["wavelen"].size):
        for irhreff in range(NBRH_OR_REFF):
            ext_out[irhreff,iwav] = ds["ext"][iwav,irhreff]
            ssa_out[irhreff,iwav] = ds["ssa"][iwav,irhreff]
            for istk in range (NBSTK):
                # ntheta (wl, reff, stk)
                nth = ds["ntheta"][iwav,irhreff,istk].data
                # theta (wl, reff, stk, ntheta)
                th = ds["theta"][iwav,irhreff,istk,:].data
                pha_out[irhreff,iwav,istk,:] = np.interp(theta, th[:nth], phase[iwav,irhreff,istk,:nth],  period=np.inf)
    
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
    ds_out.attrs = {'name': os.path.basename(filename),
                    'H_mix_min': '0.',
                    'H_mix_max': '2',
                    'H_free_min': '2',
                    'H_free_max': '12',
                    'H_stra_max': '12',
                    'Z_mix': '8',
                    'Z_free': '8',
                    'Z_stra': '99'}

    return ds_out


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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):
        mol_sca = np.array([0., 0.5])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        wl = 550.

        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wl)
        surf  = None
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):
        mol_sca = np.array([0., 0.1])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        wl = 550.

        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wl)
        surf  = LambSurface(ALB=Albedo_cst(0.3))
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

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
        NTH = 18001
        theta = np.linspace(0, 180, NTH)
        wl = np.array([350.])
        nwl = len(wl)
        file_aer_phase = OPT_PROP_PATH_PHASE3 + "waso.mie.cdf"
        aer_phase = read_phase_nth_cte(filename=file_aer_phase, nb_theta=NTH, normalize=True)
        nstk = aer_phase.shape[2]

        aer_pha = np.zeros((nwl, nz, nstk, NTH), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            aer_pha[:,iz,:,:] = aer_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        aer_phase = LUT(aer_pha, axes=[wl, z[1:], None, theta], names=['wav', 'z', 'stk', 'theta'])
        pha_atm, ipha_atm = calc_iphase(aer_phase, np.array([wl]), z)
        lpha_lut = []
        for i in range (0, pha_atm.shape[0]):
            lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, np.linspace(0, 180, NTH)], names=['stk', 'theta_atm'])) 

        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut)).calc(wl, phase=False)
        surf  = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa 
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

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
        NTH = 18001
        theta = np.linspace(0, 180, NTH)
        wl = np.array([350.])
        nwl = len(wl)

        file_aer_phase = OPT_PROP_PATH_PHASE3 + "sizedistr_spheroid.cdf"
        aer_phase = read_phase_nth_cte(filename=file_aer_phase, nb_theta=NTH, normalize=True)
        nstk = aer_phase.shape[2]

        aer_pha = np.zeros((nwl, nz, nstk, NTH), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            aer_pha[:,iz,:,:] = aer_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        aer_phase = LUT(aer_pha, axes=[wl, z[1:], None, theta], names=['wav', 'z', 'stk', 'theta'])

        pha_atm, ipha_atm = calc_iphase(aer_phase, np.array([wl]), z)
        lpha_lut = []
        for i in range (0, pha_atm.shape[0]):
            lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, np.linspace(0, 180, NTH)], names=['stk', 'theta_atm'])) 

        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut)).calc(wl, phase=False)
        surf  = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.])
        wl = np.array([350.])
        NTH = 1801

        ds_spheroid = aer2smartg(OPT_PROP_PATH_PHASE3 + "sizedistr_spheroid.cdf",
                                 nb_theta=NTH, rh_or_reff='hum', rh_reff=np.array([0.]))

        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'spheroid_d4.nc'
            ds_spheroid.to_netcdf(file_path)
            aer = AerOPAC(str(file_path), 0.2, 350., H_mix_min=0., H_mix_max=120.,
                          H_free_min=120., H_free_max=120., H_stra_min=120., H_stra_max=120., Z_mix=1e6,
                          rh_mix=0.)
        pro = AtmAFGL('afglt', comp=[aer], grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wl, phase=True, NBTHETA=NTH)
        surf  = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z, ntheta=NTH)

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
    z = np.array([120., 0.])

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
        NTH = 18001
        wl = np.array([800.])
        nwl = len(wl)
        theta = np.linspace(0, 180, NTH)
        file_cld_phase = OPT_PROP_PATH_PHASE3 + "watercloud.mie.cdf"
        cld_phase = read_phase_nth_cte(filename=file_cld_phase, nb_theta=NTH, normalize=True)
        nstk = cld_phase.shape[2]

        cld_pha = np.zeros((nwl, nz, nstk, NTH), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            cld_pha[:,iz,:,:] = cld_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        cld_phase = LUT(cld_pha, axes=[wl, z[1:], None, theta], names=['wav', 'z', 'stk', 'theta'])

        pha_atm, ipha_atm = calc_iphase(cld_phase, np.array([wl]), z)
        lpha_lut = []
        for i in range (0, pha_atm.shape[0]):
            lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, theta], names=['stk', 'theta_atm']))

        # atmosphere profil
        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs, prof_aer=prof_aer,
                    prof_phases=(ipha_atm, lpha_lut)).calc(wl, phase=False)
        surf  = None

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        mol_sca = np.array([0., 0.1])[None,:]
        mol_abs= np.array([0., 0.])[None,:]
        z = np.array([120., 0.]) 
        wl = 550.
        
        # atmosphere profil
        pro = AtmAFGL('afglt', grid=z, prof_ray=mol_sca, prof_abs=mol_abs).calc(wl)
        surf  = RoughSurface(WIND=2., BRDF=True, WAVE_SHADOW=True, NH2O=1.33)

        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('d6', sza, saa, vza, vaa, z,
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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 + "tau_rayleigh_450nm_usstd.dat"
        wl = 450.
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        ZS = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        pro = AtmAFGL('afglt', grid=z, prof_ray=sca, prof_abs=np.zeros_like(sca)).calc(wl)
        surf = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 + "tau_rayleigh_320nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 + "tau_absorption_320nm_usstd.dat"
        wl = 320.
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        ZS = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        pro = AtmAFGL('afglt', grid=z, prof_ray=sca, prof_abs=abs).calc(wl)
        surf = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 + "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 + "tau_absorption_450nm_usstd.dat"
        wl = np.array([450.])
        nwl= len(wl)
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        ZS = len(z)
        nz = len(z[1:])
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)

        # aerosol extinction and single scattering albedo
        aer_tau_ext = np.zeros_like(sca)
        aer_tau_ext[0,-3:] = 0.5 / 3. # constant AOT between 0 and 3km
        ds_desert = xr.open_dataset(OPT_PROP_PATH_PHASE3 + "desert.cdf")
        ssa_450 = ds_desert['ssa'].values[1,0]
        aer_ssa  = np.full_like(aer_tau_ext, ssa_450)
        prof_aer = (aer_tau_ext, aer_ssa)

        # aerosol phase matrix
        NTH = 18001 
        theta = np.linspace(0, 180, NTH)
        file_aer_phase = OPT_PROP_PATH_PHASE3 + "desert.cdf"
        aer_phase = read_phase_nth_cte(filename=file_aer_phase, nb_theta=NTH, normalize=True)
        nstk = aer_phase.shape[2]
        aer_pha = np.zeros((nwl, nz, nstk, NTH), dtype=np.float32)
        # Same phase for all altitude (here only one)
        for iz in range (0, nz):
            aer_pha[:,iz,:,:] = aer_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        aer_phase = LUT(aer_pha, axes=[wl, z[1:], None, theta], names=['wav', 'z', 'stk', 'theta'])
        pha_atm, ipha_atm = calc_iphase(aer_phase, wl, z)
        lpha_lut = []
        for i in range (0, pha_atm.shape[0]):
            lpha_lut.append(LUT(pha_atm[i,:,:], axes=[None, theta], names=['stk', 'theta_atm']))

        pro = AtmAFGL('afglt', grid=z, prof_ray=sca, prof_abs=abs,
                      prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut)).calc(wl, phase=False)
        surf = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

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
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 + "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 + "tau_absorption_450nm_usstd.dat"
        wl = np.array([450.])
        nwl= len(wl)
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        ZS = len(z)
        nz = len(z[1:])
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)

        # aerosol extinction and single scattering albedo
        aer_tau_ext = np.zeros_like(sca)
        aer_ssa  = np.full_like(aer_tau_ext, 1)
        # desert aer between 0 and 3km
        aer_tau_ext[0, np.logical_and(z>=0,z<3)] = 0.5 / 3. # constant AOT between 0 and 3km
        ds_desert = xr.open_dataset(OPT_PROP_PATH_PHASE3 + "desert.cdf")
        aer_ssa[0, np.logical_and(z>=0,z<3)] = ds_desert['ssa'].values[1,0] # ssa at 450nm
        # sulfate aer between 20 and 21km
        aer_tau_ext[0, np.logical_and(z>=20,z<21)] = 0.05
        ds_sulfate = xr.open_dataset(OPT_PROP_PATH_PHASE3 + "sulfate.cdf")
        aer_ssa[0, np.logical_and(z>=20,z<21)] = ds_sulfate['ssa'].values[1,0] # ssa at 450nm
        prof_aer = (aer_tau_ext, aer_ssa)

        # aerosol phase matrix
        NTH = 18001 
        theta = np.linspace(0, 180, NTH)
        file_aer1_phase = OPT_PROP_PATH_PHASE3 + "desert.cdf"
        file_aer2_phase = OPT_PROP_PATH_PHASE3 + "sulfate.cdf"
        aer1_phase = read_phase_nth_cte(filename=file_aer1_phase, nb_theta=NTH, normalize=True)
        aer2_phase = read_phase_nth_cte(filename=file_aer2_phase, nb_theta=NTH, normalize=True)
        nstk = 6
        n_species = 2
        aer_pha = np.zeros((nwl, n_species, nstk, NTH), dtype=np.float32) # 2 different aer
        ipha_atm = np.zeros((1,nz+1), dtype=np.int32)
        lpha_lut = []
        # desert aer between 0 and 3km
        ipha_atm[0,np.argwhere(np.logical_and(z>=0,z<3))] = 0 # put pha index from z=0 to 3km.
        aer_pha[:,0,:,:] = aer1_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        lpha_lut.append(LUT(aer_pha[0,0,:,:], axes=[None, theta], names=['stk', 'theta_atm']))
        # sulfate aer between 20 and 21km
        ipha_atm[0,np.argwhere(np.logical_and(z>=20,z<21))] = 1 # put pha index from z=0 to 3km.
        aer_pha[:,1,:,:] = aer2_phase.sub()[:,0,:,:].sub({'wav_phase':Idx(wl)}).sub({'theta_atm':Idx(theta)}).data
        lpha_lut.append(LUT(aer_pha[0,1,:,:], axes=[None, theta], names=['stk', 'theta_atm']))

        pro = AtmAFGL('afglt', grid=z, prof_ray=sca, prof_abs=abs,
                        prof_aer=prof_aer, prof_phases=(ipha_atm, lpha_lut)).calc(wl, phase=False)
        surf = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e4', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


def case_E4_bis(nphotons=1e8, overwrite=True, output_dir='./'):
    
    dir_output = Path(output_dir)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    fboa_name = f'iprt_phase3_e4_bis_boa.nc'
    ftoa_name = f'iprt_phase3_e4_bis_toa.nc'
    fboa_path = dir_output / fboa_name
    ftoa_path = dir_output / ftoa_name
    fboa_exist = fboa_path.exists()
    ftoa_exist = ftoa_path.exists()


    sza = np.array([30., 60., 80., 87., 90., 93., 96., 99.])
    saa = np.array([0.])
    vza = np.array([0., 9., 18., 26., 34., 41., 48., 54., 60., 65., 70.,
                        74., 78., 81., 84., 86., 88., 89., 90.])
    vaa = np.linspace(0., 180., 19)
    z = np.array([120., 0.])

    if (overwrite      or 
        not fboa_exist or 
        not ftoa_exist  ):

        # atmosphere profil
        mol_sca_filename  =  OPT_PROP_PATH_PHASE3 + "tau_rayleigh_450nm_usstd.dat"
        mol_abs_filename  =  OPT_PROP_PATH_PHASE3 + "tau_absorption_450nm_usstd.dat"
        wl = np.array([450.])
        z = np.squeeze(pd.read_csv(mol_sca_filename, header=None, usecols=[0], dtype=float, skiprows=1, sep=r'\s+', comment='#').values)
        ZS = len(z)
        sca = pd.read_csv(mol_sca_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        abs = pd.read_csv(mol_abs_filename, header=None, usecols=[1], dtype=float, skiprows=1, sep=r'\s+', comment='#').values.reshape(1,ZS)
        NTH = 18001

        file_aer1_phase = OPT_PROP_PATH_PHASE3 + "desert.cdf"
        file_aer2_phase = OPT_PROP_PATH_PHASE3 + "sulfate.cdf"
        ds_desert = aer2smartg(file_aer1_phase, nb_theta=NTH, rh_or_reff='hum', rh_reff=np.array([0.]))
        ds_sulfate = aer2smartg(file_aer2_phase, nb_theta=NTH, rh_or_reff='hum', rh_reff=np.array([0.]))
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'desert_e4.nc'
            ds_desert.to_netcdf(file_path)
            aer1 = AerOPAC(str(file_path), 0.5, wl[0], H_mix_min=0., H_mix_max=3.,
                           H_free_min=2., H_free_max=2., H_stra_min=12., H_stra_max=12., Z_mix=1e6,
                           rh_mix=0.)
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir)/'sulfate_e4.nc'
            ds_sulfate.to_netcdf(file_path)
            aer2 = AerOPAC(str(file_path), 0.05, wl[0], H_mix_min=20., H_mix_max=21.,
                           H_free_min=2., H_free_max=2., H_stra_min=12., H_stra_max=12., Z_mix=1e6,
                           rh_mix=0.)

        pro = AtmAFGL('afglt', comp=[aer1, aer2], grid=z, prof_ray=sca, 
                      prof_abs=abs, pfgrid=[120., 21., 20., 3., 0.]).calc(wl, phase=True, NBTHETA=NTH)
        surf = None
        
        nvza = len(vza)
        nvaa = len(vaa)
        phi = -vaa
        dep = 0.03
        earth_r = 6371.

        count_lvl = np.zeros_like(sza, dtype=np.int32)
        phi_0 = -saa # To follow iprt anti-clockwise convention
        le     = {'th_deg':sza, 'phi_deg':phi_0, 'count_level':count_lvl}

        # run simulations and create intermediate files
        run_sim(overwrite, fboa_exist, ftoa_exist, fboa_path, ftoa_path,
                sza, vza, vaa, phi, nvza, nvaa, earth_r, nphotons, wl, le, surf, pro, dep, z)

    # open intermediate files and convert to iprt phase3 output format 
    to_iprt_output('e4_bis', sza, saa, vza, vaa, z,
                   overwrite=overwrite, output_dir=output_dir)


if __name__ == '__main__':
    # D - Test cases for fully spherical geometry with one layer
    # case_D1(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_D2(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_D3(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_D4(nphotons=1e8, overwrite=False, output_dir='./res_iprt_phase3_1e8photons/')
    case_D4_bis(nphotons=1e8, overwrite=True, output_dir='./res_iprt_phase3_1e8photons_new/')
    # case_D5(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_D6(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')

    # E - Test cases for fully spherical geometry for a vertically inhomogeneous atmosphere
    # case_E1(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_E2(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    # case_E3(nphotons=1e7, overwrite=False, output_dir='./res_iprt_phase3_1e7photons/')
    #case_E4(nphotons=1e8, overwrite=False, output_dir='./res_iprt_phase3_1e8photons/')
    case_E4_bis(nphotons=1e8, overwrite=False, output_dir='./res_iprt_phase3_1e8photons_new/')
