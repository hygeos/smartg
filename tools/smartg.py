#!/usr/bin/env python
# encoding: utf-8



import subprocess
import numpy as np
from pyhdf.SD import SD, SDC
from profile.profil import AeroOPAC, Profile, REPTRAN, REPTRAN_IBAND
from water.iop_spm import IOP_SPM
from water.iop_mm import IOP_MM
from os.path import dirname, realpath, join, exists, basename, isdir
from os import makedirs, remove
import textwrap
import tempfile
from luts import merge, read_lut_hdf, read_mlut_hdf


#
# set up default directories
#
dir_install = dirname(dirname(realpath(__file__)))    # base smartg directory is one directory above here
dir_tmp = join(dir_install, 'tmp/')
dir_list_pf_aer = join(dir_tmp, 'list_pf_aer/')
dir_list_pf_oce = join(dir_tmp, 'list_pf_oce/')
dir_phase_water = join(dir_tmp, 'phase_water/')
dir_phase_aero = join(dir_tmp, 'phase_aerosols/')
dir_albedo = join(dir_tmp, 'albedo/')
dir_cmdfiles = join(dir_tmp, 'command_files/')
dir_profil_aer = join(dir_tmp, 'profile_aer/')
dir_profil_oce = join(dir_tmp, 'profile_oce/')
dir_output = join(dir_tmp, 'results')



class Smartg(object):
    '''
    Run a SMART-G job

    Arguments:
        - exe: smart-g executable
        - wl: wavelength in nm (float)
              or a list of wavelengths, or an array
              used for phase functions calculation (always)
              and profile calculation (if iband is None)   # FIXME
        - iband: a REPTRAN_BAND object describing the internal band
            default None (no reptran mode)
        - output: the name of the file to create. If None (default),
          automatically choose a filename in directory dir
        - dir: directory for automatic filename if output=None
        - overwrite: if True, remove output file first if it exists
        - skip_existing: skip existing file if overwrite is False
                         otherwise (False, default), raise an Exception
        - cmdfile: the name of the command file to use. If None (default),
          automatically choose a filename.
        - atm: Profile object
            default None (no atmosphere)
        - surf: Surface object
            default None (no surface)
        - water: Iop object, providing options relative to the ocean surface
            default None (no ocean)
        - env: environment effect parameters (dictionary)
            default None (no environment effect)
        The other acuments (NBPHOTONS, THVDEG, etc) are passed directly to the
        command file.

    Attributes:
        - output: the name of the result file
    '''
    def __init__(self, exe, wl, output=None, dir=dir_output,
           overwrite=False, skip_existing=False,
           cmdfile=None, iband=None,
           atm=None, surf=None, water=None, env=None,
           NBPHOTONS=1e8, DEPO=0.0279, THVDEG=0., SEED=-1,
           NBTHETA=45, NBPHI=45,
           NFAER=1000000, NFOCE=10000000, WRITE_PERIOD=-1,
           OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
           NBLOOP=5000):
        #
        # initialization
        #

        if cmdfile is None:
            cmdfile = tempfile.mktemp(suffix='.txt',
                    prefix='smartg_command_',
                    dir=dir_cmdfiles)

        assert isinstance(wl, (float, list, np.ndarray))
        assert (iband is None) or isinstance(iband, REPTRAN_IBAND)

        if output is None:
            #
            # default file name
            #
            assert dir is not None
            list_filename = [exe]   # executable

            if iband is None:
                pass
            else:
                list_filename.append('WL={:.2f}'.format(iband.w))

            list_filename.append('THV={:.1f}'.format(THVDEG))

            if atm is not None: list_filename.append(str(atm))
            if surf is not None: list_filename.append(str(surf))
            if water is not None: list_filename.append(str(water))
            if env is not None: list_filename.append(str(env))

            filename = '_'.join(list_filename)
            filename = join(dir, filename + '.hdf')
            output = filename

        self.output = output

        if exists(output):
            if overwrite:
                remove(output)
            else:
                if skip_existing:
                    return
                else:
                    raise Exception('File {} exists'.format(output))

        ensure_dir_exists(output)
        ensure_dir_exists(cmdfile)

        #
        # make dictionary of parameters
        #
        D = {
                'NBPHOTONS': str(int(NBPHOTONS)),
                'LAMBDA': wl,
                'THVDEG': THVDEG,
                'DEPO': DEPO,
                'SEED': SEED,
                'NBTHETA': NBTHETA,
                'NBPHI': NBPHI,
                'NFAER': NFAER,
                'NFOCE': NFOCE,
                'WRITE_PERIOD': WRITE_PERIOD,
                'OUTPUT_LAYERS': OUTPUT_LAYERS,
                'XBLOCK': XBLOCK,
                'YBLOCK': 1,
                'XGRID': XGRID,
                'YGRID': 1,
                'NBLOOP': NBLOOP,
                }

        # we use a separate disctionary to store the default parameters
        # which should not override the specified ones
        Ddef = {}

        # determine SIM
        if (atm is not None) and (surf is None) and (water is None):
            SIM = -2  # atmosphere only
        elif (atm is None) and (surf is not None) and (water is None):
            SIM = -1  # surface only
        elif (atm is None) and (surf is not None) and (water is not None):
            SIM = 0  # ocean + dioptre
        elif (atm is not None) and (surf is not None) and (water is None):
            SIM = 1  # atmosphere + dioptre
        elif (atm is not None) and (surf is not None) and (water is not None):
            SIM = 2  # atmosphere + dioptre + ocean
        elif (atm is None) and (surf is None) and (water is not None):
            SIM = 3  # ocean only
        else:
            raise Exception('Error in SIM')

        D.update(SIM=SIM)

        # output file
        D.update(PATHRESULTATSHDF=output)

        #
        # atmosphere
        #
        if atm is not None:
            # write the profile
            if iband is None:
                ensure_dir_exists(dir_profil_aer)
                ensure_dir_exists(dir_phase_aero)
                ensure_dir_exists(dir_list_pf_aer)
                file_profile, file_list_pf_aer = atm.write(wl,
                            dir_profile=dir_profil_aer,
                            dir_phases=dir_phase_aero,
                            dir_list_phases=dir_list_pf_aer,
                            )
                D.update(PATHDIFFAER=file_list_pf_aer)
            else:
                file_profile = atm.write(iband, dir=dir_profil_aer)
            D.update(PATHPROFILATM=file_profile)

        else:  # no atmosphere
            Ddef.update(PATHDIFFAER='None')
            Ddef.update(PATHPROFILATM='None')

        #
        # surface
        #
        if surf is None:
            # default surface parameters
            surf = FlatSurface()
            Ddef.update(surf.dict)
        else:
            D.update(surf.dict)


        #
        # ocean profile
        #
        if water is None:
            # use default water values
            Ddef.update(PATHPROFILOCE='None', PATHDIFFOCE='None')
        else:
            ensure_dir_exists(dir_list_pf_oce)
            ensure_dir_exists(dir_profil_oce)
            ensure_dir_exists(dir_phase_water)
            profil_oce, file_list_pf_ocean = water.write(wl, dir_profile=dir_profil_oce,
                    dir_phases=dir_phase_water, dir_list_phases=dir_list_pf_oce)

            D.update(PATHPROFILOCE=profil_oce, PATHDIFFOCE=file_list_pf_ocean)

        #
        # environment effect
        #
        if env is None:
            # default values (no environment effect)
            env = Environment()
            Ddef.update(env.dict)
        else:
            D.update(env.dict)

        #
        # update the dictionary with the default parameters
        #
        for k, v in Ddef.items():
            if not k in D:
                D.update({k: v})

        #
        # write the albedo file
        #
        file_alb = tempfile.mktemp(dir=dir_albedo, prefix='albedo_')
        ensure_dir_exists(dir_albedo)
        if 'SURFALB' in D:
            surf_alb = D['SURFALB']
        else:
            surf_alb = -999.
        if water is None:
            seafloor_alb = -999.
        else:
            seafloor_alb = water.alb

        with open(file_alb, 'w') as f:
            f.write('# Surface_alb Seafloor_alb\n')
            f.write('{} {}\n'.format(surf_alb, seafloor_alb))

        D.update(PATHALB=file_alb)

        #
        # write the command file
        #
        fo = open(cmdfile, "w")
        fo.write(command_file_template(D))
        fo.close()

        #
        # run smart-g
        #
        if dirname(exe) == '':
            exe = join(dir_install, exe)
        assert exists(exe)
        cmd = '{} {}'.format(exe, cmdfile)
        ret = subprocess.call(cmd, shell=True)
        if ret:
            raise Exception('Error in SMART-G')


    def read(self, dataset=None):
        '''
        read SMARTG result as a LUT (if dataset is provided) or MLUT (default)
        '''
        if dataset is not None:
            return read_lut_hdf(self.output, dataset)
        else:
            return read_mlut_hdf(self.output)


    def view(self, QU=False, field='up (TOA)'):
        '''
        visualization of a smartg result

        Options:
            QU: show Q and U also (default, False)
        '''
        from smartg_view import smartg_view

        smartg_view(self.read(), QU=QU, field=field)


def ensure_dir_exists(file_or_dir):
    if isdir(file_or_dir):
        dir_name = file_or_dir
    else:
        dir_name = dirname(file_or_dir)
    if not exists(dir_name):
        makedirs(dir_name)


def command_file_template(dict):
    '''
    returns the content of the command file based on dict
    '''
    return textwrap.dedent("""
        ################ SIMULATION #####################
        # Number of "Photons" to inject (unsigned long long)
        NBPHOTONS = {NBPHOTONS}

        # View Zenith Angle in degree (float)
        THVDEG = {THVDEG}

        # Eventually wavelenghth (nm) (just for information purposes, not used in the simulation) 
        LAMBDA = {LAMBDA}

        # Number of output azimut angle boxes from 0 to PI
        NBPHI = {NBPHI}
        # Number of output zenith angle boxes from 0 to PI
        NBTHETA = {NBTHETA}

        # Simulation type
            # -2 Atmosphere only
            # -1 Dioptre only
            #  0 Ocean and dioptre
            #  1 Atmosphere and dioptre
            #  2 Atmosphere, dioptre and ocean
            #  3 Ocean only
        SIM = {SIM}

        # Dioptre type
            # 0 = plan
            # 1 = roughened sea surface with multiple reflections
            # 2 = roughened sea surface without multiple reflections
            # 3 = lambertian reflector (LAND)
        DIOPTRE = {DIOPTRE}

        # Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        SUR = {SUR}

        # Output layers as a binary flag 
            # 0  TOA always present
            # 1  Add BOA (0+) downward and BOA (0-) upward
            # 2  Add BOA (0+) upward and BOA (0-) downward
        OUTPUT_LAYERS = {OUTPUT_LAYERS}

        # Absolute name of output file 
        PATHRESULTATSHDF = {PATHRESULTATSHDF}

        # SEED for random number series
            # SEED > 0 Random series generated from this SEED (allow to redo the same simulation)
            # SEED =-1 A SEED is randomly generated
        SEED = {SEED}

        ################ ATMOSPHERE #####################
        # Depolarization coefficient of air
        DEPO = {DEPO}

        # Absolute name of file containing the vertical profile of atmosphere
            # Format 
        PATHPROFILATM = {PATHPROFILATM}

        # Absolute name of file containing the atmospheric phase matrix 
            # Format 
        PATHDIFFAER = {PATHDIFFAER}

        ################ SURFACE #####################
        # Absolute name of file containing the Land and Seafloor lambertian albedo 
            # Format 
        PATHALB = {PATHALB}

        # Windspeed (m/s) (if DIOPTRE = 1,2 or 4)
        WINDSPEED = {WINDSPEED}

        # Relatibve refarctive index air/water
        NH2O = {NH2O}

        #_______________ Environement effects _____________________#

        # Environment effects (circular target surrrounded by environment)
          # 0  No effect (target horizontally homogeneous)
          # 1  Effect included
        ENV = {ENV}
        # Target radius (km)
        ENV_SIZE= {ENV_SIZE}
        # X0 horizontal shift (in km) in X direction between the center of the target and the point on Earth viewed
        X0= {X0}
        # Y0 horizontal shift (in km) in Y direction between the center of the target and the point on Earth viewed
        Y0= {Y0}

        ################ OCEAN   #####################
        # Absolute name of file containing the Vertical profile of Ocean optical depth and single scattering albedo
            # Format 
        PATHPROFILOCE = {PATHPROFILOCE}
        # Absolute name of file containing the Ocean phase function 
            # Format 
        PATHDIFFOCE = {PATHDIFFOCE}

        ################ PARAMETERS   #####################

        # number of samples for the computation of the Cumulative Distribution Function of the aerosol phase matrix
        NFAER = {NFAER}

        # number of samples for the computation of the Cumulative Distribution Function of the ocean phase matrix
        NFOCE = {NFOCE}

        # LOOP number in the kernel for each thread
        NBLOOP = {NBLOOP} 

        #_______________ GPU _____________________#

        # Threads organized as BLOCKS of size XBLOCK*YBLOCK, with XBLOCK a multiple of 32
        XBLOCK = {XBLOCK} 
        YBLOCK = {YBLOCK} 

        # et les blocks sont eux-même rangés dans un grid de taille XGRID*YGRID (limite par watchdog il faut tuer X + limite XGRID<65535 et YGRID<65535)
        # BLOCKS organized as GRID of size XGRID*YGRID with XGRID<65535 and YGRID<65535
        XGRID = {XGRID} 
        YGRID = {YGRID} 

        # Device selection (-1 to select the 1st available device)
        DEVICE  -1
    """).format(**dict)


class FlatSurface(object):
    '''
    Definition of a flat sea surface

    Arguments:
        SUR: Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        NH2O: Relative refarctive index air/water
    '''
    def __init__(self, SUR=3, NH2O=1.33):
        self.dict = {
                'SUR': SUR,
                'DIOPTRE': 0,
                'WINDSPEED': -999.,
                'NH2O': NH2O,
                }
    def __str__(self):
        return 'FLATSURF-SUR={SUR}'.format(**self.dict)

class RoughSurface(object):
    '''
    Definition of a roughened sea surface

    Arguments:
        MULT: include multiple reflections at the surface
              (True => DIOPTRE=1 ; False => DIOPTRE=2)
        WIND: wind speed (m/s)
        SUR: Processes at the surface dioptre
            # 1 Forced reflection
            # 2 Forced transmission
            # 3 Reflection and transmission
        NH2O: Relative refarctive index air/water
    '''
    def __init__(self, MULT=False, WIND=5., SUR=3, NH2O=1.33):
        self.dict = {
                'SUR': SUR,
                'DIOPTRE': {True:1, False:2}[MULT],
                'WINDSPEED': WIND,
                'NH2O': NH2O,
                }
    def __str__(self):
        return 'ROUGHSUR={SUR}-WIND={WINDSPEED}-DI={DIOPTRE}'.format(**self.dict)


class LambSurface(object):
    '''
    Definition of a lambertian reflector

    ALB: Albedo of the reflector
    '''
    def __init__(self, ALB=0.5):
        self.dict = {
                'SUR': 1,
                'DIOPTRE': 3,
                'SURFALB': ALB,
                'WINDSPEED': -999.,
                'NH2O': -999.,
                }
    def __str__(self):
        return 'LAMBSUR-ALB={SURFALB}'.format(**self.dict)

class Environment(object):
    '''
    Stores the smartg parameters relative the the environment effect
    '''
    def __init__(self, ENV=0, ENV_SIZE=0., X0=0., Y0=0., ALB=0.5):
        self.dict = {
                'ENV': ENV,
                'ENV_SIZE': ENV_SIZE,
                'X0': X0,
                'Y0': Y0,
                'SURFALB': ALB,
                }

    def __str__(self):
        return 'ENV={ENV_SIZE}-X={X0:.1f}-Y={Y0:.1f}'.format(**self.dict)


def reptran_merge(files, ibands, output=None):
    '''
    merge (average) results from several correlated-k bands

    Arguments:
        * files: a list of smartg files to merge
        * ibands: a list of corresponding REPTRAN_IBANDs
        * output: the hdf file to create
            if None (default), the output file is determined by extracting the
            common prefix and suffix of all input files, and insert the band
            name inbetween

    Returns the output file name
    '''

    if output is None:
        # determine the common prefix and suffix of all files
        # and insert the band name between those
        base = basename(files[0])
        i = base.find('_WL')
        i += 3
        j = base.find('_', i)
        output = join(dirname(files[0]),
                    (base[:i]               # prefix
                    + ibands[0].band.name  # iband name
                    + base[j:]))            # suffix

    print 'Merging {} files into {}'.format(len(files), output)

    hdf_out = SD(output, SDC.WRITE|SDC.CREATE)
    hdf_ref = SD(files[0])
    for dataset in hdf_ref.datasets():

        sdsref = hdf_ref.select(dataset)
        rank = sdsref.info()[1]
        shape = sdsref.info()[2]
        dtype  = sdsref.info()[3]
        if rank < 2:
            # axis: write the axis as-is
            S = sdsref.get()
        else:
            # average all files
            S, norm = 0., 0.
            for i in xrange(len(files)):

                file = files[i]
                iband = ibands[i]

                hdf = SD(file)
                data = hdf.select(dataset).get()
                hdf.end()

                S += data * iband.weight * iband.extra
                norm += iband.weight * iband.extra

            S /= norm
            S = S.astype(data.dtype)

        # write the dataset
        sds = hdf_out.create(dataset, dtype, shape)
        sds.setcompress(SDC.COMP_DEFLATE, 9)
        sds[:] = S[:]
        # copy sds attributes from first file
        for a in sdsref.attributes().keys():
            setattr(sds, a, sdsref.attributes()[a])
        sds.endaccess()

    # copy global attributes from first file
    for a in hdf_ref.attributes():
        setattr(hdf_out, a, hdf_ref.attributes()[a])

    hdf_out.end()

    return output


def test_rayleigh():
    '''
    Basic Rayleigh example
    '''
    return Smartg('SMART-G-PP', wl=500., NBPHOTONS=1e9, atm=Profile('afglt'), overwrite=True)


def test_kokhanovsky():
    '''
    Just Rayleigh : kokhanovsky test case
    '''
    return Smartg('SMART-G-PP', wl=500., DEPO=0., NBPHOTONS=1e9,
            atm=Profile('afglt', grid='100[75]25[5]10[1]0'),
            output=join(dir_output, 'example_kokhanovsky.hdf'))


def test_rayleigh_aerosols():
    '''
    with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550.)
    pro = Profile('afglms', aer=aer)

    return Smartg('SMART-G-PP', wl=490., atm=pro, NBPHOTONS=1e9)


def test_atm_surf():
    # lambertian surface of albedo 10%
    return Smartg('SMART-G-PP', 490., NBPHOTONS=1e9,
            output=join(dir_output, 'test_atm_surf.hdf'),
            atm=Profile('afglms'),
            surf=LambSurface(ALB=0.1),
            overwrite=True)


def test_atm_surf_ocean():
    return Smartg('SMART-G-PP', 490., NBPHOTONS=1e7,
            atm=Profile('afglms', aer=AeroOPAC('maritime_clean', 0.2, 550)),
            surf=RoughSurface(),
            NBTHETA=30,
            water=IOP_MM(chl=1., NANG=1000),
            overwrite=True)


def test_surf_ocean():
    return Smartg('SMART-G-PP', 490., THVDEG=30., NBPHOTONS=2e6,
            surf=RoughSurface(),
            water=IOP_MM(1., pfwav=[400.]))


def test_ocean():
    return Smartg('SMART-G-PP', 560., THVDEG=30.,
            water=IOP_MM(1.), NBPHOTONS=5e6)


def test_reptran():
    '''
    using reptran
    '''
    aer = AeroOPAC('maritime_polluted', 0.4, 550.)
    pro = Profile('afglms.dat', aer=aer, grid='100[75]25[5]10[1]0')
    files, ibands = [], []
    for iband in REPTRAN('reptran_solar_msg').band('msg1_seviri_ch008').ibands():
        job = Smartg('SMART-G-PP', wl=np.mean(iband.band.awvl),
                NBPHOTONS=5e8,
                iband=iband, atm=pro)
        files.append(job.output)
        ibands.append(iband)

    reptran_merge(files, ibands)


def test_ozone_lut():
    '''
    Ozone Gaseous transmission for MERIS
    '''
    from itertools import product

    list_TCO = [350., 400., 450.]   # ozone column in DU
    list_AOT = [0.05, 0.1, 0.4]     # aerosol optical thickness

    luts = []
    for TCO, AOT in product(list_TCO, list_AOT):

        aer = AeroOPAC('maritime_clean', AOT, 550.)
        pro = Profile('afglms', aer=aer, O3=TCO)

        job = Smartg('SMART-G-PP', wl=490., atm=pro, NBTHETA=50, NBPHOTONS=5e6)

        lut = job.read('I_up (TOA)')
        lut.attrs.update({'TCO':TCO, 'AOT': AOT})
        luts.append(lut)

    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()
    merged.save(join(dir_output, 'test_ozone.hdf'))


def test_multispectral():
    '''
    process multiple bands at once
    '''

    pro = Profile('afglt',
            grid=[100, 75, 50, 30, 20, 10, 5, 1, 0.],  # optional, otherwise use default grid
            pfgrid=[100, 20, 0.],   # optional, otherwise use a single band 100-0
            pfwav=[400, 500, 600], # optional, otherwise phase functions are calculated at all bands
            aer=AeroOPAC('maritime_clean', 0.3, 550.))

    Smartg('SMART-G-PP', wl = np.linspace(400, 600, 10.),
            atm=pro,
            surf=RoughSurface(),
            water=IOP_SPM(1., pfwav=[500.]),
            overwrite=True)


if __name__ == '__main__':
    test_rayleigh()
    test_kokhanovsky()
    test_rayleigh_aerosols()
    test_atm_surf()
    test_atm_surf_ocean()
    test_surf_ocean()
    test_ocean()
    # test_reptran()
    test_ozone_lut()
    test_multispectral()
