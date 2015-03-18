#!/usr/bin/env python
# encoding: utf-8



import subprocess
import numpy as np
from pyhdf.SD import SD, SDC
from profile.profil import AeroOPAC, Profile, REPTRAN, REPTRAN_IBAND
from water.water_spm_model import WaterModelSPM
from os.path import dirname, realpath, join, exists, basename
from os import makedirs, remove
import textwrap
import tempfile
from luts import read_lut_hdf, merge


#
# set up default directories
#
dir_install = dirname(dirname(realpath(__file__)))    # base smartg directory is one directory above here
dir_data = join(dir_install, 'data/')   # directory for storing profiles, phase functions, etc.
dir_phase_water = join(dir_data, 'phase_water/')
dir_phase_aero = join(dir_data, 'phase_aerosols/')
dir_tmp = join(dir_install, 'tmp/')
dir_cmdfiles = join(dir_tmp, 'command_files/')
dir_profiles = join(dir_tmp, 'profiles/')
dir_output = dir_tmp



def smartg(exe, wl, output=None, dir=dir_output,
           overwrite=False,
           cmdfile=None, iband=None,
           atm=None, surf=None, water=None, env=None,
           NBPHOTONS=1e8, DEPO=0.0279, THVDEG=0., SEED=-1,
           NBTHETA=45, NBPHI=45,
           NFAER=1000000, NFOCE=10000000, WRITE_PERIOD=-1,
           OUTPUT_LAYERS=0, XBLOCK=256, XGRID=256,
           NBLOOP=5000):
    '''
    Run a SMART-G job, returns the result file

    Arguments:
        - exe: smart-g executable
        - wl: wavelength in nm (float)
             used for phase functions calculation (always)
             and profile calculation (if iband is None)
        - iband: a REPTRAN_BAND object describing the internal band
            default None (no reptran mode)
        - output: the name of the file to create. If None (default),
          automatically choose a filename in directory dir
        - dir: directory for automatic filename if output=None
        - overwrite: if True, remove output file first if it exists
                     if False, raise an exception with attribute filename
                     example:  try:
                                   out = smartg(...)
                               except Exception, e:
                                   out = e.filename
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
    '''
    #
    # initialization
    #

    if cmdfile is None:
        cmdfile = tempfile.mktemp(suffix='.txt',
                prefix='smartg_command_',
                dir=dir_cmdfiles)

    assert isinstance(wl, float)
    assert (iband is None) or isinstance(iband, REPTRAN_IBAND)

    if output is None:
        #
        # default file name
        #
        assert dir is not None
        list_filename = [exe]   # executable

        if iband is None:
            list_filename.append('WL={:.2f}'.format(wl))
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

    if exists(output):
        if overwrite:
            remove(output)
        else:
            ex = Exception('File {} exists'.format(output))
            setattr(ex, 'filename', output)
            raise ex

    if not exists(dirname(output)):
        makedirs(dirname(output))
    if not exists(dirname(cmdfile)):
        makedirs(dirname(cmdfile))

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
            file_profile = atm.write(wl, dir=dir_profiles)
        else:
            file_profile = atm.write(iband, dir=dir_profiles)
        D.update(PATHPROFILATM=file_profile)

        # aerosols
        if atm.aer is None:
            D.update(PATHDIFFAER='None')
        else:
            D.update(PATHDIFFAER=atm.aer.phase(wl, dir_phase_aero))
    else:  # no atmosphere
        D.update(PATHDIFFAER='None')
        D.update(PATHPROFILATM='None')

    #
    # surface
    #
    if surf is None:
        # default surface parameters
        surf = Surface()
    D.update(surf.dict)

    #
    # ocean parameters
    #
    if water is not None:
        # TODO: if iband is provided, use iband wavelength to calculate
        # atot and btot, and wl to calculate the phase function
        atot, btot, file_phase = water.calc(wl, dir_phase_water)
        D.update(ATOT=atot, BTOT=btot, PATHDIFFOCE=file_phase)
    else:
        # use default water values
        D.update(ATOT=0., BTOT=0., PATHDIFFOCE='None')

    #
    # environment effect
    #
    if env is None:
        # default values (no environment effect)
        env = Environment()
    D.update(env.dict)

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

    return output

def command_file_template(dict):
    '''
    returns the content of the command file based on dict
    '''
    return textwrap.dedent("""
        # Nombre de photons a lancer (unsigned long long) (limite par le tableau du poids des photons en unsigned long long)
        NBPHOTONS = {NBPHOTONS}

        # Angle zenithal de visée en degrés (float)
        THVDEG = {THVDEG}

        # Longueur d'onde [nm] (float)
        LAMBDA = {LAMBDA}

        # Nom absolu du fichier de la matrice de phase des aérosol
            # Données commencant sur la première ligne
            # %lf\t%lf\t%lf\t%lf\t%lf => Correspondant à angle	p2	p1	p3	p4
        PATHDIFFAER = {PATHDIFFAER}

        # Type de simulation
            # -2 pour atmosphere seule
            # -1 pour dioptre seul
            # 0 pour ocean et dioptre
            # 1 pour atmosphere et dioptre
            # 2 pour atmosphere dioptre et ocean
            # 3 pour ocean seul
        SIM = {SIM}

        # Nom absolu du fichier du profil vertical atmosphérique utilisateur
            # Le format du fichier doit être le suivant 
            # I	ALT		hmol(I)		haer(I)		H(I)		XDEL(I)		YDEL(I)  ZABS(I) => Première ligne indicative, pas de données dessus
            # %d\t%f\t%f\t%f\t%f\t%f\t%f\t%f  => Format pour toutes les autres lignes
        PATHPROFILATM = {PATHPROFILATM}

        # Type de reflexion de la surface
            # 1 pour reflexion forcee sur le dioptre
            # 2 transmission forcee
            # 3 reflexion et transmission
        SUR = {SUR}

        # Type de dioptre 
            # 0 = plan
            # 1 = agite avec reflexion multiple
            # 2 = agite sans reflexion multiple
            # 3 = surface lambertienne (uniquement sans océan)
            # 4 = glitter + surface lambertienne (2 en reflexion + 3 pour transmission) - Use SUR=3 in this case
        DIOPTRE = {DIOPTRE}

        # Albedo simple de diffusion de la surface lambertienne (float)
        W0LAM = {W0LAM}

        # Vitesse du vent [m/s] (utilise si idioptre=1) (modele de Cox et Munk) (float)
        WINDSPEED = {WINDSPEED}

        # Indice de refraction relatif air/eau (float)
        NH2O = {NH2O}

        # Coefficients d'absorption et de diffusion totaux de l'eau
        # et fonction de phase totale pour la diffusion dans l'eau
        ATOT = {ATOT}
        BTOT = {BTOT}
        PATHDIFFOCE = {PATHDIFFOCE}

        # Coefficient de dépolarisation
        DEPO = {DEPO}

        # Effet d environnement
        # 0 pas d effet
        # 1 effet d environnement de type cible circulaire
        # ENV_SIZE rayon de la cible en km
        # X0 decalage en km entre la coordonnee X du centre de la cible et le point visee
        # Y0 decalage en km entre la coordonnee Y du centre de la cible et le point visee
        # !! l environnement est lambertien avec un albedo W0LAM
        ENV= {ENV}
        ENV_SIZE= {ENV_SIZE}
        X0= {X0}
        Y0= {Y0}

        ###
        ## Paramètres autres de la simulation ##
        #_____________________________________#

        # Graine avec laquelle on initialise les generateurs aleatoires (int)
        # si SEED est positif on l'utilise comme graine (cela permet d'avoir les memes nombres aleatoires d'une simulation a l'autre, et
        # donc une meme simulation)
        # si SEED=-1 on crée une graine aleatoirement (donc toutes les simulations sont differentes les unes des autres)
        SEED = {SEED}

        # Le demi-espace dans lequel on récupère les photons est divisé en boites
        # theta parcourt 0..PI
        # phi parcourt 0..PI
        NBTHETA = {NBTHETA}
        NBPHI = {NBPHI}

        # number of samples for the inversion of the aerosol phase function
        NFAER = {NFAER}

        # number of samples for the inversion of the ocean phase function
        NFOCE = {NFOCE} 

        ###
        ## Controle des sorties ##
        #_______________________#        

        # Chemin du fichiers de sortie et témoin
        PATHRESULTATSHDF = {PATHRESULTATSHDF}

        # Période d'écriture du fichier témoin en min (-1 pour désactiver)
        WRITE_PERIOD = {WRITE_PERIOD}

        # Output layers (binary flags)
        # 1 -> BOA (0+) downward
        OUTPUT_LAYERS    {OUTPUT_LAYERS}     

        ###
        ## Paramètres de la carte graphique  ##
        #____________________________________#

        # Les threads sont rangés dans des blocks de taille XBLOCK*YBLOCK (limite par watchdog il faut tuer X + limite XBLOCK*YBLOCK =< 256
        #ou 512)
        # XBLOCK doit être un multiple de 32
        # à laisser à 256 (tests d'optimisation effectues)
        XBLOCK = {XBLOCK} 
        YBLOCK = {YBLOCK} 

        # et les blocks sont eux-même rangés dans un grid de taille XGRID*YGRID (limite par watchdog il faut tuer X + limite XGRID<65535 et YGRID<65535)
        XGRID = {XGRID} 
        YGRID = {YGRID} 

        # Nombre de boucles dans le kernel (unsigned int) (limite par watchdog, il faut tuer X)
        # Ne pas mettre NBLOOP à une valeur trop importante, ce qui conduit à des erreurs dans le résultats de sorti
        # Si les résultats manquent de lissage pour les valeurs importantes de réflectance, réduire ce chiffre, relancer et comparer
        # Une valeur de 5000 semble satisfaisante
        NBLOOP = {NBLOOP} 

        DEVICE = -1
    """).format(**dict)


class Surface(object):
    '''
    Stores the smartg parameters relative to the surface
    '''
    def __init__(self, SUR=1, DIOPTRE=2, W0LAM=0., WINDSPEED=5., NH2O=1.33):
        self.dict = {
                'SUR': SUR,
                'DIOPTRE': DIOPTRE,
                'W0LAM': W0LAM,
                'WINDSPEED': WINDSPEED,
                'NH2O': NH2O,
                }
    def __str__(self):
        return 'SUR={SUR}-DI={DIOPTRE}-W0={W0LAM}'.format(**self.dict)

class Environment(object):
    '''
    Stores the smartg parameters relative the the environment effect
    '''
    def __init__(self, ENV=0, ENV_SIZE=0., X0=0., Y0=0.):
        self.dict = {
                'ENV': ENV,
                'ENV_SIZE': ENV_SIZE,
                'X0': X0,
                'Y0': Y0,
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
    return smartg('SMART-G-PP', wl=500., NBPHOTONS=1e9, atm=Profile('afglt'))


def test_kokhanovsky():
    '''
    Just Rayleigh : kokhanovsky test case
    '''
    return smartg('SMART-G-PP', wl=500., DEPO=0., NBPHOTONS=1e9,
            atm=Profile('afglt', grid='100[75]25[5]10[1]0'),
            output=join(dir_output, 'example_kokhanovsky.hdf'))


def test_rayleigh_aerosols():
    '''
    with aerosols
    '''
    aer = AeroOPAC('maritime_clean', 0.4, 550., layer_phase=-1)
    pro = Profile('afglms', aer=aer)

    return smartg('SMART-G-PP', wl=490., atm=pro, NBPHOTONS=1e9)


def test_atm_surf():
    return smartg('SMART-G-PP', 490., NBPHOTONS=1e9,
            output=join(dir_output, 'test_atm_surf.hdf'),
            atm=Profile('afglms'),
            surf=Surface(SUR=1, DIOPTRE=2, W0LAM=1., WINDSPEED=5.))


def test_atm_surf_ocean():
    return smartg('SMART-G-PP', 490., NBPHOTONS=1e9,
            atm=Profile('afglms'),
            surf=Surface(SUR=1, DIOPTRE=2, W0LAM=0., WINDSPEED=5.),
            water=WaterModelSPM(SPM=1.))


def test_surf_ocean():
    return smartg('SMART-G-PP', 490., THVDEG=30., NBPHOTONS=2e6,
            surf=Surface(SUR=3, DIOPTRE=2, W0LAM=0., WINDSPEED=5.),
            water=WaterModelSPM(SPM=1.))


def test_ocean():
    return smartg('SMART-G-PP', 560., THVDEG=30.,
            water=WaterModelSPM(SPM=1.), NBPHOTONS=5e6)


def test_reptran():
    '''
    using reptran
    '''
    aer = AeroOPAC('maritime_polluted', 0.4, 550., layer_phase=-1)
    pro = Profile('afglms.dat', aer=aer, grid='100[75]25[5]10[1]0')
    files, ibands = [], []
    for iband in REPTRAN('reptran_solar_msg').band('msg1_seviri_ch008').ibands():
        f = smartg('SMART-G-PP', wl=np.mean(iband.band.awvl),
                NBPHOTONS=5e8,
                iband=iband, atm=pro)
        files.append(f)
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

        aer = AeroOPAC('maritime_clean', AOT, 550., layer_phase=5)
        pro = Profile('afglms', aer=aer, O3=TCO)

        res = smartg('SMART-G-PP', wl=490., atm=pro, NBTHETA=50, NBPHOTONS=5e6)

        lut = read_lut_hdf(res, 'I_up (TOA)', ['Azimut angles', 'Zenith angles'])
        lut.attrs.update({'TCO':TCO, 'AOT': AOT})
        luts.append(lut)

    merged = merge(luts, ['TCO', 'AOT'])
    merged.print_info()
    merged.save(join(dir_output, 'test_ozone.hdf'))


if __name__ == '__main__':
    test_rayleigh()
    # test_kokhanovsky()
    # test_rayleigh_aerosols()
    # test_atm_surf()
    # test_atm_surf_ocean()
    # test_surf_ocean()
    # test_ocean()
    # test_reptran()
    # test_ozone_lut()

