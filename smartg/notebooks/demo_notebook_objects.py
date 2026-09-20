# %% [markdown]
# # Smart-G demo notebook for Objects
#
# This is an interactive document allowing to run Smart-G with python and visualize the results. <br>
# *Tips*: cells can be executed with shift-enter. Tooltips can be obtained with shift-tab. More information [here](http://ipython.org/notebook.html) or in the help menu. [A table of content can also be added](https://github.com/minrk/ipython_extensions#table-of-contents).

# %% [markdown]
# ## Symbols used in this notebook
#
# | symbol | meaning |
# |---|---|
# | `sza`, `saa` | solar zenith and azimuth angle, in degrees |
# | `phi` | sensor azimuth, `180 - saa`, the smartg convention |
# | `w_mx`, `w_my` | heliostat half-width along x and y, in km |
# | `w_rx`, `w_ry` | receiver half-width along x and y, in km |
# | `p_rec`, `p_sensor` | receiver and sensor position, a `geoclide.Point` |
# | `helio_pos` | heliostat positions, a list of `geoclide.Point` |
# | `lobj<n>` | the list of objects handed to `run(my_objects=...)` |
# | `p_min`, `p_max` | corners of the bounding box given to `interval=` |
# | `I`, `Q`, `U`, `V` | Stokes components of the radiance |
#
# Lengths are in kilometres, the smartg unit.

# %%
# %matplotlib inline
# the next 2 lines allow to automatically reload modules that have
# been changed externally
# %reload_ext autoreload
# %autoreload 2

import subprocess
import sys
from pathlib import Path

try:
    check = subprocess.check_call(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # Root Git Path
    ROOTPATH = subprocess.Popen(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    ROOTPATH = Path(ROOTPATH)
except subprocess.CalledProcessError:
    ROOTPATH = Path.cwd()
sys.path.insert(0, str(ROOTPATH))

import warnings

import numpy as np
from IPython.display import display

from smartg.albedo import AlbedoCst
from smartg.atmosphere import AerOPAC, Atm1D
from smartg.config import DIR_AUXDATA
from smartg.objects3d import (
    CusBackward,
    CusForward,
    Entity,
    Heliostat,
    Matte,
    Mirror,
    Plane,
    Transformation,
    extract_points,
    generate_h_a,
    generate_h_p,
)
from smartg.sensor import Sensor
from smartg.smartg import LocalEstimate, Smartg
from smartg.surface import LambSurface
from smartg.view import cat_view, receiver_view, smartg_view, visualize_entity
from smartg.xarray import drop_axes

warnings.simplefilter('always', DeprecationWarning)
import geoclide as gc

# %%
# Uncomment below for 3D views
# %matplotlib widget

# %% [markdown]
# # Quick Start

# %% [markdown]
# ## Creation of objects (simple example)

# %%
# We want to create a simple case with a receiver and four
# heliostats (mir_a to mir_d). The receiver will be created at
# x = 1km, the first heliostat at x = 950m, the second heliostat at
# x = 900m and so on. The mirrors will be oriented such that the
# direct rays they reflect totally reach the receiver.

# The sun zenith angle
sza = 14.3
# Create the heliostats mir_a, mir_b, mir_c and mir_d
# (smartg unit is the kilometer)
w_mx = 0.004725
w_my = 0.00642
mir_a = Entity(
    name="reflector",
    material_front=Mirror(reflectivity=0.88),
    material_back=Matte(),
    geo=Plane(p1=gc.Point(-w_mx, -w_my, 0.),
              p2=gc.Point(w_mx, -w_my, 0.),
              p3=gc.Point(-w_mx, w_my, 0.),
              p4=gc.Point(w_mx, w_my, 0.)),
    transformation=Transformation(
        rotation=np.array([0., 20.281725, 0.]),
        translation=np.array([-0.05, 0., 0.00517])))

# mir_b, mir_c and mir_d are exact copies of mir_a, but with
# different transformations
mir_b = Entity(mir_a)
mir_c = Entity(mir_a)
mir_d = Entity(mir_a)
mir_b.set_transformation(Transformation(
    rotation=np.array([0., 29.460753, 0.]),
    translation=np.array([-0.1, 0., 0.00517])))
mir_c.set_transformation(Transformation(
    rotation=np.array([0., 35.129831, 0.]),
    translation=np.array([-0.15, 0., 0.00517])))
mir_d.set_transformation(Transformation(
    rotation=np.array([0., 38.715473, 0.]),
    translation=np.array([-0.2, 0., 0.00517])))

# Create the receiver rec1
w_rx = 0.006
w_ry = 0.007
# tc = Taille cellule. The receiver is divided in several cells to
# visualize the flux distribution
rec1 = Entity(
    name="receiver", tc=0.0005,
    material_front=Matte(reflectivity=0.),
    material_back=Matte(reflectivity=0.),
    geo=Plane(p1=gc.Point(-w_rx, -w_ry, 0.),
              p2=gc.Point(w_rx, -w_ry, 0.),
              p3=gc.Point(-w_rx, w_ry, 0.),
              p4=gc.Point(w_rx, w_ry, 0.)),
    transformation=Transformation(
        rotation=np.array([0., -101.5, 0.]),
        translation=np.array([0., 0., 0.1065])))

# Create the list containing all the objects
lobj1 = [mir_a, mir_b, mir_c, mir_d, rec1]

# %% [markdown]
# ## How to verify the drawing of objects

# %%
# 3D print of objects (the notebook give only a 2D print of the 3D one)
fig1 = visualize_entity(entities=[mir_a, mir_b, mir_c, mir_d, rec1],
                        theta_deg=sza, show_rays=True, sr_view=1,
                        rs_fac=1)

# %% [markdown]
# ## Run the simulation

# %%
# In this simulation the photons from TOA are launched to fill only
# the mirrors (ray tracing method or restricted forward method)
# --> cus_l = CusForward(mode="RF")
# By default the sun is a ponctual sun source targeting the origin
# (0,0,0) --> cus_l = None
# For a full forward mode e.g. specify in Smartg.run()
# --> cus_l = CusForward(cfx=10, cfy=10, mode="FF")
# where cfx is the size in kilometer along the x axis of the
# rectangle. Be careful, the full forward mode needs a big number of
# photons to obtain a good accuracy!

w2 = 0.5
p_min = [-w2, -w2, -0.005]
p_max = [w2, w2, 0.125]
# the interval earns some computational time, very useful in FF mode
interval0 = [p_min, p_max]
aer = AerOPAC('desert', 0.25, 550.)
pro = Atm1D('afglms', comp=[aer], p0=877, tcwp=1.2)
m = Smartg(double=True, obj3d=True).run(
    surface=LambSurface(alb=AlbedoCst(0.25)),
    th_deg=sza, n_icdf=1e6, wavelength=550., n_photons=4e7,
    n_loop=5e6, atmosphere=pro, my_objects=lobj1, interval=interval0,
    cus_l=CusForward(mode="RF"))
m_ds = m

# %% [markdown]
# ## How to show the results ?

# %%
# Show the description of the simulation
print(m_ds)
print(m_ds.attrs)

# %%
# print the infomation at the receiver
output = cat_view(m_ds)
print("\noutput:")
display(output)

# %%
# print the infomation at the receiver
output = cat_view(m_ds)
print("\noutput:")
display(output)

# %%
# print the total flux distribution at the receiver
receiver_view(ds_sg_out=m_ds)

# %%
# print the flux distribution at the receiver, direct D only (cat1)
receiver_view(ds_sg_out=m_ds, cat=1)

# %%
# print the flux distribution at the receiver, process H only (cat2)
receiver_view(ds_sg_out=m_ds, cat=2)

# %%
# print the flux distribution at the receiver, process E only (cat3)
receiver_view(ds_sg_out=m_ds, cat=3)

# %%
# print the flux distribution at the receiver, process A only (cat4)
receiver_view(ds_sg_out=m_ds, cat=4)

# %%
# print the flux distribution at the receiver, H and A only (cat5)
receiver_view(ds_sg_out=m_ds, cat=5)

# %%
# print the flux distribution at the receiver, H and E only (cat6)
receiver_view(ds_sg_out=m_ds, cat=6)

# %%
# print the flux distribution at the receiver, E and A only (cat7)
receiver_view(ds_sg_out=m_ds, cat=7)

# %%
# print the flux distribution at the receiver, H, E and A (cat8)
receiver_view(ds_sg_out=m_ds, cat=8)

# %% [markdown]
# # Complex cases

# %% [markdown]
# ## Quick heliostat generation by giving two angles

# %%
# Specify the sun zenith angle
sza = 14.3

# Creation of a receiver
w_rx = 0.006
w_ry = 0.007

rec2 = Entity(
    name="receiver", tc=0.0005,
    material_front=Matte(reflectivity=0.),
    material_back=Matte(reflectivity=0.),
    geo=Plane(p1=gc.Point(-w_rx, -w_ry, 0.),
              p2=gc.Point(w_rx, -w_ry, 0.),
              p3=gc.Point(-w_rx, w_ry, 0.),
              p4=gc.Point(w_rx, w_ry, 0.)),
    transformation=Transformation(
        rotation=np.array([0., -101.5, 0.]),
        translation=np.array([0., 0., 0.1065])))

p_rec = gc.Point(rec2.transformation.transx,
                 rec2.transformation.transy,
                 rec2.transformation.transz)

# Generation of heliostats thanks to two angles, min_ang_deg and
# max_ang_deg.
lobj2 = generate_h_a(theta_deg=sza, receiver_pos=p_rec, min_ang_deg=150,
                     max_ang_deg=210, gap_ang_deg=10, first_dist=0.1,
                     n_heliostats=3, gap_dist=0.008, helio_size_x=0.005,
                     helio_size_y=0.01, pillar_height=0.00517,
                     reflectivity=0.88)

# Without forgetting to add the receiver to the list of objects
lobj2.append(rec2)

# %% [markdown]
# ### Drawing verification

# %%
fig1 = visualize_entity(lobj2, theta_deg=sza)

# %% [markdown]
# ### Run the simulation

# %%
aer = AerOPAC('desert', 1, 550.)
pro = Atm1D('afglms', comp=[aer])
launch_mode1 = CusForward(mode="RF")

m2 = Smartg(double=True, obj3d=True).run(
    surface=LambSurface(alb=AlbedoCst(0.25)),
    th_deg=sza, n_icdf=1e6, wavelength=550., n_photons=1e7,
    n_loop=1e6, atmosphere=pro, my_objects=lobj2, cus_l=launch_mode1)
m2_ds = m2

# %%
print(m2_ds)
print(m2_ds.attrs)

# %% [markdown]
# ### Results

# %%
cat_view(m2_ds, accuracy=3)

# %%
receiver_view(ds_sg_out=m2_ds, log_color_scale=True)

# %% [markdown]
# ## Quick heliostat generation by giving positions from a file

# %%
# You need a file with the x, y and z positions, see the file
# HPOS_STP1.dat as example
# Points of heliostats are extracted from the given file as a list of
# class point
helio_pos = extract_points(fname=DIR_AUXDATA / 'STPs' / 'STP1.dat')

# Specify the solar zenith angle
sza = 14.3

# Creation of a receiver
w_rx = 0.006
w_ry = 0.007

rec3 = Entity(
    name="receiver", tc=0.0005,
    material_front=Matte(reflectivity=0.),
    material_back=Matte(reflectivity=0.),
    geo=Plane(p1=gc.Point(-w_rx, -w_ry, 0.),
              p2=gc.Point(w_rx, -w_ry, 0.),
              p3=gc.Point(-w_rx, w_ry, 0.),
              p4=gc.Point(w_rx, w_ry, 0.)),
    transformation=Transformation(
        rotation=np.array([0., -101.5, 0.]),
        translation=np.array([0., 0., 0.1065])))

# Coordinate of the center of the created receiver, needed for
# heliostat generation
p_rec = gc.Point(rec3.transformation.transx,
                 rec3.transformation.transy,
                 rec3.transformation.transz)

# Generate heliostats thanks to a list of Points, helio_pos.
lobj3 = generate_h_p(theta_deg=sza, heliostat_pos_list=helio_pos,
                     receiver_pos=p_rec, helio_size_x=0.00945,
                     helio_size_y=0.01284, reflectivity=0.88)

# Without forgetting to add the receiver to the list of objects
lobj3.append(rec3)

# %% [markdown]
# ### Drawing verification

# %%
fig1 = visualize_entity(lobj3, theta_deg=sza)

# %% [markdown]
# ### Run the simulation

# %%
p_min = [-0.12, -0.05, -0.05]
p_max = [0.05, 0.05, 0.125]
# the interval earns some computational time, very useful in FF mode
interval0 = [p_min, p_max]
aer = AerOPAC('desert', 0.2, 550.)
pro = Atm1D('afglms', comp=[aer])
launch_mode = CusForward(cfx=0.06, cfy=0.08, cftx=-0.08, cfty=0.,
                         mode="FF")
m3 = Smartg(double=True, obj3d=True).run(
    surface=LambSurface(AlbedoCst(0.25)),
    th_deg=sza, n_icdf=1e6, wavelength=550., n_photons=4e7,
    n_loop=2e6, atmosphere=pro, my_objects=lobj3, interval=interval0,
    cus_l=launch_mode)
m3_ds = m3

# %%
print(m3_ds)
print(m3_ds.attrs)

# %% [markdown]
# ### Results

# %%
cat_view(m3_ds)

# %%
receiver_view(ds_sg_out=m3_ds, vmin=0)

# %% [markdown]
# # More complex cases

# %% [markdown]
# ## Deals with 3D object in backward mode

# %% [markdown]
# ### Creation of the objects

# %%
# Zenith and Azimuth angle of the sun, respectively Theta and phi
sza = 50
saa = 45
phi = 180. - saa

# Position of heliostats (list) [helio_pos] and position of the
# sensor p_sensor
helio_pos = [gc.Point(-0.05, 0., 0.00517)]  # here only one
p_sensor = gc.Point(0., 0., 0.1065)

# create the heliostats: here we need the zenith and azimuth angles
lobj4 = generate_h_p(theta_deg=sza, phi_deg=phi,
                     heliostat_pos_list=helio_pos,
                     receiver_pos=p_sensor, helio_size_x=0.00945,
                     helio_size_y=0.01284, reflectivity=0.88)

# Modify the rugosity ? here -->
lobj4[0].material_front = Mirror(reflectivity=0.88, roughness=0.1,
                                 shadow=True)

# Creation of the sensor
# the sensor direction is described by a vector
v_sensor = gc.Vector(helio_pos[0] - p_sensor)
sensor = Sensor(pos_x=p_sensor.x, pos_y=p_sensor.y, pos_z=p_sensor.z,
                loc='ATMOS', fov=0., sensor_type=0,
                direction=v_sensor)

# %% [markdown]
# ### Run the simulation

# %%
aer = AerOPAC('desert', 0.2, 550.)
pro = Atm1D('afglms', comp=[aer]).calc(550.)

m4 = Smartg(double=True, obj3d=True, back=True).run(
    surface=LambSurface(alb=AlbedoCst(0.25)),
    n_icdf=1e6, wavelength=550., n_photons=1e8, n_loop=1e7,
    atmosphere=pro, my_objects=lobj4, sensor=sensor)

# %% [markdown]
# ### See the results

# %%
ind_az = [int(np.abs(m4['Azimuth angles'].values - a).argmin())
          for a in [0, 22, 44, 68, 90, 112, 134, 156]]
_=smartg_view(m4, qu=True, ind=ind_az)

# %% [markdown]
# ## STP construction with heliostats composed of facets

# %%
# Specify the sun zenith angle and sun azimuth angle
sza = 50
saa = 100
phi = 180. - saa

# Creation of a receiver
w_rx = 0.006
w_ry = 0.007

rec5 = Entity(
    name="receiver", tc=0.0005,
    material_front=Matte(reflectivity=0.),
    material_back=Matte(reflectivity=0.),
    geo=Plane(p1=gc.Point(-w_rx, -w_ry, 0.),
              p2=gc.Point(w_rx, -w_ry, 0.),
              p3=gc.Point(-w_rx, w_ry, 0.),
              p4=gc.Point(w_rx, w_ry, 0.)),
    transformation=Transformation(
        rotation=np.array([0., -101.5, 0.]),
        translation=np.array([0., 0., 0.1065])))

p_rec = gc.Point(rec5.transformation.transx,
                 rec5.transformation.transy,
                 rec5.transformation.transz)

# Generation of heliostats with facets
helio_type = Heliostat(n_facets_x=2, n_facets_y=2,
                       helio_size_x=0.00945, helio_size_y=0.01284)
lobj5 = generate_h_a(theta_deg=sza, phi_deg=phi, receiver_pos=p_rec,
                     min_ang_deg=140, max_ang_deg=220, gap_ang_deg=10,
                     first_dist=0.15, n_heliostats=4, gap_dist=0.03,
                     pillar_height=0.00517, reflectivity=0.88,
                     roughness=0.001, heliostat_type=helio_type)

# Without forgetting to add the receiver to the list of objects
lobj5.append(rec5)

# %% [markdown]
# ### Drawing verification

# %%
# visualize without facets
lobj5_bis = generate_h_a(theta_deg=sza, phi_deg=phi,
                         receiver_pos=p_rec, min_ang_deg=140,
                         max_ang_deg=220, gap_ang_deg=10,
                         first_dist=0.15, n_heliostats=4,
                         gap_dist=0.03, pillar_height=0.00517,
                         helio_size_x=0.00945, helio_size_y=0.01284,
                         reflectivity=0.88, roughness=0.001)
lobj5_bis.append(rec5)
fig1 = visualize_entity(lobj5_bis, theta_deg=sza, phi_deg=phi,
                        show_rays=True)

# %% [markdown]
# ### Run the simulation

# %%
p_min = [-0.6, -0.6, 0.]
p_max = [0.6, 0.6, 0.125]
# the interval earns some computational time, very useful in FF mode
interval0 = [p_min, p_max]
aer = AerOPAC('desert', 0.5, 550.)
pro = Atm1D('afglms', comp=[aer])
launch_mode = CusForward(cfx=8, cfy=8, cftx=-0.1, cfty=0., mode="FF",
                         sampling='isotropic', fov=0.266)
m5 = Smartg(double=True, obj3d=True).run(
    surface=LambSurface(alb=AlbedoCst(0.25)),
    th_deg=sza, ph_deg=phi, n_icdf=1e6, wavelength=550.,
    n_photons=4e9, n_loop=5e7, atmosphere=pro, my_objects=lobj5,
    interval=interval0, cus_l=launch_mode, direct=True)
m5_ds = m5

# %% [markdown]
# ### Results

# %%
print(m5_ds)
print(m5_ds.attrs)

# %%
print("kernel time(s)=", m5.attrs['kernel time (s)'])
cat_view(m5_ds, output_unit='FLUX', flux_unit='kW')

# %%
receiver_view(ds_sg_out=m5_ds, vmin=0, flux_unit='kW')

# %% [markdown]
# ## The Same in backward

# %%
# Specify the sun zenith angle and sun azimuth angle
sza = 50
saa = 100
phi = 180. - saa

# Creation of a receiver
w_rx = 0.006
w_ry = 0.007

rec6 = Entity(
    name="receiver", tc=0.0005,
    material_front=Matte(reflectivity=0.),
    material_back=Matte(reflectivity=0.),
    geo=Plane(p1=gc.Point(-w_rx, -w_ry, 0.),
              p2=gc.Point(w_rx, -w_ry, 0.),
              p3=gc.Point(-w_rx, w_ry, 0.),
              p4=gc.Point(w_rx, w_ry, 0.)),
    transformation=Transformation(
        rotation=np.array([0., -101.5, 0.]),
        translation=np.array([0., 0., 0.1065])))

p_rec = gc.Point(rec6.transformation.transx,
                 rec6.transformation.transy,
                 rec6.transformation.transz)

# Generation of heliostats with facets
helio_type = Heliostat(n_facets_x=2, n_facets_y=2,
                       helio_size_x=0.00945, helio_size_y=0.01284)
lobj6 = generate_h_a(theta_deg=sza, phi_deg=phi, receiver_pos=p_rec,
                     min_ang_deg=140, max_ang_deg=220, gap_ang_deg=10,
                     first_dist=0.15, n_heliostats=4, gap_dist=0.03,
                     pillar_height=0.00517, reflectivity=0.88,
                     roughness=0.001, heliostat_type=helio_type)

# Without forgetting to add the receiver to the list of objects
lobj6.append(rec6)

# %% [markdown]
# ### Drawing verification

# %%
# visualize without facets
lobj6_bis = generate_h_a(theta_deg=sza, phi_deg=phi,
                         receiver_pos=p_rec, min_ang_deg=140,
                         max_ang_deg=220, gap_ang_deg=10,
                         first_dist=0.15, n_heliostats=4,
                         gap_dist=0.03, pillar_height=0.00517,
                         helio_size_x=0.00945, helio_size_y=0.01284,
                         reflectivity=0.88, roughness=0.001)
lobj6_bis.append(rec6)
fig1 = visualize_entity(lobj6_bis, theta_deg=sza, phi_deg=phi)

# %% [markdown]
# ### Run the simulation

# %%
normal_rec = gc.Vector(0, 0, 1)
rot_y = gc.get_rotate_y_tf(rec6.transformation.rotation[1])
normal_rec = rot_y(normal_rec)
normal_rec = gc.normalize(normal_rec)

p_min = [-0.6, -0.6, 0.]
p_max = [0.6, 0.6, 0.03]
# the interval earns some computational time, very useful in FF mode
interval0 = [p_min, p_max]
aer = AerOPAC('desert', 0.5, 550.)
pro = Atm1D('afglms', comp=[aer])
v_sun = gc.ang2vec(sza, phi, vec_view='nadir')
# sun_fov gives the sun its angular size here, where the run has no
# local estimate; with the le parameter it is le_fov=0.266 instead
launch_mode = CusBackward(position=p_rec, normal=normal_rec,
                          receiver_fov=90, mode="BR", receiver=rec6,
                          v_sun=v_sun, sun_fov=0.266)
m6 = Smartg(double=True, obj3d=True, back=True).run(
    surface=LambSurface(alb=AlbedoCst(0.25)),
    n_icdf=1e6, wavelength=550., n_photons=1e9, n_loop=1e7,
    atmosphere=pro, my_objects=lobj6, interval=interval0,
    cus_l=launch_mode, direct=True)
m6_ds = m6

# %% [markdown]
# ### Results

# %%
print("kernel time(s)=", m6.attrs['kernel time (s)'])
cat_view(m6_ds, output_unit='FLUX', flux_unit='kW')

# %%
receiver_view(ds_sg_out=m6_ds, vmin=0, flux_unit='kW')

# %% [markdown]
# # 2D Coastal Bathymetry — depth-varying ocean reflectance
#
# This section demonstrates how ocean **depth** drives spectral reflectance changes along a coastal transect.
#
# ## Physical concept
#
# In a shallow-water scenario:
# - Photons enter the ocean at the surface, travel through the water column, reflect off the seafloor, and exit.
# - The water column absorbs and scatters photons as a function of depth and wavelength.
# - **Shallow water** → stronger seafloor contribution → higher reflectance, less spectrally selective.
# - **Deep water** → seafloor signal attenuated exponentially → lower reflectance, selective (blue > red, because water absorbs red faster).
#
# ## Scene setup
#
# ```
# x < 0 km : Land   (Lambertian, albedo = 0.10)
# x > 0 km : Ocean  (FlatSurface + HydrosolPR, sandy seafloor albedo = 0.20)
# ```
#
# The ocean depth increases linearly from the coast (`d_coast` at x=0) to `d_max` at x=5 km.
# Because SmartG's ocean is 1D (single water profile per run), three separate runs are performed at
# representative depths **d = 5 m**, **20 m**, and **100 m** and composited into a synthetic transect.

# %%
# Additional imports needed for this section
import matplotlib.pyplot as plt

from smartg.albedo import AlbedoMap
from smartg.surface import Environment, RoughSurface
from smartg.water import HydrosolPR, Water1D

# %%
# ── Scene parameters ────────────────────────────────────────────────
d_km = 5.                  # half-domain extent [km]
n_sensors = 11             # number of sensors in the transect
depths = [5., 20., 100.]   # ocean depths to simulate [m]

# Wavelengths: three key ocean-colour bands
wavelength_bath = np.array([490., 550., 670.])   # [nm]

# ── Coastal environment (AlbedoMap, env=5) ──────────────────────────
#   x < 0 : land  (environment index 0 → AlbedoCst(0.10))
#   x ≥ 0 : ocean (environment index -1 → surface + water from run())
#
# AlbedoMap x-bins: (-inf, 0] = land, (0, +inf) = ocean
x_env = np.array([0., 1e8])   # [km] upper edges of x-bins
y_env = np.array([1e8])       # single y-bin (whole domain)
ai_coast = np.array([[0],     # bin x<=0 : land (alb list index 0)
                     [-1]])   # bin x>0  : ocean (negative → surface)
alb_land = AlbedoCst(0.10)
alb_coast = AlbedoMap(ai_coast, x_env, y_env, [alb_land])
env_coast = Environment(env=5, x0=0., y0=0., alb=alb_coast)

# ── Surface and atmosphere ──────────────────────────────────────────
surf_ocean = RoughSurface()   # flat air/water interface
atm_bath = Atm1D('afglt')     # standard mid-latitude atmosphere

# ── Sensor transect: nadir-looking, from x=-4 km to x=+4 km ─────────
x_sensors = np.linspace(-d_km + 0.5, d_km - 0.5, n_sensors)   # [km]
sensors_bath = [Sensor(pos_z=120., pos_x=xi, pos_y=0., loc='ATMOS',
                       th_deg=180.)
                for xi in x_sensors]

# Sun zenith 30°, azimuth 0°
le_bath = LocalEstimate(th_deg=np.array([30.]),
                        phi_deg=np.array([0.]), zip=True)

print(f"Transect x positions [km]: {x_sensors}")
print(f"Ocean depths to simulate [m]: {depths}")

# %%
# ── Run one simulation per depth ────────────────────────────────────
#
#  ocean model: HydrosolPR(chl=0.1) — coastal turbid water
#               (chl = 0.1 mg/m³)
#  seafloor:    AlbedoCst(0.20) — sandy/coral bottom (20% reflectance)
#  alb in Water1D is the *seafloor* albedo; grid sets the water-column
#  thickness.
#
#  Note: SmartG's ocean is 1-D (a single global water profile per
#  run), so we run once per representative depth and then assemble a
#  synthetic spatial transect.

res_bath = {}   # key = depth [m]

for d in depths:
    water_d = Water1D(grid=[0., -d], comp=[HydrosolPR(chl=0.1)],
                      alb=AlbedoCst(0.20))
    m_d = Smartg().run(
        wavelength=wavelength_bath,
        le=le_bath,
        sensor=sensors_bath,
        atmosphere=atm_bath,
        surface=surf_ocean,
        environment=env_coast,
        water=water_d,
        n_photons=5e7,
        beer=1,
        russian_roulette=0,
    )
    m_d = drop_axes(m_d, 'Zenith angles')
    res_bath[d] = m_d
    print(f"depth = {d:6.1f} m  →  mean ocean reflectance @ 490 nm: "
          f"{m_d['I_up (TOA)'].data[x_sensors > 0, 0].mean():.4f}")

# %% [markdown]
# ### Visualise the depth-dependent reflectance
#
# Expected physics:
# - **Left plot**: land pixels (x < 0) are spectrally flat at ~0.10; ocean pixels decrease and become more blue-dominant as depth increases.
# - **Right plot / depth curve**: at 670 nm (red, strong water absorption) the bottom signal vanishes quickly; at 490 nm (blue, low water absorption) the seafloor remains visible at greater depths.
# - The coastline transition (x = 0) shows an abrupt change between land and ocean surface types.

# %%
# ── Visualisation ───────────────────────────────────────────────────
#
# Left panel  : reflectance transect (land → ocean) at three
#               wavelengths, one line per depth.
# Right panel : reflectance spectra at x=+2 km (ocean pixel) for each
#               depth.

colors_d = {5.: '#1f77b4', 20.: '#ff7f0e', 100.: '#2ca02c'}
ls_wl = {490.: '-', 550.: '--', 670.: ':'}
labels_wl = {490.: '490 nm', 550.: '550 nm', 670.: '670 nm'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── LEFT : spatial transect ─────────────────────────────────────────
ax = axes[0]
for d in depths:
    m = res_bath[d]
    refl = m['I_up (TOA)'].data           # shape (n_sensors, n_wl)
    for i_w, wavelength_nm in enumerate(wavelength_bath):
        lbl = (f"{int(wavelength_nm)} nm, d={int(d)} m"
               if wavelength_nm == 490. else None)
        ax.plot(x_sensors, refl[:, i_w],
                color=colors_d[d],
                ls=ls_wl[wavelength_nm],
                label=lbl)

ax.axvline(0, color='k', lw=0.8, ls='--', label='coastline')
ax.axvspan(-d_km, 0, alpha=0.08, color='saddlebrown', label='land')
ax.axvspan(0, d_km, alpha=0.08, color='steelblue', label='ocean')
ax.set_xlabel('Cross-shore position [km]')
ax.set_ylabel('Reflectance  I/F  [–]')
ax.set_title('Reflectance transect (SZA=30°, nadir sensor)')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)

# ── RIGHT : depth vs reflectance spectra (ocean only) ───────────────
ax2 = axes[1]
# Pick the sensor closest to x=+2 km (ocean pixel)
i_oc = np.argmin(np.abs(x_sensors - 2.0))
for d in depths:
    m = res_bath[d]
    r_oc = m['I_up (TOA)'].data[i_oc, :]   # shape (n_wl,)
    ax2.plot(wavelength_bath, r_oc, 'o-', color=colors_d[d],
             label=f"depth = {int(d)} m")

ax2.set_xlabel('Wavelength [nm]')
ax2.set_ylabel('Reflectance  I/F  [–]')
ax2.set_title(
    f'Spectral signature at x≈{x_sensors[i_oc]:.1f} km (ocean pixel)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ── Depth–reflectance curves at fixed wavelengths ───────────────────
fig2, ax3 = plt.subplots(figsize=(6, 4))
i_oc = np.argmin(np.abs(x_sensors - 2.0))
for i_w, wavelength_nm in enumerate(wavelength_bath):
    r_vs_d = [res_bath[d]['I_up (TOA)'].data[i_oc, i_w] for d in depths]
    ax3.plot(depths, r_vs_d, 'o-', label=labels_wl[wavelength_nm])

ax3.set_xscale('log')
ax3.set_xlabel('Ocean depth [m]')
ax3.set_ylabel('Reflectance  I/F  [–]')
ax3.set_title('Depth–reflectance relationship '
              '(coastal turbid water, seafloor alb = 0.20)')
ax3.legend()
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.show()
