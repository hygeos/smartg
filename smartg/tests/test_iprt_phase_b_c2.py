"""Non-regression tests of the 3D mode on the IPRT C2 cubic cloud.

The IPRT phase B cubic cloud case (C2) is run with opt3d=True, in
backward and in forward mode, without and with a Rayleigh atmosphere,
with and without the GT truncation, and compared with MYSTIC. The
atmospheres, the sensors, the runs and the plots come from
smartg.iprt.phase_b.

Tested with the following GPUs: 5070 Ti
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from smartg import conftest
from smartg.grid3d import Grid3D
from smartg.iprt.phase_b import (
    ATM_CASE_OFFSET,
    CASES,
    FORWARD_GROUPS,
    MYSTIC_RES_C2,
    ForwardGroup,
    PhaseBAtmosphere,
    backward_run_kwargs,
    build_atm_c2,
    find_optimal_xb_xg,
    forward_run_kwargs,
    plot_camera_difference,
    plot_camera_iquv,
    read_iprt_iquv,
    run_case_backward,
    run_group_forward,
    sensor_grid_c2,
    smartg_iquv,
)
from smartg.smartg import Smartg
from smartg.tests.iprt_checks import ReferenceChecks
from smartg.truncation import GT_trunc

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The reference delta_m values below
# were measured with this seed.
SEED = 1234
N_PHOTONS = 49e9  # notebook values: required for the reference
N_LOOP = 1e8  # delta_m values below to be reproducible
N_THETA = 18001  # 1801 is not enough for case 6

# Every test runs in two tiers. The slow one uses the photon counts of
# the IPRT benchmark and validates against MYSTIC: it is what the
# reference delta_m values below were measured with, and it is
# deselected by default (see pytest.ini). The fast one divides every
# photon count by PHOTON_DIVIDER and is the one that runs routinely.
TIERS = ["fast", pytest.param("slow", marks=pytest.mark.slow)]
PHOTON_DIVIDER = {"fast": 30, "slow": 1}

# What the two tiers are worth, measured by scaling the cloud extinction
# coefficient of the backward case 1 and looking at what fires:
#
#     extinction   mean of I   delta_m of I   fast tier   slow tier
#         +1%        +0.34%        2.923         pass       fails
#         +3%        +0.73%        3.094         pass         -
#         +5%        +1.08%        3.349        fails         -
#        +10%        +1.89%        4.557        fails         -
#
# So the slow tier catches a 1% error and the fast tier a 5% one. The
# gap is the price of 30 times fewer photons, and it is why the slow
# tier is kept. Note that at +5% it is the mean that fires, delta_m
# being still inside its band: without the mean check the fast tier
# would only catch 10%.

# Two sided fractional band around the reference delta_m. The fast tier
# gets a wider one because dividing the photons by 30 multiplies its MC
# noise by sqrt(30): its delta_m values are about 5.5 times the slow
# ones, and they move more from one noise realisation to the next.
# Measured by rerunning the fast tier with another SEED: the worst case
# then uses 77% of the 0.25 band, hence the margin taken here.
DELTAM_TOL = {"fast": 0.4, "slow": 0.25}

# Second observable, and the sensitive one at the fast tier, where
# delta_m is dominated by the MC noise and a small systematic bias would
# hide inside its band. The spatial mean of a Stokes component over the
# 4900 sensors averages that noise down by a factor ~70, and being a
# linear functional it is an unbiased estimator: its expected value
# depends neither on the photon count nor on the noise realisation. One
# reference therefore serves both tiers.
#
# It is compared with an absolute tolerance of MEAN_TOL times the mean
# of I, for all four components: the means of Q, U and V pass through
# zero from one case to the next, so a relative band on them would be
# meaningless. Rerunning the fast tier with another SEED uses at worst
# 41% of this band, and never failed, which is what a linear functional
# is expected to do.
MEAN_TOL = 0.01

# A component whose mean absolute value falls below SIGNAL_FLOOR times
# the one of I carries no usable signal in these configurations: it is
# Monte Carlo noise, which is why the IPRT benchmark itself reports
# delta_m of 300 to 900% on V. Such a component is logged but not
# asserted, by any of the checks.
#
# This is not a way of hiding a failure, it is what the measurements
# force. The mean of |V| was tried as an observable, precisely to cover
# the components whose signed mean cancels; being nonlinear, it depends
# on the noise level and has fat tails, and rerunning the fast tier with
# another SEED moved it by up to 11 times a 2% band. No cheap statistic
# is stable on a quantity that is pure noise. Below the floor, delta_m
# is not stable either: the same rerun moved the V of the with
# atmosphere case 5 by twice its 25% band.
SIGNAL_FLOOR = 1e-3

# CUDA block/grid: the optimal pair is GPU-dependent and could be
# measured at runtime, but the RNG is seeded per thread index over a
# XBLOCK*XGRID state buffer (smartg/smartg.py:3820), so changing the
# pair changes the noise realisation even at fixed SEED. The pair is
# therefore pinned, and the search is kept for benchmarking only.
FIND_OPTIMAL_XB_XG = False
X_BLOCKS = [32, 64, 128]  # candidate XBLOCK values
X_GRIDS = [512, 1024]  # candidate XGRID values
CHECK_N_PHOTONS = 1e8  # short runs used only for timing
CHECK_N_LOOP = 1e8
XBLOCK = 128  # used when FIND_OPTIMAL_XB_XG is False
XGRID = 1024  # (values accepted by most GPUs after 10xx)

SCALE = 1  # can be useful for grid with very small cells
N_SENSORS = 70  # along x and along y, as in sensor_grid_c2
ROOT_PATH = Path(__file__).resolve().parent.parent

# GT truncation, as in Iwabuchi and Suzuki (2009), with the parameters
# of the notebook notebooks/demo_notebook.py: simple GT truncation
# without correction, i.e. scheme S of the paper. Truncating the
# forward peak of the cloud phase matrix converges much faster, so the
# truncated tests use fewer photons.
GT_TRUNC = GT_trunc(
    trunc_frac=0.435,
    theta_tol=20,
    theta_tr=None,
    integral_method="lobatto",
    lobatto_optimization=True,
)
N_PHOTONS_TRUNC = N_PHOTONS / 50

# The with atmosphere cases add a homogeneous Rayleigh layer of total
# optical depth 0.5, without depolarization. Only a subset of the cases
# is covered: 1 and 5 in backward (a transmittance and a reflectance
# one), and in forward the first group only, which is about 3 times
# faster than the second one.
TAU_RAYLEIGH = 0.5
DEPO_ATM = 0.0
ATM_BACKWARD_CASES = (1, 5)
ATM_FORWARD_GROUP = 1
# The case 5, in the nadir direction, is the slowest one: the notebook
# reduces its photon count, which is kept here.
N_PHOTONS_ATM_B = {1: N_PHOTONS, 5: 1e9}

# Reference delta_m values (in percent) of I, Q, U and V, measured with
# the settings above (SEED, XBLOCK, XGRID, N_PHOTONS). They are within
# a few percent of the values of the notebook
# notebooks/validation_smartg_iprt_phase_b_c2.py, which holds no output.
# Regenerate them with the same settings from the log if the physics
# legitimately changes. Each table holds one sub table per tier, the
# fast one being noisier by construction.
DELTAM_REF_NOATM_B = {
    "slow": {
        1: (0.507, 2.126, 65.313, 404.483),
        2: (0.481, 2.267, 2.644, 304.021),
        3: (0.458, 3.689, 2.070, 248.201),
        4: (0.393, 4.605, 32.565, 315.668),
        5: (0.142, 1.081, 24.303, 262.352),
        6: (0.255, 30.129, 89.739, 981.808),
        7: (0.177, 1.507, 1.391, 70.308),
        8: (0.157, 17.658, 6.626, 365.500),
        9: (0.176, 11.334, 69.407, 424.826),
    },
    "fast": {
        1: (2.750, 11.927, 377.639, 2394.825),
        2: (2.535, 12.288, 11.876, 1862.977),
        3: (2.473, 21.523, 11.996, 1196.923),
        4: (1.926, 24.758, 168.878, 1874.313),
        5: (0.756, 6.612, 168.916, 1508.347),
        6: (0.982, 176.833, 449.371, 5032.999),
        7: (0.918, 7.237, 7.092, 386.257),
        8: (0.826, 95.702, 37.147, 1929.606),
        9: (0.724, 55.769, 359.445, 2273.760),
    },
}

# Same, for the forward simulations. The notebook only saved the cases
# 1 to 4 (0.587, 2.362, 77.761, 492.204 / 0.539, 2.409, 2.796, 356.972 /
# 0.514, 4.429, 2.552, 321.536 / 0.411, 4.612, 37.424, 397.968), the
# output of its cases 5 to 9 cell is empty. The values below are within
# 23% of those four, and their I agrees with the backward table above,
# as expected for the same configuration computed the other way round.
DELTAM_REF_NOATM_F = {
    "slow": {
        1: (0.496, 2.177, 86.529, 429.089),
        2: (0.601, 2.577, 3.179, 371.001),
        3: (0.520, 4.393, 2.607, 302.005),
        4: (0.431, 4.849, 41.652, 307.330),
        5: (0.137, 1.108, 25.284, 223.950),
        6: (0.271, 33.639, 86.133, 908.246),
        7: (0.179, 1.528, 1.331, 91.253),
        8: (0.175, 18.513, 7.059, 355.751),
        9: (0.157, 10.352, 64.669, 388.765),
    },
    "fast": {
        1: (2.921, 11.676, 443.890, 3045.734),
        2: (2.929, 12.982, 14.585, 1907.866),
        3: (2.699, 25.300, 12.991, 1734.547),
        4: (1.855, 26.066, 186.764, 1710.735),
        5: (0.654, 5.398, 139.970, 1181.154),
        6: (0.978, 158.543, 498.021, 4162.765),
        7: (0.891, 7.019, 7.685, 394.590),
        8: (0.877, 96.036, 39.688, 2183.070),
        9: (0.903, 68.540, 387.049, 2272.791),
    },
}

# Same, for the forward simulations with the GT truncation. They are
# expected to differ from the untruncated ones above, by the truncation
# bias and by the MC noise left by 50 times fewer photons. Measured
# separately on the cases 5 to 9: at N_PHOTONS_TRUNC the truncated run
# is 6 to 8 times less noisy than the untruncated one, which is the
# point of the truncation. Their I rising from ~0.17 to ~0.6 is mostly
# that residual noise, plus a small bias: at the full photon count the
# truncated I only comes back down to ~0.5.
# The case 6 is the exception. Its I of 2.731 is truncation bias
# alone, in the exact backscattering direction: multiplying the photon
# count by 50 leaves it at 2.721. Pinning it here is deliberate, it is
# a stable property of the GT scheme S, which is uncorrected.
DELTAM_REF_NOATM_F_GT = {
    "slow": {
        1: (0.532, 1.836, 65.012, 379.106),
        2: (0.495, 2.044, 2.269, 290.020),
        3: (0.456, 3.575, 2.081, 218.844),
        4: (0.543, 4.360, 25.772, 275.192),
        5: (0.636, 2.279, 19.292, 189.583),
        6: (2.731, 24.971, 71.257, 719.034),
        7: (0.594, 3.166, 3.282, 65.703),
        8: (0.584, 14.176, 5.457, 284.017),
        9: (0.632, 9.282, 55.220, 363.366),
    },
    "fast": {
        1: (1.710, 5.699, 197.084, 1326.524),
        2: (1.529, 5.562, 6.938, 911.727),
        3: (1.477, 11.021, 5.557, 696.117),
        4: (1.365, 10.915, 98.020, 867.338),
        5: (1.130, 2.893, 60.490, 536.987),
        6: (2.747, 81.563, 198.725, 2363.258),
        7: (1.148, 4.171, 4.046, 205.615),
        8: (1.146, 46.239, 17.065, 834.945),
        9: (1.183, 26.760, 180.659, 1039.182),
    },
}

# Same, with the Rayleigh atmosphere. The notebook saved no output at
# all for its with atmosphere section, so unlike the tables above these
# have no independent counterpart to be compared with. The backward
# and the forward case 1 agree on I, Q and U (0.145 / 0.181 / 15.606
# against 0.153 / 0.179 / 15.250), which is the only cross-check
# available here.
DELTAM_REF_ATM_B = {
    "slow": {
        1: (0.145, 0.181, 15.606, 29.253),
        # 49 times fewer photons than the case 1, hence the larger
        # values
        5: (0.655, 2.749, 198.337, 161.134),
    },
    "fast": {
        1: (0.785, 0.906, 82.579, 161.015),
        5: (1.986, 8.024, 602.427, 797.055),
    },
}
DELTAM_REF_ATM_F = {
    "slow": {
        1: (0.153, 0.179, 15.250, 64.613),
        2: (0.172, 0.297, 0.296, 53.137),
        3: (0.166, 1.158, 0.456, 65.336),
        4: (0.168, 1.766, 21.157, 111.205),
    },
    "fast": {
        1: (0.746, 0.936, 77.002, 366.130),
        2: (0.929, 1.633, 1.471, 315.728),
        3: (0.784, 5.818, 2.258, 346.504),
        4: (0.894, 8.669, 96.733, 512.546),
    },
}

# Same, with the GT truncation. This is the only test where the
# truncation rescaling of the optical coefficients sees a non zero
# Rayleigh contribution. As without atmosphere, the truncated values
# stay of the same order as the untruncated ones, with 50 times fewer
# photons.
DELTAM_REF_ATM_F_GT = {
    "slow": {
        1: (0.327, 0.446, 24.791, 52.853),
        2: (0.325, 0.607, 0.590, 46.042),
        3: (0.343, 2.214, 0.862, 46.240),
        4: (0.344, 3.285, 39.361, 76.719),
    },
    "fast": {
        1: (1.047, 1.452, 75.684, 187.962),
        2: (1.024, 1.938, 1.846, 127.613),
        3: (1.082, 6.784, 2.820, 135.141),
        4: (1.079, 10.569, 124.495, 264.687),
    },
}

# Reference spatial means of I, Q, U and V, one table per group of
# cases and shared by the two tiers, see MEAN_TOL above. They are
# measured on the slow tier, which is the most precise estimate
# available, and the fast tier is required to reproduce them.
MEAN_REF_NOATM_B = {
    1: (3.275212e-03, 1.139225e-04, 2.620983e-09, -2.459797e-08),
    2: (4.203413e-03, 1.099513e-04, -9.746786e-05, 1.016942e-08),
    3: (6.523648e-03, 6.522401e-05, -1.195332e-04, -7.090443e-09),
    4: (8.526505e-03, 5.744536e-05, -2.261445e-07, -2.479307e-09),
    5: (4.939198e-02, -3.443307e-04, 1.177905e-07, -3.983131e-08),
    6: (5.232794e-02, -1.303586e-05, 1.096235e-08, -2.042112e-08),
    7: (4.991454e-02, 3.990680e-04, 4.420852e-04, -1.966214e-08),
    8: (4.916192e-02, -1.794464e-05, 4.456556e-05, -1.070468e-07),
    9: (4.920845e-02, -1.897888e-05, 3.975927e-08, -1.902957e-08),
}
MEAN_REF_NOATM_F = {
    1: (3.276558e-03, 1.142856e-04, -4.394040e-07, 9.818503e-08),
    2: (4.206401e-03, 1.099452e-04, -9.710456e-05, 2.713082e-08),
    3: (6.519265e-03, 6.514093e-05, -1.188836e-04, -4.159776e-08),
    4: (8.532151e-03, 5.721916e-05, 1.708930e-07, 3.962629e-08),
    5: (4.939118e-02, -3.454032e-04, -1.080490e-07, 6.631268e-09),
    6: (5.232837e-02, -1.256582e-05, 2.639346e-07, -1.700126e-08),
    7: (4.991480e-02, 4.003881e-04, 4.434539e-04, -1.623734e-08),
    8: (4.916164e-02, -1.811675e-05, 4.482573e-05, -8.949405e-08),
    9: (4.920869e-02, -1.917033e-05, 7.009311e-08, -5.067205e-10),
}
MEAN_REF_NOATM_F_GT = {
    1: (3.276109e-03, 1.144488e-04, -3.337631e-07, 3.605540e-08),
    2: (4.203009e-03, 1.102029e-04, -9.754915e-05, -3.736773e-08),
    3: (6.525858e-03, 6.552968e-05, -1.199829e-04, -4.270017e-08),
    4: (8.537454e-03, 5.808903e-05, 2.201372e-07, -4.272513e-09),
    5: (4.939808e-02, -3.521395e-04, -7.316375e-08, 3.331810e-08),
    6: (5.256640e-02, -1.232451e-05, 1.483331e-07, 1.992798e-08),
    7: (4.992617e-02, 4.112837e-04, 4.561402e-04, -1.586470e-08),
    8: (4.916009e-02, -1.770847e-05, 4.416655e-05, -8.095702e-08),
    9: (4.920474e-02, -1.891985e-05, -3.580468e-08, -3.310602e-08),
}
MEAN_REF_ATM_B = {
    1: (6.131054e-02, -2.226966e-02, -1.180943e-07, 7.971769e-08),
    5: (7.786467e-02, -9.874307e-03, 2.528186e-06, -1.046942e-07),
}
MEAN_REF_ATM_F = {
    1: (6.130909e-02, -2.227019e-02, 9.447113e-08, 3.520619e-08),
    2: (6.577108e-02, -1.329901e-02, 1.346168e-02, 9.553408e-07),
    3: (7.694625e-02, -3.336877e-03, 9.013561e-03, 1.155397e-06),
    4: (8.403251e-02, -2.320688e-03, 6.699170e-07, -7.903470e-08),
}
MEAN_REF_ATM_F_GT = {
    1: (6.130797e-02, -2.226847e-02, 1.724001e-06, -3.318604e-08),
    2: (6.576571e-02, -1.329756e-02, 1.345997e-02, 9.201621e-07),
    3: (7.694768e-02, -3.336521e-03, 9.012680e-03, 1.112200e-06),
    4: (8.403481e-02, -2.318281e-03, 7.218640e-07, 1.157812e-08),
}

# Mean absolute value of each Stokes component, measured on the fast
# tier. These are not checked: they are the signal scale against
# which SIGNAL_FLOOR above decides which components are worth
# asserting at all.
SIGNAL_REF_NOATM_B = {
    1: (3.277895e-03, 1.126179e-04, 1.124248e-05, 1.923929e-06),
    2: (4.202616e-03, 1.084060e-04, 9.734038e-05, 1.969278e-06),
    3: (6.527285e-03, 6.448222e-05, 1.192485e-04, 1.846019e-06),
    4: (8.528416e-03, 5.820745e-05, 1.215699e-05, 1.932994e-06),
    5: (4.938628e-02, 3.400484e-04, 1.428447e-05, 1.142657e-06),
    6: (5.233631e-02, 2.276313e-05, 1.778811e-05, 1.576814e-06),
    7: (4.992846e-02, 4.030145e-04, 4.430098e-04, 1.768967e-06),
    8: (4.916764e-02, 2.336304e-05, 4.934094e-05, 1.842389e-06),
    9: (4.920635e-02, 2.641819e-05, 1.210677e-05, 1.597636e-06),
}
SIGNAL_REF_NOATM_F = {
    1: (3.265877e-03, 1.150597e-04, 1.248991e-05, 2.238306e-06),
    2: (4.203065e-03, 1.087849e-04, 9.623822e-05, 2.104932e-06),
    3: (6.517572e-03, 6.896376e-05, 1.191270e-04, 2.448761e-06),
    4: (8.525142e-03, 5.673033e-05, 1.355871e-05, 2.024701e-06),
    5: (4.938933e-02, 3.466164e-04, 1.300644e-05, 9.462992e-07),
    6: (5.233936e-02, 2.033053e-05, 1.901580e-05, 1.511728e-06),
    7: (4.991222e-02, 3.985720e-04, 4.408366e-04, 1.817938e-06),
    8: (4.916201e-02, 2.116997e-05, 4.742968e-05, 1.987739e-06),
    9: (4.921867e-02, 2.535930e-05, 1.264315e-05, 1.761671e-06),
}
SIGNAL_REF_NOATM_F_GT = {
    1: (3.279608e-03, 1.143074e-04, 6.642794e-06, 1.147470e-06),
    2: (4.201359e-03, 1.106473e-04, 9.818132e-05, 1.085059e-06),
    3: (6.530253e-03, 6.581518e-05, 1.196587e-04, 1.182340e-06),
    4: (8.535052e-03, 5.875215e-05, 9.363746e-06, 1.066312e-06),
    5: (4.939188e-02, 3.510618e-04, 8.052788e-06, 4.758920e-07),
    6: (5.256557e-02, 1.390592e-05, 8.756961e-06, 8.413767e-07),
    7: (4.992719e-02, 4.117032e-04, 4.552191e-04, 1.144152e-06),
    8: (4.915838e-02, 1.902941e-05, 4.542510e-05, 9.076485e-07),
    9: (4.920186e-02, 2.142646e-05, 6.803261e-06, 8.672769e-07),
}
SIGNAL_REF_ATM_B = {
    1: (6.130788e-02, 2.227450e-02, 1.979370e-04, 4.638922e-06),
    5: (7.784973e-02, 9.878597e-03, 3.554605e-04, 7.975883e-06),
}
SIGNAL_REF_ATM_F = {
    1: (6.130959e-02, 2.227075e-02, 1.517125e-04, 5.660074e-06),
    2: (6.578507e-02, 1.330023e-02, 1.346101e-02, 6.800714e-06),
    3: (7.695004e-02, 3.333331e-03, 9.015917e-03, 6.585973e-06),
    4: (8.402842e-02, 2.329950e-03, 1.136152e-04, 4.943300e-06),
}
SIGNAL_REF_ATM_F_GT = {
    1: (6.132178e-02, 2.227153e-02, 2.085445e-04, 5.843341e-06),
    2: (6.578884e-02, 1.329916e-02, 1.346745e-02, 6.213064e-06),
    3: (7.697145e-02, 3.327661e-03, 9.012209e-03, 6.095580e-06),
    4: (8.406343e-02, 2.323938e-03, 2.153372e-04, 5.325432e-06),
}

# The viewing and sun geometries of the cases are phase_b.CASES
# **********************************************************************

# **************************** logging *********************************
LOG_DIR = ROOT_PATH / "tests" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "iprt_phase_b_c2.log"
LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)

# Errors on the console, everything in the log file
logger = logging.getLogger("test_phase_b_c2")
logger.setLevel(logging.INFO)
for handler, level in (
    (logging.StreamHandler(), logging.ERROR),
    (logging.FileHandler(LOG_FILE, mode="w"), logging.INFO),
):
    handler.setLevel(level)
    handler.setFormatter(LOG_FORMATTER)
    logger.addHandler(handler)
# **********************************************************************

# The delta_m and mean checks against the reference tables above
CHECKS = ReferenceChecks(logger, SIGNAL_FLOOR, MEAN_TOL)


@pytest.fixture(scope="module")
def s3db() -> Smartg:
    """Backward compilation in 3D."""
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=True, double=True, bias=True
    )


@pytest.fixture(scope="module")
def s3df() -> Smartg:
    """Forward compilation in 3D."""
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=False, double=True, bias=True
    )


@pytest.fixture(scope="module")
def atm_c2_noatm() -> PhaseBAtmosphere:
    """IPRT C2 atmosphere without the molecular contribution."""
    return build_atm_c2(n_theta=N_THETA, scale=SCALE)


@pytest.fixture(scope="module")
def atm_c2_noatm_gt() -> PhaseBAtmosphere:
    """Build atm_c2_noatm with the GT truncated phase matrices."""
    return build_atm_c2(truncation=GT_TRUNC, n_theta=N_THETA, scale=SCALE)


@pytest.fixture(scope="module")
def atm_c2_atm() -> PhaseBAtmosphere:
    """IPRT C2 atmosphere with a homogeneous Rayleigh layer."""
    return build_atm_c2(tau_ray=TAU_RAYLEIGH, n_theta=N_THETA, scale=SCALE)


@pytest.fixture(scope="module")
def atm_c2_atm_gt() -> PhaseBAtmosphere:
    """Build atm_c2_atm with the GT truncated phase matrices."""
    return build_atm_c2(
        tau_ray=TAU_RAYLEIGH, truncation=GT_TRUNC, n_theta=N_THETA,
        scale=SCALE,
    )


@pytest.fixture(scope="module")
def sensor_grid() -> Grid3D:
    """Build the 70x70 sensor grid, identical for the 9 cases."""
    return sensor_grid_c2(SCALE)


def _xblock_xgrid(sg: Smartg, **run_kwargs: Any) -> tuple[int, int]:
    """Return the CUDA block and grid sizes of a run.

    They are XBLOCK and XGRID, unless FIND_OPTIMAL_XB_XG asks for the
    fastest pair, measured with short runs of the same geometry.

    Parameters
    ----------
    sg : Smartg
        The compiled SMART-G.
    **run_kwargs
        The Smartg.run arguments of the run, without n_photons, n_loop,
        xblock and xgrid.

    Returns
    -------
    xblock, xgrid : int
        The CUDA block and grid sizes.
    """
    if not FIND_OPTIMAL_XB_XG:
        return XBLOCK, XGRID
    return find_optimal_xb_xg(
        sg, X_BLOCKS, X_GRIDS, CHECK_N_PHOTONS, CHECK_N_LOOP, **run_kwargs
    )


def _run_case_backward(
    s3db: Smartg,
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    n_photons: float = N_PHOTONS,
    depo: float | None = None,
) -> tuple[xr.Dataset, float]:
    """Run one backward C2 case with the pinned settings.

    Parameters
    ----------
    s3db : Smartg
        The backward compilation.
    atm : PhaseBAtmosphere
        The atmosphere.
    sensor_grid : Grid3D
        The sensor grid.
    case : int
        The case number.
    n_photons : float
        Number of photons.
    depo : float, optional
        The depolarization factor. It is only given with a Rayleigh
        atmosphere, otherwise the SMART-G default is left alone.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps.
    """
    options: dict[str, Any] = {"n_icdf": N_THETA}
    if depo is not None:
        options["depo"] = depo
    xblock, xgrid = _xblock_xgrid(
        s3db, **backward_run_kwargs(atm, sensor_grid, case), **options
    )
    return run_case_backward(
        s3db, atm, sensor_grid, case, n_photons, n_loop=N_LOOP,
        xblock=xblock, xgrid=xgrid, seed=SEED, **options,
    )


def _run_group_forward(
    s3df: Smartg,
    atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    group: ForwardGroup,
    n_photons: float = N_PHOTONS,
    depo: float | None = None,
) -> tuple[xr.Dataset, float]:
    """Run one forward group of C2 cases with the pinned settings.

    A single kernel run covers the whole group: the viewing directions
    of its cases are zipped in the local estimate.

    Parameters
    ----------
    s3df : Smartg
        The forward compilation.
    atm : PhaseBAtmosphere
        The atmosphere.
    sensor_grid : Grid3D
        The sensor grid.
    group : ForwardGroup
        The group of cases.
    n_photons : float
        Number of photons.
    depo : float, optional
        The depolarization factor, see _run_case_backward.

    Returns
    -------
    ds : xr.Dataset
        The output of the run.
    norm : float
        The normalisation of the maps.
    """
    options: dict[str, Any] = {"n_icdf": N_THETA}
    if depo is not None:
        options["depo"] = depo
    xblock, xgrid = _xblock_xgrid(
        s3df, **forward_run_kwargs(atm, sensor_grid, group), **options
    )
    return run_group_forward(
        s3df, atm, sensor_grid, group, n_photons, n_loop=N_LOOP,
        xblock=xblock, xgrid=xgrid, seed=SEED, **options,
    )


def _plot_case(
    request: pytest.FixtureRequest,
    iquv_sg: tuple[np.ndarray, ...],
    iquv_my: tuple[np.ndarray, ...],
    case: int,
    sensor_grid: Grid3D,
    title_suffix: str,
    i_vmin: float | None = None,
    v_diff_frac: float = 0.05,
) -> None:
    """Save the SMART-G maps and their differences with MYSTIC.

    The two figures go to the pytest html report.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request of the test.
    iquv_sg, iquv_my : tuple of ndarray
        The SMART-G and the MYSTIC I, Q, U and V maps.
    case : int
        The case number, for the titles.
    sensor_grid : Grid3D
        The sensor grid.
    title_suffix : str
        The end of the titles.
    i_vmin : float, optional
        The lower bound of the I colour scale. By default the minimum
        of abs(I).
    v_diff_frac : float
        The bound of the V difference colour scale, as a fraction of
        the maximum of abs(V).
    """
    head = f"C2 - case {case}"
    plot_camera_iquv(
        iquv_sg, sensor_grid.xgrid, sensor_grid.ygrid,
        title=f"{head} - SMART-G - {title_suffix}", i_vmin=i_vmin,
    )
    conftest.savefig(request, bbox_inches="tight")
    plot_camera_difference(
        iquv_sg, iquv_my, sensor_grid.xgrid, sensor_grid.ygrid,
        title=f"{head} - dif(SMART-G - MYSTIC) - {title_suffix}",
        v_diff_frac=v_diff_frac,
    )
    conftest.savefig(request, bbox_inches="tight")


def _check_group_forward(
    request: pytest.FixtureRequest,
    ds: xr.Dataset,
    norm: float,
    group: ForwardGroup,
    sensor_grid: Grid3D,
    refs: dict[int, tuple[float, ...]],
    mean_refs: dict[int, tuple[float, ...]],
    signal_refs: dict[int, tuple[float, ...]],
    tol: float,
    title_suffix: str,
    label_suffix: str,
    mystic_offset: int = 0,
    i_vmin: float | None = None,
    v_diff_frac: float = 0.015,
) -> list[str]:
    """Plot and check every case of a single forward run.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The request of the test.
    ds : xr.Dataset
        The output of the forward run.
    norm : float
        The normalisation of the maps.
    group : ForwardGroup
        The group of cases of the run.
    sensor_grid : Grid3D
        The sensor grid.
    refs, mean_refs, signal_refs : dict
        The reference delta_m values, means and mean absolute values,
        by case.
    tol : float
        The fractional band around the reference delta_m.
    title_suffix, label_suffix : str
        The end of the figure titles and of the case labels.
    mystic_offset : int
        Added to the case number to reach the MYSTIC block:
        ATM_CASE_OFFSET for the cases with atmosphere.
    i_vmin : float, optional
        The lower bound of the I colour scale. By default the minimum
        of abs(I).
    v_diff_frac : float
        The bound of the V difference colour scale, see _plot_case.

    Returns
    -------
    list of str
        The failure messages of the whole group, so that one noisy
        case does not hide the others.
    """
    errors = []
    for direction, case in enumerate(group.cases):
        # U and V follow the IPRT convention with the forward signs
        iquv_sg = smartg_iquv(
            ds, norm, N_SENSORS, level=group.level, direction=direction,
            u_sign=-1.0, v_sign=1.0,
        )
        iquv_my = read_iprt_iquv(MYSTIC_RES_C2, case + mystic_offset,
                                 N_SENSORS)

        _plot_case(request, iquv_sg, iquv_my, case, sensor_grid,
                   title_suffix, i_vmin, v_diff_frac)

        label = f"C2 - case {case} - {label_suffix}"
        signal_ref = signal_refs.get(case)
        errors += CHECKS.check_deltam(
            refs.get(case), signal_ref, iquv_my, iquv_sg, label, tol
        )
        errors += CHECKS.check_means(
            mean_refs.get(case), signal_ref, iquv_sg, label
        )

    return errors


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "case", list(CASES), ids=[f"case{i}" for i in CASES]
)
def test_c2_noatm_backward(
    request: pytest.FixtureRequest,
    s3db: Smartg,
    atm_c2_noatm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, backward, without atmosphere."""
    print(f"=== Test C2 case {case} - backward - without atmosphere - {tier}")

    ds, norm = _run_case_backward(
        s3db,
        atm_c2_noatm,
        sensor_grid,
        case,
        n_photons=N_PHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = smartg_iquv(ds, norm, N_SENSORS)
    iquv_my = read_iprt_iquv(MYSTIC_RES_C2, case, N_SENSORS)

    _plot_case(request, iquv_sg, iquv_my, case, sensor_grid,
               title_suffix="without atm", v_diff_frac=0.015)

    label = f"C2 - case {case} - {tier}"
    signal_ref = SIGNAL_REF_NOATM_B[case]
    errors = CHECKS.check_deltam(
        DELTAM_REF_NOATM_B[tier][case],
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += CHECKS.check_means(
        MEAN_REF_NOATM_B[case], signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "group", list(FORWARD_GROUPS), ids=[f"group{i}" for i in FORWARD_GROUPS]
)
def test_c2_noatm_forward(
    request: pytest.FixtureRequest,
    s3df: Smartg,
    atm_c2_noatm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    group: int,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, forward, without atmosphere."""
    cases = FORWARD_GROUPS[group].cases
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - without atmosphere - {tier}"
    )

    ds, norm = _run_group_forward(
        s3df,
        atm_c2_noatm,
        sensor_grid,
        FORWARD_GROUPS[group],
        n_photons=N_PHOTONS / PHOTON_DIVIDER[tier],
    )

    errors = _check_group_forward(
        request,
        ds,
        norm,
        FORWARD_GROUPS[group],
        sensor_grid,
        DELTAM_REF_NOATM_F[tier],
        MEAN_REF_NOATM_F,
        SIGNAL_REF_NOATM_F,
        DELTAM_TOL[tier],
        title_suffix="without atm - forward",
        label_suffix=f"F - {tier}",
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "group", list(FORWARD_GROUPS), ids=[f"group{i}" for i in FORWARD_GROUPS]
)
def test_c2_noatm_forward_gt(
    request: pytest.FixtureRequest,
    s3df: Smartg,
    atm_c2_noatm_gt: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    group: int,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, forward, without atmosphere.

    With the GT truncated cloud phase matrices.
    """
    cases = FORWARD_GROUPS[group].cases
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - without atmosphere - GT truncation - {tier}"
    )

    ds, norm = _run_group_forward(
        s3df,
        atm_c2_noatm_gt,
        sensor_grid,
        FORWARD_GROUPS[group],
        n_photons=N_PHOTONS_TRUNC / PHOTON_DIVIDER[tier],
    )

    errors = _check_group_forward(
        request,
        ds,
        norm,
        FORWARD_GROUPS[group],
        sensor_grid,
        DELTAM_REF_NOATM_F_GT[tier],
        MEAN_REF_NOATM_F_GT,
        SIGNAL_REF_NOATM_F_GT,
        DELTAM_TOL[tier],
        title_suffix="without atm - forward - GT trunc",
        label_suffix=f"F GT - {tier}",
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "case", ATM_BACKWARD_CASES, ids=[f"case{i}" for i in ATM_BACKWARD_CASES]
)
def test_c2_atm_backward(
    request: pytest.FixtureRequest,
    s3db: Smartg,
    atm_c2_atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    case: int,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, backward, with atmosphere."""
    print(f"=== Test C2 case {case} - backward - with atmosphere - {tier}")

    ds, norm = _run_case_backward(
        s3db,
        atm_c2_atm,
        sensor_grid,
        case,
        n_photons=N_PHOTONS_ATM_B[case] / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )
    iquv_sg = smartg_iquv(ds, norm, N_SENSORS)
    iquv_my = read_iprt_iquv(MYSTIC_RES_C2, case + ATM_CASE_OFFSET,
                             N_SENSORS)

    _plot_case(request, iquv_sg, iquv_my, case, sensor_grid,
               title_suffix="with atm", i_vmin=0.0, v_diff_frac=0.05)

    label = f"C2 - case {case} - atm - {tier}"
    signal_ref = SIGNAL_REF_ATM_B.get(case)
    errors = CHECKS.check_deltam(
        DELTAM_REF_ATM_B[tier].get(case),
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += CHECKS.check_means(
        MEAN_REF_ATM_B.get(case), signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
def test_c2_atm_forward(
    request: pytest.FixtureRequest,
    s3df: Smartg,
    atm_c2_atm: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, forward, with atmosphere."""
    group = FORWARD_GROUPS[ATM_FORWARD_GROUP]
    cases = group.cases
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - with atmosphere - {tier}"
    )

    ds, norm = _run_group_forward(
        s3df,
        atm_c2_atm,
        sensor_grid,
        group,
        n_photons=N_PHOTONS / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )

    errors = _check_group_forward(
        request,
        ds,
        norm,
        group,
        sensor_grid,
        DELTAM_REF_ATM_F[tier],
        MEAN_REF_ATM_F,
        SIGNAL_REF_ATM_F,
        DELTAM_TOL[tier],
        title_suffix="with atm - forward",
        label_suffix=f"F atm - {tier}",
        mystic_offset=ATM_CASE_OFFSET,
        i_vmin=0.0,
        v_diff_frac=0.05,
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
def test_c2_atm_forward_gt(
    request: pytest.FixtureRequest,
    s3df: Smartg,
    atm_c2_atm_gt: PhaseBAtmosphere,
    sensor_grid: Grid3D,
    tier: str,
) -> None:
    """IPRT phase B, cubic cloud C2, forward, with atmosphere.

    With the GT truncated cloud phase matrices.
    """
    group = FORWARD_GROUPS[ATM_FORWARD_GROUP]
    cases = group.cases
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - with atmosphere - GT truncation - {tier}"
    )

    ds, norm = _run_group_forward(
        s3df,
        atm_c2_atm_gt,
        sensor_grid,
        group,
        n_photons=N_PHOTONS_TRUNC / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )

    errors = _check_group_forward(
        request,
        ds,
        norm,
        group,
        sensor_grid,
        DELTAM_REF_ATM_F_GT[tier],
        MEAN_REF_ATM_F_GT,
        SIGNAL_REF_ATM_F_GT,
        DELTAM_TOL[tier],
        title_suffix="with atm - forward - GT trunc",
        label_suffix=f"F atm GT - {tier}",
        mystic_offset=ATM_CASE_OFFSET,
        i_vmin=0.0,
        v_diff_frac=0.05,
    )
    assert not errors, "\n".join(errors)
