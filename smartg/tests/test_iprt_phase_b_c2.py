#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Non-regression test of the 3D atmosphere mode (opt3d=True) using the
# IPRT phase B cubic cloud case (C2).
# Tested with the following GPUs: 5070 Ti
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smartg import conftest
from smartg.atmosphere import Atm1D, Atm3D, Cloud3D
from smartg.config import DIR_AUXDATA
from smartg.diff import diff1
from smartg.grid3d import Grid3D
from smartg.iprt.iprt import compute_deltam, groupIQUV
from smartg.sensor import get_sensors_grid
from smartg.view import satellite_view
from smartg.phase import read_cld_nth_cte
from smartg.albedo import AlbedoCst
from smartg.surface import LambSurface
from smartg.smartg import Smartg
from smartg.truncation import GT_trunc

# *********************** Global variable(s) ***************************
# Fixed seed: seed=-1 would derive it from the clock, giving a new
# noise realisation at every run. The reference delta_m values below
# were measured with this seed.
SEED = 1234
NBPHOTONS = 49e9  # notebook values: required for the reference
NBLOOP = 1e8  # delta_m values below to be reproducible
NTH = 18001  # 1801 is not enough for case 6

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
XB = [32, 64, 128]  # candidate XBLOCK values
XG = [512, 1024]  # candidate XGRID values
CHECK_NBPHOTONS = 1e8  # short runs used only for timing
CHECK_NBLOOP = 1e8
XBLOCK = 128  # used when FIND_OPTIMAL_XB_XG is False
XGRID = 1024  # (values accepted by most GPUs after 10xx)

SCALE = 1  # can be useful for grid with very small cells
ROOTPATH = Path(__file__).resolve().parent.parent

# GT truncation, as in Iwabuchi and Suzuki (2009), with the parameters
# of the notebook notebooks/demo_notebook.ipynb: simple GT truncation
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
NBPHOTONS_TRUNC = NBPHOTONS / 50

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
NBPHOTONS_ATM_B = {1: NBPHOTONS, 5: 1e9}

# Reference delta_m values (in percent) of I, Q, U and V, measured with
# the settings above (SEED, XBLOCK, XGRID, NBPHOTONS). They are within
# a few percent of the values of the notebook
# notebooks/validation_SMARTG_IPRT_phaseB-C2.ipynb, whose outputs are
# stripped in the repository. Regenerate them with the same settings
# from the log if the physics legitimately changes. Each table holds
# one sub table per tier, the fast one being noisier by construction.
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
# separately on the cases 5 to 9: at NBPHOTONS_TRUNC the truncated run
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
        # 49 times fewer photons than the case 1, hence the larger values
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

# Viewing and sun geometry of the 9 IPRT C2 cases:
# (POSZ key, THETA, PHI, THETA_0). PHI_0 is 180. everywhere.
CASES = {
    1: ("bottom", 40.0, 0.0, 20.0),
    2: ("bottom", 40.0, 60.0, 20.0),
    3: ("bottom", 40.0, 120.0, 20.0),
    4: ("bottom", 40.0, 180.0, 20.0),
    5: ("top", 180.0, 0.0, 40.0),
    6: ("top", 140.0, 0.0, 40.0),
    7: ("top", 140.0, 60.0, 40.0),
    8: ("top", 140.0, 120.0, 40.0),
    9: ("top", 140.0, 180.0, 40.0),
}
PHI_0 = 180.0

# In forward, a single kernel run covers a whole group of cases through
# a zipped local estimate, so the cases are grouped by sun position.
# The two groups do not only differ by their case list: the first one
# looks at the downward radiance below the cloud (count_level 1,
# OUTPUT_LAYERS 3) and the second one at the upward radiance at TOA
# (count_level 0, OUTPUT_LAYERS 1), and only the second one reverses
# the zenith angles of the local estimate.
FORWARD_GROUPS = {
    1: {
        "cases": (1, 2, 3, 4),
        "inv_th": False,
        "count_level": 1,
        "output_layers": 3,
        "layer": "_down (0+)",
    },
    2: {
        "cases": (5, 6, 7, 8, 9),
        "inv_th": True,
        "count_level": 0,
        "output_layers": 1,
        "layer": "_up (TOA)",
    },
}
# **********************************************************************

# **************************** logging *********************************
# Create log file
log_dir = ROOTPATH / "tests" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create a named logger
logger = logging.getLogger("test_phase_b_c2")
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

# Set the formatter for the console handler
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler(
    ROOTPATH / "tests" / "logs" / "iprt_phase_b_c2.log", mode="w"
)
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S%p",
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
# **********************************************************************


def _build_atm_c2(truncation=None, tau_ray=None, **atm1d_kwargs):
    """
    Build the IPRT C2 cubic cloud atmosphere.

    Only the molecular arguments of Atm1D differ between the with and
    without atmosphere sections, hence the **atm1d_kwargs. tau_ray, if
    given, adds a homogeneous Rayleigh layer of that total optical
    depth. truncation is the scattering phase truncation, applied to
    the 3D phase matrices by Atm3D.calc.

    Returns
    -------
    (pro, grid3, surf, wls)
    """
    # ========= phase matrix
    file_cld_phase = (
        DIR_AUXDATA / "IPRT" / "phaseB" / "opt_prop" / "watercloud_800.mie.cdf"
    )
    cld_phase = read_cld_nth_cte(filename=file_cld_phase, nb_theta=NTH)

    # ========= grid (reduced grid = faster)
    xgrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    ygrid = np.array([0.0, 3.0, 4.0, 7.0]) * SCALE
    zgrid = np.array([0.0, 2.0, 3.0, 5.0]) * SCALE
    grid3 = Grid3D(xgrid, ygrid, zgrid, periodic=True)

    # ========= cloud
    # First column x, second y and third z. We follow the IPRT
    # convention for indices (start at 1 instead of 0): the cubic cloud
    # is between 3 and 4 km in x and y, and between 2 and 3 km in z.
    # Its single scattering albedo is forced to 1 (non absorbing).
    cloud_indices = np.zeros((1, 3), dtype=np.int32)
    cloud_indices[0, :] = np.array([2, 2, 2])
    cld_ext_coeff = np.zeros(1, dtype=np.float64)
    cld_ext_coeff[0] = 10.0 * (1 / SCALE)
    reff = np.zeros_like(cld_ext_coeff, dtype=np.float64)
    reff[0] = 10.0
    cloud3 = Cloud3D(
        "wc",
        w_ref=800.0,
        ext_ref=cld_ext_coeff,
        cell_indices=cloud_indices,
        reff=reff,
        phase=cld_phase,
        ssa_cst=1.0,
    )

    # ========= homogeneous Rayleigh layer
    atm3_kwargs = {}
    if tau_ray is not None:
        dz = diff1(grid3.zGRID)
        tau_ray_cs = np.cumsum((dz / grid3.zGRID[-1]) * tau_ray).reshape(
            1, len(dz)
        )
        sca_ray = abs(diff1(tau_ray_cs, axis=1) / dz)
        sca_ray[np.isnan(sca_ray)] = 0
        atm3_kwargs["mol_sca_1d"] = sca_ray
        atm3_kwargs["mol_abs_1d"] = np.zeros_like(sca_ray)

    # ========= profiles computations
    wls = np.array([800.0])
    atm3 = Atm3D(
        atm_1d=Atm1D("afglt", **atm1d_kwargs),
        grid_3d=grid3,
        comp_3d=[cloud3],
        pfwav=[800.0],
        **atm3_kwargs,
    )
    pro = atm3.calc(wls, n_theta=NTH, truncation=truncation)

    surf = LambSurface(alb=AlbedoCst(0.2))

    return pro, grid3, surf, wls


@pytest.fixture(scope="module")
def s3db():
    """
    Backward compilation in 3D
    """
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=True, double=True, bias=True
    )


@pytest.fixture(scope="module")
def s3df():
    """
    Forward compilation in 3D
    """
    return Smartg(
        opt3d=True, alt_pp=True, alis=False, back=False, double=True, bias=True
    )


@pytest.fixture(scope="module")
def atm_c2_noatm():
    """
    IPRT C2 atmosphere without the molecular contribution
    """
    return _build_atm_c2(tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0)


@pytest.fixture(scope="module")
def atm_c2_noatm_gt():
    """
    Same as atm_c2_noatm, with the GT truncated phase matrices
    """
    return _build_atm_c2(
        truncation=GT_TRUNC, tau_r=0.0, no2=False, tco3=0.0, tcwp=0.0
    )


@pytest.fixture(scope="module")
def atm_c2_atm():
    """
    IPRT C2 atmosphere with a homogeneous Rayleigh layer
    """
    return _build_atm_c2(tau_ray=TAU_RAYLEIGH)


@pytest.fixture(scope="module")
def atm_c2_atm_gt():
    """
    Same as atm_c2_atm, with the GT truncated phase matrices
    """
    return _build_atm_c2(truncation=GT_TRUNC, tau_ray=TAU_RAYLEIGH)


@pytest.fixture(scope="module")
def sensor_grid():
    """
    The 70x70 sensor grid, identical for the 9 cases
    """
    return Grid3D(
        np.linspace(0.0, 7.0, 71) * SCALE,
        np.linspace(0.0, 7.0, 71) * SCALE,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) * SCALE,
        periodic=True,
    )


def _resolve_posz(sensor_grid, key):
    """
    Altitude where the sensors are placed
    """
    if key == "bottom":
        return sensor_grid.zGRID[0]
    elif key == "top":
        return sensor_grid.zGRID[-1] - 1e-6 * SCALE
    raise NameError(f"Unknown POSZ key '{key}'!")


def _find_optimal_xb_xg(sg, **run_kwargs):
    """
    Find the number of CUDA blocks and grids giving the shortest kernel
    time, using short runs with the same geometry as the real one.
    """
    if not FIND_OPTIMAL_XB_XG:
        return XBLOCK, XGRID

    k_time = np.inf
    best_xb = XBLOCK
    best_xg = XGRID
    for xg in XG:
        for xb in XB:
            m_test = sg.run(
                **run_kwargs,
                nb_photons=CHECK_NBPHOTONS,
                nb_loop=CHECK_NBLOOP,
                xblock=xb,
                xgrid=xg,
                progress=False,
            )
            time_s = float(m_test.attrs["kernel time (s)"])
            if time_s < k_time:
                k_time = time_s
                best_xb = xb
                best_xg = xg
            logger.info(
                f"time (s) = {time_s}; xblock = {xb}; xgrid = {xg}"
            )
    logger.info(f"Best xblock = {best_xb}; best xgrid = {best_xg}")

    return best_xb, best_xg


def _run_case_backward(
    s3db, atm_c2, sensor_grid, case, nbphotons=NBPHOTONS, depo=None
):
    """
    Run one backward IPRT C2 case

    depo is the depolarization factor: it is only given when there is a
    Rayleigh atmosphere, otherwise the SMART-G default is left alone.

    Returns
    -------
    (m, norm)
    """
    pro, grid3, surf, wls = atm_c2
    posz_key, theta, phi, theta_0 = CASES[case]
    posz = _resolve_posz(sensor_grid, posz_key)

    # !!!! grid3 is different than the sensors grid !!!
    sensors = get_sensors_grid(
        sensor_grid.xgrid,
        sensor_grid.ygrid,
        pos_z=posz,
        th_deg=theta,
        ph_deg=phi,
        fov=0.0,
        loc="ATMOS",
        cell_size=sensor_grid.xgrid[1] - sensor_grid.xgrid[0],
        grid_3d=grid3,
    )

    # count_level = 0 -> only COUNT TOA
    le = {
        "th_deg": np.array([theta_0]),
        "phi_deg": np.array([PHI_0]),
        "count_level": np.array([0]),
    }

    kw = dict(
        wl=wls,
        atm=pro,
        sensor=sensors,
        le=le,
        surf=surf,
        n_f=NTH,
        stdev=True,
    )
    if depo is not None:
        kw["depo"] = depo
    xb, xg = _find_optimal_xb_xg(s3db, **kw)

    m = s3db.run(
        **kw,
        nb_photons=nbphotons,
        nb_loop=NBLOOP,
        xblock=xb,
        xgrid=xg,
        seed=SEED,
    )

    return m, np.cos(np.radians(theta_0)) / np.pi


def _run_group_forward(
    s3df, atm_c2, sensor_grid, group, nbphotons=NBPHOTONS, depo=None
):
    """
    Run one forward group of IPRT C2 cases

    A single kernel run covers the whole group: the viewing directions
    of its cases are zipped in the local estimate.

    Returns
    -------
    (m, norm)
    """
    pro, grid3, surf, wls = atm_c2
    cases = group["cases"]
    # All the cases of a group share the same sun position
    theta_0 = CASES[cases[0]][3]
    posz = _resolve_posz(sensor_grid, "top")

    # In forward the sensors are the source: they are aimed at the sun
    # position instead of at the viewing direction
    sensors = get_sensors_grid(
        sensor_grid.xgrid,
        sensor_grid.ygrid,
        pos_z=posz,
        th_deg=180.0 - theta_0,
        ph_deg=180.0 - PHI_0,
        fov=0.0,
        loc="ATMOS",
        cell_size=sensor_grid.xgrid[1] - sensor_grid.xgrid[0],
        grid_3d=grid3,
    )

    theta = np.array([CASES[case][1] for case in cases])
    phi = np.array([CASES[case][2] for case in cases])
    le = {
        "th_deg": 180.0 - theta if group["inv_th"] else theta,
        "phi_deg": phi + 180.0,
        "count_level": np.full(len(cases), group["count_level"]),
        "zip": True,
    }

    kw = dict(
        th_v_deg=theta_0,
        wl=wls,
        atm=pro,
        sensor=sensors,
        le=le,
        surf=surf,
        n_f=NTH,
        output_layers=group["output_layers"],
    )
    if depo is not None:
        kw["depo"] = depo
    xb, xg = _find_optimal_xb_xg(s3df, **kw)

    m = s3df.run(
        **kw,
        nb_photons=nbphotons,
        nb_loop=NBLOOP,
        xblock=xb,
        xgrid=xg,
        seed=SEED,
    )

    return m, np.cos(np.radians(theta_0)) / np.pi


def _smartg_iquv(
    m, norm, U_sign=1, V_sign=-1, mI=None, mQ=None, mU=None, mV=None
):
    """
    Extract the normalized I, Q, U and V (70, 70) matrices

    The mI, mQ, mU and mV arguments allow to force the values, for
    example when a single forward run holds several cases.
    """
    if mI is None:
        mI = m["I_up (TOA)"][:, 0, 0]
    if mQ is None:
        mQ = m["Q_up (TOA)"][:, 0, 0]
    if mU is None:
        mU = m["U_up (TOA)"][:, 0, 0]
    if mV is None:
        mV = m["V_up (TOA)"][:, 0, 0]

    return (
        mI.reshape(70, 70) * norm,
        mQ.reshape(70, 70) * norm,
        mU.reshape(70, 70) * norm * U_sign,
        mV.reshape(70, 70) * norm * V_sign,
    )


def _mystic_iquv(tcase):
    """
    Read the MYSTIC I, Q, U and V (70, 70) matrices of a given case

    tcase is the MYSTIC case number: it is the C2 case number without
    atmosphere, and the C2 case number + 9 with atmosphere.
    """
    file_res = (
        DIR_AUXDATA / "IPRT" / "phaseB" / "mystic_res" / "iprt_case_C2_mystic.dat"
    )
    read_res = pd.read_csv(
        file_res,
        skiprows=(4900 * (tcase - 1)) + 3,
        nrows=4900,
        header=None,
        sep=r"\s+",
        dtype=float,
    ).values

    return (
        read_res[:, 7].reshape(70, 70).T,
        read_res[:, 8].reshape(70, 70).T,
        read_res[:, 9].reshape(70, 70).T,
        read_res[:, 10].reshape(70, 70).T,
    )


def _plot_case(
    request,
    m,
    iquv_sg,
    iquv_my,
    case,
    sensor_grid,
    title_suffix,
    i_vmin,
    v_diff_frac,
):
    """
    Save the SMART-G maps and the SMART-G - MYSTIC differences in the
    pytest html report
    """
    i_sg, q_sg, u_sg, v_sg = iquv_sg
    i_my, q_my, u_my, v_my = iquv_my

    stk = ["I", "Q", "U", "V"]
    wl = m.axes["wavelength"]
    xgrid = sensor_grid.xgrid
    ygrid = sensor_grid.ygrid
    max_i = np.max(i_sg)
    max_q = np.max(np.abs(q_sg))
    max_u = np.max(np.abs(u_sg))
    max_v = np.max(np.abs(v_sg))

    satellite_view(
        m,
        xgrid,
        ygrid,
        wl,
        "none",
        ["jet", "coolwarm", "coolwarm", "coolwarm"],
        figsize=(10.5, 7),
        fontsize=16,
        vmin=[i_vmin, -max_q, -max_u, -max_v],
        vmax=[max_i, max_q, max_u, max_v],
        scale=False,
        stokes=stk,
        matrices=[i_sg, q_sg, u_sg, v_sg],
        cbar_shrink=1,
        cbar_sci_format=True,
        title=f"C2 - case {case} - SMART-G - {title_suffix}",
    )
    conftest.savefig(request, bbox_inches="tight")

    lim = [
        max_i * 0.05,
        max_q * 0.05,
        max_u * 0.05,
        max_v * v_diff_frac,
    ]
    satellite_view(
        m,
        xgrid,
        ygrid,
        wl,
        "none",
        ["coolwarm", "coolwarm", "coolwarm", "coolwarm"],
        figsize=(10.5, 7),
        fontsize=16,
        vmin=[-val for val in lim],
        vmax=lim,
        scale=False,
        stokes=stk,
        matrices=[i_sg - i_my, q_sg - q_my, u_sg - u_my, v_sg - v_my],
        cbar_shrink=1,
        cbar_sci_format=True,
        title=(
            f"C2 - case {case} - dif(SMART-G - MYSTIC) - {title_suffix}"
        ),
    )
    conftest.savefig(request, bbox_inches="tight")


def _is_significant(signal_ref, istk):
    """
    Whether a Stokes component carries enough signal to be asserted on

    See SIGNAL_FLOOR. A signal_ref of None, i.e. not yet measured, keeps
    every component so that a new reference gets fully logged.
    """
    if signal_ref is None:
        return True

    return signal_ref[istk] > SIGNAL_FLOOR * signal_ref[0]


def _skipped(signal_ref):
    """
    Names of the components left unasserted, for the log
    """
    return [
        stk
        for istk, stk in enumerate(["I", "Q", "U", "V"])
        if not _is_significant(signal_ref, istk)
    ]


def _check_deltam(delta_m_ref, signal_ref, iquv_my, iquv_sg, label, tol):
    """
    Compute the delta_m values and compare them with the previous saved
    validated ones

    Returns the list of the failure messages (empty if the case is ok)
    instead of asserting, so that a forward test can report every case
    of its group instead of stopping at the first one.

    delta_m_ref can be None: the calculated values are then logged and
    the case is reported as a failure, which is how a new reference is
    measured before being written in the tables above.
    """
    iquv_mystic = groupIQUV(
        lI=[iquv_my[0]], lQ=[iquv_my[1]], lU=[iquv_my[2]], lV=[iquv_my[3]]
    )
    iquv_smartg = groupIQUV(
        lI=[iquv_sg[0]], lQ=[iquv_sg[1]], lU=[iquv_sg[2]], lV=[iquv_sg[3]]
    )

    delta_m = compute_deltam(
        obs=iquv_mystic, mod=iquv_smartg, print_res=False
    )

    if delta_m_ref is not None:
        logger.info(
            f"{label} - I={delta_m_ref[0]:.3f}; Q={delta_m_ref[1]:.3f}; "
            + f"U={delta_m_ref[2]:.3f}; V={delta_m_ref[3]:.3f} - ref delta_m:"
        )
    logger.info(
        f"{label} - I={delta_m[0]:.3f}; Q={delta_m[1]:.3f}; "
        + f"U={delta_m[2]:.3f}; V={delta_m[3]:.3f} - calculated delta_m"
    )

    if delta_m_ref is None:
        return [f"{label}: no reference delta_m, see the log for the values"]

    # Check if the test is ok by comparing the ref delta_m and the
    # calculated one
    skipped = _skipped(signal_ref)
    if skipped:
        logger.info(
            f"{label} - {', '.join(skipped)} below SIGNAL_FLOOR, "
            + "not asserted"
        )

    errors = []
    iquv_name = ["I", "Q", "U", "V"]
    for istk, stk in enumerate(iquv_name):
        if not _is_significant(signal_ref, istk):
            continue
        ref = delta_m_ref[istk]
        if abs(delta_m[istk] - ref) > tol * ref:
            errors.append(
                f"{label}: problem with {stk} values, get "
                + f"{delta_m[istk]:.5f}. {stk} must be within "
                + f"[{(1-tol)*ref:.5f}, {(1+tol)*ref:.5f}]"
            )

    return errors


def _check_means(mean_ref, signal_ref, iquv_sg, label):
    """
    Compare the spatial mean of each Stokes component with its previous
    saved validated one

    Unlike delta_m, this averages the Monte Carlo noise out, so it is
    the observable that keeps the fast tier sensitive to a systematic
    bias. Same contract as _check_deltam: returns the list of the
    failure messages, and a mean_ref of None logs the calculated values
    and reports a failure, which is how a new reference is measured.
    """
    means = tuple(float(np.mean(stk)) for stk in iquv_sg)

    if mean_ref is not None:
        logger.info(
            f"{label} - I={mean_ref[0]:.6e}; Q={mean_ref[1]:.6e}; "
            + f"U={mean_ref[2]:.6e}; V={mean_ref[3]:.6e} - ref mean:"
        )
    logger.info(
        f"{label} - I={means[0]:.6e}; Q={means[1]:.6e}; "
        + f"U={means[2]:.6e}; V={means[3]:.6e} - calculated mean"
    )

    if mean_ref is None:
        return [f"{label}: no reference mean, see the log for the values"]

    # The mean of I sets the scale of the four tolerances, the means of
    # Q, U and V being free to pass through zero
    tol = MEAN_TOL * abs(mean_ref[0])

    errors = []
    for istk, stk in enumerate(["I", "Q", "U", "V"]):
        if not _is_significant(signal_ref, istk):
            continue
        ref = mean_ref[istk]
        if abs(means[istk] - ref) > tol:
            errors.append(
                f"{label}: problem with the mean of {stk}, get "
                + f"{means[istk]:.6e}. It must be within "
                + f"[{ref-tol:.6e}, {ref+tol:.6e}]"
            )

    return errors


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "case", list(CASES), ids=[f"case{i}" for i in CASES]
)
def test_c2_noatm_backward(
    request, s3db, atm_c2_noatm, sensor_grid, case, tier
):
    """
    IPRT phase B, cubic cloud C2, backward, without atmosphere
    """
    print(f"=== Test C2 case {case} - backward - without atmosphere - {tier}")

    m, norm = _run_case_backward(
        s3db,
        atm_c2_noatm,
        sensor_grid,
        case,
        nbphotons=NBPHOTONS / PHOTON_DIVIDER[tier],
    )
    iquv_sg = _smartg_iquv(m, norm)
    iquv_my = _mystic_iquv(case)

    _plot_case(
        request,
        m,
        iquv_sg,
        iquv_my,
        case,
        sensor_grid,
        title_suffix="without atm",
        i_vmin=np.min(np.abs(iquv_sg[0])),
        v_diff_frac=0.015,
    )

    label = f"C2 - case {case} - {tier}"
    signal_ref = SIGNAL_REF_NOATM_B[case]
    errors = _check_deltam(
        DELTAM_REF_NOATM_B[tier][case],
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += _check_means(
        MEAN_REF_NOATM_B[case], signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)


def _check_group_forward(
    request,
    m,
    norm,
    group,
    sensor_grid,
    refs,
    mean_refs,
    signal_refs,
    tol,
    title_suffix,
    label_suffix,
    mystic_offset=0,
    i_vmin=None,
    v_diff_frac=0.015,
):
    """
    Plot and check every case held by a single forward run

    mystic_offset is added to the case number to reach the MYSTIC rows:
    it is 9 for the with atmosphere cases. i_vmin, if None, is taken
    from the SMART-G values themselves.

    Returns the failure messages of the whole group, so that one noisy
    case does not hide the others.
    """
    layer = group["layer"]
    errors = []
    for iza, case in enumerate(group["cases"]):
        iquv_sg = _smartg_iquv(
            m,
            norm,
            U_sign=-1,
            V_sign=1,
            mI=m[f"I{layer}"][:, iza],
            mQ=m[f"Q{layer}"][:, iza],
            mU=m[f"U{layer}"][:, iza],
            mV=m[f"V{layer}"][:, iza],
        )
        iquv_my = _mystic_iquv(case + mystic_offset)

        _plot_case(
            request,
            m,
            iquv_sg,
            iquv_my,
            case,
            sensor_grid,
            title_suffix=title_suffix,
            i_vmin=(
                np.min(np.abs(iquv_sg[0])) if i_vmin is None else i_vmin
            ),
            v_diff_frac=v_diff_frac,
        )

        label = f"C2 - case {case} - {label_suffix}"
        signal_ref = signal_refs.get(case)
        errors += _check_deltam(
            refs.get(case), signal_ref, iquv_my, iquv_sg, label, tol
        )
        errors += _check_means(
            mean_refs.get(case), signal_ref, iquv_sg, label
        )

    return errors


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize(
    "group", list(FORWARD_GROUPS), ids=[f"group{i}" for i in FORWARD_GROUPS]
)
def test_c2_noatm_forward(
    request, s3df, atm_c2_noatm, sensor_grid, group, tier
):
    """
    IPRT phase B, cubic cloud C2, forward, without atmosphere
    """
    cases = FORWARD_GROUPS[group]["cases"]
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - without atmosphere - {tier}"
    )

    m, norm = _run_group_forward(
        s3df,
        atm_c2_noatm,
        sensor_grid,
        FORWARD_GROUPS[group],
        nbphotons=NBPHOTONS / PHOTON_DIVIDER[tier],
    )

    errors = _check_group_forward(
        request,
        m,
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
    request, s3df, atm_c2_noatm_gt, sensor_grid, group, tier
):
    """
    IPRT phase B, cubic cloud C2, forward, without atmosphere, with the
    GT truncated cloud phase matrices
    """
    cases = FORWARD_GROUPS[group]["cases"]
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - without atmosphere - GT truncation - {tier}"
    )

    m, norm = _run_group_forward(
        s3df,
        atm_c2_noatm_gt,
        sensor_grid,
        FORWARD_GROUPS[group],
        nbphotons=NBPHOTONS_TRUNC / PHOTON_DIVIDER[tier],
    )

    errors = _check_group_forward(
        request,
        m,
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
def test_c2_atm_backward(request, s3db, atm_c2_atm, sensor_grid, case, tier):
    """
    IPRT phase B, cubic cloud C2, backward, with atmosphere
    """
    print(f"=== Test C2 case {case} - backward - with atmosphere - {tier}")

    m, norm = _run_case_backward(
        s3db,
        atm_c2_atm,
        sensor_grid,
        case,
        nbphotons=NBPHOTONS_ATM_B[case] / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )
    iquv_sg = _smartg_iquv(m, norm)
    iquv_my = _mystic_iquv(case + 9)

    _plot_case(
        request,
        m,
        iquv_sg,
        iquv_my,
        case,
        sensor_grid,
        title_suffix="with atm",
        i_vmin=0.0,
        v_diff_frac=0.05,
    )

    label = f"C2 - case {case} - atm - {tier}"
    signal_ref = SIGNAL_REF_ATM_B.get(case)
    errors = _check_deltam(
        DELTAM_REF_ATM_B[tier].get(case),
        signal_ref,
        iquv_my,
        iquv_sg,
        label,
        DELTAM_TOL[tier],
    )
    errors += _check_means(
        MEAN_REF_ATM_B.get(case), signal_ref, iquv_sg, label
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
def test_c2_atm_forward(request, s3df, atm_c2_atm, sensor_grid, tier):
    """
    IPRT phase B, cubic cloud C2, forward, with atmosphere
    """
    group = FORWARD_GROUPS[ATM_FORWARD_GROUP]
    cases = group["cases"]
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - with atmosphere - {tier}"
    )

    m, norm = _run_group_forward(
        s3df,
        atm_c2_atm,
        sensor_grid,
        group,
        nbphotons=NBPHOTONS / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )

    errors = _check_group_forward(
        request,
        m,
        norm,
        group,
        sensor_grid,
        DELTAM_REF_ATM_F[tier],
        MEAN_REF_ATM_F,
        SIGNAL_REF_ATM_F,
        DELTAM_TOL[tier],
        title_suffix="with atm - forward",
        label_suffix=f"F atm - {tier}",
        mystic_offset=9,
        i_vmin=0.0,
        v_diff_frac=0.05,
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("tier", TIERS)
def test_c2_atm_forward_gt(request, s3df, atm_c2_atm_gt, sensor_grid, tier):
    """
    IPRT phase B, cubic cloud C2, forward, with atmosphere, with the GT
    truncated cloud phase matrices
    """
    group = FORWARD_GROUPS[ATM_FORWARD_GROUP]
    cases = group["cases"]
    print(
        f"=== Test C2 cases {cases[0]} to {cases[-1]} - forward"
        + f" - with atmosphere - GT truncation - {tier}"
    )

    m, norm = _run_group_forward(
        s3df,
        atm_c2_atm_gt,
        sensor_grid,
        group,
        nbphotons=NBPHOTONS_TRUNC / PHOTON_DIVIDER[tier],
        depo=DEPO_ATM,
    )

    errors = _check_group_forward(
        request,
        m,
        norm,
        group,
        sensor_grid,
        DELTAM_REF_ATM_F_GT[tier],
        MEAN_REF_ATM_F_GT,
        SIGNAL_REF_ATM_F_GT,
        DELTAM_TOL[tier],
        title_suffix="with atm - forward - GT trunc",
        label_suffix=f"F atm GT - {tier}",
        mystic_offset=9,
        i_vmin=0.0,
        v_diff_frac=0.05,
    )
    assert not errors, "\n".join(errors)
