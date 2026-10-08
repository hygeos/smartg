"""Tests of the CUDA context of SMART-G next to the one of JAX.

JAX opens a CUDA context of its own on the GPU, where the kernels of
SMART-G are unknown. These tests need JAX with a GPU (the extra
environment) and are skipped otherwise. Run them in a pytest process of
their own: test_smartg_jax.py keeps JAX on the CPU for the whole
process.
"""

import os

import numpy as np
import pytest

from smartg.atmosphere import Atm1D
from smartg.smartg import LocalEstimate, Smartg


def test_run_after_jax_on_gpu() -> None:
    """Check that Smartg runs after JAX has made its context current.

    The Smartg object compiled before runs again, and a new one
    compiles and runs, with the radiance of the first run. They failed
    with 'invalid resource handle' before 2.0.3.
    """
    # a small share of the GPU memory is enough for this test
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    jax = pytest.importorskip("jax")
    if not any(dv.platform == "gpu" for dv in jax.devices()):
        pytest.skip("JAX has no GPU")
    import jax.numpy as jnp

    kwargs = {
        "atmosphere": Atm1D("afglms"),
        "le": LocalEstimate(th_deg=[0.0], phi_deg=[0.0]),
        "n_photons": 1e5,
        "seed": 1,
        "progress": False,
    }
    s1 = Smartg()
    i1 = s1.run(550.0, **kwargs)["I_up (TOA)"].values
    float(jnp.sum(jnp.arange(10.0)))
    i2 = s1.run(550.0, **kwargs)["I_up (TOA)"].values
    i3 = Smartg(alt_pp=True).run(550.0, **kwargs)["I_up (TOA)"].values
    np.testing.assert_allclose(i2, i1, rtol=0.05)
    np.testing.assert_allclose(i3, i1, rtol=0.05)
