#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.gpuarray import zeros as gpuzeros


def test_pycuda():
    """Verify that a simple PyCUDA kernel runs on the GPU.

    Compiles the ``hello_gpu`` example kernel that multiplies two
    float arrays element-wise, launches it on the default CUDA
    device, and asserts the GPU result matches the NumPy
    reference ``a * b``.
    """
    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)

    dest = np.zeros_like(a)
    multiply_them(
            drv.Out(dest), drv.In(a), drv.In(b),
            block=(400, 1, 1))

    np.testing.assert_allclose(dest-a*b, 0)
    print('Used', pycuda.autoinit.device.name())


def test_atomic_add():
    mod = SourceModule("""
    __global__ void mykernel(int *a)
    {
        atomicAdd(a, 1);
    }
    """)
    mykernel = mod.get_function("mykernel")
    a = gpuzeros(1, dtype=np.int32)  # type: ignore[arg-type]
    mykernel(a, block=(256, 1, 1), grid=(256, 1, 1))
    assert a == 256*256
    print('Used', a, pycuda.autoinit.device.name())
