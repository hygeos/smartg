/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 *
 *  The file includes several additions from:
 *    Hygeos - 165 Avenue de Bretagne, 59000 Lille, France.
 *      - Additional functions
 *      - Add of some child structures
 *      - Add of Matrix structures
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include <cuda_runtime.h>
//#include "cuda_runtime.h"
#include <math_constants.h>

#ifndef DEBUG
#define myError(expr) ((void)0)
#else
//#define myError(expr) ( (expr) ? \
						(printf("vector or matrice indices error***\n")) : (0) )
#define myError(expr) ((void)0)
#endif

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline __host__ __device__ float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline __host__ __device__ int max(int a, int b)
{
    return a > b ? a : b;
}

inline __host__ __device__ int min(int a, int b)
{
    return a < b ? a : b;
}

inline __host__ __device__ float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
//******************************************************************************
inline __host__ __device__ double fmin(double a, double b)
{
    return a < b ? a : b;
}

inline __host__ __device__ double fmax(double a, double b)
{
    return a > b ? a : b;
}

inline __host__ __device__ double rsqrt(double x)
{
    return 1.0 / sqrt(x);
}

#endif

////////////////////////////////////////////////////////////////////////////////
// child to enable indices
////////////////////////////////////////////////////////////////////////////////

// child structures
struct int2c: public int2
{
	__host__ __device__ int operator[](int idx) const
	{
		myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}

	__host__ __device__ int &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}	
};

struct int3c: public int3
{
	__host__ __device__ int operator[](int idx) const
	{
		myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}

	__host__ __device__ int &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}	
};

struct int4c: public int4
{
	__host__ __device__ int operator[](int idx) const
	{
		myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}

	__host__ __device__ int &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}	
};

struct float2c: public float2
{
	__host__ __device__ float operator[](int idx) const
	{
		myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}

	__host__ __device__ float &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}	
};

struct float3c: public float3
{
	__host__ __device__ float operator[](int idx) const
	{
		myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}

	__host__ __device__ float &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}	
};

struct float4c: public float4
{
	__host__ __device__ float operator[](int idx) const
	{
		myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}

	__host__ __device__ float &operator[](int idx)
	{

		myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}	
};

//********************************************************************
struct double2c: public double2
{
	__host__ __device__ double operator[](int idx) const
	{
		myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}

	__host__ __device__ double &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 1));
		return (&x)[idx];
	}	
};

struct double3c: public double3
{
	__host__ __device__ double operator[](int idx) const
	{
		myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}

	__host__ __device__ double &operator[](int idx)
	{
	    myError((idx < 0) || (idx > 2));
		return (&x)[idx];
	}	
};

struct double4c: public double4
{
	__host__ __device__ double operator[](int idx) const
	{
		myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}

	__host__ __device__ double &operator[](int idx)
	{

		myError((idx < 0) || (idx > 3));
		return (&x)[idx];
	}	
};

// child constructors
inline __host__ __device__ int2c make_int2c(int x, int y)
{
  int2c t; t.x = x; t.y = y; return t;
}
inline __host__ __device__ int2c make_int2c(int2 s)
{
  int2c t; t.x = s.x; t.y = s.y; return t;
}

inline __host__ __device__ int3c make_int3c(int x, int y, int z)
{
  int3c t; t.x = x; t.y = y; t.z = z; return t;
}
inline __host__ __device__ int3c make_int3c(int3 s)
{
  int3c t; t.x = s.x; t.y = s.y; t.z = s.z; return t;
}

inline __host__ __device__ int4c make_int4c(int x, int y, int z, int w)
{
  int4c t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline __host__ __device__ int4c make_int4c(int4 s)
{
  int4c t; t.x = s.x; t.y = s.y; t.z = s.z; t.w = s.w; return t;
}

inline __host__ __device__ float2c make_float2c(float x, float y)
{
  float2c t; t.x = x; t.y = y; return t;
}
inline __host__ __device__ float2c make_float2c(float2 s)
{
  float2c t; t.x = s.x; t.y = s.y; return t;
}

inline __host__ __device__ float3c make_float3c(float x, float y, float z)
{
  float3c t; t.x = x; t.y = y; t.z = z; return t;
}
inline __host__ __device__ float3c make_float3c(float3 s)
{
  float3c t; t.x = s.x; t.y = s.y; t.z = s.z; return t;
}

inline __host__ __device__ float4c make_float4c(float x, float y, float z, float w)
{
  float4c t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline __host__ __device__ float4c make_float4c(float4 s)
{
  float4c t; t.x = s.x; t.y = s.y; t.z = s.z; t.w = s.w; return t;
}

//******************************************************************************
inline __host__ __device__ double2c make_double2c(double x, double y)
{
  double2c t; t.x = x; t.y = y; return t;
}
inline __host__ __device__ double2c make_double2c(double2 s)
{
  double2c t; t.x = s.x; t.y = s.y; return t;
}

inline __host__ __device__ double3c make_double3c(double x, double y, double z)
{
  double3c t; t.x = x; t.y = y; t.z = z; return t;
}
inline __host__ __device__ double3c make_double3c(double3 s)
{
  double3c t; t.x = s.x; t.y = s.y; t.z = s.z; return t;
}

inline __host__ __device__ double4c make_double4c(double x, double y, double z, double w)
{
  double4c t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
inline __host__ __device__ double4c make_double4c(double4 s)
{
  double4c t; t.x = s.x; t.y = s.y; t.z = s.z; t.w = s.w; return t;
}

////////////////////////////////////////////////////////////////////////////////
// BEGIN MATRICES
////////////////////////////////////////////////////////////////////////////////

// structures of matrices
struct float2x2
{
    float2c r0, r1;

	__host__ __device__ float2c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 1));
		return (&r0)[idx];
	}

	__host__ __device__ float2c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 1));
		return (&r0)[idx];
	}
};

struct float3x3
{
    float3c r0, r1, r2;

	__host__ __device__ float3c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 2));
		return (&r0)[idx];
	}

	__host__ __device__ float3c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 2));
		return (&r0)[idx];
	}
};

struct float4x4
{
    float4c r0, r1, r2, r3;

	__host__ __device__ float4c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 3));
		return (&r0)[idx];
	}

	__host__ __device__ float4c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 3));
		return (&r0)[idx];
	}
};

//**********************
struct double2x2
{
    double2c r0, r1;

	__host__ __device__ double2c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 1));
		return (&r0)[idx];
	}

	__host__ __device__ double2c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 1));
		return (&r0)[idx];
	}
};

struct double3x3
{
    double3c r0, r1, r2;

	__host__ __device__ double3c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 2));
		return (&r0)[idx];
	}

	__host__ __device__ double3c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 2));
		return (&r0)[idx];
	}
};

struct double4x4
{
    double4c r0, r1, r2, r3;

	__host__ __device__ double4c operator[](int idx) const
	{
		myError((idx < 0) || (idx > 3));
		return (&r0)[idx];
	}

	__host__ __device__ double4c &operator[](int idx)
	{
		myError((idx < 0) || (idx > 3));
		return (&r0)[idx];
	}
};

// matrices constructors
inline __host__ __device__ float2x2 make_float2x2(float m00, float m01,
												  float m10, float m11){
	float2x2 M;
	M.r0.x=m00; M.r0.y=m01; // row 0
	M.r1.x=m10; M.r1.y=m11; // row 1
	return M;
}

inline __host__ __device__ float3x3 make_float3x3(float m00, float m01,float m02,
												  float m10, float m11, float m12,
												  float m20, float m21, float m22){
	float3x3 M;
	M.r0.x=m00; M.r0.y=m01; M.r0.z=m02; // row 0
	M.r1.x=m10; M.r1.y=m11; M.r1.z=m12; // row 1
	M.r2.x=m20; M.r2.y=m21; M.r2.z=m22; // row 2
	return M;
}

inline __host__ __device__ float4x4 make_float4x4(float m00, float m01, float m02, float m03,
												  float m10, float m11, float m12, float m13,
												  float m20, float m21, float m22, float m23,
												  float m30, float m31, float m32, float m33){
	float4x4 M;
	M.r0.x=m00; M.r0.y=m01; M.r0.z=m02; M.r0.w=m03; // row 0
	M.r1.x=m10; M.r1.y=m11; M.r1.z=m12; M.r1.w=m13; // row 1
	M.r2.x=m20; M.r2.y=m21; M.r2.z=m22; M.r2.w=m23; // row 2
	M.r3.x=m30; M.r3.y=m31; M.r3.z=m32; M.r3.w=m33; // row 3
	return M;
}

//**********************
inline __host__ __device__ double2x2 make_double2x2(float m00, float m01,
													float m10, float m11){
	double2x2 M;
	M.r0.x=m00; M.r0.y=m01; // row 0
	M.r1.x=m10; M.r1.y=m11; // row 1
	return M;
}

inline __host__ __device__ double3x3 make_double3x3(float m00, float m01,float m02,
													float m10, float m11, float m12,
													float m20, float m21, float m22){
	double3x3 M;
	M.r0.x=m00; M.r0.y=m01; M.r0.z=m02; // row 0
	M.r1.x=m10; M.r1.y=m11; M.r1.z=m12; // row 1
	M.r2.x=m20; M.r2.y=m21; M.r2.z=m22; // row 2
	return M;
}

inline __host__ __device__ double4x4 make_double4x4(float m00, float m01, float m02, float m03,
													float m10, float m11, float m12, float m13,
													float m20, float m21, float m22, float m23,
													float m30, float m31, float m32, float m33){
	double4x4 M;
	M.r0.x=m00; M.r0.y=m01; M.r0.z=m02; M.r0.w=m03; // row 0
	M.r1.x=m10; M.r1.y=m11; M.r1.z=m12; M.r1.w=m13; // row 1
	M.r2.x=m20; M.r2.y=m21; M.r2.z=m22; M.r2.w=m23; // row 2
	M.r3.x=m30; M.r3.y=m31; M.r3.z=m32; M.r3.w=m33; // row 3
	return M;
}

// others
inline __host__ __device__ float2x2 make_float2x2(float s)
{
    return make_float2x2(s, s, s, s);
}

inline __host__ __device__ float2x2 make_float2x2(float2c r1, float2c r2)
{
    return make_float2x2(r1.x, r1.y, r2.x, r2.y);
}

inline __host__ __device__ float3x3 make_float3x3(float s)
{
    return make_float3x3(s, s, s, s, s, s, s, s, s);
}

inline __host__ __device__ float3x3 make_diag_float3x3(float s)
{
    float z = 0.;
    return make_float3x3(s, z, z, z, s, z, z, z, s);
}

inline __host__ __device__ float3x3 make_float3x3(float3c r1, float3c r2, float3c r3)
{
    return make_float3x3(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z);
}

inline __host__ __device__ float4x4 make_float4x4(float s)
{
    return make_float4x4(s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s);
}


inline __host__ __device__ float4x4 make_float4x4(float4c r1, float4c r2, float4c r3, float4c r4)
{
    return make_float4x4(r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w, r4.x, r4.y, r4.z, r4.w);
}

inline __host__ __device__ float4x4 make_diag_float4x4(float s)
{
    float z = 0.;
    return make_float4x4(s, z, z, z, z, s, z, z, z, z, s, z, z, z, z, s);
}

//**********************
inline __host__ __device__ double2x2 make_double2x2(float s)
{
    return make_double2x2(s, s, s, s);
}

inline __host__ __device__ double2x2 make_double2x2(double2c r1, double2c r2)
{
    return make_double2x2(r1.x, r1.y, r2.x, r2.y);
}

inline __host__ __device__ double3x3 make_double3x3(double s)
{
    return make_double3x3(s, s, s, s, s, s, s, s, s);
}

inline __host__ __device__ double3x3 make_double3x3(double3c r1, double3c r2, double3c r3)
{
    return make_double3x3(r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z);
}

inline __host__ __device__ double4x4 make_double4x4(double s)
{
    return make_double4x4(s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s);
}


inline __host__ __device__ double4x4 make_double4x4(double4c r1, double4c r2, double4c r3, double4c r4)
{
    return make_double4x4(r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w, r4.x, r4.y, r4.z, r4.w);
}

inline __host__ __device__ double4x4 make_diag_double4x4(double s)
{
    double z = 0.;
    return make_double4x4(s, z, z, z, z, s, z, z, z, z, s, z, z, z, z, s);
}

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

// ****************************************************************************
inline __host__ __device__ double2 make_double2(double s)
{
    return make_double2(s, s);
}
inline __host__ __device__ double2 make_double2(double3 a)
{
    return make_double2(a.x, a.y);
}
inline __host__ __device__ double2 make_double2(int2 a)
{
    return make_double2(double(a.x), double(a.y));
}
inline __host__ __device__ double2 make_double2(uint2 a)
{
    return make_double2(double(a.x), double(a.y));
}
inline __host__ __device__ double2 make_double2(float2 a)
{
    return make_double2(double(a.x), double(a.y));
}

inline __host__ __device__ double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
inline __host__ __device__ double3 make_double3(double2 a)
{
    return make_double3(a.x, a.y, 0.0);
}
inline __host__ __device__ double3 make_double3(double2 a, double s)
{
    return make_double3(a.x, a.y, s);
}
inline __host__ __device__ double3 make_double3(double4 a)
{
    return make_double3(a.x, a.y, a.z);
}
inline __host__ __device__ double3 make_double3(int3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
inline __host__ __device__ double3 make_double3(uint3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}
inline __host__ __device__ double3 make_double3(float3 a)
{
    return make_double3(double(a.x), double(a.y), double(a.z));
}

inline __host__ __device__ double4 make_double4(double s)
{
    return make_double4(s, s, s, s);
}
inline __host__ __device__ double4 make_float4(double3 a)
{
    return make_double4(a.x, a.y, a.z, 0.0);
}
inline __host__ __device__ double4 make_float4(double3 a, double w)
{
    return make_double4(a.x, a.y, a.z, w);
}
inline __host__ __device__ double4 make_double4(int4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(uint4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}
inline __host__ __device__ double4 make_double4(float4 a)
{
    return make_double4(double(a.x), double(a.y), double(a.z), double(a.w));
}




inline __host__ __device__ float2 make_float2(double2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(double2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ float3 make_float3(double3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(double3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ float4 make_float4(double4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ int4 make_int4(double4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

//******************************************************************************
inline __host__ __device__ double2 operator-(double2 &a)
{
    return make_double2(-a.x, -a.y);
}
inline __host__ __device__ double3 operator-(double3 &a)
{
    return make_double3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ double4 operator-(double4 &a)
{
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

//******************************************************************************
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ double2 operator+(double2 a, double b)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ double2 operator+(double b, double2 a)
{
    return make_double2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(double2 &a, double b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3 &a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ double3 operator+(double b, double3 a)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(double3 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ double4 operator+(double4 a, double4 b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ double4 operator+(double4 a, double b)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ double4 operator+(double b, double4 a)
{
    return make_double4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(double4 &a, double b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}
////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

//******************************************************************************
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ double2 operator-(double2 a, double b)
{
    return make_double2(a.x - b, a.y - b);
}
inline __host__ __device__ double2 operator-(double b, double2 a)
{
    return make_double2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(double2 &a, double b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(double3 &a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(double3 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(double4 &a, double4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ double4 operator-(double4 a, double b)
{
    return make_double4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(double4 &a, double b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

//******************************************************************************
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(double2 &a, double b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(double3 &a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(double3 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ double4 operator*(double4 a, double4 b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(double4 &a, double4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ double4 operator*(double4 a, double b)
{
    return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ double4 operator*(double b, double4 a)
{
    return make_double4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(double4 &a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

//******************************************************************************
inline __host__ __device__ double2 operator/(double2 a, double2 b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(double2 &a, double2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ double2 operator/(double2 a, double b)
{
    return make_double2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(double2 &a, double b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ double2 operator/(double b, double2 a)
{
    return make_double2(b / a.x, b / a.y);
}
inline __host__ __device__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(double3 &a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ double3 operator/(double3 a, double b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(double3 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ double3 operator/(double b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ double4 operator/(double4 a, double4 b)
{
    return make_double4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(double4 &a, double4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ double4 operator/(double4 a, double b)
{
    return make_double4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(double4 &a, double b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ double4 operator/(double b, double4 a)
{
    return make_double4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

//******************************************************************************
inline __host__ __device__ double2 fmin(double2 a, double2 b)
{
    return make_double2(fmin(a.x,b.x), fmin(a.y,b.y));
}
inline __host__ __device__ double3 fmin(double3 a, double3 b)
{
    return make_double3(fmin(a.x,b.x), fmin(a.y,b.y), fmin(a.z,b.z));
}
inline __host__ __device__ double4 fmin(double4 a, double4 b)
{
    return make_double4(fmin(a.x,b.x), fmin(a.y,b.y), fmin(a.z,b.z), fmin(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}
//******************************************************************************
inline __host__ __device__ double2 fmax(double2 a, double2 b)
{
    return make_double2(fmax(a.x,b.x), fmax(a.y,b.y));
}
inline __host__ __device__ double3 fmax(double3 a, double3 b)
{
    return make_double3(fmax(a.x,b.x), fmax(a.y,b.y), fmax(a.z,b.z));
}
inline __host__ __device__ double4 fmax(double4 a, double4 b)
{
    return make_double4(fmax(a.x,b.x), fmax(a.y,b.y), fmax(a.z,b.z), fmax(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}
//******************************************************************************
inline __device__ __host__ double lerp(double a, double b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double2 lerp(double2 a, double2 b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double3 lerp(double3 a, double3 b, double t)
{
    return a + t*(b-a);
}
inline __device__ __host__ double4 lerp(double4 a, double4 b, double t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
//******************************************************************************
inline __device__ __host__ double clamp(double f, double a, double b)
{
    return fmax(a, fmin(f, b));
}

inline __device__ __host__ double2 clamp(double2 v, double a, double b)
{
    return make_double2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ double2 clamp(double2 v, double2 a, double2 b)
{
    return make_double2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ double3 clamp(double3 v, double a, double b)
{
    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ double3 clamp(double3 v, double3 a, double3 b)
{
    return make_double3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ double4 clamp(double4 v, double a, double b)
{
    return make_double4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ double4 clamp(double4 v, double4 a, double4 b)
{
    return make_double4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
//******************************************************************************
inline __host__ __device__ double dot(double2 a, double2 b)
{
    return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ double dot(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double dot(double4 a, double4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// mul - multiplication with vectors and matrices
////////////////////////////////////////////////////////////////////////////////

// mutiplication matrix x vector
inline __host__ __device__ float2 mul(float2x2 M, float2 v)
{
	float2 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	return r;
}

inline __host__ __device__ float3 mul(float3x3 M, float3 v)
{
	float3 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	r.z = dot (M.r2, v);
	return r;
}

inline __host__ __device__ float4 mul(float3x3 M, float4 v)
{
	float4 r; float3 v2=make_float3(v.x, v.y, v.z);
	r.x = dot (M.r0, v2);
	r.y = dot (M.r1, v2);
	r.z = dot (M.r2, v2);
	//r.w = 0.;
	r.w = v.w;
	return r;
}

inline __host__ __device__ float4 mul(float4x4 M, float4 v)
{
	float4 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	r.z = dot (M.r2, v);
	r.w = dot (M.r3, v);
	return r;
}

inline __host__ __device__ float2x2 mul2(float2x2 M, float f)
{
	float2x2 M2 = make_float2x2(
		M[0][0]*f,  M[0][1]*f,
		M[1][0]*f,  M[1][1]*f
		);
	return M2;
}

inline __host__ __device__ float3x3 mul2(float3x3 M, float f)
{
	float3x3 M2 = make_float3x3(
		M[0][0]*f,  M[0][1]*f, M[0][2]*f,
		M[1][0]*f,  M[1][1]*f, M[1][2]*f,
		M[2][0]*f,  M[2][1]*f, M[2][2]*f
		);
	return M2;
}

inline __host__ __device__ float4x4 mul2(float4x4 M, float f)
{
	float4x4 M2 = make_float4x4(
		M[0][0]*f,  M[0][1]*f, M[0][2]*f, M[0][3]*f,
		M[1][0]*f,  M[1][1]*f, M[1][2]*f, M[1][3]*f,
		M[2][0]*f,  M[2][1]*f, M[2][2]*f, M[2][3]*f,
		M[3][0]*f,  M[3][1]*f, M[3][2]*f, M[3][3]*f
		);
	return M2;
}

// mutiplication matrix x scalar
inline __host__ __device__ float3x3 mul(float3x3 &M, float f)
{
	float3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M[i][j] * f;
		}
	}
	return r;
}

inline __host__ __device__ float4x4 mul(float4x4 &M, float f)
{
	float4x4 r;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			r[i][j] = M[i][j] * f;
		}
	}
	return r;
}

// addition matrix + matrix
inline __host__ __device__ float2x2 add(const float2x2 &M1, const float2x2 &M2)
{
	float2x2 r;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			r[i][j] = M1[i][j] + M2[i][j];
		}
	}
	return r;
}

inline __host__ __device__ float3x3 add(const float3x3 &M1, const float3x3 &M2)
{
	float3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M1[i][j] + M2[i][j];
		}
	}
	return r;
}

inline __host__ __device__ float4x4 add(const float4x4 &M1, const float4x4 &M2)
{
	float4x4 r;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			r[i][j] = M1[i][j] + M2[i][j];
		}
	}
	return r;
}

inline __host__ __device__ double3x3 add(const double3x3 &M1, const double3x3 &M2)
{
	double3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M1[i][j] + M2[i][j];
		}
	}
	return r;
}
//******************************************************************************
// mutiplication matrix x scalar
inline __host__ __device__ double3x3 mul(double3x3 &M, double f)
{
	double3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M[i][j] * f;
		}
	}
	return r;
}

inline __host__ __device__ double2 mul(double2x2 M, double2 v)
{
	double2 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	return r;
}

inline __host__ __device__ double3 mul(double3x3 M, double3 v)
{
	double3 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	r.z = dot (M.r2, v);
	return r;
}

inline __host__ __device__ double4 mul(double3x3 M, double4 v)
{
	double4 r; double3 v2=make_double3(v.x, v.y, v.z);
	r.x = dot (M.r0, v2);
	r.y = dot (M.r1, v2);
	r.z = dot (M.r2, v2);
	//r.w = 0.;
	r.w = v.w;
	return r;
}

inline __host__ __device__ double4 mul(double4x4 M, double4 v)
{
	double4 r;
	r.x = dot (M.r0, v);
	r.y = dot (M.r1, v);
	r.z = dot (M.r2, v);
	r.w = dot (M.r3, v);
	return r;
}

// mutiplication matrix x matrix
inline __host__ __device__ float2x2 mul(const float2x2 &M1, const float2x2 &M2)
{
	float2x2 r;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j];
		}
	}
	return r;
}

inline __host__ __device__ float3x3 mul(const float3x3 &M1, const float3x3 &M2)
{
	float3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j] +
				M1[i][2] * M2[2][j];
		}
	}
	return r;
}

inline __host__ __device__ float4x4 mul(const float4x4 &M1, const float4x4 &M2)
{
	float4x4 r;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j] +
				M1[i][2] * M2[2][j] + M1[i][3] * M2[3][j];
		}
	}
	return r;
}

//******************************************************************************
inline __host__ __device__ double2x2 mul(const double2x2 &M1, const double2x2 &M2)
{
	double2x2 r;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j];
		}
	}
	return r;
}

inline __host__ __device__ double3x3 mul(const double3x3 &M1, const double3x3 &M2)
{
	double3x3 r;
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j] +
				M1[i][2] * M2[2][j];
		}
	}
	return r;
}

inline __host__ __device__ double4x4 mul(const double4x4 &M1, const double4x4 &M2)
{
	double4x4 r;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			r[i][j] = M1[i][0] * M2[0][j] + M1[i][1] * M2[1][j] +
				M1[i][2] * M2[2][j] + M1[i][3] * M2[3][j];
		}
	}
	return r;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}
// Enable to calcule the lengh between two points
inline __host__ __device__ float length(float3 p1, float3 p2)
{
	float3 v;
	v.x = p1.x-p2.x; v.y = p1.y-p2.y; v.z = p1.z-p2.z;
    return sqrtf(dot(v, v));
}

//******************************************************************************
inline __host__ __device__ double length(double2 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double3 v)
{
    return sqrt(dot(v, v));
}
inline __host__ __device__ double length(double4 v)
{
    return sqrt(dot(v, v));
}
// Enable to calcule the lengh between two points
inline __host__ __device__ double length(double3 p1, double3 p2)
{
	double3 v;
	v.x = p1.x-p2.x; v.y = p1.y-p2.y; v.z = p1.z-p2.z;
    return sqrt(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

//******************************************************************************
inline __host__ __device__ double2 normalize(double2 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double4 normalize(double4 v)
{
    double invLen = rsqrt(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

//******************************************************************************
inline __host__ __device__ double2 floord(double2 v)
{
    return make_double2(floor(v.x), floor(v.y));
}
inline __host__ __device__ double3 floord(double3 v)
{
    return make_double3(floor(v.x), floor(v.y), floor(v.z));
}
inline __host__ __device__ double4 floord(double4 v)
{
    return make_double4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

//******************************************************************************
inline __host__ __device__ double fracd(double v)
{
    return v - floor(v);
}
inline __host__ __device__ double2 fracd(double2 v)
{
    return make_double2(fracd(v.x), fracd(v.y));
}
inline __host__ __device__ double3 fracd(double3 v)
{
    return make_double3(fracd(v.x), fracd(v.y), fracd(v.z));
}
inline __host__ __device__ double4 fracd(double4 v)
{
    return make_double4(fracd(v.x), fracd(v.y), fracd(v.z), fracd(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

//******************************************************************************
inline __host__ __device__ double2 fmodd(double2 a, double2 b)
{
    return make_double2(fmod(a.x, b.x), fmod(a.y, b.y));
}
inline __host__ __device__ double3 fmodd(double3 a, double3 b)
{
    return make_double3(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z));
}
inline __host__ __device__ double4 fmodd(double4 a, double4 b)
{
    return make_double4(fmod(a.x, b.x), fmod(a.y, b.y), fmod(a.z, b.z), fmod(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabsf(v.x), fabsf(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

//******************************************************************************
inline __host__ __device__ double2 fabs(double2 v)
{
    return make_double2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ double3 fabs(double3 v)
{
    return make_double3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ double4 fabs(double4 v)
{
    return make_double4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

//******************************************************************************
inline __host__ __device__ double3 reflect(double3 i, double3 n)
{
    return i - 2.0 * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

//******************************************************************************
inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

//******************************************************************************
inline __device__ __host__ double smoothstep(double a, double b, double x)
{
    double y = clamp((x - a) / (b - a), 0.0, 1.0);
    return (y*y*(3.0 - (2.0*y)));
}
inline __device__ __host__ double2 smoothstep(double2 a, double2 b, double2 x)
{
    double2 y = clamp((x - a) / (b - a), 0.0, 1.0);
    return (y*y*(make_double2(3.0) - (make_double2(2.0)*y)));
}
inline __device__ __host__ double3 smoothstep(double3 a, double3 b, double3 x)
{
    double3 y = clamp((x - a) / (b - a), 0.0, 1.0);
    return (y*y*(make_double3(3.0) - (make_double3(2.0)*y)));
}
inline __device__ __host__ double4 smoothstep(double4 a, double4 b, double4 x)
{
    double4 y = clamp((x - a) / (b - a), 0.0, 1.0);
    return (y*y*(make_double4(3.0) - (make_double4(2.0)*y)));
}

////////////////////////////////////////////////////////////////////////////////
// Simple swap function
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ void swap(float *a, float *b)
{
	float temp = *a;
	*a = *b;
	*b = temp;
}

//******************************************************************************
inline __device__ __host__ void swap(double *a, double *b)
{
	double temp = *a;
	*a = *b;
	*b = temp;
}

////////////////////////////////////////////////////////////////////////////////
// Quadratic (At² + Bt + C = 0)
// - return true if there exist quadratic values
// - Modify t0 and t1 by the computed ones
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ bool quadratic(float *t0, float *t1,
										  float A, float B, float C)
{
    // Trouver le discriminant quadratique
	double discrim = (double)B * (double)B - 4 * (double)A * (double)C;
	if (discrim < 0) {return false;}
	double rootDiscrim = std::sqrt(discrim);

	// Calculer les valeurs de t0 et t1
	double q;
	if (B < 0) {q = -.5 * (B - rootDiscrim);}
	else {q = -.5 * (B + rootDiscrim);}
	*t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1) {swap(t0, t1);}
    return true;
}

//******************************************************************************
inline __device__ __host__ bool quadratic(double *t0, double *t1,
										  double A, double B, double C)
{
    // Trouver le discriminant quadratique
	double discrim = B * B - 4 * A * C;
	if (discrim < 0) {return false;}
	double rootDiscrim = sqrt(discrim);

	// Calculer les valeurs de t0 et t1
	double q;
	if (B < 0) {q = -.5 * (B - rootDiscrim);}
	else {q = -.5 * (B + rootDiscrim);}
	*t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1) {swap(t0, t1);}
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// FaceForward
//  Flip the Vector/Normal a if the Vector/Normal b is in the opposite direction.
//  For exemple, it can be useful to flip a surface normal so that it lies in the
//  same hemisphere as a given vector.
// - Args : Vector or Normal a, b
// - Output : Possibly fliped Vector or Normal a
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float3 faceForward(float3 a, float3 b)
{
	return (dot(a, b) < 0.f) ? a*(-1) : a;
}

inline __device__ __host__ bool isForward(float3 a, float3 b)
{
	return (dot(a, b) > 0.f);
}

inline __device__ __host__ bool isBackward(float3 a, float3 b)
{
	return (dot(a, b) < 0.f);
}

//******************************************************************************
inline __device__ __host__ double3 faceForward(double3 a, double3 b)
{
	return (dot(a, b) < 0.) ? a*(-1) : a;
}

inline __device__ __host__ bool isForward(double3 a, double3 b)
{
	return (dot(a, b) > 0.);
}

inline __device__ __host__ bool isBackward(double3 a, double3 b)
{
	return (dot(a, b) < 0.);
}

////////////////////////////////////////////////////////////////////////////////
// radians
// - convert degree to radians
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float radians(float deg) {
    return (PI/180.f) * deg;
}

inline __device__ __host__ double radiansd(double deg) {
    return (CUDART_PI/180.) * deg;
}

////////////////////////////////////////////////////////////////////////////////
// Inverse using Gauss-Jordan elimination
// - compute the inverse of square matrix (here 4x4 matrix)
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float4x4 inverse(const float4x4 &m)
{
	int4c indxc, indxr;
	int4c ipiv = make_int4c(0., 0., 0., 0.);
	float4x4 minv;
	minv = m;
	for (int i =0; i < 4; i++)
	{
		int irow = -1, icol = -1;
		float big =0.;
		// choice of the pivot
		for (int j = 0; j < 4; j++)
		{
			if (ipiv[j] != 1)
			{
				for (int k = 0; k < 4; k++)
				{
					if (ipiv[k] == 0)
					{
						if (fabsf(minv[j][k]) >= big)
						{
							big = float(fabsf(minv[j][k]));
                            irow = j;
                            icol = k;
						}
					}
					else if (ipiv[k] > 1)
                        asm("trap;");
				}
			}
		}
		++ipiv[icol];
        if (irow != icol)
		{
            for (int k = 0; k < 4; ++k)
                swap(&minv[irow][k], &minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.)
            asm("trap;");

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        float pivinv = 1.f / minv[icol][icol];
        minv[icol][icol] = 1.f;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++)
		{
            if (j != icol)
			{
                float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++)
                    minv[j][k] -= minv[icol][k]*save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
                swap(&minv[k][indxr[j]], &minv[k][indxc[j]]);
        }
    }
    return minv;
}

//******************************************************************************
inline __device__ __host__ double4x4 inverse(const double4x4 &m)
{
	int4c indxc, indxr;
	int4c ipiv = make_int4c(0., 0., 0., 0.);
	double4x4 minv;
	minv = m;
	for (int i =0; i < 4; i++)
	{
		int irow = -1, icol = -1;
		double big =0.;
		// choice of the pivot
		for (int j = 0; j < 4; j++)
		{
			if (ipiv[j] != 1)
			{
				for (int k = 0; k < 4; k++)
				{
					if (ipiv[k] == 0)
					{
						if (fabs(minv[j][k]) >= big)
						{
							big = double(fabs(minv[j][k]));
                            irow = j;
                            icol = k;
						}
					}
					else if (ipiv[k] > 1)
                        asm("trap;");
				}
			}
		}
		++ipiv[icol];
        if (irow != icol)
		{
            for (int k = 0; k < 4; ++k)
                swap(&minv[irow][k], &minv[icol][k]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.)
            asm("trap;");

        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        double pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;

        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++)
		{
            if (j != icol)
			{
                double save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++)
                    minv[j][k] -= minv[icol][k]*save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
                swap(&minv[k][indxr[j]], &minv[k][indxc[j]]);
        }
    }
    return minv;
}

////////////////////////////////////////////////////////////////////////////////
// Compare two string (works also in the device)
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ bool compStr(const char* s1, const char* s2)
{
	int i = 0;
	do
	{
		if (s1[i] != s2[i])
			return false;
		i++;
	}
	while (s1[i] != '\0' && s2[i] != '\0');

	if (s1[i] == '\0' && s2[i] == '\0')
		return true;

	return false;
}

////////////////////////////////////////////////////////////////////////////////
// Matrix transpose
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2x2 transpose(float2x2 m)
{
    return make_float2x2(
		m[0][0], m[1][0],
		m[0][1], m[1][1]
		);
}

inline __host__ __device__ float3x3 transpose(float3x3 m)
{
    return make_float3x3(
		m[0][0], m[1][0], m[2][0],
		m[0][1], m[1][1], m[2][1],
		m[0][2], m[1][2], m[2][2]
		);
}

inline __host__ __device__ float4x4 transpose(float4x4 m)
{
    return make_float4x4(
		m[0][0], m[1][0], m[2][0], m[3][0],
		m[0][1], m[1][1], m[2][1], m[3][1],
		m[0][2], m[1][2], m[2][2], m[3][2],
		m[0][3], m[1][3], m[2][3], m[3][3]
		);
}

//******************************************************************************
inline __host__ __device__ double2x2 transpose(double2x2 m)
{
    return make_double2x2(
		m[0][0], m[1][0],
		m[0][1], m[1][1]
		);
}

inline __host__ __device__ double3x3 transpose(double3x3 m)
{
    return make_double3x3(
		m[0][0], m[1][0], m[2][0],
		m[0][1], m[1][1], m[2][1],
		m[0][2], m[1][2], m[2][2]
		);
}

inline __host__ __device__ double4x4 transpose(double4x4 m)
{
    return make_double4x4(
		m[0][0], m[1][0], m[2][0], m[3][0],
		m[0][1], m[1][1], m[2][1], m[3][1],
		m[0][2], m[1][2], m[2][2], m[3][2],
		m[0][3], m[1][3], m[2][3], m[3][3]
		);
}

////////////////////////////////////////////////////////////////////////////////
// CoordinateSystem function
// - enable to get an orthogonal coordinate from a given vector
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ void coordinateSystem(const float3 &v1, float3 *v2, float3 *v3)
{
    if (fabsf(v1.x) > fabsf(v1.y))
	{
        float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
        *v2 = make_float3(-v1.z * invLen, 0.f, v1.x * invLen);
    }
    else
	{
        float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
        *v2 = make_float3(0.f, v1.z * invLen, -v1.y * invLen);
    }
    *v3 = cross(v1, *v2);
}

//******************************************************************************
inline __host__ __device__ void coordinateSystem(const double3 &v1, double3 *v2, double3 *v3)
{
    if (fabs(v1.x) > fabs(v1.y))
	{
        double invLen = 1. / sqrt(v1.x*v1.x + v1.z*v1.z);
        *v2 = make_double3(-v1.z * invLen, 0., v1.x * invLen);
    }
    else
	{
        double invLen = 1. / sqrt(v1.y*v1.y + v1.z*v1.z);
        *v2 = make_double3(0., v1.z * invLen, -v1.y * invLen);
    }
    *v3 = cross(v1, *v2);
}

////////////////////////////////////////////////////////////////////////////////
// computePsiFN function
// - Compute the angle psi of the rotation matrix given the incident direction
//   v, the incident perp direction u, the normal m of the surface impact and
//   the sinus of the angle between the normal m and the direction v sTheta
////////////////////////////////////////////////////////////////////////////////

inline __device__ float computePsiFN(float3 u, float3 v, float3 m, float sTheta)
{
	float3 crossUV = cross(u, v);
	float psi;

    psi = __fdividef(dot(m, u), sTheta);
	psi = acosf( clamp(psi, -1.F, 1.F) );
	
    return (dot(m, crossUV) < 0) ? -psi : psi;
}

////////////////////////////////////////////////////////////////////////////////
// computeRefMat function
// - Compute the reflection matrix given refraction index, the cosinus and sinus
//   of the angle theta (between the normal of the surface impact and the dir v)
////////////////////////////////////////////////////////////////////////////////

inline __device__ float4x4 computeRefMat(float nind, float cTheta, float sTheta)
{
	float cot, ncot, ncTh, rpar, rper, rpar2, rper2, rparper;

	cot = __fdividef(sTheta, nind);
	cot = sqrtf( 1.0F - cot*cot );
	ncTh = nind*cTheta;
	ncot = nind*cot;
	rpar = __fdividef(ncTh-cot, ncTh+cot); // DR Mobley 2015 sign convention
	rper = __fdividef(cTheta-ncot, cTheta+ncot);
	rpar2 = rpar*rpar;
	rper2 = rper*rper;
	rparper = rpar * rper;

	// Fill and return the reflection matrix
	return make_float4x4(
		rpar2, 0.   , 0.     , 0.     ,
		0.   , rper2, 0.     , 0.     ,
		0.   , 0.   , rparper, 0.     ,
		0.   , 0.   , 0.     , rparper 
		);
}

//******************************************************************************
inline __device__ void refMat(float nind, float cTheta, float sTheta, float4x4 *R, float *scaR)
{
	float cot, ncot, ncTh, rpar, rper, rpar2, rper2, rparper;

	cot = __fdividef(sTheta, nind);
	cot = sqrtf( 1.0F - cot*cot );
	ncTh = nind*cTheta;
	ncot = nind*cot;
	rpar = __fdividef(ncTh-cot, ncTh+cot); // DR Mobley 2015 sign convention
	rper = __fdividef(cTheta-ncot, cTheta+ncot);
	rpar2 = rpar*rpar;
	rper2 = rper*rper;
	rparper = rpar * rper;

	// Fill the reflection matrix
    *R= make_float4x4(
		rpar2, 0.   , 0.     , 0.     ,
		0.   , rper2, 0.     , 0.     ,
		0.   , 0.   , rparper, 0.     ,
		0.   , 0.   , 0.     , rparper 
		);

	*scaR = __fdividef(rpar2+rper2, 2.F);
}

////////////////////////////////////////////////////////////////////////////////
// specularFN function
// - Compute the specular reflection by giving the incoming direction and the
//   normal of the surface (and the cosine of the angle bet v and the normal)
////////////////////////////////////////////////////////////////////////////////

inline __device__ float3 specularFN(float3 vi, float3 n)
{
	float3 v = make_float3(-vi.x, -vi.y, -vi.z);
	return vi + n*(2*dot(n, v));
}

inline __device__ float3 specularFNC(float3 vi, float3 n, float cTheta)
{
	return vi + (2.F*cTheta)*n;
}

////////////////////////////////////////////////////////////////////////////////
// perfect_mirrorRM function
////////////////////////////////////////////////////////////////////////////////
inline __device__ float4x4 perfect_mirrorRF()
{
	// Fill and return the reflection matrix of a perfect mirror
	return make_float4x4(
		1.F, 0. , 0.  , 0.  ,
		0. , 1.F, 0.  , 0.  ,
		0. , 0. , -1.F, 0.  ,
		0. , 0. , 0.  , -1.F 
		);
}

////////////////////////////////////////////////////////////////////////////////
// Gamma function of limb model (see Koepke et al. 2000)
////////////////////////////////////////////////////////////////////////////////
inline __device__ double GammaL( double lambd, double r)
{
	double Beta, G;
	Beta = ( 3*(6.63e-34)*(2.998e+8)*(pow(2., 0.25)) )/( 8*(1.38e-23)*(lambd*1e-9)*5740 );

	G = ( 1 + (Beta*sqrt(1- r*r)) ) / (1+Beta);
	return G;
}

////////////////////////////////////////////////////////////////////////////////
// other useful functions
////////////////////////////////////////////////////////////////////////////////
__device__ float Gamma_eps(int n, float eps)
{
	return (n * eps) / (1 - n * eps);
}

__device__ float3 Permute(float3 v, int x, int y, int z)
{
	float3c u = make_float3c(v.x, v.y, v.z);
	return make_float3(u[x], u[y], u[z]);
}

__device__ int MaxDim(float3 v)
{
	return (v.x>v.y) ? ((v.x>v.z)?0:2) : ((v.y>v.z)?1:2);
}
__device__ double Gamma_eps(int n, double eps)
{
	return (n * eps) / (1 - n * eps);
}

__device__ double3 Permute(double3 v, int x, int y, int z)
{
	double3c u = make_double3c(v.x, v.y, v.z);
	return make_double3(u[x], u[y], u[z]);
}

__device__ int MaxDim(double3 v)
{
	return (v.x>v.y) ? ((v.x>v.z)?0:2) : ((v.y>v.z)?1:2);
}
#endif

