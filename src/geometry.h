
#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <math.h>
#include <helper_math.h>
#include <math_constants.h>
#include <limits>
#include <stdio.h>


/**********************************************************
*	> Classe(s) représentant géométriquement quelque chose
*     - ex: Un Point, Un vecteur, une normal, un rayon...
***********************************************************/


class Ray
// ========================================================
// Classe Rayon
// ========================================================
{
public:
	// Méthodes publiques du rayon
	__host__ __device__ Ray()
	{
        #if __CUDA_ARCH__ >= 200
		maxt = CUDART_INF_F;
        #elif !defined(__CUDA_ARCH__)
		maxt = std::numeric_limits<float>::max();
        #endif
		mint = 0.f; time = 0.f;
		o = make_float3c(0., 0., 0.);
		d = make_float3c(0., 0., 0.);
	}

	__host__ __device__ Ray(const Ray &r)
	{
		mint = r.mint; maxt = r.maxt, time = r.time;
		o = r.o; d = r.d;
	}

    #if __CUDA_ARCH__ >= 200
	__device__ Ray(const float3 &origin, const float3 &direction,
				   float start, float end = CUDART_INF_F, float t = 0.f)
	{
		mint = start; maxt = end, time = t;
		o = make_float3c(origin);
		d = make_float3c(direction);
	}
    #elif !defined(__CUDA_ARCH__)
    __host__ Ray(const float3 &origin, const float3 &direction,
				 float start, float end = std::numeric_limits<float>::max(),
				 float t = 0.f)
	{
		mint = start; maxt = end, time = t;
		o = make_float3c(origin);
		d = make_float3c(direction);
	}
    #endif

    __host__ __device__ float3 operator()(float t) const
	{
		return o + d*t;
	}

	// Paramètres publiques du rayon
	float3c o;           // point d'origine du rayon
	float3c d;           // vecteur de direction du rayon
	float mint, maxt;    // valeur min et max de t
	float time;          // variable t: ray = o + d*t
 
private:
};


class BBox
// ========================================================
// Classe BBox
// ========================================================
{
public:
	// Méthodes publiques de la Box
	__host__ __device__ BBox()
	{
        #if __CUDA_ARCH__ >= 200
		pMin = make_float3c( CUDART_INF_F,  CUDART_INF_F,  CUDART_INF_F);
		pMax = make_float3c(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
        #elif !defined(__CUDA_ARCH__)
		pMin = make_float3c(std::numeric_limits<float>::max(),
							std::numeric_limits<float>::max(),
							std::numeric_limits<float>::max());
		pMax = make_float3c(-std::numeric_limits<float>::max(),
							-std::numeric_limits<float>::max(),
							-std::numeric_limits<float>::max());
        #endif
	}
    __host__ __device__ BBox(const float3 &p)
		: pMin(make_float3c(p)), pMax(make_float3c(p)) { }
	__host__ __device__ BBox(const float3 &p1, const float3 &p2)
	{
        pMin = make_float3c(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
        pMax = make_float3c(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
    }
    __host__ __device__ bool Inside(const float3 &pt) const
	{
        return (pt.x >= pMin.x && pt.x <= pMax.x &&
                pt.y >= pMin.y && pt.y <= pMax.y &&
                pt.z >= pMin.z && pt.z <= pMax.z);
    }
	__host__ __device__ bool IntersectP(const Ray &ray) const
	{
		float t0 = 0., t1 = ray.maxt;

		for (int i = 0; i < 3; ++i)
		{
            // Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;
			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) {swap(&tNear, &tFar);}
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) {return false;}
		}
		return true;
	}

	// Paramètres publiques de la Box
    float3c pMin, pMax; // point min et point max
private:
};

#endif // _GEOMETRY_H_
