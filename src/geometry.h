
#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <math.h>
#include <helper_math.h>
#include <stdio.h>


/**********************************************************
*	> Classes/structures liées à l'étude de géométries
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
		mint = 0.f; maxt = INFINITY, time = 0.f;
		o = make_float3(0., 0., 0.);
		d = make_float3(0., 0., 0.);
	}

	__host__ __device__ Ray(const Ray &r)
	{
		mint = r.mint; maxt = r.maxt, time = r.time;
		o = r.o; d = r.d;
	}

    __host__ __device__ Ray(const float3 &origin, const float3 &direction,
						    float start, float end = INFINITY, float t = 0.f)
	{
		mint = start; maxt = end, time = t;
		o = origin;
		d = direction;
	}

    __host__ __device__ float3 operator()(float t) const
	{
		return o + d*t;
	}

	// Paramètres publiques du rayon
	float3 o;            // point d'origine du rayon
	float3 d;            // vecteur de direction du rayon
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
		pMin = make_float3( INFINITY,  INFINITY,  INFINITY);
		pMax = make_float3(-INFINITY, -INFINITY, -INFINITY);
	}
    __host__ __device__ BBox(const float3 &p) : pMin(p), pMax(p) { }
	__host__ __device__ BBox(const float3 &p1, const float3 &p2)
	{
        pMin = make_float3(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
        pMax = make_float3(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
    }
    __host__ __device__ bool Inside(const float3 &pt) const
	{
        return (pt.x >= pMin.x && pt.x <= pMax.x &&
                pt.y >= pMin.y && pt.y <= pMax.y &&
                pt.z >= pMin.z && pt.z <= pMax.z);
    }
	__host__ __device__ bool IntersectP(const Ray &ray) const
	{
		float t0 = ray.mint, t1 = ray.maxt;
		float invRayDir, tNear, tFar;

		// Update interval for _i_th bounding box slab
		invRayDir = 1.f / ray.d.x;
		tNear = (pMin.x - ray.o.x) * invRayDir;
		tFar  = (pMax.x - ray.o.x) * invRayDir;

		// Update parametric interval from slab intersection $t$s
		if (tNear > tFar) swap(&tNear, &tFar);
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar  < t1 ? tFar  : t1;
		if (t0 > t1) return false;

		// Update interval for _i_th bounding box slab
		invRayDir = 1.f / ray.d.y;
		tNear = (pMin.y - ray.o.y) * invRayDir;
		tFar  = (pMax.y - ray.o.y) * invRayDir;

		// Update parametric interval from slab intersection $t$s
		if (tNear > tFar) swap(&tNear, &tFar);
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar  < t1 ? tFar  : t1;
		if (t0 > t1) return false;

		// Update interval for _i_th bounding box slab
		invRayDir = 1.f / ray.d.z;
		tNear = (pMin.z - ray.o.z) * invRayDir;
		tFar  = (pMax.z - ray.o.z) * invRayDir;

		// Update parametric interval from slab intersection $t$s
		if (tNear > tFar) swap(&tNear, &tFar);
		t0 = tNear > t0 ? tNear : t0;
		t1 = tFar  < t1 ? tFar  : t1;
		if (t0 > t1) return false;

		return true;
	}
	// Paramètres publiques de la Box
    float3 pMin, pMax; // point min et point max
private:
};

#endif // _GEOMETRY_H_
