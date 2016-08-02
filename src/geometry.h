
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
#endif // _GEOMETRY_H_
