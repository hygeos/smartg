
#ifndef _SHAPES_H_
#define _SHAPES_H_

#include "geometry.h"
#include "transform.h"

#include <math.h>
#include <helper_math.h>
#include <stdio.h>


/**********************************************************
*	> Classes/structures liées à l'étude de géométries
***********************************************************/


// Variable globale (coté device)
// - La fonction kernel est appelée à plusieurs reprises, il est donc
// - nécessaire d'initialiser la variable au début de la fonction.
__device__ static unsigned long int bigCount;

class Shape // doit être défini avant la structure DifferentialGeometry
// ========================================================
// Classe Shape : commun avec toute les géométries
// ========================================================
{
public:
    // Méthodes publiques
	__host__ __device__ Shape(const Transform *o2w, const Transform *w2o)
		: ObjectToWorld(o2w), WorldToObject(w2o)
	{
        #if __CUDA_ARCH__ >= 200
		shapeId = bigCount;
		bigCount++;
		#elif !defined(__CUDA_ARCH__)
		shapeId++;
        #endif
	}
    __host__ __device__ virtual BBox ObjectBound() const = 0;
    __host__ __device__ virtual BBox WorldBound() const
	{
		return (*ObjectToWorld)(ObjectBound());
	}

    // Paramètres publiques
    #if __CUDA_ARCH__ >= 200
	unsigned long int shapeId;
    #elif !defined(__CUDA_ARCH__)
	static unsigned long int shapeId;
    #endif
    const Transform *ObjectToWorld, *WorldToObject;

private:
};

#if !defined(__CUDA_ARCH__)
unsigned long int Shape::shapeId = 1;
#endif


struct DifferentialGeometry // doit être défini avant les classes filles de Shape
// ========================================================
// Struture d'une géomértie différentielle
// ========================================================
{
    __host__ __device__ DifferentialGeometry()
	{
		p = make_float3(0, 0, 0); dpdu = make_float3(0, 0, 0);
		dpdv = make_float3(0, 0, 0); nn = make_float3(0, 0, 0);
		u = v = 0.; shape = NULL;
	}

    __host__ __device__ DifferentialGeometry(const float3 &P, const float3 &DPDU,
						 const float3 &DPDV, float uu, float vv, const Shape *sh)
		: p(P), dpdu(DPDU), dpdv(DPDV)
	{
		nn = normalize(cross(dpdu, dpdv));
		u = uu;
		v = vv;
		shape = sh;
	}

	float3 p;             // Position du point d'intersection
	float3 dpdu;          // Dérivée partielle par rapport à u
	float3 dpdv;          // Dérivée partielle par rapport à v
	float3 nn;            // Normal au point d'intersection
	float u;              // parmètre: P=f(u,v)
	float v;              // parmètre: P=f(u,v)
	const Shape *shape;   // la géométrie utilisé
};


class Sphere : public Shape
// ========================================================
// Classe Sphere
// ========================================================
{
public:
	// Méthodes publiques de la sphère
	__host__ __device__ Sphere(const Transform *o2w, const Transform *w2o,
							   float rad, float zmin, float zmax, float phiMax);
    __host__ __device__ virtual BBox ObjectBound() const;
    __host__ __device__ bool Intersect(const Ray &ray, float* tHit,
									   DifferentialGeometry *Dg) const;
    __host__ __device__ float Area() const;

private:
	// Paramètres privés de la sphère
	float radius;               // Rayon de la sphere
    float phiMax;               // phimax = 360 pour une sphere pleine
    float zmin, zmax;           // zmin = (-1)*zmax = (-1)*rayon pour une sphere pleine
    float thetaMin, thetaMax;   // calculées en fonction de zmin et zmax
};

// -------------------------------------------------------
// définitions des méthodes de la classe sphere
// -------------------------------------------------------
	Sphere::Sphere(const Transform *o2w, const Transform *w2o,
				   float rad, float z0, float z1, float pm)
	: Shape(o2w, w2o)
{
    radius = rad;
    zmin = clamp(min(z0, z1), -radius, radius);
    zmax = clamp(max(z0, z1), -radius, radius);
    thetaMin = acosf(clamp(zmin/radius, -1.f, 1.f));
    thetaMax = acosf(clamp(zmax/radius, -1.f, 1.f));
    phiMax = radians(clamp(pm, 0.0f, 360.0f));
}

BBox Sphere::ObjectBound() const
{	
	if (phiMax < PI/2)
	{
		return BBox(make_float3( 0.f, 0.f, zmin),
					make_float3( radius, radius*sinf(phiMax), zmax));
	}
	else if (phiMax < PI)
	{
		return BBox(make_float3( radius*cosf(phiMax), 0.f, zmin),
					make_float3( radius, radius, zmax));
	}
	else if (phiMax < 3*PI/2)
	{
	    return BBox(make_float3(-radius, radius*sinf(phiMax), zmin),
					make_float3( radius,  radius, zmax));
	}
	else //if (phiMax >= 3*PI/2)
	{
	    return BBox(make_float3(-radius, -radius, zmin),
					make_float3( radius,  radius, zmax));
	}
}

bool Sphere::Intersect(const Ray &r, float *tHit, DifferentialGeometry *dg) const
{
    float phi;
    float3 phit;

	Ray ray;
    // Passage (transform) du rayon "ray" dans l'espace de l'objet
	(*WorldToObject)(r, &ray);

	// printf("ray(%f) = (%f, %f, %f)\n", ray.maxt, ray(ray.maxt).x, ray(ray.maxt).y, ray(ray.maxt).z);
    // Calcul des coefficients quadratiques de la sphere
    float A = ray.d.x*ray.d.x + ray.d.y*ray.d.y + ray.d.z*ray.d.z;
    float B = 2 * (ray.d.x*ray.o.x + ray.d.y*ray.o.y + ray.d.z*ray.o.z);
    float C = ray.o.x*ray.o.x + ray.o.y*ray.o.y +
              ray.o.z*ray.o.z - radius*radius;

    // Résoudre l'équation du second degrée pour obtenir t0 et t1
    float t0, t1;
    if (!quadratic(&t0, &t1, A, B, C))
        return false;

    // Calcul à quel temps t le rayon intersecte la sphere
    if (t0 > ray.maxt || t1 < ray.mint)
        return false;
    float thit = t0;
    if (t0 < ray.mint)
	{
        thit = t1;
        if (thit > ray.maxt) {return false;}
    }

    // Calcul la position de l'intersection ainsi que la valeur de $\phi$
    phit = ray(thit);
    if (phit.x == 0.f && phit.y == 0.f) {phit.x = 1e-5f * radius;}
    phi = atan2f(phit.y, phit.x);
    if (phi < 0.) {phi += 2.f*PI;}

    // Prendre en compte les paramètres d'une sphere partiel
    if ((zmin > -radius && phit.z < zmin) ||
        (zmax <  radius && phit.z > zmax) || phi > phiMax)
	{
        if (thit == t1) {return false;}
        if (t1 > ray.maxt) {return false;}
        thit = t1;
        // Calcul la position ainsi que la valeur de $\phi$
        phit = ray(thit);
        if (phit.x == 0.f && phit.y == 0.f) {phit.x = 1e-5f * radius;}
        phi = atan2f(phit.y, phit.x);
        if (phi < 0.) {phi += 2.f*PI;}
        if ((zmin > -radius && phit.z < zmin) ||
            (zmax <  radius && phit.z > zmax) || phi > phiMax)
            return false;
    }

    // Trouve la représentation paramétrique au point d'intersection
    float u = phi / phiMax;
    float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
    float v = (theta - thetaMin) / (thetaMax - thetaMin);

    // Calcul des dérivées partielles $\dpdu$ et $\dpdv$
    float zradius = sqrtf(phit.x*phit.x + phit.y*phit.y);
    float invzradius = 1.f / zradius;
    float cosphi = phit.x * invzradius;
    float sinphi = phit.y * invzradius;
    float3 dpdu = make_float3(-phiMax * phit.y, phiMax * phit.x, 0);
    float3 dpdv = make_float3(phit.z * cosphi, phit.z * sinphi,
							  -radius * sinf(theta)) * (thetaMax-thetaMin);

    // Initialisation de  _DifferentialGeometry_ depuis les données paramétriques
    const Transform &o2w = *ObjectToWorld;
	char myP[]="Point", myN[]="Normal";
    *dg = DifferentialGeometry(o2w(phit, myP), o2w(dpdu, myN), o2w(dpdv, myN), u, v, this);

    // mise a jour de _tHit_
    *tHit = thit;

    return true;
}

float Sphere::Area() const {
    return phiMax * radius * (zmax-zmin);
}
#endif // _SHAPES_H_
