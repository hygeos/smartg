
#ifndef _SHAPES_H_
#define _SHAPES_H_

#include "geometry.h"
#include "transform.h"

#include <math.h>
#include <math_constants.h>
#include <limits>
#include <helper_math.h>
#include <stdio.h>

/**********************************************************
*	> Classes/structures liées à l'étude de géométries
***********************************************************/


// Variable globale (coté device)
// - La fonction kernel est appelée à plusieurs reprises, il est donc
// - nécessaire d'initialiser la variable au début de la fonction kernel.
__device__ static unsigned long int bigCount;

class Shape // doit être défini avant la structure DifferentialGeometry
// ========================================================
// Classe Shape : commun avec toute les géométries
// ========================================================
{
public:
    // Méthodes publiques
	__host__ __device__ Shape()
	{
        #if __CUDA_ARCH__ >= 200
		shapeId = bigCount;
		bigCount++;
		#elif !defined(__CUDA_ARCH__)
		shapeId++;
        #endif

		// Initialisation avec des transformations "nulles"
		Transform nothing;
		ObjectToWorld = &nothing;
		WorldToObject = &nothing;
	}

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
	__host__ __device__ Sphere();
	__host__ __device__ Sphere(const Transform *o2w, const Transform *w2o,
							   float rad, float zmin, float zmax, float phiMax);

    /* uniquement device pour éviter des problèmes de mémoires */
    __device__ BBox ObjectBoundSphere() const;
    __device__ BBox WorldBoundSphere() const;

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
Sphere::Sphere() : Shape()
{
    radius = 0.f;
    zmin = 0.f;
    zmax = 0.f;
    thetaMin = 0.f;
    thetaMax = 0.f;
    phiMax = 0.f;
}

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

__device__ BBox Sphere::WorldBoundSphere() const
{
	return (*ObjectToWorld)(ObjectBoundSphere());
}

__device__ BBox Sphere::ObjectBoundSphere() const
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
	
    *dg = DifferentialGeometry(o2w(Pointf(phit)), o2w(Normalf(dpdu)), o2w(Normalf(dpdv)), u, v, this);

    // mise a jour de _tHit_
    *tHit = thit;

    return true;
}

float Sphere::Area() const {
    return phiMax * radius * (zmax-zmin);
}


class Triangle : public Shape
// ========================================================
// Classe triangle
// ========================================================
{
public:
	// Méthodes publiques de classe triangle
	__host__ __device__ Triangle();
	__host__ __device__ Triangle(const Transform *o2w, const Transform *w2o);
	__host__ __device__ Triangle(const Transform *o2w, const Transform *w2o,
								 float3 a, float3 b, float3 c);

    /* uniquement device pour éviter des problèmes de mémoires */
    __device__ BBox ObjectBoundTriangle() const;
    __device__ BBox WorldBoundTriangle() const;

    __host__ __device__ bool Intersect(const Ray &ray, float* tHit,
									   DifferentialGeometry *dg) const;
	__device__ bool Intersect2(const Ray &ray, float* tHit,
									   DifferentialGeometry *dg) const;
	__host__ __device__ bool IntersectP(const Ray &ray) const;
	__device__ bool IntersectP2(const Ray &ray) const;
    __host__ __device__ float Area() const;

private:
	// Paramètres privés de la classe triangle
	float3 p1, p2, p3;
};

// -------------------------------------------------------
// définitions des méthodes de la classe Triangle
// -------------------------------------------------------
Triangle::Triangle()
// Donne la possibilité de créer un tableau de triangle
{
	p1 = make_float3(0.f, 0.f, 0.f);
	p2 = make_float3(0.f, 0.f, 0.f);
	p3 = make_float3(0.f, 0.f, 0.f);
}

Triangle::Triangle(const Transform *o2w, const Transform *w2o)
	: Shape(o2w, w2o)
{
	p1 = make_float3(0.f, 0.f, 0.f);
	p2 = make_float3(0.f, 0.f, 0.f);
	p3 = make_float3(0.f, 0.f, 0.f);
}

Triangle::Triangle(const Transform *o2w, const Transform *w2o,
				   float3 a, float3 b, float3 c)
	: Shape(o2w, w2o)
{
	p1 = a; p2 =b; p3 = c;
}

__device__ bool Triangle::Intersect2(const Ray &ray, float *tHit,
									DifferentialGeometry *dg) const
{
	/* float3 p0t, p1t, p2t; */
	/* p0t = p1 - ray.o; p1t = p2 - ray.o; p2t = p3 - ray.o; */
	
	/* int kz = MaxDim( make_float3(abs(ray.d.x),abs(ray.d.y),abs(ray.d.z)) ); */
	/* int kx = kz + 1; */
	/* if(kx == 3) kx = 0; */
	/* int ky = kx+1; */
	/* if(ky == 3) ky = 0; */
	/* float3 d = Permute(ray.d, kx, ky, kz); */
	/* p0t = Permute(p0t, kx, ky, kz); */
	/* p1t = Permute(p1t, kx, ky, kz); */
	/* p2t = Permute(p2t, kx, ky, kz); */

	/* float Sx=-d.x/d.z; float Sy=-d.y/d.z; float Sz=1.F/d.z; */
	/* p0t.x += Sx*p0t.z; p0t.y += Sy*p0t.z; */
	/* p1t.x += Sx*p1t.z; p1t.y += Sy*p1t.z; */
	/* p2t.x += Sx*p2t.z; p2t.y += Sy*p2t.z; */

	/* float e0 = p1t.x * p2t.y - p1t.y * p2t.x; */
	/* float e1 = p2t.x * p0t.y - p2t.y * p0t.x; */
	/* float e2 = p0t.x * p1t.y - p0t.y * p1t.x; */

	/* if ( e0 == 0.F || e1 == 0.F || e2 == 0.F ) */
	/* { */
	/* 	double p2txp1ty = (double)p2t.x * (double)p1t.y; */
	/* 	double p2typ1tx = (double)p2t.y * (double)p1t.x; */
	/* 	e0 = (float)(p2typ1tx - p2txp1ty); */
	/* 	double p0txp2ty = (double)p0t.x * (double)p2t.y; */
	/* 	double p0typ2tx = (double)p0t.y * (double)p2t.x; */
	/* 	e1 = (float)(p0typ2tx - p0txp2ty); */
	/* 	double p1txp0ty = (double)p1t.x * (double)p0t.y; */
	/* 	double p1typ0tx = (double)p1t.y * (double)p0t.x; */
	/* 	e2 = (float)(p1typ0tx - p1txp0ty); */
	/* } */

	/* if((e0<0 || e1<0 || e2<0) && (e0>0 || e1>0 || e2>0)) */
	/* 	return false; */
	/* float det = e0 + e1 + e2; */
	/* if(det == 0) return false; */

	/* p0t *= Sz; p1t *= Sz; p2t *= Sz; */
	/* float tScaled = e0*p0t.z + e1*p1t.z + e2*p2t.z; */
	/* if(det < 0 && (tScaled >= 0 || tScaled < ray.maxt*det)) */
	/* 	return false; */
	/* else if (det > 0 && (tScaled <= 0 || tScaled > ray.maxt*det)) */
	/* 	return false; */

	/* float invDet = 1/det; */
	/* //float b0 = e0*invDet; float b1 = e1*invDet; float b2 = e2*invDet; */
	/* float t = tScaled*invDet; */

	/* if (t < ray.mint || t > ray.maxt) */
    /*     return false; */

	/* float maxZt = max( abs(p0t.z), max( abs(p1t.z), abs(p2t.z) )  ); */
	/* float eps = machine_eps_flt() * 0.5; */
	/* float deltaZ = Gamma_eps(3, eps) * maxZt; */

	/* float maxXt = max( abs(p0t.x), max( abs(p1t.x), abs(p2t.x) )  ); */
	/* float maxYt = max( abs(p0t.y), max( abs(p1t.y), abs(p2t.y) )  ); */
	/* float deltaX = Gamma_eps(5, eps) * (maxXt + maxZt); */
	/* float deltaY = Gamma_eps(5, eps) * (maxYt + maxZt); */

	/* float deltaE = 2*(Gamma_eps(2, eps)*maxXt*maxYt + */
	/* 				  deltaY*maxXt + deltaX*maxYt); */
	/* float maxE = max( abs(e0), max( abs(e1), abs(e2) )  ); */
	/* float deltaT = 3*(Gamma_eps(3, eps)*maxE*maxZt + deltaE*maxZt + */
	/* 				  deltaZ*maxE)*abs(invDet); */
	/* if(t <= deltaT) return false; */

	/* float3 dpdu, dpdv; */
	/* float2 uv[3]; */
	/* uv[0] = make_float2(0,0); */
	/* uv[1] = make_float2(1,0); */
	/* uv[2] = make_float2(1,1); */
	
	/* float2 duv02 = uv[0]-uv[2], duv12 = uv[1]-uv[2]; */
	/* float3 dp02 = p1-p3, dp12=p2-p3; */
	/* float determinant = duv02.x*duv12.y - duv02.y*duv12.x; */
	/* bool degenerateUV = abs(determinant) < 1e-8; */
	/* if(!degenerateUV) */
	/* { */
	/* 	float invdet = 1/ determinant; */
	/* 	dpdu */
	/* } */
	return true;
}

__device__ bool Triangle::IntersectP2(const Ray &ray) const
{
	float3 p0t, p1t, p2t;
	p0t = p1 - ray.o; p1t = p2 - ray.o; p2t = p3 - ray.o;
	
	int kz = MaxDim( make_float3(abs(ray.d.x),abs(ray.d.y),abs(ray.d.z)) );
	int kx = kz + 1;
	if(kx == 3) kx = 0;
	int ky = kx+1;
	if(ky == 3) ky = 0;
	float3 d = Permute(ray.d, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	float Sx=-d.x/d.z; float Sy=-d.y/d.z; float Sz=1.F/d.z;
	p0t.x += Sx*p0t.z; p0t.y += Sy*p0t.z;
	p1t.x += Sx*p1t.z; p1t.y += Sy*p1t.z;
	p2t.x += Sx*p2t.z; p2t.y += Sy*p2t.z;

	float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	if ( e0 == 0.F || e1 == 0.F || e2 == 0.F )
	{
		double p2txp1ty = (double)p2t.x * (double)p1t.y;
		double p2typ1tx = (double)p2t.y * (double)p1t.x;
		e0 = (float)(p2typ1tx - p2txp1ty);
		double p0txp2ty = (double)p0t.x * (double)p2t.y;
		double p0typ2tx = (double)p0t.y * (double)p2t.x;
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)p1t.x * (double)p0t.y;
		double p1typ0tx = (double)p1t.y * (double)p0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	if((e0<0 || e1<0 || e2<0) && (e0>0 || e1>0 || e2>0))
		return false;
	float det = e0 + e1 + e2;
	if(det == 0) return false;

	p0t *= Sz; p1t *= Sz; p2t *= Sz;
	float tScaled = e0*p0t.z + e1*p1t.z + e2*p2t.z;
	if(det < 0 && (tScaled >= 0 || tScaled < ray.maxt*det))
		return false;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.maxt*det))
		return false;

	float invDet = 1/det;
	//float b0 = e0*invDet; float b1 = e1*invDet; float b2 = e2*invDet;
	float t = tScaled*invDet;

	if (t < ray.mint || t > ray.maxt)
        return false;

	float maxZt = max( abs(p0t.z), max( abs(p1t.z), abs(p2t.z) )  );
	float eps = machine_eps_flt() * 0.5;
	float deltaZ = Gamma_eps(3, eps) * maxZt;

	float maxXt = max( abs(p0t.x), max( abs(p1t.x), abs(p2t.x) )  );
	float maxYt = max( abs(p0t.y), max( abs(p1t.y), abs(p2t.y) )  );
	float deltaX = Gamma_eps(5, eps) * (maxXt + maxZt);
	float deltaY = Gamma_eps(5, eps) * (maxYt + maxZt);

	float deltaE = 2*(Gamma_eps(2, eps)*maxXt*maxYt +
					  deltaY*maxXt + deltaX*maxYt);
	float maxE = max( abs(e0), max( abs(e1), abs(e2) )  );
	float deltaT = 3*(Gamma_eps(3, eps)*maxE*maxZt + deltaE*maxZt +
					  deltaZ*maxE)*abs(invDet);
	if(t <= deltaT) return false;

	return true;
}

#ifndef DOUBLE
// Méthode de Möller-Trumbore pour l'intersection rayon/triangle
bool Triangle::Intersect(const Ray &ray, float *tHit,
						 DifferentialGeometry *dg) const
{
	float3 e1 = p2 - p1;
	float3 e2 = p3 - p1;
	float3 s1 = cross(ray.d, e2);
    float divisor = dot(s1, e1);

	if (divisor == 0.)
	{return false;}
	float invDivisor = 1.F/divisor;

	// Calcul de la 1er composante des coordonnées baricentriques
	float3 s = ray.o - p1;
	float b1 = dot(s, s1) * invDivisor;

    if (b1 < -0.0000001 || b1 > 1.0000001)
	{return false;}

    // Calcul de la 2nd composante des coordonnées baricentriques
    float3 s2 = cross(s, e1);
    float b2 = dot(ray.d, s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Calcul de temps t du rayon pour atteindre le point d'intersection
    float t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt)
        return false;

    // Calcul des dérivée partielles du triangle
	float3 dpdu, dpdv;
	float3c uvsC0, uvsC1; // row1 and row2
	uvsC0 = make_float3c(0., 1., 1.);
	uvsC1 = make_float3c(0., 0., 1.);
		
    // Calcul du Delta pour les dérivée partielles du triangle
	float du1 = uvsC0[0] - uvsC0[2];
	float du2 = uvsC0[1] - uvsC0[2];
    float dv1 = uvsC1[0] - uvsC1[2];
    float dv2 = uvsC1[1] - uvsC1[2];
    float3 dp1 = p1 - p3, dp2 = p2 - p3;
    float determinant = du1 * dv2 - dv1 * du2;

    if (determinant == 0.)
	{
        // Gestion du cas où le déterminant est nul
        coordinateSystem(normalize(cross(e2, e1)), &dpdu, &dpdv);
    }
    else
	{
        double invdet = 1. / determinant;
        dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
        dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    }

    // Interpolation des coordonnées paramétrique du triangle $(u,v)$
    float b0 = 1 - b1 - b2;
    float tu = b0*uvsC0[0] + b1*uvsC0[1] + b2*uvsC0[2];
    float tv = b0*uvsC1[0] + b1*uvsC1[1] + b2*uvsC1[2];

    // Initialisation de  _DifferentialGeometry_ depuis les données paramétriques
    *dg = DifferentialGeometry(ray(t), dpdu, dpdv, tu, tv, this);

    // mise a jour de _tHit_
	*tHit = t;
	return true;
}
// Méthode de Möller-Trumbore pour l'intersection rayon/triangle
bool Triangle::IntersectP(const Ray &ray) const
{
	float3 e1 = p2 - p1;
	float3 e2 = p3 - p1;
	float3 s1 = cross(ray.d, e2);
    float divisor = dot(s1, e1);

	if (divisor == 0.)
	{return false;}
	float invDivisor = 1.F/divisor;

	// Calcul de la 1er composante des coordonnées baricentriques
	float3 s = ray.o - p1;
	float b1 = dot(s, s1) * invDivisor;

    if (b1 < -0.0000001 || b1 > 1.0000001)
	{return false;}

    // Calcul de la 2nd composante des coordonnées baricentriques
    float3 s2 = cross(s, e1);
    float b2 = dot(ray.d, s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Calcul de temps t du rayon pour atteindre le point d'intersection
    float t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt)
        return false;

	return true;
}
//*******************************************************************************
#else
// Méthode de Möller-Trumbore pour l'intersection rayon/triangle
bool Triangle::Intersect(const Ray &ray, float *tHit,
						 DifferentialGeometry *dg) const
{
	double3 p1d = make_double3(double(p1.x), double(p1.y), double(p1.z));
	double3 p2d = make_double3(double(p2.x), double(p2.y), double(p2.z));
	double3 p3d = make_double3(double(p3.x), double(p3.y), double(p3.z));
	double3 dray_o = make_double3(double(ray.o.x), double(ray.o.y), double(ray.o.z));
	double3 dray_d = make_double3(double(ray.d.x), double(ray.d.y), double(ray.d.z));
	
	double3 e1 = p2d - p1d;
	double3 e2 = p3d - p1d;
	double3 s1 = cross(dray_d, e2);
    double divisor = dot(s1, e1);

	if (divisor == 0.)
	{return false;}
	double invDivisor = 1./divisor;

	// Calcul de la 1er composante des coordonnées baricentriques
	double3 s = dray_o - p1d;
	double b1 = dot(s, s1) * invDivisor;

    if (b1 < -0.0000000001 || b1 > 1.0000000001)
	{return false;}

    // Calcul de la 2nd composante des coordonnées baricentriques
    double3 s2 = cross(s, e1);
    double b2 = dot(dray_d, s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Calcul de temps t du rayon pour atteindre le point d'intersection
    double t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt)
        return false;

    // Calcul des dérivée partielles du triangle
	double3 dpdu, dpdv;
	double3c uvsC0, uvsC1; // row1 and row2
	uvsC0 = make_double3c(0., 1., 1.);
	uvsC1 = make_double3c(0., 0., 1.);
		
    // Calcul du Delta pour les dérivée partielles du triangle
	double du1 = uvsC0[0] - uvsC0[2];
	double du2 = uvsC0[1] - uvsC0[2];
    double dv1 = uvsC1[0] - uvsC1[2];
    double dv2 = uvsC1[1] - uvsC1[2];
    double3 dp1 = p1d - p3d, dp2 = p2d - p3d;
    double determinant = du1 * dv2 - dv1 * du2;

    if (determinant == 0.)
	{
        // Gestion du cas où le déterminant est nul
        coordinateSystem(normalize(cross(e2, e1)), &dpdu, &dpdv);
    }
    else
	{
        double invdet = 1. / determinant;
        dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
        dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    }

    // Interpolation des coordonnées paramétrique du triangle $(u,v)$
    double b0 = 1 - b1 - b2;
    double tu = b0*uvsC0[0] + b1*uvsC0[1] + b2*uvsC0[2];
    double tv = b0*uvsC1[0] + b1*uvsC1[1] + b2*uvsC1[2];

    // Initialisation de  _DifferentialGeometry_ depuis les données paramétriques
	float3 dpduf = make_float3(float(dpdu.x), float(dpdu.y), float(dpdu.z));
	float3 dpdvf = make_float3(float(dpdv.x), float(dpdv.y), float(dpdv.z));
	
	*dg = DifferentialGeometry(ray(float(t)), dpduf, dpdvf, float(tu), float(tv), this);

    // mise a jour de _tHit_
	*tHit = float(t);
	return true;
}

// Möller-Trumbore for ray/triangle intersection simple bool test
bool Triangle::IntersectP(const Ray &ray) const
{
	double3 p1d = make_double3(double(p1.x), double(p1.y), double(p1.z));
	double3 p2d = make_double3(double(p2.x), double(p2.y), double(p2.z));
	double3 p3d = make_double3(double(p3.x), double(p3.y), double(p3.z));
	double3 dray_o = make_double3(double(ray.o.x), double(ray.o.y), double(ray.o.z));
	double3 dray_d = make_double3(double(ray.d.x), double(ray.d.y), double(ray.d.z));
	
	double3 e1 = p2d - p1d;
	double3 e2 = p3d - p1d;
	double3 s1 = cross(dray_d, e2);
    double divisor = dot(s1, e1);

	if (divisor == 0.)
	{return false;}
	double invDivisor = 1./divisor;

	// Calcul de la 1er composante des coordonnées baricentriques
	double3 s = dray_o - p1d;
	double b1 = dot(s, s1) * invDivisor;

    if (b1 < -0.0000000001 || b1 > 1.0000000001)
	{return false;}

    // Calcul de la 2nd composante des coordonnées baricentriques
    double3 s2 = cross(s, e1);
    double b2 = dot(dray_d, s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Calcul de temps t du rayon pour atteindre le point d'intersection
    double t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt)
        return false;

	return true;
}
#endif

__device__ BBox Triangle::ObjectBoundTriangle() const
{
	BBox objectBounds((*WorldToObject)(Pointf(p1)), (*WorldToObject)(Pointf(p2)));
	return objectBounds.Union(objectBounds, (*WorldToObject)(Pointf(p3)));
}

__device__ BBox Triangle::WorldBoundTriangle() const
{
	BBox worldBounds(p1, p2);
    return worldBounds.Union(worldBounds, p3);
}

float Triangle::Area() const
{
    return 0.5f * length(cross(p2-p1, p3-p1));
}


class TriangleMesh : public Shape
// ========================================================
// Classe triangleMesh
// ========================================================
{
public:
	// Méthodes publiques de classe triangleMesh
	__host__ __device__ TriangleMesh(const Transform *o2w, const Transform *w2o,
									 int nt, int nv, int *vi, float3 *P);

    /* uniquement device pour éviter des problèmes de mémoires */
    __device__ BBox ObjectBoundTriangleMesh() const;
    __device__ BBox WorldBoundTriangleMesh() const;

    __host__ __device__ bool Intersect(const Ray &ray, float* tHit,
									   DifferentialGeometry *dg) const;
	__host__ __device__ bool IntersectP(const Ray &ray) const;
	__device__ bool IntersectP2(const Ray &ray) const;
    __host__ __device__ float Area() const;
	float3 *p;
private:
	// Paramètres privés de la classe triangleMesh
	int ntris, nverts;
	int *vertexIndex;
	//float3 *p;
	Triangle *refTri;
};

// -------------------------------------------------------
// définitions des méthodes de la classe TriangleMesh
// -------------------------------------------------------
TriangleMesh::TriangleMesh(const Transform *o2w, const Transform *w2o,
						   int nt, int nv, int *vi, float3 *P)
	: Shape(o2w, w2o)
{
	ntris = nt; nverts = nv;
	vertexIndex = vi;
	/* refTri = rt; */
	p = P;

	// Applique les transformations sur le maillage	
	for (int i = 0; i < nverts; ++i)
		p[i] = (*ObjectToWorld)(Pointf(p[i]));
}


bool TriangleMesh::Intersect(const Ray &ray, float* tHit,
							 DifferentialGeometry *dg) const
{
    bool dgbool = false;
	Transform nothing;
    #if __CUDA_ARCH__ >= 200
	*tHit = CUDART_INF_F;
    #elif !defined(__CUDA_ARCH__)
	*tHit = std::numeric_limits<float>::max();
    #endif

	for (int i = 0; i < ntris; ++i)
	{
		float triHit;
		DifferentialGeometry dgTri;
		/* // créer le triangle i en fonction de *vi et *P	 */
		float3 PA = p[vertexIndex[3*i]];
		float3 PB = p[vertexIndex[3*i + 1]];
		float3 PC = p[vertexIndex[3*i + 2]];
		Triangle rt(&nothing, &nothing, PA, PB, PC);
		if (rt.Intersect(ray, &triHit, &dgTri))
		{
			dgbool = true;
			if (*tHit > triHit)
			{
				*dg = dgTri;
				*tHit = triHit;
			}
		}
	}
	return dgbool;
}

bool TriangleMesh::IntersectP(const Ray &ray) const
{
	Transform nothing;
	for (int i = 0; i < ntris; ++i)
	{
		/* // créer le triangle i en fonction de *vi et *P	 */
		float3 PA = p[vertexIndex[3*i]];
		float3 PB = p[vertexIndex[3*i + 1]];
		float3 PC = p[vertexIndex[3*i + 2]];
		Triangle rt(&nothing, &nothing, PA, PB, PC);
		if (rt.IntersectP(ray))
			return true;
	}
	return false;
}

__device__ bool TriangleMesh::IntersectP2(const Ray &ray) const
{
	Transform nothing;
	for (int i = 0; i < ntris; ++i)
	{
		/* // créer le triangle i en fonction de *vi et *P	 */
		float3 PA = p[vertexIndex[3*i]];
		float3 PB = p[vertexIndex[3*i + 1]];
		float3 PC = p[vertexIndex[3*i + 2]];
		Triangle rt(&nothing, &nothing, PA, PB, PC);
		if (rt.IntersectP2(ray))
			return true;
	}
	return false;
}

__device__ BBox TriangleMesh::ObjectBoundTriangleMesh() const
{
	BBox objectBounds;
    for (int i = 0; i < nverts; i++) {
		float3 pW = (*WorldToObject)(Pointf(p[i]));
		objectBounds = objectBounds.Union(objectBounds, pW);}
    return objectBounds;
}

__device__ BBox TriangleMesh::WorldBoundTriangleMesh() const
{
    BBox worldBounds;
    for (int i = 0; i < nverts; i++)
		worldBounds = worldBounds.Union(worldBounds, p[i]);
    return worldBounds;
}

float TriangleMesh::Area() const
{
	float Area = 0.f;
    for (int i = 0; i < ntris; ++i)
	{
		float3 PA = p[vertexIndex[3*i]];
		float3 PB = p[vertexIndex[3*i + 1]];
		float3 PC = p[vertexIndex[3*i + 2]];
		Area += 0.5f * length(cross(PB-PA, PC-PA));
	}
    return Area;
}
#endif // _SHAPES_H_
