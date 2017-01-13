
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
	char myP[]="Point", myN[]="Normal";
    *dg = DifferentialGeometry(o2w(phit, myP), o2w(dpdu, myN), o2w(dpdv, myN), u, v, this);

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

// Méthode de Möller-Trumbore pour l'intersection rayon/triangle
bool Triangle::Intersect(const Ray &ray, float* tHit,
						 DifferentialGeometry *dg) const
{
	float3 e1 = p2 - p1;
	float3 e2 = p3 - p1;
	float3 s1 = cross(ray.d, e2);
	float divisor = dot(s1, e1);

	if (divisor == 0.)
		return false;
	float invDivisor = 1.f/divisor;

	// Calcul de la 1er composante des coordonnées baricentriques
	float3 s = ray.o - p1;
	float b1 = dot(s, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        return false;

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

__device__ BBox Triangle::ObjectBoundTriangle() const
{
	char myP[]="Point";
	BBox objectBounds((*WorldToObject)(p1, myP), (*WorldToObject)(p2, myP));
	return objectBounds.Union(objectBounds, (*WorldToObject)(p3, myP));
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
									 int nt, int nv, int *vi, float3 *P, Triangle *rt);

    /* uniquement device pour éviter des problèmes de mémoires */
    __device__ BBox ObjectBoundTriangleMesh() const;
    __device__ BBox WorldBoundTriangleMesh() const;

    __host__ __device__ bool Intersect(const Ray &ray, float* tHit,
									   DifferentialGeometry *dg) const;
    __host__ __device__ float Area() const;

private:
	// Paramètres privés de la classe triangleMesh
	int ntris, nverts;
	int *vertexIndex;
	float3 *p;
	Triangle *refTri;
};

// -------------------------------------------------------
// définitions des méthodes de la classe TriangleMesh
// -------------------------------------------------------
TriangleMesh::TriangleMesh(const Transform *o2w, const Transform *w2o,
						   int nt, int nv, int *vi, float3 *P, Triangle *rt)
	: Shape(o2w, w2o)
{
	ntris = nt; nverts = nv;
	vertexIndex = vi;
	refTri = rt;
	p = P;

	// Applique les transformations sur le maillage
	char myP[]="Point";
	for (int i = 0; i < nverts; ++i)
		p[i] = (*ObjectToWorld)(p[i], myP);
	
    // créer les triangles en fonction de *vi et *P	
	Transform nothing;
    for (int i = 0; i < ntris; ++i)
	{
		float3 PA = p[vertexIndex[3*i]];
		float3 PB = p[vertexIndex[3*i + 1]];
		float3 PC = p[vertexIndex[3*i + 2]];
		refTri[i] = Triangle(&nothing, &nothing, PA, PB, PC);
	}
}


bool TriangleMesh::Intersect(const Ray &ray, float* tHit,
							 DifferentialGeometry *dg) const
{
    bool dgbool = false;
    #if __CUDA_ARCH__ >= 200
	*tHit = CUDART_INF_F;
    #elif !defined(__CUDA_ARCH__)
	*tHit = std::numeric_limits<float>::max();
    #endif
	for (int i = 0; i < ntris; ++i)
	{
		bool mybool;
		float triHit;
		DifferentialGeometry dgTri;
		mybool = refTri[i].Intersect(ray, &triHit, &dgTri);
		if (mybool)
		{
			dgbool = true;
			if (*tHit > triHit)
			{
				*dg = dgTri;
				*tHit = triHit;
			}
		}
	}

	if (!dgbool)
		return false;
	
	return true;
}

__device__ BBox TriangleMesh::ObjectBoundTriangleMesh() const
{
	BBox objectBounds;
	char myP[]="Point";
    for (int i = 0; i < nverts; i++)
        objectBounds.Union(objectBounds, (*WorldToObject)(p[i], myP));
    return objectBounds;
}

__device__ BBox TriangleMesh::WorldBoundTriangleMesh() const
{
    BBox worldBounds;
    for (int i = 0; i < nverts; i++)
        worldBounds.Union(worldBounds, p[i]);
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
