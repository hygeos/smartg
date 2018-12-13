
#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#ifdef OBJ3D
#include "geometry.h"
#endif

#include <math.h>
#include <helper_math.h>
#include <stdio.h>


/**********************************************************
*	> Classe qui permet le déplacement d'un objet.
*     - Translation en x, y et z
*     - Rotation par rapport à x, y et z
***********************************************************/


class Transform
// ========================================================
// Classe Transformation
// ========================================================
{
public:
	// Méthodes publiques
	__host__ __device__ Transform();
	__host__ __device__ Transform(const float4x4 &mat);
	__host__ __device__ Transform(const float4x4 &mat, const float4x4 &matInv);

    __host__ __device__ bool operator<(const Transform &t2) const;  // pas encore utilisée
    __host__ __device__ bool IsIdentity() const;                    // pas encore utilisée

	inline __host__ __device__ float3 operator()(const float3 &c, const char* type) const;
	inline __host__ __device__ void operator()(const float3 &c, float3 *ctrans,
												 const char* type) const;
	#ifdef OBJ3D
    inline __host__ __device__ Ray operator()(const Ray &r) const;
    inline __host__ __device__ void operator()(const Ray &r, Ray *rt) const;
    __host__ __device__ BBox operator()(const BBox &b) const;
	#endif
    __host__ __device__ Transform operator*(const Transform &t2) const;

    __host__ __device__ const float4x4 &GetMatrix() const { return m; }
    __host__ __device__ const float4x4 &GetInverseMatrix() const { return mInv; }
    __host__ __device__ Transform Inverse(const Transform &t);
    __host__ __device__	Transform Translate(const float3 &delta); // delta doit être un vecteur
	__host__ __device__ Transform Scale(float x, float y, float z); // Echelle (facteur) en x, y et z
	__host__ __device__ Transform RotateX(float angle);             // rot par rapport à X  
	__host__ __device__ Transform RotateY(float angle);             // rot par rapport à Y  
	__host__ __device__ Transform RotateZ(float angle);             // rot par rapport à Z

	private:
	// Paramètres privés
	float4x4 m, mInv;
};

// -------------------------------------------------------
// définitions des méthodes de la classe Transformation
// -------------------------------------------------------
Transform::Transform()
{
	m = make_float4x4(
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.
		);
	mInv = m;
}

Transform::Transform(const float4x4 &mat)
{
	m = mat;
	mInv = inverse(m);
}

Transform::Transform(const float4x4 &mat, const float4x4 &matInv)
{
	m = mat;
	mInv = matInv;
}

bool Transform::operator<(const Transform &t2) const
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (m[i][j] < t2.m[i][j]) return true;
			if (m[i][j] > t2.m[i][j]) return false;
		}
	}
	return false;
}

bool Transform::IsIdentity() const
{
	return (m[0][0] == 1.f && m[0][1] == 0.f &&
			m[0][2] == 0.f && m[0][3] == 0.f &&
			m[1][0] == 0.f && m[1][1] == 1.f &&
			m[1][2] == 0.f && m[1][3] == 0.f &&
			m[2][0] == 0.f && m[2][1] == 0.f &&
			m[2][2] == 1.f && m[2][3] == 0.f &&
			m[3][0] == 0.f && m[3][1] == 0.f &&
			m[3][2] == 0.f && m[3][3] == 1.f);
}

inline float3 Transform::operator()(const float3 &c, const char* type) const
{
	float x = c.x, y = c.y, z = c.z;
	if (compStr(type, "Point"))
	{
		float xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		float yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		float zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp == 1.) return make_float3(xp, yp, zp);
		else          return make_float3(xp, yp, zp)/wp;
	}
	else if (compStr(type, "Vector"))
	{
		float xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		float yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		float zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
		return make_float3(xv, yv, zv);
	}
	else if (compStr(type, "Normal"))
	{
		float xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		float yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		float zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
		return make_float3(xn, yn, zn);
	}
	else
	{
		printf("\"%s\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
		return c;
	}
}

inline void Transform::operator()(const float3 &c, float3 *ctrans, const char* type) const
{
	float x = c.x, y = c.y, z = c.z;
	if (compStr(type, "Point"))
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp != 1.) *ctrans /= wp;
	}
	else if (compStr(type, "Vector"))
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	}
	else if (compStr(type, "Normal"))
	{
		ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	}
	else
	{
		printf("\"%s\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
	}
}

#ifdef OBJ3D
inline Ray Transform::operator()(const Ray &r) const
{
    Ray ret = r;
	char myP[]="Point", myD[]="Vector";
    (*this)(ret.o, &ret.o, myP);
    (*this)(ret.d, &ret.d, myD);
    return ret;
}


inline void Transform::operator()(const Ray &r, Ray *rt) const
{
	char myP[]="Point", myD[]="Vector";
    (*this)(r.o, &rt->o, myP);
    (*this)(r.d, &rt->d, myD);
    if (rt != &r)
	{
        rt->mint = r.mint;
        rt->maxt = r.maxt;
        rt->time = r.time;
    }
}

// les 8 coins d'une box peuvent être défini en fonction
// d'un seul point et de ces trois vecteurs unitaires
BBox Transform::operator()(const BBox &b) const
{
    const Transform &M = *this;
	char myP[]="Point", myV[]="Vector";

    // creation du point P et du vecteur V=(v1, v2, v3)
	float3 P, V1, V2, V3;

	// Application des transformations
	P = M(b.pMin, myP);
	V1 = M(make_float3(b.pMax.x-b.pMin.x, 0, 0), myV);
	V2 = M(make_float3(0, b.pMax.y-b.pMin.y, 0), myV);
	V3 = M(make_float3(0, 0, b.pMax.z-b.pMin.z), myV);

	// Creation de la box avec le 1er point P
	BBox ret(P);

    // élargir la box en prenant une face du cube
	// Face avec 4 points : P, P+V.x, P+V.y, P+(V.x, V.y)
	ret = ret.Union(ret, P+V1);
	ret = ret.Union(ret, P+V2);
	ret = ret.Union(ret, P+V1+V2);

	// un point en z est suffisant (symétrie)
	ret = ret.Union(ret, P+V3);
	/* ret = ret.Union(ret, P+V1+V3); */
	/* ret = ret.Union(ret, P+V2+V3); */
	/* ret = ret.Union(ret, P+V1+V2+V3); */
    return ret;
}
#endif

Transform Transform::operator*(const Transform &t2) const
{
    float4x4 myM = mul(m, t2.m);
    float4x4 myMinv = mul(t2.mInv, mInv);
    return Transform(myM, myMinv);
}

Transform Transform::Inverse(const Transform &t)
{
	return Transform(t.mInv, t.m);
}

Transform Transform::Translate(const float3 &delta)
{
	float4x4 myM = make_float4x4(
		1, 0, 0, delta.x,
		0, 1, 0, delta.y,
		0, 0, 1, delta.z,
		0, 0, 0,       1
		);
    float4x4 myMinv = make_float4x4(
		1, 0, 0, -delta.x,
		0, 1, 0, -delta.y,
		0, 0, 1, -delta.z,
		0, 0, 0,        1
		);
    return Transform(myM, myMinv);
}

Transform Transform::Scale(float x, float y, float z) {
    float4x4 myM = make_float4x4(
		x, 0, 0, 0,
		0, y, 0, 0,
		0, 0, z, 0,
		0, 0, 0, 1
		);
    float4x4 myMinv = make_float4x4(
		1.f/x,     0,     0,     0,
		0,     1.f/y,     0,     0,
		0,         0,     1.f/z, 0,
		0,         0,     0,     1)
		;
    return Transform(myM, myMinv);
}

Transform Transform::RotateX(float angle) {
	#if __CUDA_ARCH__ >= 200
    float sin_t = sin(radians(angle));
    float cos_t = cos(radians(angle));
	#elif !defined(__CUDA_ARCH__)
    float sin_t = sinf(radians(angle));
    float cos_t = cosf(radians(angle));
	#endif
    float4x4 myM = make_float4x4(
		1,     0,      0, 0,
		0, cos_t, -sin_t, 0,
		0, sin_t,  cos_t, 0,
		0,     0,      0, 1
		);
    return Transform(myM, transpose(myM));
}

Transform Transform::RotateY(float angle) {
    #if __CUDA_ARCH__ >= 200
    float sin_t = sin(radians(angle));
    float cos_t = cos(radians(angle));
	#elif !defined(__CUDA_ARCH__)
    float sin_t = sinf(radians(angle));
    float cos_t = cosf(radians(angle));
	#endif
    float4x4 myM = make_float4x4(
		cos_t , 0, sin_t, 0,
		0,      1,     0, 0,
		-sin_t, 0, cos_t, 0,
		0,      0,     0, 1
		);
    return Transform(myM, transpose(myM));
}

Transform Transform::RotateZ(float angle) {
	#if __CUDA_ARCH__ >= 200
    float sin_t = sin(radians(angle));
    float cos_t = cos(radians(angle));
	#elif !defined(__CUDA_ARCH__)
	float sin_t = sinf(radians(angle));
    float cos_t = cosf(radians(angle));
	#endif
    float4x4 m = make_float4x4(
		cos_t, -sin_t, 0, 0,
		sin_t,  cos_t, 0, 0,
		0,          0, 1, 0,
		0,          0, 0, 1);
    return Transform(m, transpose(m));
}

//**************************************************************
//**************************************************************
//**************************************************************
class Transformd
// ========================================================
// Classe Transformation double
// ========================================================
{
public:
	// Méthodes publiques
	__host__ __device__ Transformd();
	__host__ __device__ Transformd(const double4x4 &mat);
	__host__ __device__ Transformd(const double4x4 &mat, const double4x4 &matInv);

    __host__ __device__ bool operator<(const Transformd &t2) const;  // pas encore utilisée
    __host__ __device__ bool IsIdentity() const;                    // pas encore utilisée

	inline __host__ __device__ double3 operator()(const double3 &c, const char* type) const;
	inline __host__ __device__ void operator()(const double3 &c, double3 *ctrans,
											   const char* type) const;
    /* inline __host__ __device__ Ray operator()(const Ray &r) const; */
    /* inline __host__ __device__ void operator()(const Ray &r, Ray *rt) const; */
    /* __host__ __device__ BBox operator()(const BBox &b) const; */
    __host__ __device__ Transformd operator*(const Transformd &t2) const;

    __host__ __device__ const double4x4 &GetMatrix() const { return m; }
    __host__ __device__ const double4x4 &GetInverseMatrix() const { return mInv; }
    __host__ __device__ Transformd Inverse(const Transformd &t);
    __host__ __device__	Transformd Translate(const double3 &delta);     // delta doit être un vecteur
	__host__ __device__ Transformd Scale(double x, double y, double z); // Echelle (facteur) en x, y et z
	__host__ __device__ Transformd RotateX(double angle);               // rot par rapport à X  
	__host__ __device__ Transformd RotateY(double angle);               // rot par rapport à Y  
	__host__ __device__ Transformd RotateZ(double angle);               // rot par rapport à Z

private:
	// Paramètres privés
	double4x4 m, mInv;
};

// -------------------------------------------------------
// définitions des méthodes de la classe Transformation
// -------------------------------------------------------
Transformd::Transformd()
{
	m = make_double4x4(
		1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.
		);
	mInv = m;
}

Transformd::Transformd(const double4x4 &mat)
{
	m = mat;
	mInv = inverse(m);
}

Transformd::Transformd(const double4x4 &mat, const double4x4 &matInv)
{
	m = mat;
	mInv = matInv;
}

bool Transformd::operator<(const Transformd &t2) const
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (m[i][j] < t2.m[i][j]) return true;
			if (m[i][j] > t2.m[i][j]) return false;
		}
	}
	return false;
}

bool Transformd::IsIdentity() const
{
	return (m[0][0] == 1. && m[0][1] == 0. &&
			m[0][2] == 0. && m[0][3] == 0. &&
			m[1][0] == 0. && m[1][1] == 1. &&
			m[1][2] == 0. && m[1][3] == 0. &&
			m[2][0] == 0. && m[2][1] == 0. &&
			m[2][2] == 1. && m[2][3] == 0. &&
			m[3][0] == 0. && m[3][1] == 0. &&
			m[3][2] == 0. && m[3][3] == 1.);
}

inline double3 Transformd::operator()(const double3 &c, const char* type) const
{
	double x = c.x, y = c.y, z = c.z;
	if (compStr(type, "Point"))
	{
		double xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		double yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		double zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp == 1.) return make_double3(xp, yp, zp);
		else          return make_double3(xp, yp, zp)/wp;
	}
	else if (compStr(type, "Vector"))
	{
		double xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		double yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		double zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
		return make_double3(xv, yv, zv);
	}
	else if (compStr(type, "Normal"))
	{
		double xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		double yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		double zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
		return make_double3(xn, yn, zn);
	}
	else
	{
		printf("\"%s\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
		return c;
	}
}

inline void Transformd::operator()(const double3 &c, double3 *ctrans, const char* type) const
{
	double x = c.x, y = c.y, z = c.z;
	if (compStr(type, "Point"))
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp != 1.) *ctrans /= wp;
	}
	else if (compStr(type, "Vector"))
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	}
	else if (compStr(type, "Normal"))
	{
		ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	}
	else
	{
		printf("\"%s\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
	}
}

/* inline Ray Transformd::operator()(const Ray &r) const */
/* { */
/*     Ray ret = r; */
/* 	char myP[]="Point", myD[]="Vector"; */
/*     (*this)(ret.o, &ret.o, myP); */
/*     (*this)(ret.d, &ret.d, myD); */
/*     return ret; */
/* } */


/* inline void Transformd::operator()(const Ray &r, Ray *rt) const */
/* { */
/* 	char myP[]="Point", myD[]="Vector"; */
/*     (*this)(r.o, &rt->o, myP); */
/*     (*this)(r.d, &rt->d, myD); */
/*     if (rt != &r) */
/* 	{ */
/*         rt->mint = r.mint; */
/*         rt->maxt = r.maxt; */
/*         rt->time = r.time; */
/*     } */
/* } */

// les 8 coins d'une box peuvent être défini en fonction
// d'un seul point et de ces trois vecteurs unitaires
/* BBox Transformd::operator()(const BBox &b) const */
/* { */
/*     const Transformd &M = *this; */
/* 	char myP[]="Point", myV[]="Vector"; */

/*     // creation du point P et du vecteur V=(v1, v2, v3) */
/* 	double3 P, V1, V2, V3; */

/* 	// Application des transformations */
/* 	P = M(b.pMin, myP); */
/* 	V1 = M(make_double3(b.pMax.x-b.pMin.x, 0, 0), myV); */
/* 	V2 = M(make_double3(0, b.pMax.y-b.pMin.y, 0), myV); */
/* 	V3 = M(make_double3(0, 0, b.pMax.z-b.pMin.z), myV); */

/* 	// Creation de la box avec le 1er point P */
/* 	BBox ret(P); */

/*     // élargir la box en prenant une face du cube */
/* 	// Face avec 4 points : P, P+V.x, P+V.y, P+(V.x, V.y) */
/* 	ret = ret.Union(ret, P+V1); */
/* 	ret = ret.Union(ret, P+V2); */
/* 	ret = ret.Union(ret, P+V1+V2); */

/* 	// un point en z est suffisant (symétrie) */
/* 	ret = ret.Union(ret, P+V3); */
/* 	/\* ret = ret.Union(ret, P+V1+V3); *\/ */
/* 	/\* ret = ret.Union(ret, P+V2+V3); *\/ */
/* 	/\* ret = ret.Union(ret, P+V1+V2+V3); *\/ */
/*     return ret; */
/* } */

Transformd Transformd::operator*(const Transformd &t2) const
{
    double4x4 myM = mul(m, t2.m);
    double4x4 myMinv = mul(t2.mInv, mInv);
    return Transformd(myM, myMinv);
}

Transformd Transformd::Inverse(const Transformd &t)
{
	return Transformd(t.mInv, t.m);
}

Transformd Transformd::Translate(const double3 &delta)
{
	double4x4 myM = make_double4x4(
		1, 0, 0, delta.x,
		0, 1, 0, delta.y,
		0, 0, 1, delta.z,
		0, 0, 0,       1
		);
    double4x4 myMinv = make_double4x4(
		1, 0, 0, -delta.x,
		0, 1, 0, -delta.y,
		0, 0, 1, -delta.z,
		0, 0, 0,        1
		);
    return Transformd(myM, myMinv);
}

Transformd Transformd::Scale(double x, double y, double z) {
    double4x4 myM = make_double4x4(
		x, 0, 0, 0,
		0, y, 0, 0,
		0, 0, z, 0,
		0, 0, 0, 1
		);
    double4x4 myMinv = make_double4x4(
		1./x,     0,      0,     0,
		0,      1./y,     0,     0,
		0,         0,     1./z,  0,
		0,         0,     0,     1)
		;
    return Transformd(myM, myMinv);
}

Transformd Transformd::RotateX(double angle) {
	#if __CUDA_ARCH__ >= 200
    double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#elif !defined(__CUDA_ARCH__)
    double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#endif
    double4x4 myM = make_double4x4(
		1,     0,      0, 0,
		0, cos_t, -sin_t, 0,
		0, sin_t,  cos_t, 0,
		0,     0,      0, 1
		);
    return Transformd(myM, transpose(myM));
}

Transformd Transformd::RotateY(double angle) {
    #if __CUDA_ARCH__ >= 200
    double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#elif !defined(__CUDA_ARCH__)
    double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#endif
    double4x4 myM = make_double4x4(
		cos_t , 0, sin_t, 0,
		0,      1,     0, 0,
		-sin_t, 0, cos_t, 0,
		0,      0,     0, 1
		);
    return Transformd(myM, transpose(myM));
}

Transformd Transformd::RotateZ(double angle) {
	#if __CUDA_ARCH__ >= 200
    double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#elif !defined(__CUDA_ARCH__)
	double sin_t = sin(radiansd(angle));
    double cos_t = cos(radiansd(angle));
	#endif
    double4x4 m = make_double4x4(
		cos_t, -sin_t, 0, 0,
		sin_t,  cos_t, 0, 0,
		0,          0, 1, 0,
		0,          0, 0, 1);
    return Transformd(m, transpose(m));
}
// -------------------------------------------------------

#endif // _TRANSFORM_H_
