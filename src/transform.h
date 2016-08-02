
#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#include "geometry.h"

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

    __host__ __device__ bool operator<(const Transform &t2) const;
    __host__ __device__ bool IsIdentity() const;

	inline __host__ __device__ float3 operator()(const float3 &c, const char* type) const;
	inline __host__ __device__ void operator()(const float3 &c, float3 *ctrans,
												 const char* type) const;
    inline __host__ __device__ Ray operator()(const Ray &r) const;
    inline __host__ __device__ void operator()(const Ray &r, Ray *rt) const;

    __host__ __device__ const float4x4 &GetMatrix() const { return m; }
    __host__ __device__ const float4x4 &GetInverseMatrix() const { return mInv; }
    __host__ __device__ Transform Inverse(const Transform &t);
    __host__ __device__	Transform Translate(const float3 &delta); // delta doit être un vecteur

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
#endif // _TRANSFORM_H_
