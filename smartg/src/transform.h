
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

	inline __host__ __device__ float3 operator()(const float3 &c, const int type) const;
	inline __host__ __device__ void operator()(const float3 &c, float3 *ctrans,
												 const int type) const;
	#ifdef OBJ3D
		inline __host__ __device__ float3 operator()(const Pointf &c) const;
	inline __host__ __device__ float3 operator()(const Vectorf &c) const;
	inline __host__ __device__ float3 operator()(const Normalf &c) const;
	inline __host__ __device__ void operator()(const Pointf &c, float3 *ctrans) const;
	inline __host__ __device__ void operator()(const Vectorf &c, float3 *ctrans) const;
	inline __host__ __device__ void operator()(const Normalf &c, float3 *ctrans) const;
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
	__host__ __device__ Transform Rotate(float angle, const float3 &axis);
	__host__ __device__ Transform vec2transform(float3 vi);

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

inline float3 Transform::operator()(const float3 &c, const int type) const
{
	float x = c.x, y = c.y, z = c.z;
	
	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		float xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		float yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		float zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp == 1.) return make_float3(xp, yp, zp);
		else          return make_float3(xp, yp, zp)/wp;
	}
	else if (type == 2)
	{
		float xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		float yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		float zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
		return make_float3(xv, yv, zv);
	}
	else if (type == 3)
	{
		float xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		float yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		float zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
		return make_float3(xn, yn, zn);
	}
	else
	{
		printf("\"%i\" is an unknown type \n", type);
		printf(" P = \"lol\" \n");
		printf("Please select a type between: Point, Vector and Normal.\n");
		return c;
	}
}

inline void Transform::operator()(const float3 &c, float3 *ctrans, const int type) const
{
	float x = c.x, y = c.y, z = c.z;

	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp != 1.) *ctrans /= wp;
	}
	else if (type == 2)
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	}
	else if (type == 3)
	{
		ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	}
	else
	{
		printf("\"%i\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
	}
}

#ifdef OBJ3D
inline float3 Transform::operator()(const Pointf &c) const
{
	float x = c.x, y = c.y, z = c.z;
	
	float xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	float yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	float zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
	float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	
	if (wp == 1.) return make_float3(xp, yp, zp);
	else          return make_float3(xp, yp, zp)/wp;
}

inline float3 Transform::operator()(const Vectorf &c) const
{
	float x = c.x, y = c.y, z = c.z;
	
	float xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	float yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	float zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	
	return make_float3(xv, yv, zv);
}

inline float3 Transform::operator()(const Normalf &c) const
{
	float x = c.x, y = c.y, z = c.z;

	float xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	float yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	float zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	
	return make_float3(xn, yn, zn);
}

inline void Transform::operator()(const Pointf &c, float3 *ctrans) const
{
	float x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];

	float wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	if (wp != 1.) *ctrans /= wp;
}

inline void Transform::operator()(const Vectorf &c, float3 *ctrans) const
{
	float x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
}

inline void Transform::operator()(const Normalf &c, float3 *ctrans) const
{
	float x = c.x, y = c.y, z = c.z;

	ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
}

inline Ray Transform::operator()(const Ray &r) const
{
    Ray ret = r;
	(*this)(Pointf(ret.o), &ret.o);
    (*this)(Vectorf(ret.d), &ret.d);
    return ret;
}


inline void Transform::operator()(const Ray &r, Ray *rt) const
{
	(*this)(Pointf(r.o), &rt->o);
    (*this)(Vectorf(r.d), &rt->d);
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

    // creation du point P et du vecteur V=(v1, v2, v3)
	float3 P, V1, V2, V3;

	// Application des transformations
	P = M(Pointf(b.pMin));
	V1 = M(Vectorf(b.pMax.x-b.pMin.x, 0, 0));
	V2 = M(Vectorf(0, b.pMax.y-b.pMin.y, 0));
	V3 = M(Vectorf(0, 0, b.pMax.z-b.pMin.z));

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

Transform Transform::Rotate(float angle, const float3 &axis) {
	
	float3 a = normalize(axis);
	
	#if __CUDA_ARCH__ >= 200
    float s = sin(radians(angle));
    float c = cos(radians(angle));
	#elif !defined(__CUDA_ARCH__)
	float s = sinf(radians(angle));
    float c = cosf(radians(angle));
	#endif
	
	float4x4 m = make_diag_float4x4(1);
	
	m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
	m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
	m[0][2] = a.x * a.z * (1.f - c) + a.y * s;

	m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
	m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
	m[1][2] = a.y * a.z * (1.f - c) - a.x * s;

	m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
	m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
	m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;

    return Transform(m, transpose(m));
}

Transform Transform::vec2transform(float3 vi)
{
	// this function gives the needed transform to get the vector vi from
	// an intial vector=(0,0,1)
    double acc=1e-4;
    double3 v_ini = make_double3(0., 0., 1.);
    double roty_rad;
    double rotz_rad;
    double cosphi;
    double theta;
    double phi;
    Transform tf;
    Transform nothing;
    double3 v_ini_rotated;
	double3 v = make_double3(vi.x, vi.y, vi.z);
	v = normalize(v);


    // In case v = v_ini -> no rotations
    if (abs(v.x-v_ini.x) < acc & abs(v.y-v_ini.y) < acc & abs(v.z-v_ini.z) < acc)
    {
        return nothing;
    }

    for (int icase = 1; icase < 6; ++icase)
	{
        if (icase == 1)
        {
            roty_rad = acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = acos(cosphi);
        }
        else if(icase == 2)
        {
            roty_rad = acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = -acos(cosphi);
        }
        else if(icase == 3)
        {
            roty_rad = -acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = acos(cosphi);
        }
        else if(icase == 4)
        {
            roty_rad = -acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = -acos(cosphi);
        }
        else
        {
            return nothing;
        }

        
        theta = roty_rad * (180./CUDART_PI);
        phi = rotz_rad * (180./CUDART_PI);
        
        tf = nothing.RotateZ(phi)*nothing.RotateY(theta);
        v_ini_rotated = normalize(make_double3(tf(make_float3(v_ini), 2)));

        if (abs(v.x-v_ini_rotated.x) < acc & abs(v.y-v_ini_rotated.y) < acc & abs(v.z-v_ini_rotated.z) < acc)
        {
			return tf;
        }
	}
	return nothing;
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

	inline __host__ __device__ double3 operator()(const double3 &c, const int type) const;
	inline __host__ __device__ void operator()(const double3 &c, double3 *ctrans,
											   const int type) const;
	#ifdef OBJ3D
		inline __host__ __device__ double3 operator()(const Pointd &c) const;
	inline __host__ __device__ double3 operator()(const Vectord &c) const;
	inline __host__ __device__ double3 operator()(const Normald &c) const;
	inline __host__ __device__ void operator()(const Pointd &c, double3 *ctrans) const;
	inline __host__ __device__ void operator()(const Vectord &c, double3 *ctrans) const;
	inline __host__ __device__ void operator()(const Normald &c, double3 *ctrans) const;
	/* inline __host__ __device__ Ray operator()(const Ray &r) const; */
    /* inline __host__ __device__ void operator()(const Ray &r, Ray *rt) const; */
    /* __host__ __device__ BBox operator()(const BBox &b) const; */
	#endif
    __host__ __device__ Transformd operator*(const Transformd &t2) const;

    __host__ __device__ const double4x4 &GetMatrix() const { return m; }
    __host__ __device__ const double4x4 &GetInverseMatrix() const { return mInv; }
    __host__ __device__ Transformd Inverse(const Transformd &t);
    __host__ __device__	Transformd Translate(const double3 &delta);     // delta doit être un vecteur
	__host__ __device__ Transformd Scale(double x, double y, double z); // Echelle (facteur) en x, y et z
	__host__ __device__ Transformd RotateX(double angle);               // rot par rapport à X  
	__host__ __device__ Transformd RotateY(double angle);               // rot par rapport à Y  
	__host__ __device__ Transformd RotateZ(double angle);               // rot par rapport à Z
	__host__ __device__ Transformd Rotate(double angle, const double3 &axis);
	__host__ __device__ Transformd vec2transform(double3 vi);

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

inline double3 Transformd::operator()(const double3 &c, const int type) const
{
	double x = c.x, y = c.y, z = c.z;

	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		double xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		double yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		double zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp == 1.) return make_double3(xp, yp, zp);
		else          return make_double3(xp, yp, zp)/wp;
	}
	else if (type == 2)
	{
		double xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		double yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		double zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
		return make_double3(xv, yv, zv);
	}
	else if (type == 3)
	{
		double xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		double yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		double zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
		return make_double3(xn, yn, zn);
	}
	else
	{
		printf("\"%i\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
		return c;
	}
}

inline void Transformd::operator()(const double3 &c, double3 *ctrans, const int type) const
{
	double x = c.x, y = c.y, z = c.z;
	
	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp != 1.) *ctrans /= wp;
	}
	else if (type == 2)
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	}
	else if (type == 3)
	{
		ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	}
	else
	{
		printf("\"%i\" is an unknown type\n", type);
		printf("Please select a type between: Point, Vector and Normal.\n");
	}
}

#ifdef OBJ3D
inline double3 Transformd::operator()(const Pointd &c) const
{
	double x = c.x, y = c.y, z = c.z;
	
	double xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	double yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	double zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
	double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	
	if (wp == 1.) return make_double3(xp, yp, zp);
	else          return make_double3(xp, yp, zp)/wp;
}

inline double3 Transformd::operator()(const Vectord &c) const
{
	double x = c.x, y = c.y, z = c.z;
	
	double xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	double yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	double zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	
	return make_double3(xv, yv, zv);
}

inline double3 Transformd::operator()(const Normald &c) const
{
	double x = c.x, y = c.y, z = c.z;

	double xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	double yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	double zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	
	return make_double3(xn, yn, zn);
}

inline void Transformd::operator()(const Pointd &c, double3 *ctrans) const
{
	double x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];

	double wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	if (wp != 1.) *ctrans /= wp;
}

inline void Transformd::operator()(const Vectord &c, double3 *ctrans) const
{
	double x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
}

inline void Transformd::operator()(const Normald &c, double3 *ctrans) const
{
	double x = c.x, y = c.y, z = c.z;

	ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
}
#endif

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

Transformd Transformd::Rotate(double angle, const double3 &axis) {
	
	double3 a = normalize(axis);
	
	#if __CUDA_ARCH__ >= 200
    double s = sin(radiansd(angle));
    double c = cos(radiansd(angle));
	#elif !defined(__CUDA_ARCH__)
	double s = sin(radiansd(angle));
    double c = cos(radiansd(angle));
	#endif
	
	double4x4 m = make_diag_double4x4(1);
	
	m[0][0] = a.x * a.x + (1. - a.x * a.x) * c;
	m[0][1] = a.x * a.y * (1. - c) - a.z * s;
	m[0][2] = a.x * a.z * (1. - c) + a.y * s;

	m[1][0] = a.x * a.y * (1. - c) + a.z * s;
	m[1][1] = a.y * a.y + (1. - a.y * a.y) * c;
	m[1][2] = a.y * a.z * (1. - c) - a.x * s;

	m[2][0] = a.x * a.z * (1. - c) - a.y * s;
	m[2][1] = a.y * a.z * (1. - c) + a.x * s;
	m[2][2] = a.z * a.z + (1. - a.z * a.z) * c;

    return Transformd(m, transpose(m));
}

Transformd Transformd::vec2transform(double3 vi)
{
	// this function gives the needed transform to get the vector vi from
	// an intial vector=(0,0,1)
    double acc=1e-4;
    double3 v_ini = make_double3(0., 0., 1.);
    double roty_rad;
    double rotz_rad;
    double cosphi;
    double theta;
    double phi;
    Transformd tf;
    Transformd nothing;
    double3 v_ini_rotated;
	double3 v = make_double3(vi.x, vi.y, vi.z);
	v = normalize(v);


    // In case v = v_ini -> no rotations
    if (abs(v.x-v_ini.x) < acc & abs(v.y-v_ini.y) < acc & abs(v.z-v_ini.z) < acc)
    {
        return nothing;
    }

    for (int icase = 1; icase < 6; ++icase)
	{
        if (icase == 1)
        {
            roty_rad = acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = acos(cosphi);
        }
        else if(icase == 2)
        {
            roty_rad = acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = -acos(cosphi);
        }
        else if(icase == 3)
        {
            roty_rad = -acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = acos(cosphi);
        }
        else if(icase == 4)
        {
            roty_rad = -acos(v.z);
            if (v.x == 0 & roty_rad == 0) cosphi = 0.;
            else cosphi = clamp(v.x/sin(roty_rad), -1., 1.);
            rotz_rad = -acos(cosphi);
        }
        else
        {
            return nothing;
        }

        
        theta = roty_rad * (180./CUDART_PI);
        phi = rotz_rad * (180./CUDART_PI);
        
        tf = nothing.RotateZ(phi)*nothing.RotateY(theta);
        v_ini_rotated = normalize(tf(v_ini, 2));

        if (abs(v.x-v_ini_rotated.x) < acc & abs(v.y-v_ini_rotated.y) < acc & abs(v.z-v_ini_rotated.z) < acc)
        {
			return tf;
        }
	}
	return nothing;
}
// -------------------------------------------------------

#endif // _TRANSFORM_H_
