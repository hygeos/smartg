
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

template <typename T = float> //T -> float / double
class Transform
// ========================================================
// Transformation class
// ========================================================
{
public:
	using U4x4 = mat4x4<T>;
	using U3  = vec3<T>;
	
	// Public methods
	__host__ __device__ Transform();
	__host__ __device__ Transform(const U4x4 &mat);
	__host__ __device__ Transform(const U4x4 &mat, const U4x4 &matInv);
    __host__ __device__ bool operator<(const Transform<T> &t2) const;
    __host__ __device__ bool IsIdentity() const;

	inline __host__ __device__ U3 operator()(const U3 &c, const int type) const;
	inline __host__ __device__ void operator()(const U3 &c, U3 *ctrans, const int type) const;

	#ifdef OBJ3D
	inline __host__ __device__ U3 operator()(const Point<T> &c) const;
	inline __host__ __device__ U3 operator()(const Vector<T> &c) const;
	inline __host__ __device__ U3 operator()(const Normal<T> &c) const;
	inline __host__ __device__ void operator()(const Point<T> &c, U3 *ctrans) const;
	inline __host__ __device__ void operator()(const Vector<T> &c, U3 *ctrans) const;
	inline __host__ __device__ void operator()(const Normal<T> &c, U3 *ctrans) const;
    inline __host__ __device__ Ray<T> operator()(const Ray<T> &r) const;
    inline __host__ __device__ void operator()(const Ray<T> &r, Ray<T> *rt) const;
    __host__ __device__ BBox<T> operator()(const BBox<T> &b) const;
	#endif
    __host__ __device__ Transform<T> operator*(const Transform<T> &t2) const;

    __host__ __device__ const U4x4 &GetMatrix() const { return m; }
    __host__ __device__ const U4x4 &GetInverseMatrix() const { return mInv; }
    __host__ __device__ Transform<T> Inverse(const Transform<T> &t);
    __host__ __device__	Transform<T> Translate(const U3 &delta); // delta doit être un vecteur
	__host__ __device__ Transform<T> Scale(T x, T y, T z); // Echelle (facteur) en x, y et z
	__host__ __device__ Transform<T> RotateX(T angle);             // rot par rapport à X  
	__host__ __device__ Transform<T> RotateY(T angle);             // rot par rapport à Y  
	__host__ __device__ Transform<T> RotateZ(T angle);             // rot par rapport à Z
	__host__ __device__ Transform<T> Rotate(T angle, const U3 &axis);
	__host__ __device__ Transform<T> vec2transform(U3 vi);

	private:
	// Private parameters
	U4x4 m, mInv;
};

// -------------------------------------------------------
// définitions des méthodes de la classe Transformation
// -------------------------------------------------------
template <typename T>
Transform<T>::Transform()
{
	m = make_mat4x4<T>(
		T(1), T(0), T(0), T(0),
		T(0), T(1), T(0), T(0),
		T(0), T(0), T(1), T(0),
		T(0), T(0), T(0), T(1)
		);
	mInv = m;
}

template <typename T>
Transform<T>::Transform(const mat4x4<T> &mat)
{
	m = make_mat4x4<T>(
		mat.r0.x, mat.r0.y, mat.r0.z, mat.r0.w,
		mat.r1.x, mat.r1.y, mat.r1.z, mat.r1.w,
		mat.r2.x, mat.r2.y, mat.r2.z, mat.r2.w,
		mat.r3.x, mat.r3.y, mat.r3.z, mat.r3.w
		);
	mInv = inverse(m);
}

template <typename T>
Transform<T>::Transform(const mat4x4<T> &mat, const mat4x4<T> &matInv)
{
	m = make_mat4x4<T>(
		mat.r0.x, mat.r0.y, mat.r0.z, mat.r0.w,
		mat.r1.x, mat.r1.y, mat.r1.z, mat.r1.w,
		mat.r2.x, mat.r2.y, mat.r2.z, mat.r2.w,
		mat.r3.x, mat.r3.y, mat.r3.z, mat.r3.w
		);
	mInv = make_mat4x4<T>(
		   matInv.r0.x, matInv.r0.y, matInv.r0.z, matInv.r0.w,
		   matInv.r1.x, matInv.r1.y, matInv.r1.z, matInv.r1.w,
		   matInv.r2.x, matInv.r2.y, matInv.r2.z, matInv.r2.w,
		   matInv.r3.x, matInv.r3.y, matInv.r3.z, matInv.r3.w
		   );
}

template <typename T>
bool Transform<T>::operator<(const Transform<T> &t2) const
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

template <typename T>
bool Transform<T>::IsIdentity() const
{
	return (m[0][0] == T(1) && m[0][1] == T(0) &&
			m[0][2] == T(0) && m[0][3] == T(0) &&
			m[1][0] == T(0) && m[1][1] == T(1) &&
			m[1][2] == T(0) && m[1][3] == T(0) &&
			m[2][0] == T(0) && m[2][1] == T(0) &&
			m[2][2] == T(1) && m[2][3] == T(0) &&
			m[3][0] == T(0) && m[3][1] == T(0) &&
			m[3][2] == T(0) && m[3][3] == T(1));
}

template <typename T>
inline vec3<T> Transform<T>::operator()(const vec3<T> &c, const int type) const
{
	T x = c.x, y = c.y, z = c.z;
	
	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		T xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		T yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		T zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp == T(1)) return make_vec3<T>(xp, yp, zp);
		else            return make_vec3<T>(xp, yp, zp)/wp;
	}
	else if (type == 2)
	{
		T xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
		T yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
		T zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
		return make_vec3<T>(xv, yv, zv);
	}
	else if (type == 3)
	{
		T xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
		T yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
		T zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
		return make_vec3<T>(xn, yn, zn);
	}
	else
	{
		printf("\"%i\" is an unknown type \n", type);
		printf(" P = \"lol\" \n");
		printf("Please select a type between: Point, Vector and Normal.\n");
		return c;
	}
}

template <typename T>
inline void Transform<T>::operator()(const vec3<T> &c, vec3<T> *ctrans, const int type) const
{
	T x = c.x, y = c.y, z = c.z;

	/* 1 = Point, 2 = Vector and 3 = Normal */
	if (type == 1)
	{
		ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
		ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
		ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
		T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
		if (wp != T(1)) *ctrans /= wp;
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
template <typename T>
inline vec3<T> Transform<T>::operator()(const Point<T> &c) const
{
	T x = c.x, y = c.y, z = c.z;
	
	T xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	T yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	T zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
	T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	
	if (wp == T(1)) return make_vec3<T>(xp, yp, zp);
	else            return make_vec3<T>(xp, yp, zp)/wp;
}

template <typename T>
inline vec3<T> Transform<T>::operator()(const Vector<T> &c) const
{
	T x = c.x, y = c.y, z = c.z;
	
	T xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	T yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	T zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	
	return make_vec3<T>(xv, yv, zv);
}

template <typename T>
inline vec3<T> Transform<T>::operator()(const Normal<T> &c) const
{
	T x = c.x, y = c.y, z = c.z;

	T xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	T yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	T zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	
	return make_vec3<T>(xn, yn, zn);
}

template <typename T>
inline void Transform<T>::operator()(const Point<T> &c, vec3<T> *ctrans) const
{
	T x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];

	T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	if (wp != T(1)) *ctrans /= wp;
}

template <typename T>
inline void Transform<T>::operator()(const Vector<T> &c, vec3<T> *ctrans) const
{
	T x = c.x, y = c.y, z = c.z;

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
}

template <typename T>
inline void Transform<T>::operator()(const Normal<T> &c, vec3<T> *ctrans) const
{
	T x = c.x, y = c.y, z = c.z;

	ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
}

template <typename T>
inline Ray<T> Transform<T>::operator()(const Ray<T> &r) const
{
    Ray<T> ret = r;
	(*this)(Point<T>(ret.o), &ret.o);
    (*this)(Vector<T>(ret.d), &ret.d);
    return ret;
}

template <typename T>
inline void Transform<T>::operator()(const Ray<T> &r, Ray<T> *rt) const
{
	(*this)(Point<T>(r.o), &rt->o);
    (*this)(Vector<T>(r.d), &rt->d);
    if (rt != &r)
	{
        rt->mint = r.mint;
        rt->maxt = r.maxt;
        rt->time = r.time;
    }
}

template <typename T>
BBox<T> Transform<T>::operator()(const BBox<T> &b) const
{
    const Transform<T> &M = *this;

    // creation du point P et du vecteur V=(v1, v2, v3)
	vec3<T> P, V1, V2, V3;

	// Application des transformations
	P = M(Point<T>(b.pMin));
	V1 = M(Vector<T>(b.pMax.x-b.pMin.x, T(0), T(0)));
	V2 = M(Vector<T>(T(0), b.pMax.y-b.pMin.y, T(0)));
	V3 = M(Vector<T>(T(0), T(0), b.pMax.z-b.pMin.z));

	// Creation de la box avec le 1er point P
	BBox<T> ret(P);

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

template <typename T>
Transform<T> Transform<T>::operator*(const Transform<T> &t2) const
{
    mat4x4<T> myM = mul(m, t2.m);
    mat4x4<T> myMinv = mul(t2.mInv, mInv);
    return Transform<T>(myM, myMinv);
}

template <typename T>
Transform<T> Transform<T>::Inverse(const Transform<T> &t)
{
	return Transform<T>(t.mInv, t.m);
}

template <typename T>
Transform<T> Transform<T>::Translate(const vec3<T> &delta)
{
	mat4x4<T> myM = make_mat4x4<T>(
		T(1), T(0), T(0), delta.x,
		T(0), T(1), T(0), delta.y,
		T(0), T(0), T(1), delta.z,
		T(0), T(0), T(0),     T(1)
		);
    mat4x4<T> myMinv = make_mat4x4<T>(
		T(1), T(0), T(0), -delta.x,
		T(0), T(1), T(0), -delta.y,
		T(0), T(0), T(1), -delta.z,
		T(0), T(0), T(0),      T(1)
		);
    return Transform<T>(myM, myMinv);
}

template <typename T>
Transform<T> Transform<T>::Scale(T x, T y, T z) {
    mat4x4<T> myM = make_mat4x4<T>(
		x,    T(0), T(0), T(0),
		T(0), y,    T(0), T(0),
		T(0), T(0), z,    T(0),
		T(0), T(0), T(0), T(1)
		);
    mat4x4<T> myMinv = make_mat4x4<T>(
		T(1)/x, T(0),   T(0),   T(0),
		T(0),   T(1)/y, T(0),   T(0),
		T(0),   T(0),   T(1)/z, T(0),
		T(0),   T(0),   T(0),   T(1))
		;
    return Transform<T>(myM, myMinv);
}

template <typename T>
Transform<T> Transform<T>::RotateX(T angle) {
	T sin_t = get_func_sin(get_func_radians(angle));
	T cos_t = get_func_cos(get_func_radians(angle));
    mat4x4<T> myM = make_mat4x4<T>(
		T(1),  T(0),   T(0), T(0),
		T(0), cos_t, -sin_t, T(0),
		T(0), sin_t,  cos_t, T(0),
		T(0),  T(0),   T(0), T(1)
		);
    return Transform<T>(myM, transpose(myM));
}

template <typename T>
Transform<T> Transform<T>::RotateY(T angle) {
	T sin_t = get_func_sin(get_func_radians(angle));
	T cos_t = get_func_cos(get_func_radians(angle));
    mat4x4<T> myM = make_mat4x4<T>(
		cos_t , T(0), sin_t, T(0),
		T(0),   T(1),  T(0), T(0),
		-sin_t,    0, cos_t, T(0),
		T(0),   T(0),  T(0), T(1)
		);
    return Transform<T>(myM, transpose(myM));
}

template <typename T>
Transform<T> Transform<T>::RotateZ(T angle) {
	T sin_t = get_func_sin(get_func_radians(angle));
	T cos_t = get_func_cos(get_func_radians(angle));
    mat4x4<T> m = make_mat4x4<T>(
		cos_t, -sin_t, T(0), T(0),
		sin_t,  cos_t, T(0), T(0),
		T(0),   T(0),  T(1), T(0),
		T(0),   T(0),  T(0), T(1));
    return Transform<T>(m, transpose(m));
}

template <typename T>
Transform<T> Transform<T>::Rotate(T angle, const vec3<T> &axis) {
	
	vec3<T> a = normalize(axis);
	T s = get_func_sin(get_func_radians(angle));
	T c = get_func_cos(get_func_radians(angle));
	mat4x4<T> m = make_diag_mat4x4<T>(T(1));
	
	m[0][0] = a.x * a.x + (T(1) - a.x * a.x) * c;
	m[0][1] = a.x * a.y * (T(1) - c) - a.z * s;
	m[0][2] = a.x * a.z * (T(1) - c) + a.y * s;

	m[1][0] = a.x * a.y * (T(1) - c) + a.z * s;
	m[1][1] = a.y * a.y + (T(1) - a.y * a.y) * c;
	m[1][2] = a.y * a.z * (T(1) - c) - a.x * s;

	m[2][0] = a.x * a.z * (T(1) - c) - a.y * s;
	m[2][1] = a.y * a.z * (T(1) - c) + a.x * s;
	m[2][2] = a.z * a.z + (T(1) - a.z * a.z) * c;

    return Transform<T>(m, transpose(m));
}

template <typename T>
Transform<T> Transform<T>::vec2transform(vec3<T> vi)
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
    Transform<T> tf;
    Transform<T> nothing;
    double3 v_ini_rotated;
	double3 v = make_double3(vi.x, vi.y, vi.z);
	v = normalize(v);
	vec3<T> v_tmp;


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
        
        tf = nothing.RotateZ(T(phi))*nothing.RotateY(T(theta));
		v_tmp = tf(make_float3(v_ini), 2);
		v_ini_rotated.x = T(v_tmp.x);
		v_ini_rotated.y = T(v_tmp.y);
		v_ini_rotated.z = T(v_tmp.z);
        v_ini_rotated = normalize(v_ini_rotated);

        if (abs(v.x-v_ini_rotated.x) < acc & abs(v.y-v_ini_rotated.y) < acc & abs(v.z-v_ini_rotated.z) < acc)
        {
			return tf;
        }
	}
	return nothing;
}

#endif // _TRANSFORM_H_
