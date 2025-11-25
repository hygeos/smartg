
#ifndef _TRANSFORM_H_
#define _TRANSFORM_H_

#ifdef OBJ3D
#include "geometry.h"
#endif

#include <math.h>
#include <helper_math.h>
#include <stdio.h>


/**********************************************************
*	> The class bellow enables the movement of objects
*     - Translations in x, y et z
*     - Rotations in à x, y et z
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
	template <typename U>
	__host__ __device__ Transform(const Transform<U> &t2);
	template <typename M4x4,
	          typename Enable = typename std::enable_if<
			  !std::is_base_of<Transform<typename M4x4::value_type>, M4x4>::value
			  >::type>
	__host__ __device__ Transform(const M4x4 &mat);
	template <typename M4x4_1, typename M4x4_2>
	__host__ __device__ Transform(const M4x4_1 &mat, const M4x4_2 &matInv);
	template <typename U>
    __host__ __device__ bool operator<(const Transform<U> &t2) const;
    __host__ __device__ bool IsIdentity() const;
	template <typename C3>
	inline __host__ __device__ U3 operator()(const C3 &c, const int type) const;
	template <typename C3_1, typename C3_2>
	inline __host__ __device__ void operator()(const C3_1 &c, C3_2 *ctrans, const int type) const;
	#ifdef OBJ3D
	template <typename U> inline __host__ __device__ U3 operator()(const Point<U> &c) const;
	template <typename U> inline __host__ __device__ U3 operator()(const Vector<U> &c) const;
	template <typename U> inline __host__ __device__ U3 operator()(const Normal<U> &c) const;
	template <typename U, typename C3>
	inline __host__ __device__ void operator()(const Point<U> &c, C3 *ctrans) const;
	template <typename U, typename C3>
	inline __host__ __device__ void operator()(const Vector<U> &c, C3 *ctrans) const;
	template <typename U, typename C3>
	inline __host__ __device__ void operator()(const Normal<U> &c, C3 *ctrans) const;
    template <typename U> inline __host__ __device__ Ray<T> operator()(const Ray<U> &r) const;
	template <typename U_1, typename U_2>
    inline __host__ __device__ void operator()(const Ray<U_1> &r, Ray<U_2> *rt) const;
    template <typename U> __host__ __device__ BBox<T> operator()(const BBox<U> &b) const;
	#endif
    template <typename U> __host__ __device__ Transform<T> operator*(const Transform<U> &t2) const;
    __host__ __device__ const U4x4 &GetMatrix() const { return m; }
    __host__ __device__ const U4x4 &GetInverseMatrix() const { return mInv; }
    template <typename U> __host__ __device__ Transform<T> Inverse(const Transform<U> &t);
    template <typename C3> __host__ __device__	Transform<T> Translate(const C3 &delta); // delta must be a vector
	template <typename U_1, typename U_2, typename U_3>
	__host__ __device__ Transform<T> Scale(U_1 x, U_2 y, U_3 z);                         // scale (factor) in x, y et z
	template <typename U> __host__ __device__ Transform<T> RotateX(U angle);             // rot in X  
	template <typename U> __host__ __device__ Transform<T> RotateY(U angle);             // rot in Y  
	template <typename U> __host__ __device__ Transform<T> RotateZ(U angle);             // rot in Z
	template <typename U, typename C3> __host__ __device__ Transform<T> Rotate(U angle, const C3 &axis);
	template <typename C3> __host__ __device__ Transform<T> vec2transform(C3 vi);

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
template <typename M4x4, typename Enable>
Transform<T>::Transform(const M4x4 &mat)
{
	m = make_mat4x4<T>(
		   T(mat.r0.x), T(mat.r0.y), T(mat.r0.z), T(mat.r0.w),
		   T(mat.r1.x), T(mat.r1.y), T(mat.r1.z), T(mat.r1.w),
		   T(mat.r2.x), T(mat.r2.y), T(mat.r2.z), T(mat.r2.w),
		   T(mat.r3.x), T(mat.r3.y), T(mat.r3.z), T(mat.r3.w)
		   );
	mInv = inverse(m);
}

template <typename T>
template <typename M4x4_1, typename M4x4_2>
Transform<T>::Transform(const M4x4_1 &mat, const M4x4_2 &matInv)
{
	m = make_mat4x4<T>(
			T(mat.r0.x), T(mat.r0.y), T(mat.r0.z), T(mat.r0.w),
			T(mat.r1.x), T(mat.r1.y), T(mat.r1.z), T(mat.r1.w),
			T(mat.r2.x), T(mat.r2.y), T(mat.r2.z), T(mat.r2.w),
		    T(mat.r3.x), T(mat.r3.y), T(mat.r3.z), T(mat.r3.w)
		    );
	mInv = make_mat4x4<T>(
		   	T(matInv.r0.x), T(matInv.r0.y), T(matInv.r0.z), T(matInv.r0.w),
		   	T(matInv.r1.x), T(matInv.r1.y), T(matInv.r1.z), T(matInv.r1.w),
		   	T(matInv.r2.x), T(matInv.r2.y), T(matInv.r2.z), T(matInv.r2.w),
		   	T(matInv.r3.x), T(matInv.r3.y), T(matInv.r3.z), T(matInv.r3.w)
		   	);
}

template <typename T>
template <typename U>
Transform<T>::Transform(const Transform<U> &t2)
{
	m = make_mat4x4<T>(
		T(t2.m.r0.x), T(t2.m.r0.y), T(t2.m.r0.z), T(t2.m.r0.w),
		T(t2.m.r1.x), T(t2.m.r1.y), T(t2.m.r1.z), T(t2.m.r1.w),
		T(t2.m.r2.x), T(t2.m.r2.y), T(t2.m.r2.z), T(t2.m.r2.w),
		T(t2.m.r3.x), T(t2.m.r3.y), T(t2.m.r3.z), T(t2.m.r3.w)
		);
	mInv = make_mat4x4<T>(
		   T(t2.mInv.r0.x), T(t2.mInv.r0.y), T(t2.mInv.r0.z), T(t2.mInv.r0.w),
		   T(t2.mInv.r1.x), T(t2.mInv.r1.y), T(t2.mInv.r1.z), T(t2.mInv.r1.w),
		   T(t2.mInv.r2.x), T(t2.mInv.r2.y), T(t2.mInv.r2.z), T(t2.mInv.r2.w),
		   T(t2.mInv.r3.x), T(t2.mInv.r3.y), T(t2.mInv.r3.z), T(t2.mInv.r3.w)
		   );
}

template <typename T>
template <typename U>
bool Transform<T>::operator<(const Transform<U> &t2) const
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (m[i][j] < T(t2.m[i][j])) return true;
			if (m[i][j] > T(t2.m[i][j])) return false;
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
template <typename C3>
inline vec3<T> Transform<T>::operator()(const C3 &c, const int type) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);
	
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
template <typename C3_1, typename C3_2>
inline void Transform<T>::operator()(const C3_1 &c, C3_2 *ctrans, const int type) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);

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
template <typename U>
inline vec3<T> Transform<T>::operator()(const Point<U> &c) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);
	
	T xp = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	T yp = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	T zp = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];
	T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	
	if (wp == T(1)) return make_vec3<T>(xp, yp, zp);
	else            return make_vec3<T>(xp, yp, zp)/wp;
}

template <typename T>
template <typename U>
inline vec3<T> Transform<T>::operator()(const Vector<U> &c) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);
	
	T xv = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	T yv = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	T zv = m[2][0]*x + m[2][1]*y + m[2][2]*z;
	
	return make_vec3<T>(xv, yv, zv);
}

template <typename T>
template <typename U>
inline vec3<T> Transform<T>::operator()(const Normal<U> &c) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);

	T xn = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	T yn = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	T zn = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
	
	return make_vec3<T>(xn, yn, zn);
}

template <typename T>
template <typename U, typename C3>
inline void Transform<T>::operator()(const Point<U> &c, C3 *ctrans) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z + m[0][3];
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z + m[1][3];
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z + m[2][3];

	T wp = m[3][0]*x + m[3][1]*y + m[3][2]*z + m[3][3];
	if (wp != T(1)) *ctrans /= wp;
}

template <typename T>
template <typename U, typename C3>
inline void Transform<T>::operator()(const Vector<U> &c, C3 *ctrans) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);

	ctrans->x = m[0][0]*x + m[0][1]*y + m[0][2]*z;
	ctrans->y = m[1][0]*x + m[1][1]*y + m[1][2]*z;
	ctrans->z = m[2][0]*x + m[2][1]*y + m[2][2]*z;
}

template <typename T>
template <typename U, typename C3>
inline void Transform<T>::operator()(const Normal<U> &c, C3 *ctrans) const
{
	T x = T(c.x), y = T(c.y), z = T(c.z);

	ctrans->x = mInv[0][0]*x + mInv[1][0]*y + mInv[2][0]*z;
	ctrans->y = mInv[0][1]*x + mInv[1][1]*y + mInv[2][1]*z;
	ctrans->z = mInv[0][2]*x + mInv[1][2]*y + mInv[2][2]*z;
}

template <typename T>
template <typename U>
inline Ray<T> Transform<T>::operator()(const Ray<U> &r) const
{
    Ray<T> ret(r);
	(*this)(Point<T>(ret.o), &ret.o);
    (*this)(Vector<T>(ret.d), &ret.d);
    return ret;
}

template <typename T>
template <typename U_1, typename U_2>
inline void Transform<T>::operator()(const Ray<U_1> &r, Ray<U_2> *rt) const
{
	(*this)(Point<T>(r.o), &rt->o);
    (*this)(Vector<T>(r.d), &rt->d);
	if constexpr (std::is_same_v<U_1, U_2>)
	{
		if (rt != &r)
		{
			rt->mint = r.mint;
			rt->maxt = r.maxt;
			rt->time = r.time;
		}
	}
	else
	{
		rt->mint = r.mint;
		rt->maxt = r.maxt;
		rt->time = r.time;
	}
}

template <typename T>
template <typename U>
BBox<T> Transform<T>::operator()(const BBox<U> &b) const
{
    const Transform<T> &M = *this;

    // Create point P and vectors V1, V2 and V3
	vec3<T> P, V1, V2, V3;

	// Apply transformations
	P = M(Point<T>(b.pMin));
	V1 = M(Vector<T>(T(b.pMax.x)-T(b.pMin.x), T(0), T(0)));
	V2 = M(Vector<T>(T(0), T(b.pMax.y)-T(b.pMin.y), T(0)));
	V3 = M(Vector<T>(T(0), T(0), T(b.pMax.z)-T(b.pMin.z)));

	// Create the box with the first point P
	BBox<T> ret(P);

    // Enlarge the box by taking one cube face
	// Face with 4 points : P, P+V.x, P+V.y, P+(V.x, V.y)
	ret = ret.Union(ret, P+V1);
	ret = ret.Union(ret, P+V2);
	ret = ret.Union(ret, P+V1+V2);

	// A point in z is enough (symetry)
	ret = ret.Union(ret, P+V3);
	/* ret = ret.Union(ret, P+V1+V3); */
	/* ret = ret.Union(ret, P+V2+V3); */
	/* ret = ret.Union(ret, P+V1+V2+V3); */
    return ret;
}
#endif

template <typename T>
template <typename U>
Transform<T> Transform<T>::operator*(const Transform<U> &t2) const
{
	Transform<T> t2_(t2);
    mat4x4<T> myM = mul(m, t2_.m);
    mat4x4<T> myMinv = mul(t2_.mInv, mInv);
    return Transform<T>(myM, myMinv);
}

template <typename T>
template <typename U>
Transform<T> Transform<T>::Inverse(const Transform<U> &t)
{
	return Transform<T>(t.mInv, t.m);
}

template <typename T>
template <typename C3>
Transform<T> Transform<T>::Translate(const C3 &delta)
{
	mat4x4<T> myM = make_mat4x4<T>(
		T(1), T(0), T(0), T(delta.x),
		T(0), T(1), T(0), T(delta.y),
		T(0), T(0), T(1), T(delta.z),
		T(0), T(0), T(0),       T(1)
		);
    mat4x4<T> myMinv = make_mat4x4<T>(
		T(1), T(0), T(0), T(-delta.x),
		T(0), T(1), T(0), T(-delta.y),
		T(0), T(0), T(1), T(-delta.z),
		T(0), T(0), T(0),        T(1)
		);
    return Transform<T>(myM, myMinv);
}

template <typename T>
template <typename U_1, typename U_2, typename U_3>
Transform<T> Transform<T>::Scale(U_1 x, U_2 y, U_3 z) {
    mat4x4<T> myM = make_mat4x4<T>(
		T(x), T(0), T(0), T(0),
		T(0), T(y), T(0), T(0),
		T(0), T(0), T(z), T(0),
		T(0), T(0), T(0), T(1)
		);
    mat4x4<T> myMinv = make_mat4x4<T>(
		T(1)/T(x), T(0),      T(0),      T(0),
		T(0),      T(1)/T(y), T(0),      T(0),
		T(0),      T(0),      T(1)/T(z), T(0),
		T(0),      T(0),      T(0),      T(1))
		;
    return Transform<T>(myM, myMinv);
}

template <typename T>
template <typename U>
Transform<T> Transform<T>::RotateX(U angle) {
	T sin_t = get_func_sin(get_func_radians(T(angle)));
	T cos_t = get_func_cos(get_func_radians(T(angle)));
    mat4x4<T> myM = make_mat4x4<T>(
		T(1),  T(0),   T(0), T(0),
		T(0), cos_t, -sin_t, T(0),
		T(0), sin_t,  cos_t, T(0),
		T(0),  T(0),   T(0), T(1)
		);
    return Transform<T>(myM, transpose(myM));
}

template <typename T>
template <typename U>
Transform<T> Transform<T>::RotateY(U angle) {
	T sin_t = get_func_sin(get_func_radians(T(angle)));
	T cos_t = get_func_cos(get_func_radians(T(angle)));
    mat4x4<T> myM = make_mat4x4<T>(
		cos_t , T(0), sin_t, T(0),
		T(0),   T(1),  T(0), T(0),
		-sin_t,    0, cos_t, T(0),
		T(0),   T(0),  T(0), T(1)
		);
    return Transform<T>(myM, transpose(myM));
}

template <typename T>
template <typename U>
Transform<T> Transform<T>::RotateZ(U angle) {
	T sin_t = get_func_sin(get_func_radians(T(angle)));
	T cos_t = get_func_cos(get_func_radians(T(angle)));
    mat4x4<T> m = make_mat4x4<T>(
		cos_t, -sin_t, T(0), T(0),
		sin_t,  cos_t, T(0), T(0),
		T(0),   T(0),  T(1), T(0),
		T(0),   T(0),  T(0), T(1));
    return Transform<T>(m, transpose(m));
}

template <typename T>
template <typename U, typename C3>
Transform<T> Transform<T>::Rotate(U angle, const C3 &axis) {
	
	vec3<T> a = make_vec3<T>(T(axis.x), T(axis.y), T(axis.z));
	a = normalize(a);
	T s = get_func_sin(get_func_radians(T(angle)));
	T c = get_func_cos(get_func_radians(T(angle)));
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
template <typename C3>
Transform<T> Transform<T>::vec2transform(C3 vi)
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
		v_tmp = tf(make_vec3<T>(v_ini.x, v_ini.y, v_ini.z), 2);
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

using Transformf = Transform<float>;
using Transformd = Transform<double>;

#endif // _TRANSFORM_H_
