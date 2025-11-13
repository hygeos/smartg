
#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <math.h>
#include <helper_math.h>
#include <math_constants.h>
#include <limits>
#include <stdio.h>

#if __CUDA_ARCH__ >= 200
__device__ float machine_eps_flt() {
    typedef union {
        int i32;
        float f32;
    } flt_32;

    flt_32 s;

    s.f32 = 1.;
    s.i32++;
    return (s.f32 - 1.);
}
__device__ float machine_eps_dbl() {
    typedef union {
        long long i64;
        double d64;
    } dbl_64;

    dbl_64 s;

    s.d64 = 1.;
    s.i64++;
    return (s.d64 - 1.);
}
#endif

#ifndef DEBUG
#define myError(expr) ((void)0)
#else
//#define myError(expr) ( (expr) ? \
						(printf("Point, Vector or Normal indices error***\n")) : (0) )
#define myError(expr) ((void)0)
#endif

#include <iterator>
/**********************************************************
*	> Classe(s) représentant géométriquement quelque chose
*     - ex: Un Point, Un vecteur, une normal, un rayon...
***********************************************************/

template <typename T = float>
class Vector {
// ========================================================
// Classe Vector
// ========================================================
public:
    // Vector Public Methods
    __host__ __device__ Vector() { x = 0; y = 0; z = 0; }
	
    __host__ __device__ Vector(T xx, T yy, T zz)
	{ x = xx; y = yy; z = zz; }

	template <typename U>
	__host__ __device__ Vector(U v)
	{x = (T)v.x; y = (T)v.y; z = (T)v.z; }

    __host__ __device__ Vector<T> &operator=(const Vector<T>  &v) {
        x = v.x; y = v.y; z = v.z;
        return *this;
    }

    __host__ __device__ Vector<T>  operator+(const Vector<T>  &v) const
	{ return Vector<T>(x + v.x, y + v.y, z + v.z); }
    
    __host__ __device__ Vector<T>  operator+=(const Vector<T>  &v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    __host__ __device__ Vector<T>  operator-(const Vector<T>  &v) const
	{ return Vector<T>(x - v.x, y - v.y, z - v.z); }
    
    __host__ __device__ Vector<T> &operator-=(const Vector<T>  &v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }

	template <typename U>
    __host__ __device__ Vector<T>  operator*(U f) const { return Vector<T>(f*x, f*y, f*z); }

	template <typename U>
    __host__ __device__ Vector<T> &operator*=(U f) {
        x *= f; y *= f; z *= f;
        return *this;
    }

	template <typename U>
    __host__ __device__ Vector<T> operator/(U f) const {
        T inv = T(1) / f;
        return Vector<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Vector<T> &operator/=(U f) {
        T inv = T(1) / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    __host__ __device__ Vector<T> operator-() const { return Vector<T>(-x, -y, -z); }
	
    __host__ __device__ T operator[](int idx) const {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }
    
    __host__ __device__ T &operator[](int idx) {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }

    __host__ __device__ bool operator==(const Vector<T> &v) const
	{ return x == v.x && y == v.y && z == v.z; }
	
    __host__ __device__ bool operator!=(const Vector<T> &v) const
	{ return x != v.x || y != v.y || z != v.z; }

    // Vector Public Attributs
    T x, y, z;
private:
};

using Vectorf=Vector<float>;
using Vectord=Vector<double>;


template <typename T = float>
class Point {
// ========================================================
// Classe Point
// ========================================================
public:
    // Point Public Methods
    __host__ __device__ Point() { x = 0; y = 0; z = 0; }
	
    __host__ __device__ Point(T xx, T yy, T zz)
	{ x = xx; y = yy; z = zz; }

	template <typename U>
	__host__ __device__ Point(U p)
	{ x = (T)p.x; y = (T)p.y; z = (T)p.z; }
    
    __host__ __device__ Point<T> &operator=(const Point<T>  &p) {
        x = p.x; y = p.y; z = p.z;
        return *this;
    }

    __host__ __device__ Point<T>  operator+(const Point<T>  &p) const
	{ return Point<T>(x + p.x, y + p.y, z + p.z); }
    
    __host__ __device__ Point<T>  operator+=(const Point<T>  &p) {
        x += p.x; y += p.y; z += p.z;
        return *this;
    }
    __host__ __device__ Point<T>  operator-(const Point<T>  &p) const
	{ return Point<T>(x - p.x, y - p.y, z - p.z); }
    
    __host__ __device__ Point<T> &operator-=(const Point<T>  &p) {
        x -= p.x; y -= p.y; z -= p.z;
        return *this;
    }

	template <typename U>
    __host__ __device__ Point<T>  operator*(U f) const { return Point<T>(f*x, f*y, f*z); }

	template <typename U>
    __host__ __device__ Point<T> &operator*=(U f) {
        x *= f; y *= f; z *= f;
        return *this;
    }

	template <typename U>
    __host__ __device__ Point<T> operator/(U f) const {
        T inv = T(1) / f;
        return Point<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Point<T> &operator/=(U f) {
        T inv = T(1) / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    __host__ __device__ Point<T> operator-() const { return Point<T>(-x, -y, -z); }
	
    __host__ __device__ T operator[](int idx) const {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }
    
    __host__ __device__ T &operator[](int idx) {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }

    __host__ __device__ bool operator==(const Point<T> &p) const
	{ return x == p.x && y == p.y && z == p.z; }
	
    __host__ __device__ bool operator!=(const Point<T> &p) const
	{ return x != p.x || y != p.y || z != p.z; }

    // Vector Public Data
    T x, y, z;
private:
};

using Pointf=Point<float>;
using Pointd=Point<double>;


template <typename T = float>
class Normal {
// ========================================================
// Classe Normal
// ========================================================
public:
    // Point Public Methods
    __host__ __device__ Normal() { x = 0; y = 0; z = 0; }
	
    __host__ __device__ Normal(T xx, T yy, T zz)
	{ x = xx; y = yy; z = zz; }

	template <typename U>
	__host__ __device__ Normal(U n)
	{ x = (T)n.x; y = (T)n.y; z = (T)n.z; }
    
    __host__ __device__ Normal<T> &operator=(const Normal<T>  &n) {
        x = n.x; y = n.y; z = n.z;
        return *this;
    }

    __host__ __device__ Normal<T>  operator+(const Normal<T>  &n) const
	{ return Normal<T>(x + n.x, y + n.y, z + n.z); }
    
    __host__ __device__ Normal<T>  operator+=(const Normal<T>  &n) {
        x += n.x; y += n.y; z += n.z;
        return *this;
    }
    __host__ __device__ Normal<T>  operator-(const Normal<T>  &n) const
	{ return Normal<T>(x - n.x, y - n.y, z - n.z); }
    
    __host__ __device__ Normal<T> &operator-=(const Normal<T>  &n) {
        x -= n.x; y -= n.y; z -= n.z;
        return *this;
    }

	template <typename U>
    __host__ __device__ Normal<T>  operator*(U f) const { return Normal<T>(f*x, f*y, f*z); }

	template <typename U>
    __host__ __device__ Normal<T> &operator*=(U f) {
        x *= f; y *= f; z *= f;
        return *this;
    }

	template <typename U>
    __host__ __device__ Normal<T> operator/(U f) const {
        T inv = T(1) / f;
        return Normal<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Normal<T> &operator/=(U f) {
        T inv = T(1) / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    __host__ __device__ Normal<T> operator-() const { return Normal<T>(-x, -y, -z); }
	
    __host__ __device__ T operator[](int idx) const {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }
    
    __host__ __device__ T &operator[](int idx) {
		myError((idx < 0) || (idx > 1));
        return (&x)[idx];
    }

    __host__ __device__ bool operator==(const Normal<T> &n) const
	{ return x == n.x && y == n.y && z == n.z; }
	
    __host__ __device__ bool operator!=(const Normal<T> &n) const
	{ return x != n.x || y != n.y || z != n.z; }

    // Vector Public Data
    T x, y, z;
private:
};

using Normalf=Normal<float>;
using Normald=Normal<double>;


template <typename T = float> //T -> float / double
class Ray
// ========================================================
// Ray class
// ========================================================
{
public:
	// Public methods of class Ray
	using U3  = vec3<T>;
    using U3c = vec3c<T>;

	__host__ __device__ Ray()
	{
		//maxt = RAY_INF;
		mint = 0.; time = 0.;
		maxt = get_const_inf(T{});
		o = make_vec3c<T>(T(0), T(0), T(0));
		d = make_vec3c<T>(T(0), T(0), T(0));
	}

	template <typename U>
	__host__ __device__ Ray(const Ray<U> &r)
	{
		mint = T(r.mint); maxt = T(r.maxt), time = T(r.time);
		o = make_vec3c<T>(T(r.o.x), T(r.o.y), T(r.o.z));
		d = make_vec3c<T>(T(r.d.x), T(r.d.y), T(r.d.z));
	}

	__host__ __device__ Ray(const U3 &origin, const U3 &direction, T start=T(0.),
				   T end = get_const_inf(T{}), T t = T(0.))
	{
		mint = start; maxt = end; time = t;
		o = make_vec3c<T>(T(origin.x), T(origin.y), T(origin.z));
		d = make_vec3c<T>(T(direction.x), T(direction.y), T(direction.z));
	}

	template <typename U>
    __host__ __device__ U3 operator()(U t) const
	{
		return o + d*T(t);
	}

	// Public parameters
	U3c o;           // point d'origine du rayon
	U3c d;           // vecteur de direction du rayon
	T mint, maxt;    // valeur min et max de t
	T time;          // variable t: ray = o + d*t
 
private:
};

using Rayf = Ray<float>;
using Rayd = Ray<double>;


template <typename T = float> //T -> float / double
class BBox
// ========================================================
// Classe BBox
// ========================================================
{
public:
    using U3  = vec3<T>;
    using U3c = vec3c<T>;

	// Public methods
	__host__ __device__ BBox()
	{
        pMin = make_vec3c<T>(get_const_inf(T{}), get_const_inf(T{}), get_const_inf(T{}));
        pMax = make_vec3c<T>(-get_const_inf(T{}), -get_const_inf(T{}), -get_const_inf(T{}));
	}

    __host__ __device__ BBox(const float3 &p)
		: pMin(make_vec3c<T>(T(p.x),T(p.y),T(p.z))), pMax(make_vec3c<T>(T(p.x),T(p.y),T(p.z))) { }
    
    __host__ __device__ BBox(const double3 &p)
		: pMin(make_vec3c<T>(T(p.x),T(p.y),T(p.z))), pMax(make_vec3c<T>(T(p.x),T(p.y),T(p.z))) { }


	__host__ __device__ BBox(const float3 &p1, const float3 &p2)
	{
        // min instead of fmin to avoid ignoring a nan value
        // also min is for both float and double
        pMin = make_vec3c<T>(min(T(p1.x), T(p2.x)), min(T(p1.y), T(p2.y)), min(T(p1.z), T(p2.z)));
        pMax = make_vec3c<T>(max(T(p1.x), T(p2.x)), max(T(p1.y), T(p2.y)), max(T(p1.z), T(p2.z)));
    }

	__host__ __device__ BBox(const float3 &p1, const double3 &p2)
	{
        pMin = make_vec3c<T>(min(T(p1.x), T(p2.x)), min(T(p1.y), T(p2.y)), min(T(p1.z), T(p2.z)));
        pMax = make_vec3c<T>(max(T(p1.x), T(p2.x)), max(T(p1.y), T(p2.y)), max(T(p1.z), T(p2.z)));
    }

    __host__ __device__ BBox(const double3 &p1, const float3 &p2)
	{
        pMin = make_vec3c<T>(min(T(p1.x), T(p2.x)), min(T(p1.y), T(p2.y)), min(T(p1.z), T(p2.z)));
        pMax = make_vec3c<T>(max(T(p1.x), T(p2.x)), max(T(p1.y), T(p2.y)), max(T(p1.z), T(p2.z)));
    }

	__host__ __device__ BBox(const double3 &p1, const double3 &p2)
	{
        pMin = make_vec3c<T>(min(T(p1.x), T(p2.x)), min(T(p1.y), T(p2.y)), min(T(p1.z), T(p2.z)));
        pMax = make_vec3c<T>(max(T(p1.x), T(p2.x)), max(T(p1.y), T(p2.y)), max(T(p1.z), T(p2.z)));
    }

    template <typename U>
	__host__ __device__ BBox<T> Union(const BBox<U> &b, const float3 &p)
	{
		BBox<T> ret;
		ret.pMin.x = min(T(b.pMin.x), T(p.x));
		ret.pMin.y = min(T(b.pMin.y), T(p.y));
		ret.pMin.z = min(T(b.pMin.z), T(p.z));
		ret.pMax.x = max(T(b.pMax.x), T(p.x));
		ret.pMax.y = max(T(b.pMax.y), T(p.y));
		ret.pMax.z = max(T(b.pMax.z), T(p.z));
		return ret;
	}

    template <typename U>
	__host__ __device__ BBox<T> Union(const BBox<U> &b, const double3 &p)
	{
		BBox<T> ret;
		ret.pMin.x = min(T(b.pMin.x), T(p.x));
		ret.pMin.y = min(T(b.pMin.y), T(p.y));
		ret.pMin.z = min(T(b.pMin.z), T(p.z));
		ret.pMax.x = max(T(b.pMax.x), T(p.x));
		ret.pMax.y = max(T(b.pMax.y), T(p.y));
		ret.pMax.z = max(T(b.pMax.z), T(p.z));
		return ret;
	}

	template <typename U_1, typename U_2>
	__host__ __device__ BBox<T> Union(const BBox<U_1> &b, const BBox<U_2> &b2)
	{
		BBox<T> ret;
		ret.pMin.x = min(T(b.pMin.x), T(b2.pMin.x));
		ret.pMin.y = min(T(b.pMin.y), T(b2.pMin.y));
		ret.pMin.z = min(T(b.pMin.z), T(b2.pMin.z));
		ret.pMax.x = max(T(b.pMax.x), T(b2.pMax.x));
		ret.pMax.y = max(T(b.pMax.y), T(b2.pMax.y));
		ret.pMax.z = max(T(b.pMax.z), T(b2.pMax.z));
		return ret;
	}

    __host__ __device__ bool Inside(const float3 &pt) const
	{
        return (T(pt.x) >= pMin.x && T(pt.x) <= pMax.x &&
                T(pt.y) >= pMin.y && T(pt.y) <= pMax.y &&
                T(pt.z) >= pMin.z && T(pt.z) <= pMax.z);
    }

    __host__ __device__ bool Inside(const double3 &pt) const
	{
        return (T(pt.x) >= pMin.x && T(pt.x) <= pMax.x &&
                T(pt.y) >= pMin.y && T(pt.y) <= pMax.y &&
                T(pt.z) >= pMin.z && T(pt.z) <= pMax.z);
    }

    __host__ __device__ bool AlmostInside(const float3 &pt) const
	{
        T EPS = 1e-4;
        return (T(pt.x) >= (pMin.x-EPS) && T(pt.x) <= (pMax.x+EPS) &&
                T(pt.y) >= (pMin.y-EPS) && T(pt.y) <= (pMax.y+EPS) &&
                T(pt.z) >= (pMin.z-EPS) && T(pt.z) <= (pMax.z+EPS));
    }

    __host__ __device__ bool AlmostInside(const double3 &pt) const
	{
        T EPS = 1e-4;
        return (T(pt.x) >= (pMin.x-EPS) && T(pt.x) <= (pMax.x+EPS) &&
                T(pt.y) >= (pMin.y-EPS) && T(pt.y) <= (pMax.y+EPS) &&
                T(pt.z) >= (pMin.z-EPS) && T(pt.z) <= (pMax.z+EPS));
    }

    __host__ __device__ U3 RoundInside(const float3 &pt)
	{
        U3 ret;
        ret =  make_vec3<T>(
               min(max(T(pt.x), pMin.x), pMax.x),
               min(max(T(pt.y), pMin.y), pMax.y),
               min(max(T(pt.z), pMin.z), pMax.z));
		return ret;
    }

    __host__ __device__ U3 RoundInside(const double3 &pt)
	{
        U3 ret;
        ret =  make_vec3<T>(
               min(max(T(pt.x), pMin.x), pMax.x),
               min(max(T(pt.y), pMin.y), pMax.y),
               min(max(T(pt.z), pMin.z), pMax.z));
		return ret;
    }

    __host__ __device__ U3 RoundAlmostInside(const float3 &pt)
	{
        T EPS = 1e-5;
        U3 ret;
        ret =  make_vec3<T>(
               min(max(T(pt.x), pMin.x+EPS), pMax.x-EPS),
               min(max(T(pt.y), pMin.y+EPS), pMax.y-EPS),
               min(max(T(pt.z), pMin.z+EPS), pMax.z-EPS));
		return ret;
    }

    __host__ __device__ U3 RoundAlmostInside(const double3 &pt)
	{
        T EPS = 1e-5;
        U3 ret;
        ret =  make_vec3<T>(
               min(max(T(pt.x), pMin.x+EPS), pMax.x-EPS),
               min(max(T(pt.y), pMin.y+EPS), pMax.y-EPS),
               min(max(T(pt.z), pMin.z+EPS), pMax.z-EPS));
		return ret;
    }

    template <typename U>
	__host__ __device__ bool IntersectP(const Ray<U> &ray, T *hitt0 = NULL,
										T *hitt1 = NULL) const
	{
		T t0 = T(0.), gamma3;
        #if __CUDA_ARCH__ >= 200
		T epsi = machine_eps_flt() * T(0.5);
		#elif !defined(__CUDA_ARCH__)
		T epsi = (std::numeric_limits<T>::epsilon() * 0.5);
		#endif
        T t1 = get_const_inf(T{});

		gamma3 = (3*epsi)/(1 - 3*epsi);

		for (int i = 0; i < 3; ++i)
		{
            // Update interval for _i_th bounding box slab
            T invRayDir;
			if(T(ray.d[i]) != T(0.)) invRayDir = T(1.) / T(ray.d[i]);
            else invRayDir  = 1e32;
			T tNear = (pMin[i] - T(ray.o[i])) * invRayDir;
			T tFar  = (pMax[i] - T(ray.o[i])) * invRayDir;
			
			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) {swap(&tNear, &tFar);}
			
			// Update _tFar_ to ensure robust ray--bounds intersection
			tFar *= 1 + 2*gamma3;
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) {return false;}
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	// Paramètres publiques de la Box
    U3c pMin, pMax; // point min et point max
private:
};

using BBoxf = BBox<float>;
using BBoxd = BBox<double>;

#endif // _GEOMETRY_H_
