
#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <math.h>
#include <helper_math.h>
#include <math_constants.h>
#include <limits>
#include <stdio.h>

#ifndef DEBUG
#define myError(expr) ((void)0)
#else
//#define myError(expr) ( (expr) ? \
						(printf("Point, Vector or Normal indices error***\n")) : (0) )
#define myError(expr) ((void)0)
#endif

#include <iterator>
/**********************************************************
*	> Basic geometric classe(s), e.g., a point, a vector,
      a normal, a ray...
***********************************************************/

template <typename T = float>
class Vector {
// ========================================================
// Vector class
// ========================================================
public:
    // Public methods
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

    // Public parameters
    T x, y, z;
private:
};

using Vectorf=Vector<float>;
using Vectord=Vector<double>;


template <typename T = float>
class Point {
// ========================================================
// Point class
// ========================================================
public:
    // Public methods
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

    // Public parameters
    T x, y, z;
private:
};

using Pointf=Point<float>;
using Pointd=Point<double>;


template <typename T = float>
class Normal {
// ========================================================
// Normal class
// ========================================================
public:
    // Public methods
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

    template <typename P3, typename V3>
	__host__ __device__ Ray(const P3 &origin, const V3 &direction, T start=T(0.),
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
	U3c o;           // ray origin point
	U3c d;           // ray direction vector
	T mint, maxt;    // t min and max values
	T time;          // variable t such that ray = o + d*t
 
private:
};

using Rayf = Ray<float>;
using Rayd = Ray<double>;


template <typename T = float> //T -> float / double
class BBox
// ========================================================
// BBox class
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
    
    template <typename P3, // P3 for float3/double3
              typename = typename std::enable_if<
                         !std::is_base_of<BBox<T>, std::decay_t<P3>>::value
                         >::type>
    __host__ __device__ BBox(const P3 &p)
		: pMin(make_vec3c<T>(T(p.x),T(p.y),T(p.z))), pMax(make_vec3c<T>(T(p.x),T(p.y),T(p.z))) { }

    template <typename U>
    __host__ __device__ BBox(const BBox<U> &b)
    {
        pMin = make_vec3c<T>(T(b.pMin.x), T(b.pMin.y), T(b.pMin.z));
        pMax = make_vec3c<T>(T(b.pMax.x), T(b.pMax.y), T(b.pMax.z));
    }

    template <typename P3_1, typename P3_2> // P3_1 and P3_2 for float3/double3  
    __host__ __device__ BBox(const P3_1 &p1, const P3_2 &p2)
	{
        // min instead of fmin to avoid ignoring a nan value
        // also min is for both float and double
        pMin = make_vec3c<T>(min(T(p1.x), T(p2.x)), min(T(p1.y), T(p2.y)), min(T(p1.z), T(p2.z)));
        pMax = make_vec3c<T>(max(T(p1.x), T(p2.x)), max(T(p1.y), T(p2.y)), max(T(p1.z), T(p2.z)));
    }

    template <typename U, typename P3>
	__host__ __device__ BBox<T> UnionPoint(const BBox<U> &b, const P3 &p)
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
	__host__ __device__ BBox<T> Union(const BBox<U> &b, const float3 &p){return UnionPoint<T>(b, p);}
    template <typename U>
	__host__ __device__ BBox<T> Union(const BBox<U> &b, const double3 &p){return UnionPoint<T>(b, p);}

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

    template <typename P3>
    __host__ __device__ bool Inside(const P3 &pt) const
	{
        return (T(pt.x) >= pMin.x && T(pt.x) <= pMax.x &&
                T(pt.y) >= pMin.y && T(pt.y) <= pMax.y &&
                T(pt.z) >= pMin.z && T(pt.z) <= pMax.z);
    }

    template <typename P3>
    __host__ __device__ bool AlmostInside(const P3 &pt) const
	{
        T EPS = 1e-4;
        return (T(pt.x) >= (pMin.x-EPS) && T(pt.x) <= (pMax.x+EPS) &&
                T(pt.y) >= (pMin.y-EPS) && T(pt.y) <= (pMax.y+EPS) &&
                T(pt.z) >= (pMin.z-EPS) && T(pt.z) <= (pMax.z+EPS));
    }

    template <typename P3>
    __host__ __device__ U3 RoundInside(const P3 &pt)
	{
        U3 ret;
        ret =  make_vec3<T>(
               min(max(T(pt.x), pMin.x), pMax.x),
               min(max(T(pt.y), pMin.y), pMax.y),
               min(max(T(pt.z), pMin.z), pMax.z));
		return ret;
    }

    template <typename P3>
    __host__ __device__ U3 RoundAlmostInside(const P3 &pt)
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
		T t0 = T(0), gamma3;
		T epsi = get_const_machine_eps(T{}) * T(0.5);
        T t1 = get_const_inf(T{});

		gamma3 = (T(3)*epsi)/(T(1) - T(3)*epsi);

		for (int i = 0; i < 3; ++i)
		{
            // Update interval for _i_th bounding box slab
            T invRayDir;
			if(T(ray.d[i]) != T(0)) invRayDir = T(1) / T(ray.d[i]);
            else invRayDir  = 1e32;
			T tNear = (pMin[i] - T(ray.o[i])) * invRayDir;
			T tFar  = (pMax[i] - T(ray.o[i])) * invRayDir;
			
			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) {swap(&tNear, &tFar);}
			
			// Update _tFar_ to ensure robust ray--bounds intersection
			tFar *= T(1) + T(2)*gamma3;
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) {return false;}
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	// Public parameters
    U3c pMin, pMax; // min and max points (U3c -> float3c/double3c)
private:
};

using BBoxf = BBox<float>;
using BBoxd = BBox<double>;

#endif // _GEOMETRY_H_
