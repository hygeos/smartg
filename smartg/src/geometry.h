
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

template <typename T>
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
        float inv = 1.f / f;
        return Vector<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Vector<T> &operator/=(U f) {
        float inv = 1.f / f;
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

typedef Vector<double> Vectord;
typedef Vector<float> Vectorf;

template <typename T>
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
        float inv = 1.f / f;
        return Point<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Point<T> &operator/=(U f) {
        float inv = 1.f / f;
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

typedef Point<double> Pointd;
typedef Point<float> Pointf;


template <typename T>
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
        float inv = 1.f / f;
        return Normal<T>(x * inv, y * inv, z * inv);
    }

	template <typename U>
    __host__ __device__ Normal<T> &operator/=(U f) {
        float inv = 1.f / f;
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

typedef Normal<double> Normald;
typedef Normal<float> Normalf;


template <typename T = float> //T -> float / double
class Ray
// ========================================================
// Ray class
// ========================================================
{
public:
	// Public methods of class Ray

	// Constexpr infinity value based on T, using a lambda for initialization
    // Constant class attribute
	static constexpr T RAY_INF =
	#if __CUDA_ARCH__ >= 200
		std::is_same_v<T, float> ? CUDART_INF_F : CUDART_INF;
	#else
		std::numeric_limits<T>::max();
	#endif

	__host__ __device__ Ray()
	{
		maxt = RAY_INF;
		mint = 0.; time = 0.;
		if constexpr (std::is_same_v<T, float>)
		{
			o = make_float3c(0., 0., 0.);
			d = make_float3c(0., 0., 0.);
		}
		else //double
		{
			o = make_double3c(0., 0., 0.);
			d = make_double3c(0., 0., 0.);
		}
	}

	template <typename U>
	__host__ __device__ Ray(const Ray<U> &r)
	{
		mint = T(r.mint); maxt = T(r.maxt), time = T(r.time);
		if constexpr (std::is_same_v<T, float>)
		{
			o = make_float3c(r.o);
			d = make_float3c(r.d);
		}
		else
		{
			o = make_double3c(r.o);
			d = make_double3c(r.d);
		}
	}

	template <typename U3>
	__device__ Ray(const U3 &origin, const U3 &direction, T start=0.,
				   T end = RAY_INF, T t = 0.)
	{
		mint = start; maxt = end, time = t;
		if constexpr (std::is_same_v<T, float>)
		{
			o = make_float3c(origin);
			d = make_float3c(direction);
		}
		else
		{
			o = make_double3c(origin);
			d = make_double3c(direction);
		}
	}

	using U3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;
	template <typename U>
    __host__ __device__ U3 operator()(U t) const
	{
		return o + d*T(t);
	}

	using U3c = std::conditional_t<std::is_same_v<T, float>, float3c, double3c>;
	// Public parameters
	U3c o;           // point d'origine du rayon
	U3c d;           // vecteur de direction du rayon
	T mint, maxt;    // valeur min et max de t
	T time;          // variable t: ray = o + d*t
 
private:
};

using Rayd = Ray<double>;
using Rayf = Ray<float>;


class BBox
// ========================================================
// Classe BBox
// ========================================================
{
public:
	// Méthodes publiques de la Box
	__host__ __device__ BBox()
	{
        #if __CUDA_ARCH__ >= 200
		pMin = make_float3c( CUDART_INF_F,  CUDART_INF_F,  CUDART_INF_F);
		pMax = make_float3c(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
        #elif !defined(__CUDA_ARCH__)
		pMin = make_float3c(std::numeric_limits<float>::max(),
							std::numeric_limits<float>::max(),
							std::numeric_limits<float>::max());
		pMax = make_float3c(-std::numeric_limits<float>::max(),
							-std::numeric_limits<float>::max(),
							-std::numeric_limits<float>::max());
        #endif
	}
    __host__ __device__ BBox(const float3 &p)
		: pMin(make_float3c(p)), pMax(make_float3c(p)) { }

	__host__ __device__ BBox(const float3 &p1, const float3 &p2)
	{
		#if __CUDA_ARCH__ >= 200
        pMin = make_float3c(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = make_float3c(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
		#elif !defined(__CUDA_ARCH__)
		pMin = make_float3c(min(p1.x, p2.x), min(p1.y, p2.y), min(p1.z, p2.z));
        pMax = make_float3c(max(p1.x, p2.x), max(p1.y, p2.y), max(p1.z, p2.z));
		#endif
    }

	__host__ __device__ BBox Union(const BBox &b, const float3 &p)
	{
		BBox ret = b;
        #if __CUDA_ARCH__ >= 200
		ret.pMin.x = fmin(b.pMin.x, p.x);
		ret.pMin.y = fmin(b.pMin.y, p.y);
		ret.pMin.z = fmin(b.pMin.z, p.z);
		ret.pMax.x = fmax(b.pMax.x, p.x);
		ret.pMax.y = fmax(b.pMax.y, p.y);
		ret.pMax.z = fmax(b.pMax.z, p.z);
		#elif !defined(__CUDA_ARCH__)
		ret.pMin.x = min(b.pMin.x, p.x);
		ret.pMin.y = min(b.pMin.y, p.y);
		ret.pMin.z = min(b.pMin.z, p.z);
		ret.pMax.x = max(b.pMax.x, p.x);
		ret.pMax.y = max(b.pMax.y, p.y);
		ret.pMax.z = max(b.pMax.z, p.z);
		#endif
		return ret;
	}
	
	__host__ __device__ BBox Union(const BBox &b, const BBox &b2)
	{
		BBox ret;
		#if __CUDA_ARCH__ >= 200
		ret.pMin.x = fmin(b.pMin.x, b2.pMin.x);
		ret.pMin.y = fmin(b.pMin.y, b2.pMin.y);
		ret.pMin.z = fmin(b.pMin.z, b2.pMin.z);
		ret.pMax.x = fmax(b.pMax.x, b2.pMax.x);
		ret.pMax.y = fmax(b.pMax.y, b2.pMax.y);
		ret.pMax.z = fmax(b.pMax.z, b2.pMax.z);
		#elif !defined(__CUDA_ARCH__)
		ret.pMin.x = min(b.pMin.x, b2.pMin.x);
		ret.pMin.y = min(b.pMin.y, b2.pMin.y);
		ret.pMin.z = min(b.pMin.z, b2.pMin.z);
		ret.pMax.x = max(b.pMax.x, b2.pMax.x);
		ret.pMax.y = max(b.pMax.y, b2.pMax.y);
		ret.pMax.z = max(b.pMax.z, b2.pMax.z);
		#endif
		return ret;
	}

    __host__ __device__ bool Inside(const float3 &pt) const
	{
        return (pt.x >= pMin.x && pt.x <= pMax.x &&
                pt.y >= pMin.y && pt.y <= pMax.y &&
                pt.z >= pMin.z && pt.z <= pMax.z);
    }

    __host__ __device__ bool AlmostInside(const float3 &pt) const
	{
        float EPS = 1e-4;
        return (pt.x >= (pMin.x-EPS) && pt.x <= (pMax.x+EPS) &&
                pt.y >= (pMin.y-EPS) && pt.y <= (pMax.y+EPS) &&
                pt.z >= (pMin.z-EPS) && pt.z <= (pMax.z+EPS));
    }

    __host__ __device__ float3 RoundInside(const float3 &pt)
	{
        float3 ret;
		#if __CUDA_ARCH__ >= 200
        ret =  make_float3(
               fmin(fmax(pt.x, pMin.x), pMax.x),
               fmin(fmax(pt.y, pMin.y), pMax.y),
               fmin(fmax(pt.z, pMin.z), pMax.z));
		#elif !defined(__CUDA_ARCH__)
        ret =  make_float3(
               min(max(pt.x, pMin.x), pMax.x),
               min(max(pt.y, pMin.y), pMax.y),
               min(max(pt.z, pMin.z), pMax.z));
		#endif
		return ret;
    }


    __host__ __device__ float3 RoundAlmostInside(const float3 &pt)
	{
        float EPS = 1e-5;
        float3 ret;
		#if __CUDA_ARCH__ >= 200
        ret =  make_float3(
               fmin(fmax(pt.x, pMin.x+EPS), pMax.x-EPS),
               fmin(fmax(pt.y, pMin.y+EPS), pMax.y-EPS),
               fmin(fmax(pt.z, pMin.z+EPS), pMax.z-EPS));
		#elif !defined(__CUDA_ARCH__)
        ret =  make_float3(
               min(max(pt.x, pMin.x+EPS), pMax.x-EPS),
               min(max(pt.y, pMin.y+EPS), pMax.y-EPS),
               min(max(pt.z, pMin.z+EPS), pMax.z-EPS));
		#endif
		return ret;
    }



	__host__ __device__ bool IntersectP(const Ray<float> &ray, float *hitt0 = NULL,
										float *hitt1 = NULL) const
	{
		float t0 = 0.F, gamma3;
        #if __CUDA_ARCH__ >= 200
		float epsi = machine_eps_flt() * 0.5;
		float t1 = CUDART_INF_F;
		#elif !defined(__CUDA_ARCH__)
		float epsi = (std::numeric_limits<float>::epsilon() * 0.5);
		float t1 = std::numeric_limits<float>::max();
		#endif

		gamma3 = (3*epsi)/(1 - 3*epsi);

		for (int i = 0; i < 3; ++i)
		{
            // Update interval for _i_th bounding box slab
            float invRayDir;
			if(ray.d[i] != 0.) invRayDir = 1.F / ray.d[i];
            else invRayDir  = 1e32;
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;
			
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
    float3c pMin, pMax; // point min et point max
private:
};
#endif // _GEOMETRY_H_
