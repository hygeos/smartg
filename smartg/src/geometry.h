
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
#define myError(expr) ( (expr) ? \
						(printf("Point, Vector or Normal indices error***\n")) : (0) )
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

class Ray
// ========================================================
// Classe Rayon
// ========================================================
{
public:
	// Méthodes publiques du rayon
	__host__ __device__ Ray()
	{
        #if __CUDA_ARCH__ >= 200
		maxt = CUDART_INF_F;
        #elif !defined(__CUDA_ARCH__)
		maxt = std::numeric_limits<float>::max();
        #endif
		mint = 0.f; time = 0.f;
		o = make_float3c(0., 0., 0.);
		d = make_float3c(0., 0., 0.);
	}

	__host__ __device__ Ray(const Ray &r)
	{
		mint = r.mint; maxt = r.maxt, time = r.time;
		o = r.o; d = r.d;
	}

	__device__ Ray(const float3 &origin, const float3 &direction, float start,
				   #if __CUDA_ARCH__ >= 200
				   float end = CUDART_INF_F, float t = 0.f)
		           #elif !defined(__CUDA_ARCH__)
		           float end = std::numeric_limits<float>::max(), float t = 0.f)
				   #endif
	{
		mint = start; maxt = end, time = t;
		o = make_float3c(origin);
		d = make_float3c(direction);
	}

    __host__ __device__ float3 operator()(float t) const
	{
		return o + d*t;
	}

	// Paramètres publiques du rayon
	float3c o;           // point d'origine du rayon
	float3c d;           // vecteur de direction du rayon
	float mint, maxt;    // valeur min et max de t
	float time;          // variable t: ray = o + d*t
 
private:
};


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
	__host__ __device__ bool IntersectP(const Ray &ray, float *hitt0 = NULL,
										float *hitt1 = NULL) const
	{
		float t0 = 0., t1 = ray.maxt; //float epsi = (std::numeric_limits<float>::epsilon() * 0.5);

		for (int i = 0; i < 3; ++i)
		{
            // Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;
			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar) {swap(&tNear, &tFar);}
			//tFar *= 1 + 2 * ( (3*epsi)/( 1-(3*epsi) ) );
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
