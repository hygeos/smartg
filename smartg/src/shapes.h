
#ifndef _SHAPES_H_
#define _SHAPES_H_

#include "geometry.h"
#include "transform.h"

#include <math.h>
#include <math_constants.h>
#include <limits>
#include <helper_math.h>
#include <stdio.h>

/************************************************************************
*	> Class(es)/structure(s) representing a shape (sphere, triangle, ...)
************************************************************************/

template <typename T = float> //T -> float / double
class Shape // must be defined before the DifferentialGeometry structure
// ========================================================
// Shape class. Parent to all shape classes
// ========================================================
{
public:
    // Public parameters
	__host__ __device__ Shape()
	{
		// Initialize transformations with an empty Transform object
		Transform<T> nothing;
		ObjectToWorld = &nothing;
		WorldToObject = &nothing;
		shapeId = 0; // for the moment not used at all
	}

	template <typename U_1, typename U_2>
	__host__ __device__ Shape(const Transform<U_1> *o2w, const Transform<U_2> *w2o)
		{
			Transform<T> o2w_prv(o2w->GetMatrix(), o2w->GetInverseMatrix());
			Transform<T> w2o_prv(w2o->GetMatrix(), w2o->GetInverseMatrix());
			ObjectToWorld = &o2w_prv;
			WorldToObject = &w2o_prv;
			shapeId = 0; // for the moment not used at all
		}

    // Public parameters
	unsigned long int shapeId;
    const Transform<T> *ObjectToWorld, *WorldToObject;

private:
};

using Shapef = Shape<float>;
using Shaped = Shape<double>;


template <typename T = float> //T -> float / double
struct DifferentialGeometry // must be defined before shape child classes
// ========================================================
// DifferentialGeometry class
// ========================================================
{
	using U3  = vec3<T>;

    __host__ __device__ DifferentialGeometry()
	{
		p = make_vec3<T>(T(0), T(0), T(0)); dpdu = make_vec3<T>(T(0), T(0), T(0));
		dpdv = make_vec3<T>(T(0), T(0), T(0)); nn = make_vec3<T>(T(0), T(0), T(0));
		u = v = T(0); shape = NULL;
	}

	template <typename C3_1, typename C3_2, typename C3_3, typename U_1, typename U_2, typename U_3>
    __host__ __device__ DifferentialGeometry(const C3_1 &P, const C3_2 &DPDU,
						                     const C3_3 &DPDV, U_1 uu, U_2 vv, const Shape<U_3> *sh)
	{
		p = make_vec3<T>(T(P.x), T(P.y), T(P.z));
		dpdu = make_vec3<T>(T(DPDU.x), T(DPDU.y), T(DPDU.z));
		dpdv = make_vec3<T>(T(DPDV.x), T(DPDV.y), T(DPDV.z));
		nn = normalize(cross(dpdu, dpdv));
		u = T(uu);
		v = T(vv);
		shape = reinterpret_cast<const Shape<T>*>(sh);
	}

	U3 p;             // intersection point position
	U3 dpdu;          // partial derivative with respect to u
	U3 dpdv;          // partial derivative with respect to v
	U3 nn;            // normal at intersection point
	T u;              // parameter: P=f(u,v)
	T v;              // parameter: P=f(u,v)
	const Shape<T> *shape;   // the geometry used
};

using DifferentialGeometryf = DifferentialGeometry<float>;
using DifferentialGeometryd = DifferentialGeometry<double>;


template <typename T = float, typename TF = float> //T -> float / double
// T is type for the BBox/attributs of the class and for main operation inside the methods
// TF is type used in parent Shape class i.e., transformation
class Sphere : public Shape<TF>
// ========================================================
// Sphere class
// ========================================================
{
public:
	// Public methods
	__host__ __device__ Sphere();
	template <typename U_1, typename U_2, typename U_3, typename U_4,
	          typename U_5, typename U_6>
	__host__ __device__ Sphere(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
							   U_3 rad, U_4 zmin, U_5 zmax, U_6 phiMax);

    __device__ BBox<T> ObjectBoundSphere() const;
    __device__ BBox<T> WorldBoundSphere() const;

	template <typename U_1, typename U_2, typename U_3>
    __host__ __device__ bool Intersect(const Ray<U_1> &ray, U_2* tHit,
									   DifferentialGeometry<U_3> *Dg) const;
    __host__ __device__ T Area() const;

private:
	// private parameters
	T radius;               // sphere radius
    T phiMax;               // phimax = 360° for a complete sphere
    T zmin, zmax;           // zmin = (-1)*zmax = (-1)*radius for a complete sphere
    T thetaMin, thetaMax;   // computed as function of zmin et zmax
};

// -------------------------------------------------------
// Definitions of Sphere class methods
// -------------------------------------------------------
template <typename T, typename TF> 
Sphere<T,TF>::Sphere() : Shape<TF>()
{
    radius = T(0);
    zmin = T(0);
    zmax = T(0);
    thetaMin = T(0);
    thetaMax = T(0);
    phiMax = T(0);
}

template <typename T, typename TF> 
template <typename U_1, typename U_2, typename U_3, typename U_4,
	      typename U_5, typename U_6>
Sphere<T,TF>::Sphere(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
			         U_3 rad, U_4 z0, U_5 z1, U_6 pm)
	                : Shape<TF>(o2w, w2o)
{
    radius = T(rad);
    zmin = clamp(T(min(T(z0), T(z1))), T(-radius), T(radius));
    zmax = clamp(T(max(T(z0), T(z1))), T(-radius), T(radius));
    thetaMin = get_func_acos(clamp(zmin/radius, T(-1), T(1)));
    thetaMax = get_func_acos(clamp(zmax/radius, T(-1), T(1)));
    phiMax = get_func_radians(clamp(T(pm), T(0), T(360)));
}

template <typename T, typename TF> 
__device__ BBox<T> Sphere<T,TF>::WorldBoundSphere() const
{
	return (*this->ObjectToWorld)(ObjectBoundSphere());
}

template <typename T, typename TF> 
__device__ BBox<T> Sphere<T,TF>::ObjectBoundSphere() const
{
	if (phiMax < get_const_pi(T{})/T(2))
	{
		return BBox<T>(make_vec3<T>( T(0), T(0), zmin),
					   make_vec3<T>( radius, radius*get_func_sin(phiMax), zmax));
	}
	else if (phiMax < get_const_pi(T{}))
	{
		return BBox<T>(make_vec3<T>( radius*get_func_cos(phiMax), T(0), zmin),
					   make_vec3<T>( radius, radius, zmax));
	}
	else if (phiMax < T(3)*get_const_pi(T{})/T(2))
	{
	    return BBox<T>(make_vec3<T>(-radius, radius*get_func_sin(phiMax), zmin),
					   make_vec3<T>( radius,  radius, zmax));
	}
	else //if (phiMax >= 3*PI/2)
	{
	    return BBox<T>(make_vec3<T>(-radius, -radius, zmin),
					   make_vec3<T>( radius,  radius, zmax));
	}
}

template <typename T, typename TF>
template <typename U_1, typename U_2, typename U_3>
bool Sphere<T,TF>::Intersect(const Ray<U_1> &r, U_2 *tHit, DifferentialGeometry<U_3> *dg) const
{
    T phi;
    vec3<T> phit;

	Ray<T> ray(r);
    // Passing the "ray" into the sphere space
	(*this->WorldToObject)(r, &ray);

    // Compute the quadratic coefficients of the sphere
    T A = ray.d.x*ray.d.x + ray.d.y*ray.d.y + ray.d.z*ray.d.z;
    T B = 2 * (ray.d.x*ray.o.x + ray.d.y*ray.o.y + ray.d.z*ray.o.z);
    T C = ray.o.x*ray.o.x + ray.o.y*ray.o.y +
              ray.o.z*ray.o.z - radius*radius;

    // Solve the equation of second order to get t0 and t1
    T t0, t1;
    if (!quadratic(&t0, &t1, A, B, C))
        return false;

    // Calculate the factor t at which the ray is reaching the sphere
    if (t0 > ray.maxt || t1 < ray.mint)
        return false;
    T thit = t0;
    if (t0 < ray.mint)
	{
        thit = t1;
        if (thit > ray.maxt) {return false;}
    }

    // Compute the intersection position and $\phi$
    phit = ray(thit);
    if (phit.x == T(0) && phit.y == T(0)) {phit.x = (1e-5) * radius;}
    phi = get_func_atan2(phit.y, phit.x);
    if (phi < T(0)) {phi += T(2)*get_const_pi(T{});}

    // Consider additional parameters in case of partial sphere
    if ((zmin > -radius && phit.z < zmin) ||
        (zmax <  radius && phit.z > zmax) || phi > phiMax)
	{
        if (thit == t1) {return false;}
        if (t1 > ray.maxt) {return false;}
        thit = t1;
        // Compute instersection position and $\phi$
        phit = ray(thit);
        if (phit.x == T(0) && phit.y == T(0)) {phit.x = T(1e-5) * radius;}
        phi = get_func_atan2(phit.y, phit.x);
        if (phi < T(0)) {phi += T(2)*get_const_pi(T{});}
        if ((zmin > -radius && phit.z < zmin) ||
            (zmax <  radius && phit.z > zmax) || phi > phiMax)
            return false;
    }

    // Find the parameteric representation at the intersection point
    T u = phi / phiMax;
    T theta = get_func_acos(clamp(phit.z / radius, T(-1), T(1)));
    T v = (theta - thetaMin) / (thetaMax - thetaMin);

    // Compute the partial derivatives $\dpdu$ et $\dpdv$
    T zradius = get_func_sqrt(phit.x*phit.x + phit.y*phit.y);
    T invzradius = T(1) / zradius;
    T cosphi = phit.x * invzradius;
    T sinphi = phit.y * invzradius;
    vec3<T> dpdu = make_vec3<T>(-phiMax * phit.y, phiMax * phit.x, T(0));
    vec3<T> dpdv = make_vec3<T>(phit.z * cosphi, phit.z * sinphi,
							    -radius * get_func_sin(theta)) * (thetaMax-thetaMin);

    // Create the DifferentialGeometry object
    const Transform<TF> &o2w = *this->ObjectToWorld;
    *dg = DifferentialGeometry<U_3>(o2w(Point<U_3>(phit)),
								    o2w(Normal<U_3>(dpdu)),
								    o2w(Normal<U_3>(dpdv)),
								    u, v, this);

    // Update tHit
    *tHit = thit;

    return true;
}

template <typename T, typename TF> 
T Sphere<T,TF>::Area() const {
    return phiMax * radius * (zmax-zmin);
}

using Sphereff = Sphere<float,float>;
using Spherefd = Sphere<float,double>;
using Spheredf = Sphere<double,float>;
using Spheredd = Sphere<double,double>;


template <typename T = float, typename TF = float> //T/TF -> float / double
// T is type for the BBox/attributs of the class and for main operation inside the methods
// TF is type used in parent Shape class i.e., transformation
class Triangle : public Shape<TF>
// ========================================================
// Triangle class
// ! Care ! transformation are not used here!
// ========================================================
{
public:
	using U3  = vec3<T>;
	using U2  = vec2<T>;   // used in Intersect_v3 method
	using U3c  = vec3c<T>; // used in Intersect_v2 method

	// Public methods
	__host__ __device__ Triangle();
	template <typename U_1, typename U_2>
	__host__ __device__ Triangle(const Transform<U_1> *o2w, const Transform<U_2> *w2o);
	template <typename U_1, typename U_2, typename C3_1, typename C3_2, typename C3_3>
	__host__ __device__ Triangle(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
								 C3_1 a, C3_2 b, C3_3 c);

    __device__ BBox<T> ObjectBoundTriangle() const;
    __device__ BBox<T> WorldBoundTriangle() const;

	// v2 -> use pbrtv2 method ; v3 -> pbrtv3
	// Möller-Trumbore for ray/triangle intersection
	template <typename U_1, typename U_2, typename U_3>
    __host__ __device__ bool Intersect_v2(const Ray<U_1> &r, U_2* tHit,
		                                  DifferentialGeometry<U_3> *dg) const;
	template <typename U_1, typename U_2, typename U_3>
	__host__ __device__ bool Intersect_v3(const Ray<U_1> &r, U_2* tHit,
		                                  DifferentialGeometry<U_3> *dg) const;
	template <typename U_1, typename U_2, typename U_3>
	__host__ __device__ bool Intersect(const Ray<U_1> &r, U_2* tHit,
		                               DifferentialGeometry<U_3> *dg, int version = 3) const;								   									  
	template <typename U> __host__ __device__ bool IntersectP_v2(const Ray<U> &r) const;
	template <typename U> __host__ __device__ bool IntersectP_v3(const Ray<U> &r) const;
	template <typename U>
	__host__ __device__ bool IntersectP(const Ray<U> &r, int version = 3) const;	
    __host__ __device__ T Area() const;

private:
	// Private parameters
	U3 p1, p2, p3;
};

// -------------------------------------------------------
// Definitions of Triangle class methods
// -------------------------------------------------------
template <typename T, typename TF>
Triangle<T,TF>::Triangle()
{
	p1 = make_vec3<T>(T(0), T(0), T(0));
	p2 = make_vec3<T>(T(0), T(0), T(0));
	p3 = make_vec3<T>(T(0), T(0), T(0));
}

template <typename T, typename TF>
template <typename U_1, typename U_2>
Triangle<T,TF>::Triangle(const Transform<U_1> *o2w, const Transform<U_2> *w2o)
: Shape<TF>(o2w, w2o)
{
	p1 = make_vec3<T>(T(0), T(0), T(0));
	p2 = make_vec3<T>(T(0), T(0), T(0));
	p3 = make_vec3<T>(T(0), T(0), T(0));
}

template <typename T, typename TF>
template <typename U_1, typename U_2, typename C3_1, typename C3_2, typename C3_3>
Triangle<T,TF>::Triangle(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
				         C3_1 a, C3_2 b, C3_3 c): Shape<TF>(o2w, w2o)
{
	p1 = make_vec3<T>(T(a.x), T(a.y), T(a.z));
	p2 = make_vec3<T>(T(b.x), T(b.y), T(b.z));
	p3 = make_vec3<T>(T(c.x), T(c.y), T(c.z));
}

template <typename T, typename TF>
template <typename U_1, typename U_2, typename U_3>
bool Triangle<T,TF>::Intersect_v2(const Ray<U_1> &r, U_2 *tHit,
						          DifferentialGeometry<U_3> *dg) const
{
	Ray<T> ray(r);
	U3 e1 = p2 - p1;
	U3 e2 = p3 - p1;
	U3 s1 = cross(ray.d, e2);
    T divisor = dot(s1, e1);

	if (divisor == T(0))
	{return false;}
	T invDivisor = T(1) / divisor;

	// Compute the first baricentric coordinate component
	U3 s = ray.o - p1;
	T b1 = dot(s, s1) * invDivisor;

    if (b1 < T(0) || b1 > T(1)) {return false;}

    // Compute the second component
    U3 s2 = cross(s, e1);
    T b2 = dot(ray.d, s2) * invDivisor;
	if (b2 < T(0) || b1 + b2 > T(1)) {return false;}

    // Compute t
    T t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt) {return false;}

    // Compute partial derivatives
	U3 dpdu, dpdv;
	U3c uvsC0, uvsC1; // row1 and row2
	uvsC0 = make_vec3c<T>(T(0), T(1), T(1));
	uvsC1 = make_vec3c<T>(T(0), T(0), T(1));
		
    // Compute Deltas
	T du1 = uvsC0[0] - uvsC0[2];
	T du2 = uvsC0[1] - uvsC0[2];
    T dv1 = uvsC1[0] - uvsC1[2];
    T dv2 = uvsC1[1] - uvsC1[2];
    U3 dp1 = p1 - p3, dp2 = p2 - p3;
    T determinant = du1 * dv2 - dv1 * du2;

    if (determinant == T(0))
	{
        // Manage the case where the determinant is equal to 0
        coordinateSystem(normalize(cross(e2, e1)), &dpdu, &dpdv);
    }
    else
	{
        T invdet = T(1) / determinant;
        dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
        dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    }

    // Parametric coordinate interpolations of triangle $(u,v)$
    T b0 = T(1) - b1 - b2;
    T tu = b0*uvsC0[0] + b1*uvsC0[1] + b2*uvsC0[2];
    T tv = b0*uvsC1[0] + b1*uvsC1[1] + b2*uvsC1[2];

    // Create the DifferentialGeometry object
    *dg = DifferentialGeometry<U_3>(ray(t), dpdu, dpdv, tu, tv, this);

    // Update tHit
	*tHit = t;
	return true;
}

template <typename T, typename TF>
template <typename U_1, typename U_2, typename U_3>
__device__ bool Triangle<T,TF>::Intersect_v3(const Ray<U_1> &r, U_2 *tHit,
									         DifferentialGeometry<U_3> *dg) const
{
	Ray<T> ray(r);
	U3 p0t, p1t, p2t;
	U3 P0, P1, P2;

	P0 = make_vec3<T>(T(p1.x), T(p1.y), T(p1.z));
	P1 = make_vec3<T>(T(p2.x), T(p2.y), T(p2.z));
	P2 = make_vec3<T>(T(p3.x), T(p3.y), T(p3.z));

	U3 p_o, p_d;
	p_o = make_vec3<T>(T(ray.o.x), T(ray.o.y), T(ray.o.z));
	p_d = make_vec3<T>(T(ray.d.x), T(ray.d.y), T(ray.d.z));

	p0t = P0 - p_o; p1t = P1 - p_o; p2t = P2 - p_o;
	
	int kz = MaxDim( make_vec3<T>(abs(p_d.x),abs(p_d.y),abs(p_d.z)) );
	int kx = kz + 1;
	if(kx == 3) kx = 0;
	int ky = kx+1;
	if(ky == 3) ky = 0;
	U3 d = Permute(p_d, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	T Sx=-d.x/d.z; T Sy=-d.y/d.z; T Sz=T(1)/d.z;
	p0t.x += Sx*p0t.z; p0t.y += Sy*p0t.z;
	p1t.x += Sx*p1t.z; p1t.y += Sy*p1t.z;
	p2t.x += Sx*p2t.z; p2t.y += Sy*p2t.z;

	T e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	T e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	T e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	if constexpr (std::is_same_v<T, float>)
	{
		if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)
		{
			double p2txp1ty = (double)p2t.x * (double)p1t.y;
			double p2typ1tx = (double)p2t.y * (double)p1t.x;
			e0 = (float)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0t.x * (double)p2t.y;
			double p0typ2tx = (double)p0t.y * (double)p2t.x;
			e1 = (float)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1t.x * (double)p0t.y;
			double p1typ0tx = (double)p1t.y * (double)p0t.x;
			e2 = (float)(p1typ0tx - p1txp0ty);
		}
	} 

	if((e0<0 || e1<0 || e2<0) && (e0>0 || e1>0 || e2>0)) {return false;}
	T det = e0 + e1 + e2;
	if(det == 0) {return false;}

	p0t *= Sz; p1t *= Sz; p2t *= Sz;
	T tScaled = e0*p0t.z + e1*p1t.z + e2*p2t.z;
	if(det < T(0) && (tScaled >= T(0) || tScaled < ray.maxt*det)) {return false;}
	else if (det > T(0) && (tScaled <= T(0) || tScaled > ray.maxt*det))	{return false;}

	// Compute barycentric coordinates and t value
	T invDet = T(1) / det;
	T b0 = e0*invDet; T b1 = e1*invDet; T b2 = e2*invDet;
	T t = tScaled*invDet;

	if (t < ray.mint || t > ray.maxt) {return false;}

	T maxZt = max( abs(p0t.z), max( abs(p1t.z), abs(p2t.z) )  );
	T eps = get_const_machine_eps(T{}) * T(0.5);
	T deltaZ = Gamma_eps(3, eps) * maxZt;

	T maxXt = max( abs(p0t.x), max( abs(p1t.x), abs(p2t.x) )  );
	T maxYt = max( abs(p0t.y), max( abs(p1t.y), abs(p2t.y) )  );
	T deltaX = Gamma_eps(5, eps) * (maxXt + maxZt);
	T deltaY = Gamma_eps(5, eps) * (maxYt + maxZt);

	T deltaE = 2*(Gamma_eps(2, eps)*maxXt*maxYt +
				  deltaY*maxXt + deltaX*maxYt);
	T maxE = max( abs(e0), max( abs(e1), abs(e2) )  );
	T deltaT = 3*(Gamma_eps(3, eps)*maxE*maxZt + deltaE*maxZt +
					  deltaZ*maxE)*abs(invDet);
	if(t <= deltaT) {return false;}

	U3 dpdu, dpdv;
	U2 uv[3];
	uv[0] = make_vec2<T>(0,0);
	uv[1] = make_vec2<T>(1,0);
	uv[2] = make_vec2<T>(1,1);
	
	U2 duv02 = uv[0]-uv[2], duv12 = uv[1]-uv[2];
	U3 dp02 = P0-P2, dp12=P1-P2;
	T determinant = duv02.x*duv12.y - duv02.y*duv12.x;
	bool degenerateUV = abs(determinant) < T(1e-8);
	if(!degenerateUV)
	{
		T invdet = T(1) / determinant;
		dpdu = (duv12.y*dp02 - duv02.y*dp12)*invdet;
		dpdv = (-duv12.x*dp02 - duv02.x*dp12)*invdet;
	}
	U3 croD = cross(dpdu, dpdv);
	if(degenerateUV || (croD.x*croD.x + croD.y*croD.y + croD.z*croD.z) == 0)
	{
		U3 ng = cross(P2-P0, P1-P0);
		if ((ng.x*ng.x + ng.y*ng.y + ng.z*ng.z) == T(0))
			return false; // the intersection is bogus
		coordinateSystem(normalize(ng), &dpdu, &dpdv);
	}

	U3 phit = b0*P0+b1*P1+b2*P2;

	// Create the DifferentialGeometry object
	*dg = DifferentialGeometry<U_3>(phit, dpdu, dpdv, 0, 0, this);

    // Update tHit
    *tHit = t;
	
	return true;
}

template <typename T, typename TF>
template <typename U_1, typename U_2, typename U_3>
bool Triangle<T,TF>::Intersect(const Ray<U_1> &r, U_2 *tHit,
						       DifferentialGeometry<U_3> *dg, int version) const
{
	if (version == 2) { return Intersect_v2(r, tHit, dg); }
	else { return Intersect_v3(r, tHit, dg); }
}

template <typename T, typename TF>
template <typename U>
bool Triangle<T,TF>::IntersectP_v2(const Ray<U> &r) const
{
	Ray<T> ray(r);

	U3 e1 = p2 - p1;
	U3 e2 = p3 - p1;
	U3 s1 = cross(ray.d, e2);
    T divisor = dot(s1, e1);

	if (divisor == 0.) {return false;}
	T invDivisor = T(1) / divisor;

	// Compute the first baricentric coordinate component
	U3 s = ray.o - p1;
	T b1 = dot(s, s1) * invDivisor;

    if (b1 < T(0) || b1 > T(1))
	{return false;}

    // Compute the second component
    U3 s2 = cross(s, e1);
    T b2 = dot(ray.d, s2) * invDivisor;
	if (b2 < T(0) || b1 + b2 > T(1)) {return false;}

    // Compute t
    T t = dot(e2, s2) * invDivisor;
	
    if (t < ray.mint || t > ray.maxt) {return false;}

	return true;
}

template <typename T, typename TF>
template <typename U>
__device__ bool Triangle<T,TF>::IntersectP_v3(const Ray<U> &r) const
{
	Ray<T> ray(r);

    U3 p0t, p1t, p2t;
	U3 P0, P1, P2;

	P0 = make_vec3<T>(p1.x, p1.y, p1.z);
	P1 = make_vec3<T>(p2.x, p2.y, p2.z);
	P2 = make_vec3<T>(p3.x, p3.y, p3.z);

	U3 p_o, p_d;
	p_o = ray.o;
	p_d = ray.d;

	p0t = P0 - p_o; p1t = P1 - p_o; p2t = P2 - p_o;
	
	int kz = MaxDim( make_vec3<T>(abs(p_d.x),abs(p_d.y),abs(p_d.z)) );
	int kx = kz + 1;
	if(kx == 3) kx = 0;
	int ky = kx+1;
	if(ky == 3) ky = 0;
	U3 d = Permute(p_d, kx, ky, kz);
	p0t = Permute(p0t, kx, ky, kz);
	p1t = Permute(p1t, kx, ky, kz);
	p2t = Permute(p2t, kx, ky, kz);

	T Sx=-d.x/d.z; T Sy=-d.y/d.z; T Sz=T(1)/d.z;
	p0t.x += Sx*p0t.z; p0t.y += Sy*p0t.z;
	p1t.x += Sx*p1t.z; p1t.y += Sy*p1t.z;
	p2t.x += Sx*p2t.z; p2t.y += Sy*p2t.z;

	T e0 = p1t.x * p2t.y - p1t.y * p2t.x;
	T e1 = p2t.x * p0t.y - p2t.y * p0t.x;
	T e2 = p0t.x * p1t.y - p0t.y * p1t.x;

	if constexpr (std::is_same_v<T, float>)
	{
		if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)
		{
			double p2txp1ty = (double)p2t.x * (double)p1t.y;
			double p2typ1tx = (double)p2t.y * (double)p1t.x;
			e0 = (float)(p2typ1tx - p2txp1ty);
			double p0txp2ty = (double)p0t.x * (double)p2t.y;
			double p0typ2tx = (double)p0t.y * (double)p2t.x;
			e1 = (float)(p0typ2tx - p0txp2ty);
			double p1txp0ty = (double)p1t.x * (double)p0t.y;
			double p1typ0tx = (double)p1t.y * (double)p0t.x;
			e2 = (float)(p1typ0tx - p1txp0ty);
		}
	}  

	if((e0<0 || e1<0 || e2<0) && (e0>0 || e1>0 || e2>0)) {return false;}
	T det = e0 + e1 + e2;
	if(det == 0) {return false;}

	p0t *= Sz; p1t *= Sz; p2t *= Sz;
	T tScaled = e0*p0t.z + e1*p1t.z + e2*p2t.z;
	if(det < T(0) && (tScaled >= T(0) || tScaled < ray.maxt*det)) {return false;}
	else if (det > T(0) && (tScaled <= T(0) || tScaled > ray.maxt*det)) {return false;}

	T invDet = T(1) / det;
	T t = tScaled*invDet;

	if (t < ray.mint || t > ray.maxt) {return false;}

	T maxZt = max( abs(p0t.z), max( abs(p1t.z), abs(p2t.z) )  );
	T eps = get_const_machine_eps(T{}) * T(0.5);
	T deltaZ = Gamma_eps(3, eps) * maxZt;

	T maxXt = max( abs(p0t.x), max( abs(p1t.x), abs(p2t.x) )  );
	T maxYt = max( abs(p0t.y), max( abs(p1t.y), abs(p2t.y) )  );
	T deltaX = Gamma_eps(5, eps) * (maxXt + maxZt);
	T deltaY = Gamma_eps(5, eps) * (maxYt + maxZt);

	T deltaE = 2*(Gamma_eps(2, eps)*maxXt*maxYt +
					  deltaY*maxXt + deltaX*maxYt);
	T maxE = max( abs(e0), max( abs(e1), abs(e2) )  );
	T deltaT = 3*(Gamma_eps(3, eps)*maxE*maxZt + deltaE*maxZt +
					  deltaZ*maxE)*abs(invDet);
	if(t <= deltaT) {return false;}

	U3 dpdu, dpdv;
	U2 uv[3];
	uv[0] = make_vec2<T>(0,0);
	uv[1] = make_vec2<T>(1,0);
	uv[2] = make_vec2<T>(1,1);
	
	U2 duv02 = uv[0]-uv[2], duv12 = uv[1]-uv[2];
	U3 dp02 = P0-P2, dp12=P1-P2;
	T determinant = duv02.x*duv12.y - duv02.y*duv12.x;
	bool degenerateUV = abs(determinant) < T(1e-8);
	if(!degenerateUV)
	{
		T invdet = T(1) / determinant;
		dpdu = (duv12.y*dp02 - duv02.y*dp12)*invdet;
		dpdv = (-duv12.x*dp02 - duv02.x*dp12)*invdet;
	}
	U3 croD = cross(dpdu, dpdv);
	if(degenerateUV || (croD.x*croD.x + croD.y*croD.y + croD.z*croD.z) == 0)
	{
		U3 ng = cross(P2-P0, P1-P0);
		if ((ng.x*ng.x + ng.y*ng.y + ng.z*ng.z) == T(0))
			return false; // the intersection is bogus
		coordinateSystem(normalize(ng), &dpdu, &dpdv);
	}

	return true;
}

template <typename T, typename TF>
template <typename U>
bool Triangle<T,TF>::IntersectP(const Ray<U> &r, int version) const
{
	if (version == 2) { return IntersectP_v2(r); }
	else { return IntersectP_v3(r); }
}

template <typename T, typename TF> 
__device__ BBox<T> Triangle<T,TF>::ObjectBoundTriangle() const
{
	BBox<T> objectBounds((*this->WorldToObject)(Point<T>(p1)), (*this->WorldToObject)(Point<T>(p2)));
	return objectBounds.Union(objectBounds, (*this->WorldToObject)(Point<T>(p3)));
}

template <typename T, typename TF> 
__device__ BBox<T> Triangle<T,TF>::WorldBoundTriangle() const
{
	BBox<T> worldBounds(p1, p2);
    return worldBounds.Union(worldBounds, p3);
}

template <typename T, typename TF> 
T Triangle<T,TF>::Area() const
{
    return T(0.5) * length(cross(p2-p1, p3-p1));
}

using Triangleff = Triangle<float,float>;
using Trianglefd = Triangle<float,double>;
using Trianglesf = Triangle<double,float>;
using Triangledd = Triangle<double,double>;


template <typename T = float, typename TF = float, typename TP = float > //T/TP/TP -> float / double
// T is type for the BBox/attributs of the class and for main operation inside the methods
// TF is type used in parent Shape class i.e., transformation
// TP is base type for the given triangle list of points vec3<TP> *P
class TriangleMesh : public Shape<TF>
// ========================================================
// TriangleMesh class
// ========================================================
{
public:
	using U3  = vec3<T>;
	using U3TP  = vec3<TP>;

	// Public methods
	template <typename U_1, typename U_2>
	__host__ __device__ TriangleMesh(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
									 int nt, int nv, int *vi, U3TP *P);
    __device__ BBox<T> ObjectBoundTriangleMesh() const;
    __device__ BBox<T> WorldBoundTriangleMesh() const;
	template <typename U_1, typename U_2, typename U_3>
    __host__ __device__ bool Intersect(const Ray<U_1> &r, U_2* tHit,
									   DifferentialGeometry<U_3> *dg) const;
	template <typename U > __host__ __device__ bool IntersectP(const Ray<U> &r) const;
    __host__ __device__ T Area() const;
	U3TP *p; // list of all the triangle mesh points

private:
	// Private parameters
	int ntris, nverts;
	int *vertexIndex;
};

// -------------------------------------------------------
// Definitions of TriangleMesh class methods
// -------------------------------------------------------
template <typename T, typename TF, typename TP>
template <typename U_1, typename U_2>
TriangleMesh<T,TF,TP>::TriangleMesh(const Transform<U_1> *o2w, const Transform<U_2> *w2o,
						            int nt, int nv, int *vi, U3TP *P)
	: Shape<TF>(o2w, w2o)
{
	ntris = nt; nverts = nv;
	vertexIndex = vi;
	p = P;

	// Apply the transformations to the triangle mesh
	Point<TP> pbis;
	for (int i = 0; i < nverts; ++i)
	{
		pbis.x = p[i].x; pbis.y = p[i].y; pbis.y = p[i].y;
		(*this->ObjectToWorld)(pbis, &p[i]);
	}
}

template <typename T, typename TF, typename TP>
template <typename U_1, typename U_2, typename U_3>
bool TriangleMesh<T,TF,TP>::Intersect(const Ray<U_1> &r, U_2* tHit,
							          DifferentialGeometry<U_3> *dg) const
{
    bool dgbool = false;
	Transform<TF> nothing;
	*tHit = get_const_inf(U_2{});

	U_2 triHit;
	DifferentialGeometry<U_3> dgTri;
	for (int i = 0; i < ntris; ++i)
	{
		/* // Create the triangle i as function of *vi et *P	 */
		Triangle<T> rt(&nothing, &nothing, p[vertexIndex[3*i]], p[vertexIndex[3*i + 1]],
			           p[vertexIndex[3*i + 2]]);
		if (rt.Intersect(r, &triHit, &dgTri))
		{
			dgbool = true;
			if (*tHit > triHit)
			{
				*dg = dgTri;
				*tHit = triHit;
			}
		}
	}
	return dgbool;
}

template <typename T, typename TF, typename TP>
template <typename U>
bool TriangleMesh<T,TF,TP>::IntersectP(const Ray<U> &r) const
{
	Transform<TF> nothing;
	for (int i = 0; i < ntris; ++i)
	{
		/* // Create the triangle i as function of *vi et *P	 */
		Triangle<T> rt(&nothing, &nothing, p[vertexIndex[3*i]], p[vertexIndex[3*i + 1]],
			           p[vertexIndex[3*i + 2]]);
		if (rt.IntersectP(r))
			return true;
	}
	return false;
}

template <typename T, typename TF, typename TP>
__device__ BBox<T> TriangleMesh<T,TF,TP>::ObjectBoundTriangleMesh() const
{
	BBox<T> objectBounds;
	Point<TP> pW;
    for (int i = 0; i < nverts; i++)
	{
		pW.x = TP(p[i].x); pW.y = TP(p[i].y); pW.z = TP(p[i].z);
		(*this->WorldToObject)(p[i], &pW); // pointer to keep pW type
		objectBounds = objectBounds.UnionPoint(objectBounds, pW);
	}
    return objectBounds;
}

template <typename T, typename TF, typename TP>
__device__ BBox<T> TriangleMesh<T,TF,TP>::WorldBoundTriangleMesh() const
{
    BBox<T> worldBounds;
    for (int i = 0; i < nverts; i++)
		worldBounds = worldBounds.Union(worldBounds, p[i]);
    return worldBounds;
}

template <typename T, typename TF, typename TP>
T TriangleMesh<T,TF,TP>::Area() const
{
	T Area = T(0);
    for (int i = 0; i < ntris; ++i)
	{
		U3TP pa = p[vertexIndex[3*i]];
		U3TP pb = p[vertexIndex[3*i + 1]];
		U3TP pc = p[vertexIndex[3*i + 2]];
		U3 PA = make_vec3<T>(T(pa.x), T(pa.y), T(pa.z));
		U3 PB = make_vec3<T>(T(pb.x), T(pb.y), T(pb.z));
		U3 PC = make_vec3<T>(T(pc.x), T(pc.y), T(pc.z));
		Area += T(0.5) * length(cross(PB-PA, PC-PA));
	}
    return Area;
}

using TriangleMeshfff = TriangleMesh<float,float,float>;
using TriangleMeshfdf = TriangleMesh<float,double,float>;
using TriangleMeshffd = TriangleMesh<float,float,double>;
using TriangleMeshfdd = TriangleMesh<float,double,double>;
using TriangleMeshdff = TriangleMesh<double,float,float>;
using TriangleMeshddf = TriangleMesh<double,double,float>;
using TriangleMeshdfd = TriangleMesh<double,float,double>;
using TriangleMeshddd = TriangleMesh<double,double,double>;
#endif // _SHAPES_H_
