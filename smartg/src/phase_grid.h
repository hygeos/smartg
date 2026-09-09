#ifndef PHASE_GRID_H
#define PHASE_GRID_H

/**********************************************************
*
*			phase_grid.h
*
*	The phase matrix table of a medium and the two lookups
*	on it: the index of an arbitrary scattering angle, for
*	the local estimate, and the deflection a random walk
*	draws from the very same table.
*
***********************************************************/

#ifndef PI
#define PI 3.141592654f
#endif

/* One phase matrix, tabulated on the scattering angle grid of its
   medium (struct AGrid below). Every reader of a phase matrix goes
   through this table: the local estimate, the ALIS correction, and
   the random walk, which draws its deflection from the very same
   entries through struct PGrid. It used to carry a second copy of
   the matrix at equal-probability nodes; that copy is gone, an entry
   costs 24 bytes instead of 52. */
struct Phase {
    float a_P11; /* \                          */
    float a_P12; /*  |                         */
    float a_P22; /*  | tabulated on the        */
    float a_P33; /*  | scattering angle grid   */
    float a_P43; /*  |                         */
    float a_P44; /* /                          */
};

/* Angular grid a phase table is tabulated on. One grid is shared by
   every phase function of a medium, but the atmosphere and the ocean
   have their own.
     mode 0 : theta_i = PI*i/(n-1)   (equally spaced)
     mode 1 : theta_i = ang[i]       (arbitrary, tabulated)
   Clustering the nodes towards 0 and PI resolves the forward
   diffraction peak of large particles with far fewer entries than an
   equally spaced grid, which is what keeps the table small. */
struct AGrid {
    unsigned int n;   /* angles per phase function = table entries   */
    int   mode;
    int   log2n;      /* floor(log2(n-2)), binary search trip count  */
    float *ang;       /* mode 1 only, n entries, device pointer      */
};

/* aIndex
* Lower index of theta in a phase table, returning the interpolation
* weight of the upper node.
*/
__device__ float aIndex(float theta, const struct AGrid g, int *iang)
{
	int last = (int)g.n - 2;
	float x;

	if (g.mode == 1) {
		/* Tabulated grid: bisect. The trip count is fixed, so every
		   thread of a warp runs the same iterations and the search
		   costs no divergence. */
		int lo = 0;
		#pragma unroll 1
		for (int s = g.log2n; s >= 0; s--) {
			int m = lo + (1 << s);
			lo = (m <= last && __ldg(g.ang + m) <= theta) ? m : lo;
		}
		*iang = lo;
		float a0 = __ldg(g.ang + lo), a1 = __ldg(g.ang + lo + 1);
		return __saturatef(__fdividef(theta - a0, a1 - a0));
	}

	x = theta * (g.n-1)/PI;

	/* Clamp to the last interval. Without it theta == PI, which an
	   exactly backward local estimate does reach, gives an index of
	   n-1 and an interpolation reading func[ipha*n + n]: the first
	   entry of the next phase function, or past the end of the
	   allocation for the last one. Writing the lower bound as
	   !(x > 0) also traps a NaN theta. */
	if (!(x > 0.F)) x = 0.F;
	*iang = __float2int_rd(x);
	if (*iang > last) { *iang = last; return 1.F; }

	return x - *iang;
}

/* Cumulative distribution of each phase function, at the nodes of
   the angle grid above: cdf[ipha*n + k] is the scattering probability
   below theta_k, from 0 at k = 0 to 1 at k = n-1, integrated exactly
   for the phase function the table describes, i.e. F11 linear in
   theta between nodes times the true sin(theta).

   It is tabulated on the angle grid itself and pSample inverts it
   exactly inside a bin, so the deflection a random walk draws is
   distributed exactly as the table it then reads its matrix from.
   The earlier design tabulated the inverse instead, at equally
   spaced probabilities, and interpolated it linearly: that draws
   from a staircase density, constant inside each bin, and the
   uncorrected mismatch with the smooth table is ~1e-4 per event,
   which a cloud multiplies by its ~1e3 scattering orders. On IPRT C3
   case 6 that was a 2.6% bias of the mean reflected intensity that
   only went away with 12601 nodes. This costs no memory beyond one
   float per table entry and has no knob. */
struct PGrid {
    unsigned int n;   /* = AGrid.n, nodes per phase function     */
    int   log2n;      /* floor(log2(n-2)), binary search trips   */
    float *cdf;       /* n * nphase entries, device pointer       */
};

/* F11, the phase function a deflection is drawn from, of one table
   entry, in the kernels' parallel/perpendicular convention. */
__device__ __forceinline__ float pF11(const struct Phase *e)
{
	return 0.5F * (e->a_P11 + e->a_P22 + 2.F * e->a_P12);
}

/* Node k of the angle grid. */
__device__ __forceinline__ float aNode(int k, const struct AGrid g)
{
	return (g.mode == 1) ? __ldg(g.ang + k) : PI * k / (g.n - 1);
}

/* Probability mass, up to the fraction t of the bin width, of a bin
   starting at th0 and dth wide, for F11 = f0 + df tau and the true
   sin(theta), per unit bin width: int_0^t (f0 + df tau) sin(th0 +
   dth tau) dtau. A 3 point Gauss-Legendre sum, exact to ~1e-7 for
   bins up to tens of degrees, rather than the closed form: that one
   subtracts two nearly equal terms and in float32 loses everything
   in the 0.01 degree bins of a forward peak, where it matters most.
   Every term here is positive. */
__device__ __forceinline__ float pMass(
	float t, float th0, float dth, float f0, float df)
{
	/* nodes and weights of Gauss-Legendre on [0, 1] */
	const float x0 = 0.1127016654F, x1 = 0.5F, x2 = 0.8872983346F;
	const float w0 = 0.2777777778F, w1 = 0.4444444444F;
	float a0 = t * x0, a1 = t * x1, a2 = t * x2;
	float m = w0 * (f0 + df * a0) * __sinf(th0 + dth * a0)
	        + w1 * (f0 + df * a1) * __sinf(th0 + dth * a1)
	        + w0 * (f0 + df * a2) * __sinf(th0 + dth * a2);
	return t * m;
}

/* pSample
* Scattering angle drawn from phase function ipha for u uniform on
* ]0, 1], with the table index and interpolation weight of that angle,
* so that the caller reads the matrix there without a second search.
*/
__device__ float pSample(
	float u, int ipha, const struct PGrid p, const struct AGrid g,
	const struct Phase *func, int *iang, float *zang)
{
	const float *c = p.cdf + (unsigned int)ipha * p.n;
	int last = (int)p.n - 2;

	/* bisect for the largest k <= last with cdf[k] <= u; the trip
	   count is fixed, so a warp runs the same iterations */
	int lo = 0;
	#pragma unroll 1
	for (int s = p.log2n; s >= 0; s--) {
		int m = lo + (1 << s);
		lo = (m <= last && __ldg(c + m) <= u) ? m : lo;
	}

	float c0 = __ldg(c + lo), c1 = __ldg(c + lo + 1);
	float th0 = aNode(lo, g), dth = aNode(lo + 1, g) - th0;
	const struct Phase *e = func + (unsigned int)ipha * g.n + lo;
	float f0 = pF11(e), df = pF11(e + 1) - f0;

	/* the fraction of the bin's mass below the drawn angle; a bin
	   of no mass, which float32 makes of the last entries of a very
	   fine grid, lands on its lower node */
	float v = (c1 > c0) ? __saturatef(__fdividef(u - c0, c1 - c0)) : 0.F;
	float M = pMass(1.F, th0, dth, f0, df);

	/* invert the bin's mass, which is monotone in t, by Newton from
	   the linear guess: four fixed steps, no divergence */
	float t = v;
	#pragma unroll
	for (int it = 0; it < 4; it++) {
		/* d pMass / dt: the density at t, per unit of the fraction */
		float d = (f0 + df * t) * __sinf(th0 + dth * t);
		float r = pMass(t, th0, dth, f0, df) - v * M;
		t = (d > 0.F) ? t - __fdividef(r, d) : t;
		t = __saturatef(t);
	}

	*iang = lo;
	*zang = t;
	return th0 + dth * t;
}

#endif // PHASE_GRID_H
