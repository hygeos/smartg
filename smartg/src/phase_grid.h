#ifndef PHASE_GRID_H
#define PHASE_GRID_H

/**********************************************************
*
*			phase_grid.h
*
*	The two angular grids of a medium: the one a phase
*	table is tabulated on, with the lookup that turns a
*	scattering angle into a table index, and the inverse
*	cumulative distribution a deflection is drawn from.
*
***********************************************************/

#ifndef PI
#define PI 3.141592654f
#endif

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

/* Inverse cumulative distribution of a medium: the scattering angles
   at n equally spaced cumulative probabilities over the closed
   [0, 1], so entry 0 is theta = 0 and entry n-1 is theta = PI and no
   part of the probability range is out of reach.

   Its length is deliberately independent of the grid above, because
   the two answer different questions: how finely a phase matrix must
   be resolved to be read at an arbitrary angle, against how finely
   the distribution must be resolved to be drawn from. Measured on a
   water cloud at 670 nm, the Legendre moments of the drawn
   distribution are converged to 1e-4 by 1801 nodes, where the angle
   grid is still improving well past 12601. */
struct PGrid {
    unsigned int n;   /* CDF nodes per phase function          */
    float *ang;       /* n * nphase entries, device pointer    */
};

/* pSample
* Scattering angle drawn from phase function ipha, for u uniform on
* ]0, 1].
*/
__device__ float pSample(float u, int ipha, const struct PGrid p)
{
	float x = u * (p.n - 1);
	int i = __float2int_rd(x);

	/* RAND is documented ]0;1], so the last node is reachable and
	   would interpolate one entry past the phase function */
	if (i > (int)p.n - 2) { i = (int)p.n - 2; x = 1.F; }
	else x = x - i;

	const float *a = p.ang + (unsigned int)ipha * p.n + i;

	return (1.F - x) * __ldg(a) + x * __ldg(a + 1);
}

#endif // PHASE_GRID_H
