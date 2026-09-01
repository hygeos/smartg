#ifndef PHASE_GRID_H
#define PHASE_GRID_H

/**********************************************************
*
*			phase_grid.h
*
*	Angular grid of the equal-angle ("a") half of a phase
*	table, and the lookup that turns a scattering angle
*	into a table index.
*
***********************************************************/

#ifndef PI
#define PI 3.141592654f
#endif

/* Shape of the equal-angle ("a") half of a phase table. One grid is
   shared by every phase function of a medium, but the atmosphere and
   the ocean have their own.
     mode 0 : theta_i = PI*i/(n-1)   (equally spaced)
     mode 1 : theta_i = ang[i]       (arbitrary, tabulated)
   Clustering the nodes towards 0 and PI resolves the forward
   diffraction peak of large particles with far fewer entries than an
   equally spaced grid, which is what keeps the table small. */
struct AGrid {
    unsigned int n;   /* angles per phase function = table entries   */
    int   mode;
    int   log2n;      /* floor(log2(n-2)), binary search trip count  */
    float *ang;       /* mode 2 only, n entries, device pointer      */
};

/* aIndex
* Lower index of theta in the equal-angle ("a") half of a phase table,
* returning the interpolation weight of the upper node.
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

#endif // PHASE_GRID_H
