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
     mode 0 : theta_i = PI*i/(n-1)                 (equally spaced)
     mode 1 : theta_i = PI*sin^2(PI/2 * i/(n-1))   (Chebyshev-Lobatto)
     mode 2 : theta_i = ang[i]                     (arbitrary, tabulated)
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
	float x = theta * (g.n-1)/PI;

	if (!(x > 0.F)) x = 0.F;
	*iang = __float2int_rd(x);
	if (*iang > last) { *iang = last; return 1.F; }

	return x - *iang;
}

#endif // PHASE_GRID_H
