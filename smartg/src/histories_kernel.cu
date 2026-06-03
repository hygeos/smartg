/**
 * histories_kernel.cu
 * CUDA kernel for SMART-G ALIS photon history processing.
 *
 * Processes the tabHistTot GPU array directly without transferring data to host,
 * replacing the  get_histories() + JAX BigSum()  CPU workflow from histories.py.
 *
 * ── Memory layout of tabHistTot (6-D, row-major float32) ─────────────────────
 *   Shape : (2, MAX_HIST, N3, NSENSOR, NBTHETA, NBPHI)
 *
 *   For a fixed (LEVEL, sensor, itheta, iphi), the per-photon slice is
 *   accessed with strides provided by the Python caller (stride_iph, stride_id).
 *   Within the N3 data elements per photon the layout is:
 *
 *     id  0  ..  NL-1          : D   — cumulative distances per atm layer [km]
 *     id  NL ..  NL+3          : S   — 4 Stokes components
 *     id  NL+4 .. NL+4+NLR-1  : w   — LR scattering weights
 *     id  NL+4+NLR+0           : nrrs
 *     id  NL+4+NLR+1           : nref  (Ki = number of surface reflections)
 *     id  NL+4+NLR+2           : nsif
 *     id  NL+4+NLR+3           : nvrs
 *     id  NL+4+NLR+4           : nenv
 *     id  NL+4+NLR+5           : nint
 *     id  NL+4+NLR+6           : last_scatter_layer (-1 = surface/unscattered)
 *     N3 = NL + 4 + NLR + 7
 *
 * ── What the kernel computes ──────────────────────────────────────────────────
 *   For each HR wavelength λ (one thread):
 *
 *     result[k, λ] = Σ_i  S_ik · ŵ_i(λ) · exp(−Σ_j D_ij · κ_j(λ)) · alb(λ)^{K_i}
 *
 *   where  ŵ_i(λ) = linear_interp(λ, wl_lr, w_i)
 *          κ_j(λ) = kabs[λ, j]   (gas absorption coefficient, km⁻¹)
 *
 *   This equals  BigSum(Si)(wl, kabs, alb, S, w, D, nref, wl_lr).sum(axis=1)
 *   from smartg/histories.py  (the photon sum; divide by N on the Python side).
 *
 *   When compute_sq != 0 the kernel computes Si² instead of Si
 *   (needed for variance estimation via BigSum(Si2)).
 */

/* ─────────────────────────────────────────────────────────────────────────────
 * Device helper: linear interpolation on an (irregular) sorted grid
 * ───────────────────────────────────────────────────────────────────────────── */
__device__ __forceinline__ float interp_linear_ptr(
        float          x,
        const float*   xgrid,   /* sorted ascending, length n */
        const float*   ygrid,   /* values at xgrid,  length n */
        int            n)
{
    if (n == 1)            return ygrid[0];
    if (x <= xgrid[0])     return ygrid[0];
    if (x >= xgrid[n - 1]) return ygrid[n - 1];

    /* Binary search for the interpolation interval */
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (xgrid[mid] <= x) lo = mid;
        else                 hi = mid;
    }
    float t = (x - xgrid[lo]) / (xgrid[hi] - xgrid[lo]);
    return __fmaf_rn(t, ygrid[hi] - ygrid[lo], ygrid[lo]);
}


/* ─────────────────────────────────────────────────────────────────────────────
 * process_tabHistTot_kernel  — 2-D grid, shared-memory reduction
 *
 * Grid  : (NWL,  ceil(MAX_HIST / BLOCK_PHO))
 * Block : (BLOCK_PHO, 1, 1)
 *
 * Parallelism:
 *   • blockIdx.x  = wavelength index  iwl  (one block column per λ)
 *   • blockIdx.y  = photon chunk index     (ceil(MAX_HIST/BLOCK_PHO) rows)
 *   • threadIdx.x = photon slot within the chunk
 *
 * Each thread computes the Si (or Si²) contribution of ONE photon at ONE
 * wavelength.  A shared-memory tree reduction then sums the BLOCK_PHO
 * contributions inside each block, and thread 0 writes the partial sum to
 * the global result via atomicAdd.
 *
 * result must be pre-zeroed by the caller (gpuarray.zeros).
 *
 * With NWL=301 and MAX_HIST=1e5 this launches ~1.17 × 10⁵ blocks (vs. 2
 * blocks in the old 1-D version), giving near-full GPU occupancy.
 *
 * Parameters — identical to the old 1-D version so the Python wrapper is
 * unchanged except for the grid dimensions.
 * ───────────────────────────────────────────────────────────────────────────── */

#define BLOCK_PHO 256

extern "C"
__global__ void process_tabHistTot_kernel(
        const float* __restrict__ tab,
        const float* __restrict__ kabs,
        const float* __restrict__ alb,
        const float* __restrict__ lam_hr,
        const float* __restrict__ lam_lr,
        float*                    result,   /* (4 * NWL) — must be pre-zeroed */
        long long                 base_offset,
        long long                 stride_iph,
        long long                 stride_id,
        int                       MAX_HIST,
        int                       NL,
        int                       NLR,
        int                       NWL,
        int                       compute_sq)
{
    const int iwl = (int)blockIdx.x;
    const int iph = (int)blockIdx.y * BLOCK_PHO + (int)threadIdx.x;

    if (iwl >= NWL) return;

    const float  lam     = lam_hr[iwl];
    const float  alb_wl  = alb[iwl];
    const float* kabs_wl = kabs + (long long)iwl * NL;

    /* Per-thread contribution (zero if slot is empty or out-of-range) */
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;

    if (iph < MAX_HIST) {
        const long long ph_base = base_offset + (long long)iph * stride_iph;

        /* ── Skip empty photon slots ─────────────────────────────────────── */
        if (tab[ph_base + (long long)(NL + 4) * stride_id] != 0.f) {

            /* ── Gaseous absorption: Σ_j D_j · κ_j(λ) ──────────────────── */
            float abs_od = 0.f;
            for (int il = 0; il < NL; ++il)
                abs_od = __fmaf_rn(
                            tab[ph_base + (long long)il * stride_id],
                            kabs_wl[il],
                            abs_od);

            /* ── LR weight interpolated to lam ───────────────────────────── */
            float wi;
            if (stride_id == 1) {
                wi = interp_linear_ptr(lam,
                                       lam_lr,
                                       tab + ph_base + (long long)(NL + 4),
                                       NLR);
            } else {
                float w_local[64];
                const int n = (NLR < 64) ? NLR : 64;
                for (int k = 0; k < n; ++k)
                    w_local[k] = tab[ph_base + (long long)(NL + 4 + k) * stride_id];
                wi = interp_linear_ptr(lam, lam_lr, w_local, n);
            }

            /* ── Surface albedo factor: alb(λ)^{Ki} ─────────────────────── */
            const int   Ki    = __float2int_rz(
                                    tab[ph_base + (long long)(NL + 4 + NLR + 1) * stride_id]);
            const float alb_K = (Ki == 0) ? 1.f : __powf(alb_wl, (float)Ki);

            const float factor = wi * __expf(-abs_od) * alb_K;

            /* ── Stokes contributions ────────────────────────────────────── */
            float sv0 = tab[ph_base + (long long)(NL    ) * stride_id] * factor;
            float sv1 = tab[ph_base + (long long)(NL + 1) * stride_id] * factor;
            float sv2 = tab[ph_base + (long long)(NL + 2) * stride_id] * factor;
            float sv3 = tab[ph_base + (long long)(NL + 3) * stride_id] * factor;

            if (compute_sq) {
                s0 = sv0 * sv0;  s1 = sv1 * sv1;
                s2 = sv2 * sv2;  s3 = sv3 * sv3;
            } else {
                s0 = sv0;  s1 = sv1;  s2 = sv2;  s3 = sv3;
            }
        }
    }

    /* ── Shared-memory tree reduction over the photon dimension ─────────────
     *  After the loop: thread 0 of each block holds the partial photon sum
     *  for its (iwl, chunk) pair, then atomicAdd accumulates into result.
     */
    __shared__ float sh[4][BLOCK_PHO];
    sh[0][threadIdx.x] = s0;
    sh[1][threadIdx.x] = s1;
    sh[2][threadIdx.x] = s2;
    sh[3][threadIdx.x] = s3;
    __syncthreads();

    for (int s = BLOCK_PHO >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[0][threadIdx.x] += sh[0][threadIdx.x + s];
            sh[1][threadIdx.x] += sh[1][threadIdx.x + s];
            sh[2][threadIdx.x] += sh[2][threadIdx.x + s];
            sh[3][threadIdx.x] += sh[3][threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[0 * NWL + iwl], sh[0][0]);
        atomicAdd(&result[1 * NWL + iwl], sh[1][0]);
        atomicAdd(&result[2 * NWL + iwl], sh[2][0]);
        atomicAdd(&result[3 * NWL + iwl], sh[3][0]);
    }
}
