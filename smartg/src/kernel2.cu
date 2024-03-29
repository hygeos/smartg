#include "communs.h"
#include "device.h"
#include "transform.h"
#include <math.h>
#include <math_constants.h>
#include "helper_math.h"
#include <stdio.h>
#include <cuda_fp16.h>
extern "C" {
__global__ void reduce_absorption_gpu(unsigned long long NPHOTON, unsigned long long NLAYER, unsigned long long NWVL, 
        unsigned long long NTHREAD, unsigned long long NGROUP, unsigned long long NBUNCH, unsigned long long NP_REST, unsigned long long NWVL_LOW,
        double *res, double *res_sca, double *res_rrs, double *res_sif, double *res_vrs, float *ab, float *al, float *cd, float *S, float *weight, 
        unsigned char *nrrs, unsigned char *nref, unsigned char *nsif, unsigned char *nvrs, unsigned char *nenv, unsigned char *iw_low, float *ww_low)
{
  const unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned long long n,nstart,nstop,ns;
  unsigned long long iw,ig,l,s;
  unsigned long long nl,li;
  double wabs, walb0, walb1,walb, Jwabs,Jwalb; // absorption and albedo high resolution weigths, Jacobian
  float wsca1,wsca2,wsca;
  unsigned long long iw1,iw2;

  if (idx<NTHREAD) {
    iw = idx%NWVL ; // index of current wavelength
    ig = idx/NWVL ;   // index of current group
    nstart = ig    *NBUNCH; // Start index of the photon's stack
    nstop  = (ig+1)*NBUNCH + (ig==(NGROUP-1))*NP_REST; // end index of photon's stack
                                    // last group has some remaining phton's
    iw1 = iw_low[iw];    // bracketing indices of low resolution wavelength grid
    iw2 = iw1+1;
    walb0= (double)al[iw]; // albedo of surface
    walb1= (double)al[iw +NWVL]; // albedo of surface (environment)

    for (n=nstart;n<nstop;n++) { // Loop on photon number
        //interpolating scattering 'low resolution' weights 
        wsca1 = weight[iw1+NWVL_LOW*n]; 
        wsca2 = weight[iw2+NWVL_LOW*n]; 
        wsca  = ww_low[iw] * (wsca2-wsca1) + wsca1;
        //start the computation of absorption weights
        wabs = 0.;
        for (l=0;l<NLAYER;l++) { // Loop on vertical layer
            nl = l   + NLAYER*n;
            li = iw  + NWVL*l;
            wabs += (double)cd[nl] * (double)ab[li];
        }
        Jwabs = (double)cd[NLAYER-1];
        walb  = pow(walb0, (double)nref[n]);
        Jwalb = nref[n] * pow(walb0, (double)((nref[n]-1));
        walb *= pow(walb1, (double)nenv[n]);
        Jwalb*= pow(walb1, (double)nenv[n]);
        for (s=0;s<4;s++) {
            ns = s + 4*n;
            if (!nsif[n])             atomicAdd(res    +iw+NWVL*s, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if (!nsif[n])             atomicAdd(res_sca    +iw+NWVL*s, (double)S[ns] * Jwabs * exp(-wabs) * (double)wsca * walb);
            //if (!nsif[n])             atomicAdd(res_sca+iw+NWVL*s, (double)S[ns] *              (double)wsca * walb);
            if ( nrrs[n] && !nsif[n]) atomicAdd(res_rrs+iw+NWVL*s, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nsif[n])             atomicAdd(res_sif+iw+NWVL*s, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nvrs[n] && !nsif[n]) atomicAdd(res_vrs+iw+NWVL*s, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
        }
    }
  }
}
}

#include <stdio.h>
extern "C" {
__global__ void reduce_absorption_gpu2(unsigned long long NPHOTON, unsigned long long NWVL, unsigned long long NINFO, 
        unsigned long long NOCE, unsigned long long NATM,
        unsigned long long NTHREAD, unsigned long long NGROUP, unsigned long long NBUNCH, unsigned long long NP_REST, unsigned long long NWVL_LOW,
        double *res, double *ab, float *tabHist, unsigned char *iw_low, float *ww_low)
{
  const unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned long long n,nstart,nstop,ns;
  unsigned long long iw,ig,l,s;
  unsigned long long nl,li;
  double wabs; // absorption OD of a photon at the current wavelength;
  float wsca1,wsca2,wsca;
  unsigned long long iw1,iw2;

  if (idx<NTHREAD) {
    iw = idx%NWVL ; // index of current wavelength
    ig = idx/NWVL ;   // index of current group
    nstart = ig    *NBUNCH; // Start index of the photon's stack
    nstop  = (ig+1)*NBUNCH + (ig==(NGROUP-1))*NP_REST; // end index of photon's stack
                                    // last group has some remaining phton's
    iw1 = iw_low[iw];    // bracketing indices of low resolution wavelength grid
    iw2 = iw1+1;

    for (n=nstart;n<nstop;n++) { // Loop on photon number
        //interpolating scattering 'low resolution' weights 
        wsca1 = tabHist[iw1+NATM+NOCE+4+NINFO*n]; 
        wsca2 = tabHist[iw2+NATM+NOCE+4+NINFO*n]; 
        //wsca1 = weight[iw1+NWVL_LOW*n]; 
        //wsca2 = weight[iw2+NWVL_LOW*n]; 
        wsca  = ww_low[iw] * (wsca2-wsca1) + wsca1;
        //start the computation of absorption weights
        wabs = 0.;
        for (l=0;l<NATM;l++) { // Loop on vertical layer
            nl = NOCE+l   + NINFO*n;
            li = iw  + NWVL*l;
            wabs += (double)tabHist[nl] * ab[li];
        }
        for (s=0;s<4;s++) {
            ns = s+NATM+NOCE + NINFO*n;
            //ns = s + 4*n;
            atomicAdd(res+iw+NWVL*s, (double)tabHist[ns] * exp(-wabs) * (double)wsca);
            //res[iw + NWVL*ig + NWVL*NGROUP*s] +=  (double)tabHist[ns] * exp(-wabs) * (double)wsca;
        }
    }
  }
}
}
