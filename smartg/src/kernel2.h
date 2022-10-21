#include <stdio.h>
extern "C" {
__global__ void reduce_absorption_gpu(unsigned long long NPHOTON, unsigned long long NLAYER, unsigned long long NWVL, 
        unsigned long long NTHREAD, unsigned long long NGROUP, unsigned long long NBUNCH, unsigned long long NP_REST, unsigned long long NWVL_LOW,
        double *res, double *res_sca, double *res_rrs, double *res_sif, double *res_vrs, float *ab, float *al, float *cd, float *S, float *weight, 
        unsigned char *nrrs, unsigned char *nref, unsigned char *nsif, unsigned char *nvrs, unsigned char *nenv, unsigned char *iw_low, float *ww_low);
}
