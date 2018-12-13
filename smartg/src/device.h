
#ifndef DEVICE_H
#define DEVICE_H

/**********************************************************
*
*			device.h
*
***********************************************************/


__device__ __constant__ unsigned int NBLOOPd;
__device__ __constant__ int NOCEd;
__device__ __constant__ unsigned int OUTPUT_LAYERSd;
__device__ __constant__ unsigned int NF;
__device__ __constant__ int NATMd;
__device__ __constant__ float WINDSPEEDd;
__device__ __constant__ float NH2Od;

__device__ __constant__ int XBLOCKd;
__device__ __constant__ int YBLOCKd;
__device__ __constant__ int XGRIDd;
__device__ __constant__ int YGRIDd;
__device__ __constant__ int NBTHETAd;
__device__ __constant__ int NBPHId;
__device__ __constant__ int NLAMd;
__device__ __constant__ int NPSTKd;
__device__ __constant__ int NLVLd;
__device__ __constant__ int SIMd;
__device__ __constant__ int LEd;
__device__ __constant__ int ZIPd;
__device__ __constant__ int FLUXd;
//__device__ __constant__ int MId;
__device__ __constant__ int SURd;
__device__ __constant__ int BRDFd;
__device__ __constant__ int DIOPTREd;
__device__ __constant__ int WAVE_SHADOWd;
__device__ __constant__ int SINGLEd;
__device__ __constant__ int ENVd;
__device__ __constant__ float ENV_SIZEd;		// Size of target in km
__device__ __constant__ float X0d;		// position of the target in x (km)
__device__ __constant__ float Y0d;		// position of the target in y (km)
__device__ __constant__ float STHVd;	//sinThetaView_Host
__device__ __constant__ float CTHVd;	//cosThetaView_Host

__device__ __constant__ float RTER;
__device__ __constant__ int NWLPROBA;
__device__ __constant__ int REFRACd;
__device__ __constant__ int HORIZd;
__device__ __constant__ float SZA_MAXd;
__device__ __constant__ float SUN_DISCd;
__device__ __constant__ int BEERd;
__device__ __constant__ int RRd;
__device__ __constant__ float WEIGHTRRd; // THRESHOLD for RUSSIAN ROULETTE PROCEDURE
__device__ __constant__ int NLOWd;
__device__ __constant__ int NJACd;
__device__ __constant__ int HISTd;
__device__ __constant__ int NSENSORd;
// copy en rapport avec les objets :
#ifdef OBJ3D
__device__ __constant__ int nObj;
__device__ __constant__ float Pmin_x;
__device__ __constant__ float Pmin_y;
__device__ __constant__ float Pmin_z;
__device__ __constant__ float Pmax_x;
__device__ __constant__ float Pmax_y;
__device__ __constant__ float Pmax_z;
__device__ __constant__ int IsAtm;
__device__ __constant__ float TCd;
__device__ __constant__ int nbCx;
__device__ __constant__ int nbCy;
#endif
// custum forward
__device__ __constant__ float CFXd;
__device__ __constant__ float CFYd;
__device__ __constant__ int CFMODEd;

/**********************************************************
*	> Prototypes de device.cu
***********************************************************/


/**********************************************************
*	> Kernel
***********************************************************/

extern "C" {
__global__ void launchKernel(
        struct Spectrum *spectrum, float *X0,
        struct Phase *faer, struct Phase *foce2,
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        unsigned long long *errorcount, int *nThreadsActive, void *tabPhotons, void *tabDist, void *tabHist,
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //unsigned long long *errorcount, int *nThreadsActive, void *tabPhotons,
        unsigned long long *Counter,
        unsigned long long *NPhotonsIn,
        unsigned long long *NPhotonsOut,
        float *tabthv, float *tabphi,  struct Sensor *tab_sensor,
        struct Profile *prof_atm,
        struct Profile *prof_oc,
        long long *wl_proba_icdf,
        void *rng_state
		#ifdef OBJ3D
		, void *tabObjInfo,
		struct IObjets *myObjets,
		unsigned long long *nbPhCat,
		double *wPhCat
		#endif
        );
}


/* initPhoton
*/
__device__ void initPhoton(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                           struct Sensor *tab_sensor, struct Spectrum *spectrum,float *X0,
                           unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf, float* tabthv, float* tabphi,
                           struct RNG_State*
						   #ifdef OBJ3D
						   , struct IObjets *myObjets
						   #endif
	);

#ifdef SPHERIQUE
__device__ void move_sp(Photon*, struct Profile *prof_atm, int le, int count_level, struct RNG_State*);
#endif

#ifdef ALT_PP
__device__ void move_pp2(Photon*, struct Profile *prof_atm, struct Profile* prof_oc, int le, int count_level,
                        struct RNG_State*);
#endif

// move, version plan parallèle
__device__ void move_pp(Photon*, struct Profile *prof_atm, struct Profile* prof_oc,
                        struct RNG_State*
						#ifdef OBJ3D
						, IGeo *geoS, struct IObjets *myObjets, void *tabObjInfo
						#endif
	);


/* scatter */

__device__ void choose_scatterer(Photon* ph,
								 struct Profile *prof_atm, struct Profile *prof_oc,
								 struct Spectrum *spectrum,
								 struct RNG_State*);


__device__ void scatter(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
        struct Phase *faer2, struct Phase *foce2,
        int le, float dth,
        float* tabthv, float* tabphi, int count_level,
        struct RNG_State*);


__device__ void surfaceAgitee(Photon*, int le, float* tabthv, float* tabphi, int count_level,
                              struct RNG_State*);
__device__ void surfaceBRDF(Photon*, int le, float* tabthv, float* tabphi, int count_level,
                              struct RNG_State*);
__device__ void surfaceBRDF_old(Photon*, int le, float* tabthv, float* tabphi, int count_level,
                              struct RNG_State*);


__device__ void surfaceLambert(Photon*, int le,
                                    float* tabthv, float* tabphi, struct Spectrum *spectrum,
                                    struct RNG_State*);

#ifdef OBJ3D
__device__ void surfaceLambertienne3D(Photon* ph, int le, float* tabthv, float* tabphi,
									  struct Spectrum *spectrum, struct RNG_State*, IGeo* geoS);

__device__ void surfaceRugueuse3D(Photon* ph, IGeo* geoS, struct RNG_State *rngstate);

__device__ void countPhotonObj3D(Photon* ph, void *tabObjInfo, IGeo* geoS, unsigned long long *nbPhCat, double *wPhCat);
#endif

__device__ void countPhoton(Photon* , struct Profile* prof_atm, struct Profile* prof_oc, float*, float *,
        int, unsigned long long*, void*, void*, void*, unsigned long long*);

// rotation of the stokes parameters by an angle psi
__device__ void rotateStokes(float4 s, float psi, float4 *sr);

// rotation Matrix L by an angle psi
__device__ void rotationM(float psi, float4x4 *L);

// Rotation Matrix of angle theta around unit vector u
__device__ void rotation3D_test(float, float3, float3x3*);

// Rotation Matrix of angle theta around unit vector u
__device__ float3x3 rotation3D(float, float3);

/* ComputePsi */
__device__ void ComputePsi(Photon*, float*, float);

/* ComputePsiZenith */
__device__ void ComputePsiZenith(Photon* , float* , float);

/* ComputeBox */
__device__ int ComputeBox(int*, int*, int*, Photon*,
                           unsigned long long *errorcount, int count_level);

#ifdef DEBUG_PHOTON
__device__ void display(const char* desc, Photon* ph);
#endif

__device__ void copyPhoton(Photon*, Photon*); 

__device__ void modifyUV( float3 v0, float3 u0, float cTh, float psi, float3 *v1, float3 *u1) ;

__device__ float ComputeTheta(float3 v0, float3 v1);

__device__ void ComputePsiLE(float3 u0,	float3 v0, float3 v1, float* psi, float3* u1);
#if defined(DOUBLE) && (__CUDA_ARCH__ < 600)
__device__ double DatomicAdd(double* address, double val);
#endif
__device__ float get_OD(int , struct Profile ) ;  

__device__ float Lambda(float , float ) ;
__device__ double LambdaM(double , double ) ;

__device__ void DirectionToUV(float, float, float3*, float3*) ;
__device__ float3 LocalToGlobal(float3, float3, float3, float3) ;
__device__ float3 GlobalToLocal(float3, float3, float3, float3) ;
__device__ void MakeLocalFrame(float3, float3*, float3*, float3*) ;

/* Fresnel Reflection Matrix*/
__device__ float4x4 FresnelR(float3, float3) ;

#ifdef PHILOX
/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/


/* initPhilox4x32_7Compteur
* Fonction qui initialise la partie variable du compteur des philox
*/
__global__ void initPhilox4x32_7Compteur(unsigned int*, unsigned int);

/* randomPhilox4x32_7float
* Fonction random Philox-4x32-7 qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
*/
__device__ float randomPhilox4x32_7float(philox4x32_ctr_t*, philox4x32_key_t*);

/* randomPhilox4x32_7uint
* Fonction random Philox-4x32-7 qui renvoit un uint à partir d'un generateur (etat+config)
* TODO A noter que 4 valeurs sont en fait generees, un seul uint peut etre renvoye, donc 3 sont perdus
* En pratique les valeurs generees sont des int32. Il y a donc une conversion vers uint32 de realisee
*/
__device__ unsigned int randomPhilox4x32_7uint(philox4x32_ctr_t*, philox4x32_key_t*);
#endif

#ifdef OBJ3D
/**********************************************************
*	> Fonctions liées à la création de géométries
***********************************************************/

__device__ bool geoTest(float3 o, float3 dir, int phLocPrev, float3* phit, IGeo *GeoV , struct IObjets *myObjets);
#endif
#endif // DEVICE_H
