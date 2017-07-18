
#ifndef DEVICE_H
#define DEVICE_H

/**********************************************************
*
*			device.h
*
*	> Variables externes fichier device/kernel
*	> Prototypes de device.cu
*		> Modélisation phénomènes physiques
*		> Initialisation de données dans le device
*		> Fonctions liées au générateur aléatoire
*
***********************************************************/


/**********************************************************
*	> Variables externes fichier device/kernel
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
__device__ __constant__ int FLUXd;
__device__ __constant__ int MId;
__device__ __constant__ int SURd;
__device__ __constant__ int BRDFd;
__device__ __constant__ int DIOPTREd;
__device__ __constant__ int WAVE_SHADOWd;
__device__ __constant__ int ENVd;
__device__ __constant__ float ENV_SIZEd;		// Size of target in km
__device__ __constant__ float X0d;		// position of the target in x (km)
__device__ __constant__ float Y0d;		// position of the target in y (km)
__device__ __constant__ float STHVd;	//sinThetaView_Host
__device__ __constant__ float CTHVd;	//cosThetaView_Host

__device__ __constant__ float RTER;
__device__ __constant__ int NWLPROBA;
__device__ __constant__ int BEERd;
__device__ __constant__ int NLOWd;
__device__ __constant__ int BACKd;
__device__ __constant__ float POSXd;
__device__ __constant__ float POSYd;
__device__ __constant__ float POSZd;
__device__ __constant__ float THDEGd;
__device__ __constant__ float PHDEGd;
__device__ __constant__ int LOCd;
__device__ __constant__ float FOVd;
__device__ __constant__ int TYPEd;



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
        unsigned long long *errorcount, int *nThreadsActive, void *tabPhotons,
        unsigned long long *Counter,
        unsigned long long *NPhotonsIn,
        unsigned long long *NPhotonsOut,
        float *tabthv, float *tabphi,
        struct Profile *prof_atm,
        struct Profile *prof_oc,
        long long *wl_proba_icdf,
        void *rng_state
        );
}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton2(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                           struct Spectrum *spectrum,float *X0,
                           unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf, float* tabthv, float* tabphi,
                           struct RNG_State*);
__device__ void initPhoton(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                           struct Spectrum *spectrum,float *X0,
                           unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf,
                           struct RNG_State*);


// move, version sphérique
#ifdef SPHERIQUE
__device__ void move_sp(Photon*, struct Profile *prof_atm, int le, int count_level, struct RNG_State*);
#endif

// move, version plan parallèle
__device__ void move_pp(Photon*, struct Profile *prof_atm, struct Profile* prof_oc,
                        struct RNG_State*);


/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
        struct Phase *faer2, struct Phase *foce2,
        int le,
        float* tabthv, float* tabphi, int count_level,
        struct RNG_State*);


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
*/
__device__ void surfaceAgitee(Photon*, int le, float* tabthv, float* tabphi, int count_level,
                              struct RNG_State*);
__device__ void surfaceBRDF(Photon*, int le, float* tabthv, float* tabphi, int count_level,
                              struct RNG_State*);


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon*, int le,
                                    float* tabthv, float* tabphi, struct Spectrum *spectrum,
                                    struct RNG_State*);


/* exit
* Sauve les paramètres des photons sortis dans l'espace dans la boite correspondant à la direction de sortie
*/
__device__ void countPhoton(Photon* , struct Profile* prof_atm, struct Profile* prof_oc, float*, float *,
        int, unsigned long long*, void*, unsigned long long*);



// rotation of the stokes parameters by an angle psi
__device__ void rotateStokes(float4 s, float psi, float4 *sr);

// rotation Matrix L by an angle psi
__device__ void rotationM(float psi, float4x4 *L);


/* ComputePsi
*/
__device__ void ComputePsi(Photon*, float*, float);


/* ComputeBox
* Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void ComputeBox(int*, int*, int*, Photon*,
                           unsigned long long *errorcount);

#ifdef DEBUG_PHOTON
__device__ void display(const char* desc, Photon* ph);
#endif

__device__ void copyPhoton(Photon*, Photon*); 

__device__ void modifyUV( float3 v0, float3 u0, float cTh, float psi, float3 *v1, float3 *u1) ;

__device__ float ComputeTheta(float3 v0, float3 v1);

__device__ void ComputePsiLE(float3 u0,	float3 v0, float3 v1, float* psi, float3* u1);

#ifdef DOUBLE
__device__ double DatomicAdd(double* address, double val);
#endif

__device__ float get_OD(int , struct Profile ) ;  



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

#endif // DEVICE_H
