
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
__device__ __constant__ unsigned int NFAERd;
__device__ __constant__ unsigned int NFOCEd;
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
__device__ __constant__ int SIMd;
__device__ __constant__ int SURd;
__device__ __constant__ int DIOPTREd;
__device__ __constant__ int ENVd;
__device__ __constant__ float ENV_SIZEd;		// Size of target in km
__device__ __constant__ float X0d;		// position of the target in x (km)
__device__ __constant__ float Y0d;		// position of the target in y (km)
__device__ __constant__ float STHVd;	//sinThetaView_Host
__device__ __constant__ float CTHVd;	//cosThetaView_Host

__device__ __constant__ float DELTAd;
__device__ __constant__ float DELTA_PRIMd;
__device__ __constant__ float DELTA_SECOd;
__device__ __constant__ float BETAd;
__device__ __constant__ float ALPHAd;
__device__ __constant__ float Ad;
__device__ __constant__ float ACUBEd;

__device__ __constant__ float RTER;



/**********************************************************
*	> Prototypes de device.cu
***********************************************************/


/**********************************************************
*	> Kernel
***********************************************************/

/* lancementKernel
* Kernel de lancement et gestion de la simulation
* Les fonctions de plus bas niveau sont appelées en fonction de la localisation du photon
* Il peut être important de rappeler que le kernel lance tous les threads mais effectue des calculs similaires. La boucle de la
* fonction va donc être effectuée pour chaque thread du block de la grille
*/
extern "C" {
__global__ void lancementKernelPy(Variables* var, Tableaux *tab, Init* init);
}

__device__ void launchKernel(Variables* var, Tableaux tab, Init* init);


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph, Tableaux tab
		, Init* init
		#ifdef RANDMWC
		, unsigned long long*, unsigned int*
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT*, ConfigMT*
                #endif
                #ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t*, philox4x32_key_t*
                #endif
		    );


// move, version sphérique
#ifdef SPHERIQUE
__device__ void move_sp(Photon*, Tableaux tab, Init* init
		#ifdef RANDMWC
		, unsigned long long*, unsigned int*
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT*, ConfigMT*
                #endif
                #ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t*, philox4x32_key_t*
                #endif
		    );
#endif


// move, version plan parallèle
__device__ void move_pp(Photon*,float*z, float* h, float* pMol , float *abs , float* ho
		#ifdef RANDMWC
		, unsigned long long*, unsigned int*
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT*, ConfigMT*
                #endif
                #ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t*, philox4x32_key_t*
                #endif
		    );


/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter(Photon* photon, float* faer, float* ssa , float* foce , float* sso, int* ip, int* ipo
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
		);


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
* //TODO: transmission vers l'océan et/ou reflexion totale
*/
__device__ void surfaceAgitee(Photon*, float* alb
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
                #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
		      );


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* , float* alb
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
                #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
		);


/* exit
* Sauve les paramètres des photons sortis dans l'espace dans la boite correspondant à la direction de sortie
*/
__device__ void countPhoton(Photon* , Tableaux, int
		#ifdef PROGRESSION
		, Variables*
		#endif
		    );



// rotation of the stokes parameters by an angle psi
__device__ void rotateStokes(float s1, float s2, float s3, float psi,
        float *s1r, float *s2r, float *s3r);


/* calculPsi
* Calcul du psi pour la direction de sortie du photon
*/
__device__ void calculPsi(Photon*, float*, float);


/* calculCase
* Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void calculCase(int*, int*, int*, Photon*
				#ifdef PROGRESSION
				, Variables* var
				#endif 
					);



/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/

#ifdef RANDCUDA
/* initRandCUDA
* Fonction qui initialise les generateurs du random cuda
*/
__global__ void initRandCUDA(curandState_t*, unsigned long long);
#endif
#if defined(RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
/* initRandCUDANDQRNGs
* Fonction qui initialise le generateur (scrambled) sobol 32 de curand
*/
__global__ void initRandCUDANDQRNGs(curandSTATE* etat, curandDirectionVectors32_t *rngDirections);
#endif

#ifdef RANDMT
/* initRandMTEtat
* Fonction qui initialise l'etat des generateurs du random Mersenne Twister (generateur = etat + config)
*/
__global__ void initRandMTEtat(EtatMT*, ConfigMT*);


/* randomMTfloat
* Fonction random Mersenne Twister qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
*/
__device__ float randomMTfloat(EtatMT*, ConfigMT*);


/* randomMTuint
* Fonction random Mersenne Twister qui renvoit un uint à partir d'un generateur (etat+config)
*/
__device__ unsigned int randomMTuint(EtatMT*, ConfigMT*);
#endif


#ifdef RANDMWC
/* randomMWCfloat
* Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
*/
__device__ float randomMWCfloat(unsigned long long*,unsigned int*);
#endif

#ifdef RANDPHILOX4x32_7
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
