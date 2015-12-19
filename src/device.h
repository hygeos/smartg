
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
__device__ __constant__ int LEd;
__device__ __constant__ int FLUXd;
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

extern "C" {
__global__ void launchKernel(Variables* var, Tableaux *tab, float *X0);
}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph, Tableaux tab, float *X0,
                           philox4x32_ctr_t*, philox4x32_key_t*);


// move, version sphérique
#ifdef SPHERIQUE
__device__ void move_sp(Photon*, Tableaux tab, int le, int count_level , philox4x32_ctr_t*, philox4x32_key_t*);
#endif

// move, version plan parallèle
__device__ void move_pp(Photon*,float*z, float* h, float* pMol , float *abs , float* ho , philox4x32_ctr_t*, philox4x32_key_t*);


/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter(Photon* ph, float* faer, float* ssa , float* foce , float* sso, int* ip, int* ipo, int le, float* tabthv, float* tabphi, int count_level , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr);


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
* //TODO: transmission vers l'océan et/ou reflexion totale
*/
__device__ void surfaceAgitee_old(Photon*, float* alb , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr);

__device__ void surfaceAgitee(Photon*, float* alb, int le, float* tabthv, float* tabphi, int count_level , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr);


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* , float* alb, philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr);


/* exit
* Sauve les paramètres des photons sortis dans l'espace dans la boite correspondant à la direction de sortie
*/
__device__ void countPhoton(Photon* , Tableaux, int , Variables*);



// rotation of the stokes parameters by an angle psi
__device__ void rotateStokes(float s1, float s2, float s3, float psi,
        float *s1r, float *s2r, float *s3r);


/* ComputePsi
* Calcul du psi pour la direction de sortie du photon
*/
__device__ void ComputePsi(Photon*, float*, float);


/* ComputeBox
* Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void ComputeBox(int*, int*, int*, Photon* , Variables* var);

#ifdef DEBUG_PHOTON
__device__ void display(const char* desc, Photon* ph);
#endif

__device__ void copyPhoton(Photon*, Photon*); 

__device__ void modifyUV( float vx0, float vy0, float vz0, float ux0, float uy0, float uz0,
        float cTh, float psi, 
        float *vx1, float *vy1, float *vz1, float *ux1, float *uy1, float *uz1) ;

__device__ float ComputeTheta(float vx0, float vy0, float vz0, float vx1, float vy1, float vz1);

__device__ void ComputePsiLE(float ux0, float uy0, float uz0,
			    float vx0, float vy0, float vz0,
			    float vx1, float vy1, float vz1,
			    float* psi,
			    float* ux1, float* uy1, float* uz1);
#ifdef DOUBLE
__device__ double DatomicAdd(double* address, double val);
#endif

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

#endif // DEVICE_H
