
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

__device__ __constant__ unsigned long long NBPHOTONSd;
__device__ __constant__ unsigned int NBLOOPd;
__device__ __constant__ float THSDEGd;
__device__ __constant__ float LAMBDAd;
__device__ __constant__ float TAURAYd;
__device__ __constant__ float TAUAERd;
#ifndef SPHERIQUE
__device__ __constant__ float TAUATMd;
__device__ __constant__ float TAUMAXd;	//tau initial du photon (Host)
#endif
__device__ __constant__ float W0AERd;
__device__ __constant__ float W0OCEd;
__device__ __constant__ float W0LAMd;
__device__ __constant__ unsigned int NFAERd;
__device__ __constant__ unsigned int NFOCEd;
__device__ __constant__ float HAd;
__device__ __constant__ float HRd;
__device__ __constant__ float ZMINd;
__device__ __constant__ float ZMAXd;
__device__ __constant__ int NATMd;
__device__ __constant__ int HATMd;
__device__ __constant__ float WINDSPEEDd;
__device__ __constant__ float NH2Od;
__device__ __constant__ float CONPHYd;

__device__ __constant__ int XBLOCKd;
__device__ __constant__ int YBLOCKd;
__device__ __constant__ int XGRIDd;
__device__ __constant__ int YGRIDd;
__device__ __constant__ int NBTHETAd;
__device__ __constant__ int NBPHId;
__device__ __constant__ int PROFILd;
__device__ __constant__ int SIMd;
__device__ __constant__ int SURd;
__device__ __constant__ int DIOPTREd;
__device__ __constant__ int DIFFFd;
__device__ __constant__ float THSd;		//thetaSolaire_Host en radians
__device__ __constant__ float STHSd;	//sinThetaSolaire_Host
__device__ __constant__ float CTHSd;	//cosThetaSolaire_Host

__device__ __constant__ float GAMAd;


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
__global__ void lancementKernel(Variables* var, Tableaux tab
		#ifdef SPHERIQUE
		, Init* init
		#endif
		#ifdef TABRAND
		, float*
		#endif
		#ifdef TRAJET
		, Evnt*
		#endif
			       );


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph
		#ifdef SPHERIQUE
		, Tableaux tab, Init* init
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );


/* move
* Effectue le déplacement du photon dans l'atmosphère
* Pour l'atmosphère sphèrique, l'algorithme est basé sur la formule de pythagore généralisé
* Modification des coordonnées position du photon
*/
__device__ void move(Photon*, Tableaux tab
		#ifndef SPHERIQUE
		,int flagDiff
		#endif
		#ifdef SPHERIQUE
		, Init* init
		#endif
		#ifdef DEBUG
		, Variables* var
		#endif
		#ifdef RANDMWC
		, unsigned long long*, unsigned int*
		#endif
		#ifdef RANDCUDA
		, curandState_t*
		#endif
		#ifdef RANDMT
		, EtatMT*, ConfigMT*
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );


/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter(Photon* photon, float* faer, float* foce
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		);


/* calculDiffScatter
* Regroupe l'ensemble des calculs propre à la diffusion moléculaire ou par les aérosols.
* Pour l'optimisation du programme, il est possible d'effectuer un travail de réduction au maximum de cette fonction. L'idée est
* de calculer et d'utiliser la fonction de phase moléculaire
*/
__device__ void calculDiffScatter( Photon* photon, float* cTh, float* faer, float* foce
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		);


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
* //TODO: transmission vers l'océan et/ou reflexion totale
*/
__device__ void surfaceAgitee(Photon*
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		      );


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* photon
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		);


/* exit
* Sauve les paramètres des photons sortis dans l'espace dans la boite correspondant à la direction de sortie
*/
__device__ void exit(Photon* , Variables*, Tableaux, unsigned long long*
		#ifdef PROGRESSION
		, unsigned int*
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );


/* Modifie les paramètres de stokes
* Flag permet de tester (si flag=1) ou non la valeur des paramètres avant modification
*/
__device__ void modifStokes(Photon*, float, float, float, int flag);


/* calculPsi
* Calcul du psi pour la direction de sortie du photon
*/
__device__ void calculPsi(Photon*, float*, float);


/* calculCase
* Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void calculCase(int*, int*, Photon*, Variables*);


/**********************************************************
*	> Initialisation de données dans le device
***********************************************************/

/* initConstantesDevice
* Fonction qui initialise les constantes du device calculés dans le host
* Elle doit rester dans ce fichier
*/
void initConstantesDevice();


/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/

/* initRandCUDA
* Fonction qui initialise les generateurs du random cuda
*/
__global__ void initRandCUDA(curandState_t*, unsigned long long);


/* initRandMTEtat
* Fonction qui initialise l'etat des generateurs du random Mersenne Twister (generateur = etat + config)
*/
__global__ void initRandMTEtat(EtatMT*, ConfigMT*);


/* randomMWCfloat
* Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
*/
__device__ float randomMWCfloat(unsigned long long*,unsigned int*);


/* randomMTfloat
* Fonction random Mersenne Twister qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
*/
__device__ float randomMTfloat(EtatMT*, ConfigMT*);


/* randomMTuint
* Fonction random Mersenne Twister qui renvoit un uint à partir d'un generateur (etat+config)
*/
__device__ unsigned int randomMTuint(EtatMT*, ConfigMT*);


#endif	// DEVICE_H
