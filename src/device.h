
	  ///////////////////////////////
	 // CONSTANTES FICHIER DEVICE //
	///////////////////////////////

// Création des constantes utilisées dans le kernel et le fichier device uniquement
__device__ __constant__ unsigned long long NBPHOTONSd;
__device__ __constant__ unsigned int NBLOOPd;
__device__ __constant__ float THSDEGd;
__device__ __constant__ float LAMBDAd;
__device__ __constant__ float TAURAYd;
__device__ __constant__ float TAUAERd;
__device__ __constant__ float TAUATMd;
__device__ __constant__ float W0AERd;
__device__ __constant__ float W0LAMd;
__device__ __constant__ unsigned int NFAERd;
__device__ __constant__ float HAd;
__device__ __constant__ float HRd;
__device__ __constant__ float ZMINd;
__device__ __constant__ float ZMAXd;
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
__device__ __constant__ float THSd; //thetaSolaire_Host en radians
__device__ __constant__ float STHSd; //sinThetaSolaire_Host
__device__ __constant__ float CTHSd; //cosThetaSolaire_Host

__device__ __constant__ float GAMAd;

// __device__ float* FAERd;	// Paramètres du modèle de diffusion par aérosols
// __device__ __constant__ float* TAUCOUCHEd;	// Modèle des couches atmosphèriques
// __device__ __constant__ float* PMOLd;		// Modèle des couches atmosphèriques


	  ///////////////////////
	 // PROTOTYPES DEVICE //
	///////////////////////

__global__ void lancementKernel(Variables* var, Tableaux tab, Init* init
		#ifdef TABRAND
		, float*
		#endif
		#ifdef TRAJET
		, Evnt*
		#endif
			       );
__global__ void initRandCUDA(curandState_t*, unsigned long long);

__global__ void initRandMTEtat(EtatMT*, ConfigMT*);

__device__ void initPhoton(Photon* ph, Tableaux tab, Init* init, float* hphoton, float* zphoton /*float* hph0_s, float* zph0_s*/
		#ifdef TRAJET
		, int, Evnt*
		#endif
		#ifdef SORTIEINT
		, unsigned int iloop
		#endif
		    );


__device__ void calculProfil(Photon* ph, Tableaux tab, float* hphoton, float* zphoton
		#ifdef TRAJET
		, int idx
		#endif
		);

__device__ void move(Photon*, Tableaux tab, float* hphoton, float* zphoton
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


__device__ void exit(Photon* , Variables*, Tableaux, unsigned long long*
		#ifdef PROGRESSION
		, unsigned int*
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		#ifdef SORTIEINT
		, unsigned int iloop
		#endif
		    );
			
__device__ void calculPsi(Photon*, float*, float);

/* Modifie les paramètres de stokes
 * Flag permet de tester (si flag=1) ou non la valeur des paramètres avant modification
*/
__device__ void modifStokes(Photon*, float, float, float, int flag);

__device__ void calculCase(int*, int*, Photon*, Variables*);
__device__ float randomMWCfloat(unsigned long long*,unsigned int*);
__device__ float randomMTfloat(EtatMT*, ConfigMT*);
__device__ unsigned int randomMTuint(EtatMT*, ConfigMT*);
__device__ void atomicAddULL(unsigned long long* address, unsigned int add);

void initConstantesDevice();

// Détermination de la nouvelle direction du photon lorsqu'il est diffusé par aérosol

__device__ void calculDiffScatter( Photon* photon, float* cTh, Tableaux tab
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
		
// Cette fonction regroupe les deux diffusions: molécules et aérosols
__device__ void scatter(Photon* photon, Tableaux tab
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
		
__device__ void impact(Photon* photon, Tableaux tab
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		);

