
	  ////////////////
	 // LIBRAIRIES //
	////////////////

#include <mfhdf.h>
#include <curand_kernel.h>

/*
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <hdf.h>
#include <float.h>
#include <cutil.h>
#include <math.h>
#include <limits.h>
#include <string.h>
//#include <shrUtils.h>
//#include <shrQATest.h>
//#include <cutil_inline.h>
*/

	  ////////////////////////////
	 // CONSTANTES PREDEFINIES //
	////////////////////////////

// Constantes pour la fonction random Mersenne Twister
#define MT_MM 9
#define MT_NN 19
#define MT_WMASK 0xFFFFFFFFU
#define MT_UMASK 0xFFFFFFFEU
#define MT_LMASK 0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18

// Poids initial du photon
#define WEIGHTINIT 1.F
// Au dela du poids WEIGHTMAX le photon est considéré comme une erreur
#define WEIGHTMAX 50.F
#define DEPO 0.0279F
#define PI 3.1415927F
//3.141 592 653 589F
#define DEUXPI 6.2831853F
//6.283 185 307 17F
#define DEMIPI 1.5707963F
//1.570 796 326 79F

// Precision de la recupération des poids des photons
#define SCALEFACTOR 1000000000
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F

// Localisation du photon
#define SPACE		0
#define ATMOS		1
#define SURFACE		2
#define ABSORBED	3
#define NONE		4

// DEBUG Test des differentes fonctions random
#ifdef RANDMWC
#define RAND randomMWCfloat_co(tab.x+idx,tab.a+idx)
#endif
#ifdef RANDCUDA
#define RAND curand_uniform(tab.globalRand+idx)
#endif
#ifdef RANDMT
#define RAND randomMTfloat(tab.etat+idx, tab.config+idx)
#endif

	  /////////////////////////////
	 // CONSTANTES FICHIER HOST //
	/////////////////////////////

extern unsigned long long NBPHOTONS;
extern unsigned int NBLOOP;
extern float THETASOL;
extern float LAMBDA;
extern float TAURAY;
extern float TAUAER;
extern float W0AER;
extern float HA;
extern float HR;
extern float ZMIN;
extern float ZMAX;
extern float WINDSPEED;
extern float NH2O;
extern float CONPHY;
extern int XBLOCK;
extern int YBLOCK;
extern int XGRID;
extern int YGRID;
extern int NBTHETA;
extern int NBPHI;
extern int NBSTOKES;
extern int PROFIL;
extern int SIM;
extern int SUR;
extern int DIOPTRE;
extern int DIFFF;

	  //////////////
	 // TYPEDEFS //
	//////////////

typedef struct __align__(16)
{
	// Vecteur normalisé de la vitesse du photon
	float vx;
	float vy;
	float vz;
	// Vecteur normalisé orthogonal à la vitesse du photon
	float ux;
	float uy;
	float uz;
	// Localisation du photon
	int loc;
	// Epaisseur Rayleigh du photon
	float tau;
	// Poids du photon
	float weight;
	// Paramètres de stokes du photon
	float stokes1;
	float stokes2;
	float stokes3;
	float stokes4;
}Photon;

typedef struct __align__(16)
{
	unsigned long long nbPhotons; //nombre de photons traités pour un appel du Kernel
	int erreurpoids; //nombre de photons ayant un poids anormalement élevé
	int erreurtheta; //nombre de photons ignorés (sortant dans la direction solaire)
	#ifdef PROGRESSION
	unsigned long long nbThreads; //nombre total de threads lancés
	unsigned long long nbPhotonsSor; //nombre de photons ressortis pour un appel du Kernel
	int erreurvxy; //nombre de photons sortant au zénith et donc difficiles à classer
	int erreurvy; //nombre de photons sortant à phi=0 ou phi=PI et donc difficiles à classer
	int erreurcase; // nombre de photons rangé dans une case inexistante
	#endif
}Variables; //Regroupement des variables envoyées dans le kernel

typedef struct {
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
} ConfigMT; // Parametres pour la fonction random Mersenne Twister

typedef struct {
	unsigned int mt[MT_NN];
	int iState;
	unsigned int mti1;
} EtatMT; // Etat ddu generateur pour la fonction random Mersenne Twister

typedef struct __align__(16)
{
	unsigned long long* tabPhotons;
	
	#ifdef RANDMWC
	unsigned long long* x;
	unsigned int* a;
	#endif
	#ifdef RANDCUDA
	curandState_t* globalRand;
	#endif
	#ifdef RANDMT
	ConfigMT* config;
	EtatMT* etat;
	#endif
}Tableaux; // Regroupement des tableaux envoyés dans le kernel


typedef struct __align__(16)
{
	int action;
	float tau;
	float poids;
}Evnt; // DEBUG permet de recuperer des infos sur certains photons
