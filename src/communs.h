
	  ////////////////
	 // LIBRAIRIES //
	////////////////

#include <mfhdf.h>
#include <stdio.h>
/*
#include <hdf.h>
#include <float.h>
#include <cutil.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
*/

	  //////////////////////////
	 // PARAMETRES PHYSIQUES //
	//////////////////////////

// Poids initial du photon
#define WEIGHTINIT 1.F
// Au dela du poids WEIGHTMAX le photon est considéré comme une erreur
#define WEIGHTMAX 50.F
// Angle zénithal solaire
#define THETASOL 70.F
// Epaisseur optique Rayleigh
#define TAU 0.05330F
#define DEPO 0.0279F
#define PI 3.1415927F
//3.141 592 653 589F
#define PImul2 6.2831853F
//6.283 185 307 17F
#define PIdiv2 1.5707963F
//1.570 796 326 79F

	  ///////////////////////////
	 // PARAMETRES SIMULATION //
	///////////////////////////

// Precision de la recupération des poids des photons
#define SCALEFACTOR 1000000000
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F
// Nombre de photons à lancer (limite unsigned long long
// + limite "contrôlée" du poids total des photons sortis unsigned long long)
#define NBPHOTONS 2000000000
// Nombres de Stokes pris en compte
#define NBSTOKES 2
// Nombre de boucles dans le kernel (limite unsigned int + limite watchdog 5s)
#define NBLOOP 10000
// Organisation des threads en blocks (matrice de threads) et grids (matrice de blocks)
#define XBLOCK	8
#define YBLOCK	4
#define XGRID	2
#define YGRID	6

	  //////////////////////////
	 // PARAMETRES DE SORTIE //
	//////////////////////////

// Echantillonnage d'une demi-sphère pour classer les photons sortants
// Theta parcourt PI/2
#define NBTHETA 180
// Phi parcourt 2.PI, NBPHI doit être pair
#define NBPHI 360

// Provisoire : Nombre d'itérations du programme, pour faire des tests
#define NBITERATIONS	1

	  ///////////////////////
	 // AUTRES PARAMETRES //
	///////////////////////

// Localisation du photon
#define SPACE		0
#define ATMOS		1
#define SURFACE		2
#define ABSORBED	3
#define NONE		4

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
	float thS; //thetaSolaire_Host
	float cThS; //cosThetaSolaire_Host
	float sThS; //sinThetaSolaire_Host
	float tauMax; //tau initial du photon (Host)
}Constantes;

typedef struct __align__(16)
{
	unsigned long long x;
	unsigned int a;
}Random;

typedef struct __align__(16)
{
	int action;
	float tau;
	float poids;
}Evnt;

typedef struct __align__(16)
{
	unsigned long long nbPhotons; //nombre de photons traités pour un appel du Kernel
	#ifdef PROGRESSION
	unsigned long long nbThreads; //nombre total de threads lancés
	unsigned long long nbPhotonsSor; //nombre de photons ressortis pour un appel du Kernel
	#endif
	int erreurpoids; //nombre de photons ayant un poids anormalement élevé
	int erreurtheta; //nombre de photons ignorés (sortant dans la direction solaire)
	int erreurvxy; //nombre de photons sortant au zénith et donc difficiles à classer
	int erreurvy; //nombre de photons sortant à phi=0 ou phi=PI et donc difficiles à classer
	int erreurcase; // nombre de photons rangé dans une case inexistante
}Progress;

