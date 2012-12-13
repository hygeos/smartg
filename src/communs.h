
#ifndef COMMUNS_H
#define COMMUNS_H

/**********************************************************
*
*			communs.h
*
*	> Include librairies
*	> Déclaration des constantes
*	> Variables externes fichier host
*	> Définition des structures
*
***********************************************************/


/**********************************************************
*	> Include
***********************************************************/

#include <stdio.h>
#include <mfhdf.h>
#include <curand_kernel.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>

#ifdef RANDPHILOX4x32_7
#include "philox.h"
#endif

/**********************************************************
*	> Constantes
***********************************************************/
// Codes retours des methode du projet MCCUDA
typedef int CrMCCUDA;
#define MCCUDA_OK                                           0
#define MCCUDA_KO                                           1
#define MCCUDA_ENVIRONNEMENT_GPU_NON_COMPATIBLE             2

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


/* Lié au photon */
// Poids initial du photon
#define WEIGHTINIT 1.F
// Au dela du poids WEIGHTMAX le photon est considéré comme une erreur
#define WEIGHTMAX 50.F

#define WEIGHTRR 1.F
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F
#define WEIGHTMIN 1.e-5f


/* Mathématiques */
#define PI 3.1415927F
//3.141 592 653 589F
#define DEUXPI 6.2831853F
//6.283 185 307 17F
#define DEMIPI 1.5707963F
//1.570 796 326 79F
#define DEG2RAD 0.017453293F
//0.017453293


/* Localisation du photon */
#define SPACE		0
#define ATMOS		1
#define SURFACE		2
#define ABSORBED	3
#define NONE		4
#define OCEAN		5


/* Paramètres pour l'océan */
#define DEPO 0.0279F
#define NFOCE_c = 10000000

/* Pour le calcul de la fonction de phase dans l'océan*/
#define NWAV	71

#define ANGTRONC 1999

/* Constantes propres au calcul sphérique */
#define RTER 6400


/* Option d'affichage des trajets */
#ifdef TRAJET
#define NBTRAJET 40	// Nombre de trajet à afficher pour debuggage
#endif


/* Test des differentes fonctions random */
#if defined(RANDCUDA)
typedef curandState_t curandSTATE;
#endif
#if defined(RANDCURANDSOBOL32)
typedef curandStateSobol32_t curandSTATE;
#endif
#if defined(RANDCURANDSCRAMBLEDSOBOL32)
typedef curandStateScrambledSobol32_t curandSTATE;
#endif
#ifdef RANDMWC
#define RAND randomMWCfloat(etatThr,configThr)
#endif
#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
#define RAND curand_uniform(etatThr)
// #define RAND randomXorwowFake(etatThr)
#endif
#ifdef RANDMT
#define RAND randomMTfloat(etatThr, configThr)
#endif
#ifdef RANDPHILOX4x32_7
#define RAND randomPhilox4x32_7float(etatThr, configThr)
#endif

/**********************************************************
*	> Variables externes fichier host
***********************************************************/

extern unsigned long long NBPHOTONS;
extern unsigned int NBLOOP;
extern int SEED;
extern int XBLOCK;
extern int YBLOCK;
extern int XGRID;
extern int YGRID;
extern int NBTHETA;
extern int NBPHI;
extern int PROFIL;
extern int SIM;
extern int SUR;
extern int DIOPTRE;
extern int DIFFF;

extern unsigned int LSAAER;
extern unsigned int NFAER;

extern unsigned int LSAOCE;
extern unsigned int NFOCE;

extern float THSDEG;
extern float LAMBDA;
extern float TAURAY;
extern float TAUAER;
extern float W0AER;
extern float W0LAM;
#ifdef FLAGOCEAN
extern float W0OCE;
extern float CONPHY;
#endif
extern float HA;
extern float HR;
extern float ZMIN;
extern float ZMAX;
extern int NATM;
extern float HATM;
extern float WINDSPEED;
extern float NH2O;

extern char PATHRESULTATSHDF[];
extern char PATHTEMOINHDF[];
extern char PATHDIFFAER[];
extern char PATHPROFILATM[];

extern int WRITE_PERIOD;


/**********************************************************
*	> Définition des structures
***********************************************************/

/* Photon
* Contient toutes les informations sur le photon lors de son parcours dans l'atmosphère
*/

typedef struct __align__(16)
{
	// Vecteur normalisé de la direction du photon (vitesse)
	float vx;
	float vy;
	float vz;
	// Vecteur normalisé orthogonal à la vitesse du photon (polarisation)
	float ux;
	float uy;
	float uz;
	
	// Localisation du photon
	int loc;
	
	// Poids du photon
	float weight;
	
	// Paramètres de stokes du photon
	float stokes1;
	float stokes2;
	float stokes3;
	// float stokes4;
	
	// Paramètres pour une atmosphère sphérique
	int couche;
	float prop_aer;		// Proportion d'aérosols par rapport aux molécules à l'endroit où se situe le photon
	
	float z;	// En plan parallèle, z représente le tau parcouru
	
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	int locPrec;
	
	// Position cartésienne du photon
	float x;
	float y;

	float rayon;
	float taumax;
	
	#endif
	
}	Photon;


/* Variables
* Contient toutes les variables qui sont renvoyées dans le host depuis le device suite
* à l'execution d'un kernel
*/
typedef struct __align__(16)
{
	unsigned long long nbPhotons;	// Nombre de photons traités pour un appel du Kernel
	int erreurpoids;				// Nombre de photons ayant un poids anormalement élevé
	int erreurtheta;				// Nombre de photons ignorés (sortant dans la direction solaire)
	
	#ifdef PROGRESSION
	unsigned long long nbThreads;	// Nombre total de threads lancés
	unsigned long long nbPhotonsSor;// Nombre de photons ressortis pour un appel du Kernel
	int erreurvxy;					// Nombre de photons sortant au zénith et donc difficiles à classer
	int erreurvy;					// Nombre de photons sortant à phi=0 ou phi=PI et donc difficiles à classer
	int erreurcase;					// Nombre de photons rangé dans une case inexistante
	#endif
	
}Variables;


/* ConfigMT
* Paramètres pour la fonction random Mersenne Twister
*/
typedef struct {
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
} ConfigMT;


/* EtatMT
* Etat du générateur pour la fonction random Mersenne Twister
*/
typedef struct {
	unsigned int mt[MT_NN];
	int iState;
	unsigned int mti1;
} EtatMT;


/* Tableaux
* Ensemble des tableaux envoyés par le host dans le device
* tabPhotons est également modifié par le kernel pour sauver les paramètres de stokes du photon sorti dans l'espace
*/
typedef struct __align__(16)
{
	float* tabPhotons;		// Tableau contenant l'ensemble des paramètres de stokes des photons sortis dans l'espace
	
	float* faer;			// Pointeur vers le modèle de diffusion des aérosols
	
	#ifdef FLAGOCEAN
	float* foce;			// Pointeur vers le modèle de diffusion dans l'océan
	#endif
	
	float* h;				// Pointeur vers l'épaisseur optique de chaque couches du modèle atmosphérique
	float* pMol;			// Pointeur vers la proportion de molécules dans chaque couches du modèle atmosphérique
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	
	float* z;				// Altitude de chaque couches
	
	/* Profil atmosphérique initial vu par la photon */
	float* hph0;			// Epaisseur optique vue devant le photon
	float* zph0;			// Altitude correspondante
	#endif
	
	
	#ifdef RANDMWC
	unsigned long long* etat;
	unsigned int* config;
	#endif
        #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
        curandSTATE* etat;
	#endif
	#ifdef RANDMT
	ConfigMT* config;
	EtatMT* etat;
        #endif
        #ifdef RANDPHILOX4x32_7
        /**********************/
        //parametres generaux :
        /**********************/
        //         philox4x32_ctr_t* PhiloxCompteurs; /*etat*/
        //!< ...il s'agit de 4 compteurs de 32bits.
        //!< ...trois sont fixes. Le 4eme evolue.
        //!< ...chaque generateur possede donc 2^32 valeurs (4milliards).
        //!< ...si on lance 10^11 photons necessitant 10^3 valeurs cela fait 10^14 nombres a generer
        //!< ...si l'on considere 256x256=65,536 (config. minimale supposee) threads (=generateurs)...
        //!< ...cela fait 1milliard de nombres a generer par thread. Valeur qui decroit considerablement si...
        //!< ...on augmente le nombre de threads.
        //         philox4x32_key_t* PhiloxClefs; /*config*/
        //!< ...il s'agit de deux clefs de 32bits.
        //!< ...la premiere sera utilisee pour stocker l'identifiant unique de chaque thread.
        //!< ...a partir de la, la valeur de la seconde clef importe peu,...
        //!< ...en supposant la seconde clef fixe, on obtient 2^32 generateurs differents.
        //!< ...cela represente par exemple une configuration de 1,024 x 4,194,304 et semble raisonnable
        //!< ...on utilisera la seconde clef (a priori) pour stocker celle desiree par l'utilisateur
        //!< ...de cette facon, en prenant deux fois la meme clef utilisateur on aura bien deux fois les memes sequences
        /**************************/
        //parametres minimalistes :
        /**************************/
        /*Compte tenu des precisions ci-dessus on peut se contenter de parametres minimalistes*/
        /*Eventuellement, le philox4x32 etant tres souple, on pourrait revoir les choses*/
        /*Le tout est de posseder suffisament de generateurs differents pour en attribuer un a chaque thread,
        tout en garantissant par generateur suffisament de valeurs pouvant etre crees,
        et si possible s'assurer de pouvoir simplement reproduire la meme sequence grace a la "graine" de l'utilisateur*/
        unsigned int *etat; /*compteur*/
        //!< ...la partie du compteur qui varie (a conserver entre deux appels au kernel principal)
        //!< ...si un jour l'appel a la generation d'un nombre devait se faire autant de fois pour chaque threads,...
        //!< ...i.e en dehors des methodes de la "moulinette", alors on pourrait ne conserver qu'un scalaire pour 'etat'
        //!< ...cela simplifierait egalement l'initialisation (pas de memset a faire, juste UNE valeur a fixer).
        unsigned int config; /*clef*/
        //!< ...la partie de la clef fixee par l'utilisateur
        #endif
	
}Tableaux;


#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
/* Init
* Paramètres initiaux du photon lors du premier impact avec l'atmosphère
* Les calculs sont effectués dans host.cu une seule fois
*/
typedef struct __align__(16){
	
	/* Coordonnées initiales */
	float x0;
	float y0;
	float z0;
	
	/* Paramètres liés au profil initial vu par le photon */
	float taumax0;		// Valeur maximale de l'épaisseur optique parcourue par le photon qui conduira à une 1ère intéraction
	float zintermax0;	// Distance entre le photon et une des extrémités de l'atmosphère dans le cas où il n'y as pas d'intéractoin

} Init;
#endif

/* Evnt
* DEBUG permet de recuperer des infos sur certains photons
*/
typedef struct __align__(16)
{
	int action;
	float tau;
	float poids;
}Evnt;


#endif	// COMMUNS_H
