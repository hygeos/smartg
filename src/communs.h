
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



#include <stdio.h>
#include "philox.h"

/**********************************************************
*	> Constantes
***********************************************************/

/* Lié au photon */
// Poids initial du photon
#define WEIGHTINIT 1.F

#define WEIGHTRR 1.F
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F


/* Mathématiques */
#define PI 3.1415927F
//3.141 592 653 589F
#define DEUXPI 6.2831853F
//6.283 185 307 17F
#define DEMIPI 1.5707963F
//1.570 796 326 79F



/* Localisation du photon */
#define SPACE       0
#define ATMOS       1
#define SURF0P      2   // surface (air side)
#define SURF0M      3   // surface (water side)
#define ABSORBED    4
#define NONE        5
#define OCEAN       6
#define SEAFLOOR    7

// Number of output levels
#define NLVL  5

// indexing of the output levels
#define UPTOA   0
#define DOWN0P	1
#define DOWN0M	2
#define UP0P	3
#define UP0M	4


// bitmasks for output
#define OUTPUT_BOA_DOWN_0P_UP_0M   1 // downward radiance at BOA above surface (0+) and upward radiance at BOA below surface (0-)
#define OUTPUT_BOA_DOWN_0M_UP_0P   2 // downward radiance at BOA below surface (0-) and upward radiance at BOA above surface (0+)


/* Test des differentes fonctions random */
#define RAND randomPhilox4x32_7float(etatThr, configThr)



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

	// longueur d onde du photon
	float wavel; // for Raman
    int ilam; // wavelength index

    // Angular box indices (for LE)
    int iph;
    int ith;
	
	// Paramètres de stokes du photon
	float stokes1;
	float stokes2;
	float stokes3;
	float stokes4;

	// Paramètres pour une atmosphère sphérique
	int couche;
	float prop_aer;		// Proportion d'aérosols par rapport aux molécules à l'endroit où se situe le photon
	

	float tau;	// localisation en epaisseur optique
                // atmosphère: valeurs positives
                // océan: valeurs négatives

	// Position cartésienne du photon
	float x;
	float y;
	float z;

	#ifdef SPHERIQUE

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
    int nThreadsActive;    // Number of active threads
	int erreurpoids;				// Nombre de photons ayant un poids anormalement élevé
	int erreurtheta;				// Nombre de photons ignorés (sortant dans la direction de visée)
	
	#ifdef PROGRESSION
	unsigned long long nbThreads;	// Nombre total de threads lancés
	unsigned long long nbPhotonsSor;// Nombre de photons ressortis pour un appel du Kernel
	int erreurvxy;					// Nombre de photons sortant au zénith et donc difficiles à classer
	int erreurvy;					// Nombre de photons sortant à phi=0 ou phi=PI et donc difficiles à classer
	int erreurcase;					// Nombre de photons rangé dans une case inexistante
	#endif
	
}Variables;



/* Tableaux
* Ensemble des tableaux envoyés par le host dans le device
* tabPhotons est également modifié par le kernel pour sauver les paramètres de stokes du photon sorti dans l'espace
*/
typedef struct __align__(16)
{

	unsigned long long* nbPhotonsInter;		// Tableau contenant le nb de photons injecte par interval NLAM 
	


    #ifdef DOUBLE
	double* tabPhotons;		//Tableau contenant l'ensemble des paramètres de stokes de tous les photons (évènements confondus)
    #else
	float* tabPhotons;		//Tableau contenant l'ensemble des paramètres de stokes de tous les photons (évènements confondus)
    #endif


	float* faer;			// Pointeur vers le modèle de diffusion des aérosols
	
	float* foce;			// Pointeur vers le modèle de diffusion dans l'océan
	float* ho;				// Pointeur vers l'épaisseur optique de chaque couches du modele oceanique
    float* sso;             // Pointeur vers le profil de l albedo de diffusion simple dans l'ocean
    int* ipo;               // Pointer to the vertical profile of ocean phase function index

	
	float* h;				// Pointeur vers l'épaisseur optique de chaque couches du modèle atmosphérique
	float* pMol;			// Pointeur vers la proportion de molécules dans chaque couches du modèle atmosphérique
	float* ssa;			    // Pointeur vers l'albedo de diffusion simple des aerosols dans chaque couches du modèle atmosphérique
    float* abs;             // Pointeur vers la proportion d'absorbant dans chaque couches du modèle atmosphérique
    int* ip;                // pointer to the vertical profile of atmospheric phase function index
    float* alb;             // Pointeur vers l albedo de la surface lambertienne

    float* lambda;

    float* z;

    float* tabthv;
    float* tabphi;          // Pointer to zenith and azimut angles for output in local estimate

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
	
} Tableaux;


/* Evnt
* DEBUG permet de recuperer des infos sur certains photons
*/
typedef struct __align__(16)
{
	int action;
	float tau;
	float poids;
} Evnt;


#endif	// COMMUNS_H
