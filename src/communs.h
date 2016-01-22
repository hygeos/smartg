
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

#define WEIGHTRR 0.001F
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F


/* Mathématiques */
#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F

#define MAX_LOOP 100000



/* Localisation du photon */
#define SPACE       0
#define ATMOS       1
#define SURF0P      2   // surface (air side)
#define SURF0M      3   // surface (water side)
#define ABSORBED    4
#define NONE        5
#define OCEAN       6
#define SEAFLOOR    7

// indexing of the output levels
#define UPTOA   0
#define DOWN0P	1
#define DOWN0M	2
#define UP0P	3
#define UP0M	4


// List of errors
#define ERROR_THETA      0
#define ERROR_CASE       1
#define ERROR_VXY        2
#define ERROR_MAX_LOOP   3

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


struct Spectrum {
    float lambda;
    float alb_surface;
    float alb_seafloor;
};

struct Phase {
    float p_ang; /* \                          */
    float p_P11; /*  | equally spaced in       */
    float p_P22; /*  | scattering probability  */
    float p_P33; /*  | [0, 1]                  */
    float p_P43; /* /                          */

    float a_P11; /* \                          */
    float a_P22; /*  | equally spaced in scat. */
    float a_P33; /*  | angle [0, 180]          */
    float a_P43; /* /                          */

};

struct Profile {
    float z;      // altitude
    float tau;    // cumulated optical thickness (from top)
    float pmol;   // probability of pure Rayleigh scattering event
    float ssa;    // single scattering albedo (scatterer only)
    float abs;    // absorption coefficient
    int iphase;   // phase function index
};


#endif	// COMMUNS_H
