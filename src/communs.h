
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

#define WEIGHTRR 0.1F
// Détecte les photons très proches du zenith
#define VALMIN 0.000001F


/* Mathématiques */
#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F

#define MAX_LOOP 100000
#define NEVTMAX 50 // Maximum number of scattering evts stored in photon
#define N_LOW 41



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
#define UP0M2	5


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

class Photon
{
public:
    // Vecteur normalisé de la direction du photon (vitesse)
    float3 v;
    // Vecteur normalisé orthogonal à la vitesse du photon (polarisation)
    float3 u;
	
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
    float4 stokes;

    // Paramètres pour une atmosphère sphérique
    int couche;
    float prop_aer;		// Proportion d'aérosols par rapport aux molécules à l'endroit où se situe le photon
	

    float tau;	// localisation en epaisseur optique
                // atmosphère: valeurs positives
                // océan: valeurs négatives
                //
    float tauabs; //localisation epaisseur optique d'absorption

    // Position cartésienne du photon
    float3 pos;

    #ifdef ALIS
    unsigned short nevt;  // Number  of events
    short layer_prev[NEVTMAX]; // History of layer where events occured
    float vz_prev[NEVTMAX]; // History of z cosine where events occured
    float delta_prev[NEVTMAX]; // History of proportion (between 0 and 1) within the layer where events occured
    float weight_sca[N_LOW]; // Table of scattering weigths for Importance Sampling correction
    float dtau_sca[N_LOW]; // Table of differential scattering OD of the photon for Importance Sampling correction
    #endif

    #ifdef SPHERIQUE

    float rayon;
    float taumax;
	
    #endif

private:
};


struct Spectrum {
    float lambda;
    float alb_surface;
    float alb_seafloor;
};

struct Phase {
    float p_ang; /* \                          */
    float p_P11; /*  |                         */
	float p_P12; /*  | equally spaced in       */
    float p_P22; /*  | scattering probability  */
    float p_P33; /*  | [0, 1]                  */
    float p_P43; /*  |                         */
	float p_P44; /* /                          */

    float a_P11; /* \                          */
    float a_P12; /*  |                         */
    float a_P22; /*  | equally spaced in scat. */
    float a_P33; /*  | angle [0, 180]          */
    float a_P43; /*  |                         */
    float a_P44; /* /                          */

};

struct Profile {
    float z;      // altitude
    float tau;    // cumulated extinction optical thickness (from top)
    float tausca; // cumulated scattering optical thickness (from top)
    float tauabs; // cumulated absorption optical thickness (from top)
    float pmol;   // probability of pure Rayleigh scattering event
    float ssa;    // single scattering albedo (scatterer only)
    float abs;    // absorption coefficient
    int iphase;   // phase function index
};

#endif	// COMMUNS_H
