
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

/**********************************************************
*	> Constantes
***********************************************************/

#define WEIGHTINIT 1.F

// THRESHOLD for RUSSIAN ROULETTE PROCEDURE
#define WEIGHTRR 0.1F
// THRESHOLD for SMALL ANGLE VALUE
#define VALMIN 0.000001F


/* Mathématiques */
#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F

#define MAX_LOOP 100000000
#define MAX_NEVT 100 // Maximum number of scattering evts stored in photon in the ALIS procedure
#define MAX_NLOW 101 // MAX Number of wavelengths stored in the ALIS scattering correction



/* Possible Localisation photon */
#define SPACE       0
#define ATMOS       1
#define SURF0P      2   // surface (air side)
#define SURF0M      3   // surface (water side)
#define ABSORBED    4
#define NONE        5
#define OCEAN       6
#define SEAFLOOR    7

// test
#define REMOVED     8

// indexing of the output levels
#define UPTOA   0
#define DOWN0P	1
#define DOWN0M	2
#define UP0P	3
#define UP0M	4
#define DOWNB	5
#define UP0M2	6


// List of errors
#define ERROR_THETA      0
#define ERROR_CASE       1
#define ERROR_VXY        2
#define ERROR_MAX_LOOP   3

// bitmasks for output
#define OUTPUT_BOA_DOWN_0P_UP_0M   1 // downward radiance at BOA above surface (0+) and upward radiance at BOA below surface (0-)
#define OUTPUT_BOA_DOWN_0M_UP_0P   2 // downward radiance at BOA below surface (0-) and upward radiance at BOA above surface (0+)


// pseudo-random number generator
#ifdef PHILOX
    #include "philox.h"

    struct RNG_State {
        philox4x32_key_t configThr;
        philox4x32_ctr_t etatThr;
    };

    #define RAND randomPhilox4x32_7float(&rngstate->etatThr, &rngstate->configThr)
#endif
#ifdef CURAND_PHILOX
    #include <curand.h>
    #include <curand_kernel.h>

    struct RNG_State {
        curandStatePhilox4_32_10_t state;
    };

    #define RAND curand_uniform(&rngstate->state)

#endif



/**********************************************************
*	> Définition des structures
***********************************************************/
#include "helper_math.h"

/* Photon
*/

class Photon
{
public:
    // Normalized direction vector
    float3 v;
    // Normalized vector orthogonal to the direction vector 
    float3 u;
	
    // Localization of the photon
    int loc;
    
    // photon's weight
    float weight;

    // wavelength
    float wavel; // used for inelastic scatering
    int ilam; // wavelength index in case of multispectral computation
    
    // Angular box indices (for LE)
    int iph;
    int ith;
	
    // Stokes parameters
    float4 stokes;

    int layer;
    // float prop_aer;		// Aerosol proportion within the photon current layer
	
    float tau;	// vertical coordinate in optical depth (extinction or scattering depending on BEER keyword)
                // atmosphere : positive values
                // ocean: negative values
                //
    float tau_abs; // vertical coordinate in absorption optical depth

    // Cartesian coordinates of the photon
    float3 pos;

    // Number of interaction (scattering or reflection/transmission)
    unsigned short nint;

    #ifdef ALIS
    unsigned short nevt;  // Number  of events (including exit)
    short layer_prev[MAX_NEVT]; // History of layer where events occured
    float vz_prev[MAX_NEVT]; // History of z cosine where events occured
    float epsilon_prev[MAX_NEVT]; // History of proportion (between 0 and 1) within the layer where events occured
    float weight_sca[MAX_NLOW]; // Table of scattering weigths for Importance Sampling correction
    float tau_sca[MAX_NLOW]; // Table of verical scattering OD of the photon for Importance Sampling correction
    #endif

    #ifdef SPHERIQUE
    float radius;
    float taumax;
    #endif

    #ifdef BACK
    float4x4 M;
    //float4x4 Mf;
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
    float OD;    // cumulated extinction optical thickness (from top)
    float OD_sca; // cumulated scattering optical thickness (from top)
    float OD_abs; // cumulated absorption optical thickness (from top)
    float pmol;   // probability of pure Rayleigh scattering event
    float ssa;    // single scattering albedo of the layer
    float pine;   // Fraction of inelastic scattering of the layer
    float FQY1;   // Fluorescence like Quantum Yield of 1st specie of the layer
    int iphase;   // phase function index
};

#endif	// COMMUNS_H
