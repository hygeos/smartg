
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

// THRESHOLD for SMALL ANGLE VALUE
#define VALMIN 0.000001F

/* Mathématiques */
#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F

#define MAX_LOOP 1000000000
#define MAX_NEVT 500 // Max number of scattering evts stored in photon in the ALIS procedure in basic plane parallel mode
#define MAX_NLOW 4001 // Max number of wavelengths stored in the ALIS scattering correction
#define MAX_NLAYER 200 // Max number of vertical layers recorded in ALIS procedure in spherical or alternate PP mode
#define MAX_HIST 1024*1024 // Max number of photon's histories



/* Possible Localisation photon */
#define SPACE       0
#define ATMOS       1
#define SURF0P      2   // surface (air side)
#define SURF0M      3   // surface (water side)
#define ABSORBED    4
#define NONE        5
#define OCEAN       6
#define SEAFLOOR    7
#define OBJSURF     8


/* Possible Scatterers */
#define UNDEF      -1
#define RAY         0
#define PTCLE       1
#define CHLFLUO     2


/* Possible Simulations */
#define ATM_ONLY      -2
#define SURF_ONLY     -1
#define OCEAN_SURF     0
#define SURF_ATM       1
#define OCEAN_SURF_ATM 2
#define OCEAN_ONLY     3

// test
#define REMOVED     9

// indexing of the output levels
#define UPTOA   0
#define DOWN0P	1
#define DOWN0M	2
#define UP0P	3
#define UP0M	4
#define DOWNB	5
#define UP0M2	6

#ifdef OPT3D
    #define BOUNDARY_TOA   -1
    #define BOUNDARY_0P    -2
    #define BOUNDARY_0M    -3
    #define BOUNDARY_FLOOR -4
    #define BOUNDARY_ABS   -5
#endif

// List of errors
#define ERROR_THETA      0
#define ERROR_CASE       1
#define ERROR_VXY        2
#define ERROR_MAX_LOOP   3

// bitmasks for output
#define OUTPUT_BOA_DOWN_0P_UP_0M   1 // downward radiance at BOA above surface (0+) and upward radiance at BOA below surface (0-)
#define OUTPUT_BOA_DOWN_0M_UP_0P   2 // downward radiance at BOA below surface (0-) and upward radiance at BOA above surface (0+)

#ifdef OBJ3D
// Rotation order
#define XYZ  1
#define XZY  2
#define YXZ  3
#define YZX  4
#define ZXY  5
#define ZYX  6

#define HELIOSTAT 1
#define RECEIVER  2 
#endif

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
	// Initialisation(s)
	__host__ __device__ Photon()
	{
		// Initialement le photon n'est nulle part, il doit être initialisé
		loc = NONE;
		#ifdef OBJ3D
		direct = 0;
		H = 0;
		E = 0;
		S = 0;
		#endif
	}
	
    // Normalized direction vector
    float3 v;
    // Normalized vector orthogonal to the direction vector 
    float3 u;
	
    // Localization of the photon
    int loc;
    int locPrev;
		
    // photon's weight
    float weight;

    // wavelength
    float wavel; // used for inelastic scatering
    int ilam; // wavelength index in case of multispectral computation
    
    // Angular box indices (for LE)
    int iph;
    int ith;

    // Sensor index
    int is;
	
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

	// scatterer
	short int scatterer;

    // Number of interaction (scattering or reflection/transmission)
    unsigned short nint;

    #ifdef ALIS
    #if !defined(ALT_PP) && !defined(SPHERIQUE)
    unsigned short nevt;  // Number  of events (including exit)
    short layer_prev[MAX_NEVT]; // History of layer where events occured
    float vz_prev[MAX_NEVT]; // History of z cosine where events occured
    float epsilon_prev[MAX_NEVT]; // History of proportion (between 0 and 1) within the layer where events occured
    float tau_sca[MAX_NLOW]; // Table of verical scattering OD of the photon for Importance Sampling correction
    #else
    float cdist_atm[MAX_NLAYER]; // Table of cumulative distance per layer
    float cdist_oc[MAX_NLAYER];
    #endif
    float weight_sca[MAX_NLOW]; // Table of scattering weigths for Importance Sampling correction
    #endif

    #ifdef SPHERIQUE
    float radius;
    #endif

    #ifdef BACK
    float4x4 M;
    //float4x4 Mf;
    #endif
	#ifdef OBJ3D
	int direct;
    int H, E, S;
	float weight_loss[4];
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

#ifndef OPT3D
struct Profile {
    float z;      // altitude
    float n;      // refractive index
    float OD;    // cumulated extinction optical thickness (from top)
    float OD_sca; // cumulated scattering optical thickness (from top)
    float OD_abs; // cumulated absorption optical thickness (from top)
    float pmol;   // probability of pure Rayleigh scattering event
    float ssa;    // single scattering albedo of the layer
    float pine;   // Fraction of inelastic scattering of the layer
    float FQY1;   // Fluorescence like Quantum Yield of 1st specie of the layer
    int iphase;   // phase function index
};
#else
#include <geometry.h>
struct Profile {
    float z;      // altitude
    int   i;      // Box index
    float pminx;  // Box, pmin point
    float pminy;  // Box, pmin point
    float pminz;  // Box, pmin point
    float pmaxx;  // Box, pmax point
    float pmaxy;  // Box, pmax point
    float pmaxz;  // Box, pmax point
    float n;      // refractive index
    float OD;     // extinction coefficient
    float OD_sca; // scattering coefficient
    float OD_abs; // absorption coefficient
    float pmol;   // probability of pure Rayleigh scattering event
    float ssa;    // single scattering albedo of the box
    float pine;   // Fraction of inelastic scattering of the box
    float FQY1;   // Fluorescence like Quantum Yield of 1st specie of the box
    int iphase;   // phase function index
    int neighbour1; 
    int neighbour2; 
    int neighbour3; 
    int neighbour4; 
    int neighbour5; 
    int neighbour6; 
                    // Neighbour boxes indices in order:
                    // Positive X, Negative X, Positive Y, Negative Y, Positive Z, Negative Z
};
#endif

struct Sensor {
    float POSX;   // X position of the sensor
    float POSY;   // Y position of the sensor
    float POSZ;   // Z position of the sensor (fromp Earth's center in spherical, from the ground in PP)
    float THDEG;  // zenith angle of viewing direction (Zenith> 90 for downward looking, <90 for upward, default Zenith)
    float PHDEG;  // azimut angle of viewing direction
    int LOC;      // localization (ATMOS=1, ...), see constant definitions in communs.h
    float FOV;    // sensor FOV (degree) 
    int TYPE;     // sensor type: Radiance (0), Planar flux (1), Spherical Flux (2), default 0
    int IBOX;     // box index in which the sensor is (3D)
};

#ifdef OBJ3D
// En rapport avec l'implementation des objets
#include "transform.h" // La structure IGeo a une classe transform comme attrib

struct IGeo
{	
    __device__ IGeo()
	{
		normal = make_float3(0., 0., 0.);
		material = -1;
		reflectivity = -1.;
		roughness = -1.;
		type = -1.;
		mvR = make_float3(0, 0, 0);
		normalBase = make_float3(0., 0., 0.);
	}

	__device__ IGeo(float3 nn, int mat, float ref, float rough,
					int typ, float3 mvRt, float3 nB)
	{
		normal = nn;
		material = mat;
		reflectivity = ref;
		roughness = rough;
		type = typ;
		mvR = mvRt;
		normalBase = nB;
	}
	
	float3 normal;      /* normal at the surface impact   */
	float3 normalBase;  /* front(AV) surface normal       */
	int material;       /* material of the object         */
	float reflectivity; /* albedo of the object           */
	float roughness;    /* roughness of the object        */
    Transform mvTF;     /* transformation of the object   */
	int type;           /* 1 = reflector, 2 = receiver    */
	float3 mvR;         /* rotation angles in x, y, z     */
};

struct IObjets {
    int geo;        /* 1 = sphere, 2 = plane, ...          */
	int materialAV; /* 1 = LambMirror, 2 = Matte,          */
	int materialAR;	/* 3 = Mirror, ...                     */
	int type;       /* 1 = reflector, 2 = receiver         */
	float reflectAV;/* reflectivity of the materialAV      */
	float reflectAR;/* reflectivity of the materialAR      */
	float roughAV;  /* roughness of the materialAV         */
	float roughAR;  /* roughness of the materialAR         */
	
	float p0x;      /* \             \                     */
	float p0y;      /*  | point p0    \                    */
	float p0z;      /* /               \                   */
	                /*                  |                  */
	float p1x;      /*  \               |                  */
	float p1y;      /*   | point p1     |                  */
	float p1z;      /*  /               |                  */
	                /*                  | Plane Object     */
	float p2x;      /*  \               |                  */
	float p2y;      /*   | point p2     |                  */
	float p2z;      /*  /               |                  */
	                /*                  |                  */
	float p3x;      /*  \              /                   */
	float p3y;      /*   | point p3   /                    */
	float p3z;      /*  /            /                     */
	
	float myRad;    /*  \                                  */
	float z0;       /*   | Spherical Object                */
	float z1;       /*   |                                 */
	float phi;      /*  /                                  */

	float mvRx;     /*  \                                  */
	float mvRy;     /*   | Transformation type rotation    */
	float mvRz;     /*  /                                  */
	int   rotOrder; /* rotation order: 1=XYZ; 2=XZY;...    */

	float mvTx;     /*  \                                  */
	float mvTy;     /*   | Transformation type translation */
	float mvTz;     /*  /                                  */

	float nBx;      /*  \                                  */
	float nBy;      /*   | normalBase apres transfo        */
	float nBz;      /*  /                                  */
};
#endif //END OBJ3D
#endif	// COMMUNS_H
