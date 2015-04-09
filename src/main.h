
/**********************************************************
*
*			main.h
*
*	> Définition des variables externes
*
***********************************************************/

// Creation des constantes du host (parametres de la 
unsigned long long NBPHOTONS; //Nombre de photons a lancer
unsigned int NBLOOP; //Nombre de boucles dans le kernel
int SEED; 		//Graine pour les fonctions random, si SEED=-1 on choisit une graine aléatoirement
int XBLOCK; 	//Longueur du bloc de threads
int YBLOCK; 	//Hauteur du block de threads
int XGRID; 		//Longueur de la grid de blocks
int YGRID; 		//Hauteur de la grid de blocks
int DEVICE;     //numéro du device utilisé
int NBTHETA; 	//Nombre de cases du tableau final pour theta (0..PI/2)
int NBPHI; 		//Nombre de cases du tableau final pour phi (0..2*PI)
int DIOPTRE; 	//Type de dioptre (0=plan, 1=agite)
int SIM; 		//Type de simulation
int SUR; 		//Type de reflexion de la surface
int ENV;        //Effet d'environnement(=0 pas d effet)

unsigned int LSAAER;	// Nombre d'échantillons pour les angles du modèle de diffusion aérosols
unsigned int NFAER;
unsigned int LSAOCE;	// Nombre d'échantillons pour les angles du modèle de diffusion aérosols
unsigned int NFOCE;

float THVDEG;	//Angle zénithal de visée
float LAMBDA;	//Longueur d'onde [nm]
float TAUATM;
#ifdef FLAGOCEAN
float W0OCE;	//Albedo simple de diffusion dans l'ocean
float DEPO;     //Facteur de depolarisation de l'eau
#endif
float W0LAM;	//Albedo simple de diffusion de la surface lambertienne
int NATM;		//Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
float HATM;		//Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
float WINDSPEED;	//Vitesse du vent [m/s] (utilisee pour le modele de Cox et Munk: si DIOPTRE=1)
float NH2O;		//Indice de refraction relatif air/eau
float ENV_SIZE; // Taille de la cible en km
float X0; // Position de la cible en x km
float Y0; // Position de la cible en y km
float TRANSDIR; // Direct transmission Sun-surface
#ifdef FLAGOCEAN
float ATOT, BTOT;
char PATHDIFFOCE[1024];
#endif

unsigned int OUTPUT_LAYERS;

char PATHRESULTATSHDF[1024]; //Fichier de sortie au format hdf
char PATHTEMOINHDF[1024]; //Fichier témoin au format hdf
char PATHDIFFAER[1024];	// Fichier d'entrée des données de diffusion des aérosols
char PATHPROFILATM[1024]; // Profil atmosphérique utilisateur

int WRITE_PERIOD;

