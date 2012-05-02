
	  /////////////////////////////
	 // CONSTANTES FICHIER HOST //
	/////////////////////////////

// Creation des constantes du host (parametres de la 
unsigned long long NBPHOTONS; //Nombre de photons a lancer
unsigned int NBLOOP; //Nombre de boucles dans le kernel
int SEED; //Graine pour les fonctions random, si SEED=-1 on choisit une graine aléatoirement
int XBLOCK; //Longueur du bloc de threads
int YBLOCK; //Hauteur du block de threads
int XGRID; //Longueur de la grid de blocks
int YGRID; //Hauteur de la grid de blocks
int NBTHETA; //Nombre de cases du tableau final pour theta (0..PI/2)
int NBPHI; //Nombre de cases du tableau final pour phi (0..2*PI)
int DIOPTRE; //Type de dioptre (0=plan, 1=agite)
int DIFFF; //Forcage de la premiere diffusion (0=non, 1=oui) (utilise si SIM=-2)
int PROFIL; //Type de profil atmospherique
int SIM; //Type de simulation
int SUR; //Type de reflexion de la surface

unsigned int LSAAER;	// Nombre d'échantillons pour les angles du modèle de diffusion aérosols
unsigned int NFAER;

float THSDEG; //Angle zénithal solaire
float LAMBDA; //Longueur d'onde [nm]
float TAURAY; //Epaisseur optique moleculaire (Rayleigh)
float TAUAER; //Epaisseur optique aerosol
float W0AER; //Albedo simple de diffusion des aerosols
float W0LAM; //Albedo simple de diffusion de la surface lambertienne
float HA; //Echelle de hauteur aerosol [km] (utilise si PROFIL=0)
float HR; //Echelle de hauteur moleculaire [km]
float ZMIN; //Altitude basse de la couche aerosol [km] (utilise si PROFIL=3)
float ZMAX; //Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
int NATM; //Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
int HATM; //Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
float WINDSPEED; //Vitesse du vent [m/s] (utilisee pour le modele de Cox et Munk: si DIOPTRE=1)
float NH2O; //Indice de refraction relatif air/eau
float CONPHY; //Concentration en phytoplancton [mg.m-3] (utilise si SUR>=2)

char PATHRESULTATSHDF[256]; //Fichier de sortie au format hdf
char PATHTEMOINHDF[256]; //Fichier témoin au format hdf
char PATHDIFFAER[256];	// Fichier d'entrée des données de diffusion des aérosols
char PATHPROFILATM[256]; // Profil atmosphérique utilisateur


