
	  /////////////////////////////
	 // CONSTANTES FICHIER HOST //
	/////////////////////////////

// Creation des constantes du host (parametres de la 
unsigned long long NBPHOTONS; //Nombre de photons a lancer
unsigned int NBLOOP; //Nombre de boucles dans le kernel
float THETASOL; //Angle zénithal solaire
float LAMBDA; //Longueur d'onde [nm]
float TAURAY; //Epaisseur optique moleculaire (Rayleigh)
float TAUAER; //Epaisseur optique aerosol
float W0AER; //Albedo simple de diffusion des aerosols
float HA; //Echelle de hauteur aerosol [km] (utilise si PROFIL=0)
float HR; //Echelle de hauteur moleculaire [km]
float ZMIN; //Altitude basse de la couche aerosol [km] (utilise si PROFIL=3)
float ZMAX; //Altitude haute de la couche aerosol [km] (utilise si PROFIL=3)
float WINDSPEED; //Vitesse du vent [m/s] (utilisee pour le modele de Cox et Munk: si DIOPTRE=1)
float NH2O; //Indice de refraction relatif air/eau
float CONPHY; //Concentration en phytoplancton [mg.m-3] (utilise si SUR>=2)
int XBLOCK; //Longueur du bloc de threads
int YBLOCK; //Hauteur du block de threads
int XGRID; //Longueur de la grid de blocks
int YGRID; //Hauteur de la grid de blocks
int NBTHETA; //Nombre de cases du tableau final pour theta (0..PI/2)
int NBPHI; //Nombre de cases du tableau final pour phi (0..2*PI)
int NBSTOKES; //Nombre de nombres de Stokes pris en compte
int DIOPTRE; //Type de dioptre (0=plan, 1=agite)
int DIFFF; //Forcage de la premiere diffusion (0=non, 1=oui) (utilise si SIM=-2)
int PROFIL; //Type de profil atmospherique
int SIM; //Type de simulation
int SUR; //Type de reflexion de la surface
