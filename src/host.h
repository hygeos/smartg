
#ifndef HOST_H
#define HOST_H

/**********************************************************
*
*			host.h
*
*	> Initialisation du générateur de nombres aléatoires MWC
*	> Travail sur les fichiers
*	> Initialisation des différentes structures
*	> Calculs de profils
*	> Fonctions d'affichage
*	> Calcul pour sauvegarde des résultats finaux
*	> Fichier hdf (lecture/écriture témoin, écriture résultats)
*
***********************************************************/


/**********************************************************
*	> Initialisation du générateur de nombres aléatoires MWC
***********************************************************/

/* initRandMWC
* Fonction qui initialise les generateurs du random MWC à partir d'un fichier texte
*/
int initRandMWC(unsigned long long*, unsigned int*, const unsigned int, const char*, unsigned long long);


/* initRandMTConfig
* Fonction qui initialise en partie les generateurs du random Mersenen Twister
*/
void initRandMTConfig(ConfigMT*, ConfigMT*, int);


/**********************************************************
*	> Travail sur les fichiers
***********************************************************/

/* initConstantesHost
* Fonction qui récupère les valeurs des constantes dans le fichier paramètres et initialise les constantes du host
*/
void initConstantesHost(int, char**);


/* chercheConstante
* Fonction qui cherche nomConstante dans le fichier et met la valeur de la constante dans chaineValeur (en string)
*/
void chercheConstante(FILE* fichier, const char* nomConstante, char* chaineValeur);



void init_profileATM(int *NATM, float *HATM, int *NLAM, char *PATHPROFILATM);
void init_profileOCE(int *NOCE, int *NLAM, char *PATHPROFILOCE);
void get_diff(char *PATHDIFFAER, int ilam, char *PATHDIFF) ;

int count_lines(char *PATHDIFF);


/**********************************************************
*	> Initialisation des différentes structures
***********************************************************/

/* initVariables
* Fonction qui initialise les variables à envoyer dans le kernel.
*/
void initVariables(Variables**, Variables**);


/* reinitVariables
* Fonction qui réinitialise certaines variables avant chaque envoi dans le kernel
*/
void reinitVariables(Variables*, Variables*);

/** Séparation du code pour atmosphère sphérique ou parallèle **/
	/* Code spécifique à une atmosphère sphérique */
/* initInit
* Initialisation de la structure Init contenant les paramètres initiaux du photon rentrant dans l'atmosphère.
* Ces paramètres sont utiles pour une atmosphère sphérique et sont calculés une seule fois dans le host, d'où cette fonction
* et la structure Init
*/
void initInit(Init** init_H, Init** init_D);


/* initTableaux
* Fonction qui initialise les tableaux à envoyer dans le kernel par allocation mémoire et memset
*/
void initTableaux(Tableaux*, Tableaux*);


/* freeTableaux
* Fonction qui libère l'espace mémoire de tous les tableaux alloués
*/
void freeTableaux(Tableaux*, Tableaux*);


/**********************************************************
*	> Calculs de profils
***********************************************************/

/* calculFaer
* Compute CDF of scattering phase matrices
*/
void calculF( const char* nomFichier, float* phase_H, float* phase_D , int lsa, int nf, int ip);

/* Read spectral albedo (for surface or seafloor)*/
void profilAlb( Tableaux* tab_H, Tableaux* tab_D );

/* profilOce*/
/* Read ocean extinction coefficient and single scattering albedo for ocean*/
void profilOce( Tableaux* tab_H, Tableaux* tab_D );

/* profilAtm
* Calcul du profil atmosphérique dans l'atmosphère en fonction de la couche
* Mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
*/
void profilAtm( Tableaux* tab_H, Tableaux* tab_D );



/** Séparation du code pour atmosphère sphérique ou parallèle **/
	/* Code spécifique à une atmosphère sphérique */
/* impactInit
* Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
* Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init
*/
void impactInit(Init* init_H, Init* init_D, Tableaux* tab_H, Tableaux* tab_D);


/**********************************************************
*	> Fonctions d'affichage
***********************************************************/

/* afficheParametres
* Affiche les paramètres de la simulation
*/
void afficheParametres();


/* afficheProgress
* Affiche la progression de la simulation
*/
void afficheProgress(unsigned long long, Variables*, double
		#ifdef PROGRESSION
		, unsigned long long
		#endif
		);



/**********************************************************
*	> Calcul pour sauvegarde des résultats finaux
***********************************************************/

/* calculOmega
* Fonction qui calcule l'aire normalisée de chaque boite, son theta, et son psi, sous forme de 3 tableaux
*/
void calculOmega(double* tabTh, double* tabPhi, double* tabOmega);


/* calculTabFinal
* Fonction qui remplit le tabFinal correspondant à la reflectance (R), Q et U sur tous l'espace de sorti (dans chaque boite)
*/
void calculTabFinal(double*, double*, double*, double*, unsigned long long, unsigned long long*);


/**********************************************************
*	> Fichier hdf (ecriture résultats)
***********************************************************/

/* creerHDFResultats
* Fonction qui crée le fichier .hdf contenant le résultat final pour une demi-sphère
*/
//void creerHDFResultats(double*, double*, double*, double*, double*, double*, double*, unsigned long long, Variables*, double);
void creerHDFResultats(double*, double*, double*, double*, double*, double*, double*, double*, unsigned long long, Variables*, double);


#endif	// HOST_H

