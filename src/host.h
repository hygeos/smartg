
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
void chercheConstante(FILE* fichier, char* nomConstante, char* chaineValeur);


/* definirNomFichier
* Le nom du fichier de sorti est créé automatiquement en fonction du type de simulation
* Il est également stoké dans un dossier en fonction de la date est du type de simulation
* Le chemin indiqué dans le fichier paramètres est le préfixe du chemin créé ici
*/
void definirNomFichier( char* s );


/* definirSimulation
* Défini le type de simulation pour la création du chemin et le nom du fichier résultat
*/
void definirSimulation( char* s);


/* verifierFichier
* Fonction qui vérifie l'état des fichiers temoin et résultats
* Demande à l'utilisateur s'il veut les supprimer ou non
*/
void verifierFichier();


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
#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
/* initInit
* Initialisation de la structure Init contenant les paramètres initiaux du photon rentrant dans l'atmosphère.
* Ces paramètres sont utiles pour une atmosphère sphérique et sont calculés une seule fois dans le host, d'où cette fonction
* et la structure Init
*/
void initInit(Init** init_H, Init** init_D);
#endif

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
* Calcul de la fonction de phase des aérosols
*/
void calculFaer( const char* nomFichier, Tableaux* tab_H, Tableaux* tab_D );


/* verificationFAER
* Sauvegarde la fonction de phase des aérosols calculée dans un fichier
* Permet de valider le bon calcul de la fonction de phase
*/
void verificationFAER( const char* nomFichier, Tableaux tab_H );


/* calculFoce
* Calcul de la fonction de phase dans l'océan
*/
// void calculFoce( const char* nomFichier, Tableaux* tab_H, Tableaux* tab_D );


/* profilAtm
* Calcul du profil atmosphérique dans l'atmosphère en fonction de la couche
* Mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
*/
void profilAtm( Tableaux* tab_H, Tableaux* tab_D );


/* verificationAtm
* Sauvegarde du profil atmosphérique dans un fichier
* Permet de valider le bon calcul
*/
void verificationAtm( Tableaux tab_H );


/** Séparation du code pour atmosphère sphérique ou parallèle **/
#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
/* impactInit
* Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
* Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init
*/
void impactInit(Init* init_H, Init* init_D, Tableaux* tab_H, Tableaux* tab_D);
#endif

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


#ifdef TRAJET
/* initEvnt
* Initialisation des variables à envoyer dans le kernel pour récupérer le trajet d'un photon
*/
void initEvnt(Evnt*, Evnt*);


/* afficheTrajet
* Fonction qui affiche le début du trajet du premier thread
*/
void afficheTrajet(Evnt*);
#endif


/**********************************************************
*	> Calcul pour sauvegarde des résultats finaux
***********************************************************/

/* calculOmega
* Fonction qui calcule l'aire normalisée de chaque boite, son theta, et son psi, sous forme de 3 tableaux
*/
void calculOmega(float* tabTh, float* tabPhi, float* tabOmega);


/* calculTabFinal
* Fonction qui remplit le tabFinal correspondant à la reflectance (R), Q et U sur tous l'espace de sorti (dans chaque boite)
*/
void calculTabFinal(float*, float*, float*, float*, unsigned long long);


/**********************************************************
*	> Fichier hdf (lecture/écriture témoin, écriture résultats)
***********************************************************/

/* creerHDFTemoin
* Fonction qui crée un fichier .hdf contenant les informations nécessaires à la reprise du programme
* //TODO: 	écrire moins régulièrement le témoin (non pas une écriture par appel de kernel)
*			changer le format (écrire un .bin par exemple) pour éventuellement gagner du temps (calculer le gain éventuel)
*/
void creerHDFTemoin(float*, unsigned long long, Variables*, double);


/* lireHDFTemoin
* Si un fichier temoin existe et que les paramètres correspondent à la simulation en cours, cette simulation se poursuit à
* partir de celle sauvée dans le fichier témoin.
*/
void lireHDFTemoin(Variables*, Variables*, unsigned long long*, float*, double*);


/* creerHDFResultats
* Fonction qui crée le fichier .hdf contenant le résultat final pour une demi-sphère
*/
void creerHDFResultats(float*, float*, float*, unsigned long long, Variables*, double);


#endif	// HOST_H

