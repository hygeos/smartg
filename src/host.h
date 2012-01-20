

	  /////////////////////
	 // PROTOTYPES HOST //
	/////////////////////

int initRandMWC(unsigned long long*, unsigned int*, const unsigned int, const char*, unsigned long long);
void initConstantesHost(int, char**);
void chercheConstante(FILE*, char*, char*);

void initVariables(Variables**, Variables**);
void initTableaux(Tableaux*, Tableaux*);

void initRandMTConfig(ConfigMT*, ConfigMT*, int);
void initEvnt(Evnt*, Evnt*);
void reinitVariables(Variables*, Variables*);

void calculTabFinal(float*, float*, float*, unsigned long long*, unsigned long long);
void calculOmega(float*, float*, float*);
void afficheParametres();
void afficheProgress(unsigned long long, Variables*, double
		#ifdef PROGRESSION
		, unsigned long long
		#endif
		    );
void afficheTrajet(Evnt*);

void creerHDFTemoin(unsigned long long*, unsigned long long, Variables*, double);
void lireHDFTemoin(Variables*, Variables*, unsigned long long*, unsigned long long*, double*);
void creerHDFResultats(float*, float*, float*, unsigned long long, Variables*, double);

void freeTableaux(Tableaux*, Tableaux*);


// Calcul des paramètres du modèle de diffusion des aérosols
void calculFaer( const char* nomFichier, Tableaux* tab_H, Tableaux* tab_D );

// Permet de vérifier que le modèle FAER généré est correct
void verificationFAER( const char* nomFichier, Tableaux tab_H );

// Calcul du mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
void profilAtm( Tableaux* tab_H, Tableaux* tab_D );
void verificationAtm( Tableaux tab_H );