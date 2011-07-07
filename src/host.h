
	  /////////////////////
	 // PROTOTYPES HOST //
	/////////////////////

int initRNG(unsigned long long, unsigned int, const unsigned int, const char, unsigned long long);
void initConstantes(Constantes*, Constantes*);
void freeConstantes(Constantes*);
void initRandom(Random*, Random*);
void initEvnt(Evnt*, Evnt*);
void initProgress(Progress*, Progress*);
void reinitProgress(Progress*, Progress*);
void afficheParametres();
void afficheProgress(unsigned long long, unsigned long long, Progress*);
void afficheTrajet(Evnt*);
void afficheTabStokes(unsigned long long*);
void afficheTabFinal(float*);
void afficheTabNbPhotons(unsigned long long*);
void calculOmega(float*, float*, float*);
void calculTabFinal(float*, float*, float*, unsigned long long*, Progress*, unsigned long long);
void creerHDFResultats(float*, float*, float*);
void creerHDFResultatsQuartsphere(float*, float*, float*);
void creerHDFComparaison(float*, float*, float*);
