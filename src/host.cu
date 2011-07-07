
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "host.h"

	  ////////////////////
	 // FONCTIONS HOST //
	////////////////////

// Fonction d'initialisation de la fonction random
int initRNG(unsigned long long *x, unsigned int *a, 
	     const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit)
{
	FILE *fp;
	unsigned int begin=0u;
	unsigned int fora,tmp1,tmp2;

	if (strlen(safeprimes_file) == 0)
	{
        // Try to find it in the local directory
		safeprimes_file = "safeprimes_base32.txt";
	}
	
	fp = fopen(safeprimes_file, "r");

	if(fp == NULL)
	{
		printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
		return 1;
	}

	fscanf(fp,"%u %u %u",&begin,&tmp1,&tmp2);

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if((xinit == 0ull) | (((unsigned int)(xinit>>32))>=(begin-1)) | (((unsigned int)xinit)>=0xfffffffful))
	{
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n",xinit);
		return 1;
	}

	for(unsigned int i=0;i < n_rng;i++)
	{
		fscanf(fp,"%u %u %u",&fora,&tmp1,&tmp2);
		a[i]=fora;
		x[i]=0;
		while( (x[i]==0) | (((unsigned int)(x[i]>>32))>=(fora-1)) | (((unsigned int)x[i])>=0xfffffffful))
		{
			//generate a random number
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i]=(unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			x[i]=x[i]<<32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			x[i]+=(unsigned int) xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
	}
	fclose(fp);

	return 0;
}

void initConstantes(Constantes* constantes_H, Constantes* constantes_D)
{
	constantes_H->thS = THETASOL/180*PI; //thetaSolaire_Host
	constantes_H->cThS = cosf(constantes_H->thS); //cosThetaSolaire_Host
	constantes_H->sThS = sinf(constantes_H->thS); //sinThetaSolaire_Host
	constantes_H->tauMax = TAU/constantes_H->cThS; //tau initial du photon (Host)
	cudaMemcpy(constantes_D, constantes_H, sizeof(Constantes), cudaMemcpyHostToDevice);
}

void initRandom(Random* random_H, Random* random_D)
{
	unsigned long long seed = (unsigned long long) time(NULL); //Default, use time(NULL) as seed
	unsigned long long x[XBLOCK * YBLOCK * XGRID * YGRID]; //x
	unsigned int a[XBLOCK * YBLOCK * XGRID * YGRID]; //a
	initRNG(x, a, XBLOCK * YBLOCK * XGRID * YGRID, "safeprimes_base32.txt", seed);
	for(int i = 0; i < XBLOCK * YBLOCK * XGRID * YGRID; i++)
	{
		random_H[i].a = a[i];
		random_H[i].x = x[i];
	}
	cudaMemcpy(random_D, random_H, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(Random), cudaMemcpyHostToDevice);
}

void initEvnt(Evnt* evnt_H, Evnt* evnt_D)
{
	for(int i = 0; i < 20; i++)
	{
		evnt_H[i].action = 0;
	}
	cudaMemcpy(evnt_D, evnt_H, 20 * sizeof(Evnt), cudaMemcpyHostToDevice);
}

void initProgress(Progress* progress_H, Progress* progress_D)
{
	#ifdef PROGRESSION
	progress_H->nbThreads = 0;
	#endif

	progress_H->erreurpoids = 0;
	progress_H->erreurtheta = 0;
	progress_H->erreurvxy = 0;
	progress_H->erreurvy = 0;
	progress_H->erreurcase = 0;

	cudaMemcpy(progress_D, progress_H, sizeof(Progress), cudaMemcpyHostToDevice);
}

void reinitProgress(Progress* progress_H, Progress* progress_D)
{
	// Le nombre de photons traités pour un appel du Kernel est remis à zéro
	progress_H->nbPhotons = 0;
	
	#ifdef PROGRESSION
	// Le nombre de photons ressortis pour un appel du Kernel est remis à zéro
	progress_H->nbPhotonsSor = 0;
	#endif
	
	cudaMemcpy(progress_D, progress_H, sizeof(Progress), cudaMemcpyHostToDevice);
}

// Fonction qui affiche les paramètres de la simulation
void afficheParametres()
{
	printf("\n");
	printf("WEIGHTINIT = %f",WEIGHTINIT);
	printf("\n");
	printf("WEIGHTMAX = %f",WEIGHTMAX);
	printf("\n");
	printf("THETASOL = %f (degrés)",THETASOL);
	printf("\n");
	printf("TAU = %f",TAU);
	printf("\n");
	printf("DEPO = %f",DEPO);
	printf("\n");
	printf("PI = %f",PI);
	printf("\n");
	printf("SCALEFACTOR = %d",SCALEFACTOR);
	printf("\n");
	printf("VALMIN = %f", VALMIN);
	printf("\n");
	printf("NBPHOTONS = %d",NBPHOTONS);
	printf("\n");
	printf("NBSTOKES = %d",NBSTOKES);
	printf("\n");
	printf("NBLOOP = %d",NBLOOP);
	printf("\n");
	printf("XBLOCK = %d",XBLOCK);
	printf("\n");
	printf("YBLOCK = %d",YBLOCK);
	printf("\n");
	printf("XGRID = %d",XGRID);
	printf("\n");
	printf("YGRID = %d",YGRID);
	printf("\n");
	printf("NBTHETA = %d",NBTHETA);
	printf("\n");
	printf("NBPHI = %d",NBPHI);
	printf("\n");
}

#ifdef PROGRESSION
// Fonction qui affiche la progression de la simulation
void afficheProgress(unsigned long long nbPhotonsTot,unsigned long long nbPhotonsSorTot, Progress* progress)
{
	printf("%d%% - threads: %lu - phot sortis: %lu - phot traités: %lu - erreur poids/theta/vxy/vy/case: %d/%d/%d/%d/%d", (int)(100*nbPhotonsTot/NBPHOTONS), progress->nbThreads, nbPhotonsSorTot, nbPhotonsTot, progress->erreurpoids, progress->erreurtheta, progress->erreurvxy, progress->erreurvy, progress->erreurcase);
	printf("\n");
}
#endif

// Fonction qui affiche le trajet du premier thread
void afficheTrajet(Evnt* evnt_H)
{
	printf("\nTrajet d'un thread :\n");
	for(int i = 0; i < 20; i++)
	{
		if(evnt_H[i].action == 1)
			printf("init : ");
		else if(evnt_H[i].action == 2)
			printf("move : ");
		else if(evnt_H[i].action == 3)
			printf("scat : ");
		else if(evnt_H[i].action == 4)
			printf("surf : ");
		#ifdef CONTROLE
		else if(evnt_H[i].action != 5)
			printf("\nERREUR : host afficheTrajet\n");
		#endif
		else printf("exit : ");
		printf("tau=%f ", evnt_H[i].tau);
		printf("poids=%f", evnt_H[i].poids);
		printf("\n");
	}
}

// Fonction qui affiche les tableaux "finaux" pour chaque nombre de Stokes
void afficheTabStokes(unsigned long long* tabPhotonsTot)
{
	printf("\nTableau Stokes1 :\n");
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			printf("%lu ", tabPhotonsTot[0 * NBPHI * NBTHETA + ith * NBPHI + iphi]);
		}
		printf("\n");
	}
	printf("\nTableau Stokes2 :\n");
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			printf("%lu ", tabPhotonsTot[1 * NBPHI * NBTHETA + ith * NBPHI + iphi]);
		}
		printf("\n");
	}
}

// Fonction qui affiche le tableau regroupant le nombre de photons ressortis sur une demi-sphère
void afficheTabNbPhotons(unsigned long long* tabNbPhotonsTot)
{
	printf("\nTableau nombre de photons :\n");
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			printf("%lu ", tabNbPhotonsTot[ith * NBPHI + iphi]);
		}
		printf("\n");
	}
}

#ifdef TABFINAL
// Fonction qui affiche le tableau final
void afficheTabFinal(float* tabFinal)
{
	// Affichage du tableau final
	printf("\nTableau Final :\n");
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			printf("%f ", tabFinal[ith*NBPHI+iphi]);
		}
		printf("\n");
	}
}
#endif

// Fonction host qui calcule pour chaque morceau de sphère son aire normalisée, son angle theta, et son angle psi, sous forme de 3 tableaux
void calculOmega(float* tabTh, float* tabPhi, float* tabOmega)
{
	// Tableau contenant l'angle theta de chaque morceau de sphère
	memset(tabTh, 0, NBTHETA * sizeof(float));
	float dth = PIdiv2 / NBTHETA;
	tabTh[0] = dth / 2;
	for(int ith = 1; ith < NBTHETA; ith++) tabTh[ith] = tabTh[ith-1] + dth;
	// Tableau contenant l'angle psi de chaque morceau de sphère
	memset(tabPhi, 0, NBPHI * sizeof(float));
	float dphi = PImul2 / NBPHI;
	tabPhi[0] = dphi / 2;
	for(int iphi = 1; iphi < NBPHI; iphi++) tabPhi[iphi] = tabPhi[iphi-1] + dphi;
	// Tableau contenant l'aire de chaque morceau de sphère
	float sumds = 0;
	float tabds[NBTHETA * NBPHI];
	memset(tabds, 0, NBTHETA * NBPHI * sizeof(float));
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			tabds[ith * NBPHI + iphi] = sin(tabTh[ith]) * dth * dphi;
			sumds += tabds[ith * NBPHI + iphi];
		}
	}
	// Normalisation de l'aire de chaque morceau de sphère
	memset(tabOmega, 0, NBTHETA * NBPHI * sizeof(float));
	for(int ith = 0; ith < NBTHETA; ith++)
		for(int iphi = 0; iphi < NBPHI; iphi++)
			tabOmega[ith * NBPHI + iphi] = tabds[ith * NBPHI + iphi] / sumds;
}

// Fonction qui remplit le tableau final ainsi que tabTh et tabPhi avec les valeurs de theta et psi pour chaque morceau de spère
void calculTabFinal(float* tabFinal, float* tabTh, float* tabPhi, unsigned long long* tabPhotonsTot, Progress* progress, unsigned long long nbPhotonsTot)
{
	float tabOmega[NBTHETA * NBPHI]; //tableau contenant l'aire de chaque morceau de sphère
	// Remplissage des tableaux tabTh, tabPhi, et tabOmega
	calculOmega(tabTh, tabPhi, tabOmega);
	// Remplissage du tableau final
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			tabFinal[ith*NBPHI+iphi] = (tabPhotonsTot[0*NBPHI*NBTHETA+ith*NBPHI+iphi] + tabPhotonsTot[1*NBPHI*NBTHETA+ith*NBPHI+iphi]) / (2 * nbPhotonsTot * tabOmega[ith*NBPHI+iphi] * SCALEFACTOR * cosf(tabTh[ith]));
		}
	}
}

// Fonction qui crée le fichier .hdf contenant le réultat final pour une demi-sphère
void creerHDFResultats(float* tabFinal, float* tabTh, float* tabPhi)
{
	// Création du fichier de sortie
	char nomFichier[20] = "Resultats.hdf";
	int sdFichier = SDstart(nomFichier, DFACC_CREATE);
	// Pour chaque phi on ajoute au fichier le tableau représentant le résultat final en fonction de theta
	for(int iphi = 0; iphi < NBPHI; iphi++)
	{
		// Création et remplissage d'un tableau du fichier
		// Les tab[*][0] sont les résultats finaux et les tab[*][1] sont les valeurs de theta
		float tab[NBTHETA][2] = {0};
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			tab[ith][0] = tabFinal[ith*NBPHI+iphi];
			tab[ith][1] = tabTh[ith];
		}
		char nomTab[20]; //nom du tableau
		sprintf(nomTab,"Resultats (iphi = %d)", iphi);
		int nbDimsTab = 2; //nombre de dimensions du tableau
		int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
		valDimsTab[0] = NBTHETA;
		valDimsTab[1] = 2;
		int typeTab = DFNT_FLOAT32; //type des éléments du tableau
		// Création du tableau
		int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
		int startTab[nbDimsTab]; //début de la lecture du tableau
		startTab[0]=0;
		startTab[1]=0;
		// Ecriture du tableau dans le fichier
		int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tab);
		// Vérification du bon fonctionnement de l'écriture
		#ifdef CONTROLE
		if(status) printf("\nERREUR : write hdf resultats\n");
		#endif
		// Ecriture d'informations sur le tableau
		char description[20];
		sprintf(description, "phi = %f", tabPhi[iphi]);
		if(strcmp(description, "") != 0)
			SDsetattr(sdsTab, "Tableau pour ", DFNT_CHAR8, strlen(description), description);
		// Fermeture du tableau
		SDendaccess(sdsTab);
	}
	// Fermeture du fichier
	SDend(sdFichier);
}

void creerHDFResultatsQuartsphere(float* tabFinal, float* tabTh, float* tabPhi)
{
	// Création du fichier de sortie
	char nomFichier[20] = "Quart.hdf";
	int sdFichier = SDstart(nomFichier, DFACC_CREATE);
	// Pour chaque phi de la demi-sphère on ajoute au fichier le tableau représentant la moyenne du résultat final des 2 demi-sphères en fonction de theta
	for(int iphi = 0; iphi < NBPHI/2; iphi++)
	{
		// Création et remplissage d'un tableau du fichier
		// Les tab[*][0] sont les résultats finaux et les tab[*][1] sont les valeurs de theta
		float tab[NBTHETA][2] = {0};
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			int jphi = NBPHI - 1 - iphi;
			tab[ith][0] = (tabFinal[ith*NBPHI+iphi]+tabFinal[ith*NBPHI+jphi])/2;
			tab[ith][1] = tabTh[ith];
		}
		char nomTab[20]; //nom du tableau
		sprintf(nomTab,"Quart (iphi = %d)", iphi);
		int nbDimsTab = 2; //nombre de dimensions du tableau
		int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
		valDimsTab[0] = NBTHETA;
		valDimsTab[1] = 2;
		int typeTab = DFNT_FLOAT32; //type des éléments du tableau
		// Création du tableau
		int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
		int startTab[nbDimsTab]; //début de la lecture du tableau
		startTab[0]=0;
		startTab[1]=0;
		// Ecriture du tableau dans le fichier
		int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tab);
		// Vérification du bon fonctionnement de l'écriture
		#ifdef CONTROLE
		if(status) printf("\nERREUR : write hdf quart\n");
		#endif
		// Ecriture d'informations sur le tableau
		char description[20];
		sprintf(description, "phi = %f", tabPhi[iphi]);
		if(strcmp(description, "") != 0)
			SDsetattr(sdsTab, "Tableau pour ", DFNT_CHAR8, strlen(description), description);
		// Fermeture du tableau
		SDendaccess(sdsTab);
	}
	// Fermeture du fichier
	SDend(sdFichier);
}

void creerHDFComparaison(float* tabFinal, float* tabTh, float* tabPhi)
{
	// Création du fichier de sortie
	char nomFichier[20] = "Comparaison.hdf";
	int sdFichier = SDstart(nomFichier, DFACC_CREATE);
	// Création du tableau à insérer dans le fichier
	float tab[NBTHETA][3] = {0};
	// Pour chaque phi de la demi-sphère on ajoute au fichier le tableau représentant le résultat final du quart de sphère gauche en fonction de theta puis de celui de droite en fonction de theta
	for(int iphi = 0; iphi < NBPHI/2; iphi++)
	{
		// Remplissage du tableau
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			int jphi = NBPHI - 1 - iphi;
			tab[ith][0] = tabTh[ith];
			tab[ith][1] = tabFinal[ith*NBPHI+iphi];
			tab[ith][2] = tabFinal[ith*NBPHI+jphi];
		}
		
		char nomTab[20]; //nom du tableau
		sprintf(nomTab,"Comparaison (iphi = %d)", iphi);
		int nbDimsTab = 2; //nombre de dimensions du tableau
		int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
		valDimsTab[0] = NBTHETA;
		valDimsTab[1] = 3;
		int typeTab = DFNT_FLOAT32; //type des éléments du tableau
		// Création du tableau
		int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
		// Départ de la lecture du tableau
		int startTab[nbDimsTab];
		startTab[0]=0;
		startTab[1]=0;
		// Ecriture du tableau dans le fichier
		int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tab);
		// Vérification du bon fonctionnement de l'écriture
		#ifdef CONTROLE
		if(status) printf("\nERREUR : write hdf comparaison\n");
		#endif
		// Ecriture d'informations sur le tableau
		char description[20];
		sprintf(description, "phi = %f", tabPhi[iphi]);
		if (strcmp(description, "") != 0) {
			SDsetattr(sdsTab, "Tableau pour ", DFNT_CHAR8, strlen(description), description);
		}
		// Fermeture du tableau
		SDendaccess(sdsTab);
	}
	// Fermeture du fichier
	SDend(sdFichier);
}
