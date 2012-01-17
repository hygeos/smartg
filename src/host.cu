
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "host.h"
#include "device.h"

	  ////////////////////
	 // FONCTIONS HOST //
	////////////////////

// Fonction qui initialise les generateurs du random MWC
int initRandMWC(unsigned long long *etat, unsigned int *config, 
	     const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit)
{
	FILE *fp;
	unsigned int begin=0u;
	unsigned int fora,tmp1,tmp2;
	if (strlen(safeprimes_file) == 0)
	{
        	// Try to find it in the local directory
		safeprimes_file = "MWC.txt";
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
		config[i]=fora;
		etat[i]=0;
		while( (etat[i]==0) | (((unsigned int)(etat[i]>>32))>=(fora-1)) | (((unsigned int)etat[i])>=0xfffffffful))
		{
			//generate a random number
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);
			//calculate c and store in the upper 32 bits of x[i]
			etat[i]=(unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			etat[i]=etat[i]<<32;
			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			etat[i]+=(unsigned int) xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
	}
	fclose(fp);
	return 0;
}

// Fonction qui récupère les valeurs des constantes dans Parametres.txt et initialise les constantes du host
void initConstantesHost(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("ERREUR : lecture argv");
		exit(1);
	}
	
	FILE* parametres = fopen( argv[1], "r" );
	
	if( parametres == NULL ){
		printf("ERREUR: Impossible d'ouvrir le fichier %s\n", argv[1] );
		exit(1);
	}
	
	char* s = (char*)malloc(100 * sizeof(char));

	strcpy(s,"");
	chercheConstante( parametres, "NBPHOTONS", s);
	NBPHOTONS = strtoull(s, NULL, 10);
	
	strcpy(s,"");
	chercheConstante(parametres, "NBLOOP", s);
	NBLOOP = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "SEED", s);
	SEED = atoi(s);
	if(SEED == -1) SEED = static_cast<int> (time(NULL));

	strcpy(s,"");
	chercheConstante(parametres, "XBLOCK", s);
	XBLOCK= atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "YBLOCK", s);
	YBLOCK = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "XGRID", s);
	XGRID = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "YGRID", s);
	YGRID = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "NBTHETA", s);
	NBTHETA = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "NBPHI", s);
	NBPHI = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "NBSTOKES", s);
	NBSTOKES = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "PROFIL", s);
	PROFIL = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "SIM", s);
	SIM = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "SUR", s);
	SUR = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "DIOPTRE", s);
	DIOPTRE= atoi(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "DIFFF", s);
	DIFFF = atoi(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "THSDEG", s);
	THSDEG = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "LAMBDA", s);
	LAMBDA = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "TAURAY", s);
	TAURAY = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "TAUAER", s);
	TAUAER = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "W0AER", s);
	W0AER = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "LSAAER", s);
	LSAAER = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "NFAER", s);
	NFAER = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "HA", s);
	HA = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "HR", s);
	HR = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "ZMIN", s);
	ZMIN = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "ZMAX", s);
	ZMAX = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "WINDSPEED", s);
	WINDSPEED = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "NH2O", s);
	NH2O = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "CONPHY", s);
	CONPHY = atof(s);
	
	chercheConstante(parametres, "PATHRESULTATSHDF", PATHRESULTATSHDF);
	
	chercheConstante(parametres, "PATHTEMOINHDF", PATHTEMOINHDF);
	
	chercheConstante( parametres, "PATHDIFFAER", PATHDIFFAER );

	chercheConstante( parametres, "PATHPROFILATM", PATHPROFILATM );
	
	free(s);
	fclose( parametres );
}

// Fonction qui cherche nomConstante dans le fichier et met la valeur de la constante dans chaineValeur (en string)
void chercheConstante(FILE* fichier, char* nomConstante, char* chaineValeur)
{
		int longueur = strlen(nomConstante);
		char ligne[100];
		int motTrouve = 0;
		
		// Tant que la constante n'est pas trouvee et qu'on n'est pas à la fin du fichier on lit la ligne
		while(fgets(ligne, 100, fichier) && !motTrouve)
		{
			// Si le debut de la ligne est nomConstante suivi d'un espace ou un egal on va chercher la valeur
			if((strncmp(ligne, nomConstante, longueur) == 0) && (ligne[longueur] == ' ' || ligne[longueur] == '='))
			{
				char* ptr = ligne; //pointeur du debut de la ligne
				// on avance jusqu'au prochain espace ou egal
				while (*ptr != ' ' && *ptr != '=') ptr++;
				// on avance jusqu'à la valeur de la constante
				while(*ptr == ' ' || *ptr == '=') ptr++;
				if (*ptr == '\n')
				{
					printf("ERREUR : lecture Parametre.txt\n");
					exit(1);
				}
				// On met la chaine de la valeur de la constante dans chaineValeur
				strcpy(chaineValeur, ptr);
				chaineValeur[strlen(chaineValeur)-1] = '\0';
				motTrouve = 1;
			}
		}
		rewind(fichier);
		
		if(motTrouve == 0)
		{
			printf("ERREUR : lecture Parametres.txt\n");
			exit(1);
		}

}

// Fonction qui initialise les variables à envoyer dans le kernel
void initVariables(Variables** var_H, Variables** var_D)
{
	// Initialisation de la version host des variables
	*var_H = (Variables*)malloc(sizeof(Variables));
	memset(*var_H, 0, sizeof(Variables));
	// Initialisation de la version device des variables
	cudaMalloc(var_D, sizeof(Variables));
	cudaMemset(*(var_D), 0, sizeof(Variables));
}

// Fonction qui initialise les tableaux à envoyer dans le kernel
void initTableaux(Tableaux* tab_H, Tableaux* tab_D)
{
#ifdef RANDMWC
	// Création des tableaux de generateurs pour la fonction Random MWC
	tab_H->etat = (unsigned long long*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long));
	cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long));
	tab_H->config = (unsigned int*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int));
	cudaMalloc(&(tab_D->config), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int));
	// Initialisation des tableaux host à l'aide du fichier et du seed
	initRandMWC(tab_H->etat, tab_H->config, XBLOCK * YBLOCK * XGRID * YGRID, "MWC.txt", (unsigned long long)SEED);
	// Copie dans les tableaux device
	cudaMemcpy(tab_D->etat, tab_H->etat, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(tab_D->config, tab_H->config, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int), cudaMemcpyHostToDevice);
#endif
#ifdef RANDCUDA
	// Création du tableau de generateurs (=etat+config) pour la fonction Random Cuda
	
	cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(curandState_t));
	// Initialisation du tableau dans une fonction du kernel
	initRandCUDA<<<XGRID * YGRID, XBLOCK * YBLOCK>>>(tab_D->etat, (unsigned long long)SEED);
#endif
#ifdef RANDMT
	// Création des tableaux de generateurs pour la fonction Random Mersenne Twister
	cudaMalloc(&(tab_D->config), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(ConfigMT));
	cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(EtatMT));
	tab_H->config = (ConfigMT*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(ConfigMT));
	tab_H->etat = (EtatMT*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(EtatMT));
	// Initialisation du tableau des configs à l'aide du fichier
	initRandMTConfig(tab_H->config, tab_D->config, XBLOCK * YBLOCK * XGRID * YGRID);
	// Initialisation du tableau des etats dans le kernel
	initRandMTEtat<<<XGRID * YGRID, XBLOCK * YBLOCK>>>(tab_D->etat, tab_D->config);
#endif
	
	// Tableau du poids des photons ressortis
	tab_H->tabPhotons = (unsigned long long*)malloc(NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
	cudaMalloc(&(tab_D->tabPhotons), NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
	
	// Modèle de diffusion des aérosols
	tab_H->faer = (float*) malloc(5*NFAER*sizeof(float));
	cudaMalloc( &(tab_D->faer), 5*NFAER*sizeof(float) );
	
	// Modèle de l'atmosphère
	tab_H->tauCouche =  (float*)malloc((NATM+1)*sizeof(float));
	cudaMalloc( &(tab_D->tauCouche), (NATM+1)*sizeof(float) );
	tab_H->pMol =  (float*)malloc((NATM+1)*sizeof(float));
	cudaMalloc( &(tab_D->pMol), (NATM+1)*sizeof(float) );
}

// Fonction qui initialise en partie les generateurs du random Mersenen Twister
void initRandMTConfig(ConfigMT* config_H, ConfigMT* config_D, int nbThreads)
{
	// Ouverture du fichier
	const char *fname = "MersenneTwister.dat";
	FILE* fd = fopen(fname, "rb");
	if(!fd)
	{
		printf("ERREUR: ouverture fichier MT");
		exit(0);
	}
	// Lecture et initialisation de la config pour chaque generateur (= pour chaque thread)
	for(int i = 0; i < nbThreads; i++)
	{
		// Le fichier ne contient que 4096 configs, on reutilise donc les memes configs pour les threads en trop mais les nombres aléatoires restent independants car les etats des threads sont differents
		if(i%4096 == 0)
		{
			fseek(fd, 0, 0);
		}
		if(!fread(config_H+i, sizeof(ConfigMT), 1, fd))
		{
			printf("ERREUR: lecture fichier MT");
			exit(0);
		}
	}
	fclose(fd);
	srand((unsigned int)SEED);
	// Creation des seeds aleatoires pour que les threads aient des etats differents
	for(int i = 0; i < nbThreads; i++) config_H[i].seed = (unsigned int)rand();
	cudaMemcpy(config_D, config_H, nbThreads * sizeof(ConfigMT), cudaMemcpyHostToDevice);
}

// DEBUG : Initialisation des variables à envoyer dans le kernel pour récupérer le trajet d'un photon
void initEvnt(Evnt* evnt_H, Evnt* evnt_D)
{
	for(int i = 0; i < 20; i++) evnt_H[i].action = 0;
	cudaMemcpy(evnt_D, evnt_H, 20 * sizeof(Evnt), cudaMemcpyHostToDevice);
}

// Fonction qui réinitialise certaines variables avant chaque envoi dans le kernel
void reinitVariables(Variables* var_H, Variables* var_D)
{
	// Le nombre de photons traités pour un appel du Kernel est remis à zéro
	var_H->nbPhotons = 0;
	#ifdef PROGRESSION
	// Le nombre de photons ressortis pour un appel du Kernel est remis à zéro
	var_H->nbPhotonsSor = 0;
	#endif
	// On copie le nouveau var_H dans var_D
	cudaMemcpy(var_D, var_H, sizeof(Variables), cudaMemcpyHostToDevice);
}

// Fonction qui calcule pour chaque l'aire normalisée de chaque boite, son theta, et son psi, sous forme de 3 tableaux
void calculOmega(float* tabTh, float* tabPhi, float* tabOmega)
{
	// Tableau contenant l'angle theta de chaque morceau de sphère
	memset(tabTh, 0, NBTHETA * sizeof(float));
	float dth = DEMIPI / NBTHETA;
	tabTh[0] = dth / 2;
	for(int ith = 1; ith < NBTHETA; ith++) tabTh[ith] = tabTh[ith-1] + dth;
	// Tableau contenant l'angle psi de chaque morceau de sphère
	memset(tabPhi, 0, NBPHI * sizeof(float));
	float dphi = DEUXPI / NBPHI;
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

// Fonction qui remplit le tabFinal, tabTh et tabPhi
void calculTabFinal(float* tabFinal, float* tabTh, float* tabPhi, unsigned long long* tabPhotonsTot, unsigned long long nbPhotonsTot)
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

// Fonction qui affiche les paramètres de la simulation
void afficheParametres()
{
	printf("\n");
	printf("NBPHOTONS = %lu", NBPHOTONS);
	printf("\n");
	printf("NBLOOP = %u", NBLOOP);
	printf("\n");
	printf("SEED = %d", SEED);
	printf("\n");
	printf("XBLOCK = %d", XBLOCK);
	printf("\n");
	printf("YBLOCK = %d", YBLOCK);
	printf("\n");
	printf("XGRID = %d", XGRID);
	printf("\n");
	printf("YGRID = %d", YGRID);
	printf("\n");
	printf("NBTHETA = %d", NBTHETA);
	printf("\n");
	printf("NBPHI = %d", NBPHI);
	printf("\n");
	printf("NBSTOKES = %d", NBSTOKES);
	printf("\n");
	printf("THSDEG = %f (degrés)", THSDEG);
	printf("\n");
	printf("LAMBDA = %f", LAMBDA);
	printf("\n");
	printf("TAURAY = %f", TAURAY);
	printf("\n");
	printf("TAUAER = %f", TAUAER);
	printf("\n");
	printf("W0AER = %f", W0AER);
	printf("\n");
	printf("LSAAER = %u", LSAAER);
	printf("\n");
	printf("NFAER = %u", NFAER);
	printf("\n");
	printf("PROFIL = %d", PROFIL);
	printf("\n");
	printf("HA = %f", HA);
	printf("\n");
	printf("HR = %f", HR);
	printf("\n");
	printf("ZMIN = %f", ZMIN);
	printf("\n");
	printf("ZMAX = %f", ZMAX);
	printf("\n");
	printf("WINDSPEED = %f", WINDSPEED);
	printf("\n");
	printf("NH2O = %f", NH2O);
	printf("\n");
	printf("SIM = %d", SIM);
	printf("\n");
	printf("SUR = %d", SUR);
	printf("\n");
	printf("DIOPTRE = %d", DIOPTRE);
	printf("\n");
	printf("CONPHY = %f", CONPHY);
	printf("\n");
	printf("DIFFF = %d", DIFFF);
	printf("\n");
	printf("PATHRESULTATSHDF = %s", PATHRESULTATSHDF);
	printf("\n");
	printf("PATHTEMOINHDF = %s", PATHTEMOINHDF);
	printf("\n");
	printf("PATHDIFFAER = %s", PATHDIFFAER);
	printf("\n");
	printf("PATHPROFILATM = %s", PATHPROFILATM);
	printf("\n");
}

// Fonction qui affiche la progression de la simulation
void afficheProgress(unsigned long long nbPhotonsTot, Variables* var, double tempsPrec
		#ifdef PROGRESSION
		, unsigned long long nbPhotonsSorTot
		#endif
		    )
{
	// Calcul la date et l'heure courante
	time_t dateTime = time(NULL);
	struct tm* date = localtime(&dateTime);
	// Calcul du temps ecoule et restant
	double tempsProg = (double)(clock() / CLOCKS_PER_SEC);
	double tempsTot = tempsProg + tempsPrec;
	int tempsEcoule = (int)tempsTot;
	int hEcoulees = tempsEcoule / 3600;
	int minEcoulees = (tempsEcoule%3600) / 60;
	int secEcoulees = tempsEcoule%60;
	int tempsRestant = (int)(tempsTot * ((double)NBPHOTONS / (double)nbPhotonsTot - 1.));
	if(tempsRestant < 0) tempsRestant = 0;
	int hRestantes = tempsRestant / 3600;
	int minRestantes = (tempsRestant%3600) / 60;
	int secRestantes = tempsRestant%60;
	// Calcul du pourcentage de photons traités
	int pourcent = (int)(100 * nbPhotonsTot / NBPHOTONS);
	
	// Affichage
	printf("\n");
	printf(" --------------------------------------\n");
	printf("  Photons lances : %12lu (%3d%%)\n", nbPhotonsTot, pourcent);
	printf("  Photons pb     : %12d\n", var->erreurpoids + var->erreurtheta);
	printf("  Temps ecoule   : %d h %2d min %2d sec\n", hEcoulees, minEcoulees, secEcoulees);
	printf("  Temps restant  : %d h %2d min %2d sec\n", hRestantes, minRestantes, secRestantes);
	printf("  Date actuelle  : %02u/%02u/%04u %02u:%02u:%02u\n", date->tm_mday, date->tm_mon+1, 1900 + date->tm_year, date->tm_hour, date->tm_min, date->tm_sec);
	printf(" --------------------------------------\n");
	
	#ifdef PROGRESSION
	printf("%d%% - ", (int)(100*nbPhotonsTot/NBPHOTONS));
	printf("Temps: %d - ", tempsEcoule);
	printf("phot sortis: %lu - ", nbPhotonsSorTot);
	printf("phot traités: %lu - ", nbPhotonsTot);
	printf("erreur poids/theta/vxy/vy/case: %d/%d/%d/%d/%d", var->erreurpoids, var->erreurtheta, var->erreurvxy, var->erreurvy, var->erreurcase);
	printf("\n");
	#endif
}

// Fonction qui affiche le début du trajet du premier thread
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
		else if(evnt_H[i].action != 5)
		{
			printf("\nERREUR : host afficheTrajet\n");
			exit(1);
		}
		else printf("exit : ");
		printf("tau=%f ", evnt_H[i].tau);
		printf("poids=%f", evnt_H[i].poids);
		printf("\n");
	}
}

// Fonction qui crée un fichier .hdf contenant les informations nécessaires à la reprise du programme
void creerHDFTemoin(unsigned long long* tabPhotonsTot, unsigned long long nbPhotonsTot, Variables* var, double tempsPrec)
{
	// Création du fichier de sortie
	int sdFichier = SDstart(PATHTEMOINHDF, DFACC_CREATE);
	// Création et remplissage du tableau du fichier (en double car le format hdf n'accepte pas int64)
	double* tab;
	tab = (double*)malloc(NBTHETA * NBPHI * NBSTOKES * sizeof(double));
	memset(tab, 0, NBTHETA * NBPHI * NBSTOKES * sizeof(double));
	// On remplit le tableau en convertissant de unsigned long long a double car le hdf n'accepte pas ull
	for(int i = 0; i < NBTHETA * NBPHI * NBSTOKES; i++)
		tab[i] = (double)tabPhotonsTot[i];
	char nomTab[20]; //nom du tableau
	sprintf(nomTab,"Temoin (%d%%)", (int)(100 * nbPhotonsTot / NBPHOTONS));
	int nbDimsTab = 1; //nombre de dimensions du tableau
	int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
	valDimsTab[0] = NBTHETA * NBPHI * NBSTOKES;
	int typeTab = DFNT_FLOAT64 ; //type des éléments du tableau
	// Création du tableau
	int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	int startTab[nbDimsTab]; //début de la lecture du tableau
	startTab[0]=0;
	// Ecriture du tableau dans le fichier
	int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tab);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf temoin\n");
		exit(1);
	}
	// Liberation du tableau
	free(tab);
	
	// Ecriture de toutes les informations sur la simulation : paramètres, nbphotons, nbErreurs, tempsEcoule
	double NBPHOTONSdouble = (double)NBPHOTONS; // on convertit en double car le hdf n'accepte pas ull
	double nbPhotonsTotdouble = (double)nbPhotonsTot; // on convertit en double car le hdf n'accepte pas ull
	double tempsEcouledouble = tempsPrec + (double)(clock() / CLOCKS_PER_SEC);
	
	SDsetattr(sdsTab, "NBPHOTONS", DFNT_FLOAT64, 1, &NBPHOTONSdouble);
	SDsetattr(sdsTab, "NBLOOP", DFNT_UINT32, 1, &NBLOOP);
	SDsetattr(sdsTab, "SEED", DFNT_UINT32, 1, &SEED);
	SDsetattr(sdsTab, "XBLOCK", DFNT_INT32, 1, &XBLOCK);
	SDsetattr(sdsTab, "YBLOCK", DFNT_INT32, 1, &YBLOCK);
	SDsetattr(sdsTab, "XGRID", DFNT_INT32, 1, &XGRID);
	SDsetattr(sdsTab, "YGRID", DFNT_INT32, 1, &YGRID);
	SDsetattr(sdsTab, "NBTHETA", DFNT_INT32, 1, &NBTHETA);
	SDsetattr(sdsTab, "NBPHI", DFNT_INT32, 1, &NBPHI);
	SDsetattr(sdsTab, "NBSTOKES", DFNT_INT32, 1, &NBSTOKES);
	SDsetattr(sdsTab, "DIOPTRE", DFNT_INT32, 1, &DIOPTRE);
	SDsetattr(sdsTab, "DIFFF", DFNT_INT32, 1, &DIFFF);
	SDsetattr(sdsTab, "PROFIL", DFNT_INT32, 1, &PROFIL);
	SDsetattr(sdsTab, "SIM", DFNT_INT32, 1, &SIM);
	SDsetattr(sdsTab, "SUR", DFNT_INT32, 1, &SUR);
	SDsetattr(sdsTab, "THSDEG", DFNT_FLOAT32, 1, &THSDEG);
	SDsetattr(sdsTab, "LAMBDA", DFNT_FLOAT32, 1, &LAMBDA);
	SDsetattr(sdsTab, "TAURAY", DFNT_FLOAT32, 1, &TAURAY);
	SDsetattr(sdsTab, "TAUAER", DFNT_FLOAT32, 1, &TAUAER);
	SDsetattr(sdsTab, "W0AER", DFNT_FLOAT32, 1, &W0AER);
	
	SDsetattr(sdsTab, "LSAAER", DFNT_UINT32, 1, &LSAAER);
	SDsetattr(sdsTab, "NFAER", DFNT_UINT32, 1, &NFAER);
	
	SDsetattr(sdsTab, "HA", DFNT_FLOAT32, 1, &HA);
	SDsetattr(sdsTab, "HR", DFNT_FLOAT32, 1, &HR);
	SDsetattr(sdsTab, "ZMIN", DFNT_FLOAT32, 1, &ZMIN);
	SDsetattr(sdsTab, "ZMAX", DFNT_FLOAT32, 1, &ZMAX);
	SDsetattr(sdsTab, "WINDSPEED", DFNT_FLOAT32, 1, &WINDSPEED);
	SDsetattr(sdsTab, "NH2O", DFNT_FLOAT32, 1, &NH2O);
	SDsetattr(sdsTab, "CONPHY", DFNT_FLOAT32, 1, &CONPHY);
	SDsetattr(sdsTab, "PATHRESULTATSHDF", DFNT_CHAR8, strlen(PATHRESULTATSHDF), PATHRESULTATSHDF);
	SDsetattr(sdsTab, "PATHTEMOINHDF", DFNT_CHAR8, strlen(PATHTEMOINHDF), PATHTEMOINHDF);
	SDsetattr(sdsTab, "PATHDIFFAER", DFNT_CHAR8, strlen(PATHDIFFAER), PATHDIFFAER);
	SDsetattr(sdsTab, "PATHPROFILATM", DFNT_CHAR8, strlen(PATHPROFILATM), PATHPROFILATM);
	
	SDsetattr(sdsTab, "nbPhotonsTot", DFNT_FLOAT64, 1, &nbPhotonsTotdouble);
	SDsetattr(sdsTab, "nbErreursPoids", DFNT_INT32, 1, &(var->erreurpoids));
	SDsetattr(sdsTab, "nbErreursTheta", DFNT_INT32, 1, &(var->erreurtheta));
	SDsetattr(sdsTab, "tempsEcoule", DFNT_FLOAT64, 1, &tempsEcouledouble);

	// Fermeture du tableau
	SDendaccess(sdsTab);
	// Fermeture du fichier
	SDend(sdFichier);
}

void lireHDFTemoin(Variables* var_H, Variables* var_D,
		unsigned long long* nbPhotonsTot, unsigned long long* tabPhotonsTot, double* tempsEcoule)
{
	// Ouverture du fichier temoin
	int sdFichier = SDstart(PATHTEMOINHDF, DFACC_READ);
	if(sdFichier != -1)
	{
		// Ouverture de l'unique tableau du fichier temoin
		int sdsIndex = 0;
		int sdsTab = SDselect (sdFichier, sdsIndex);
		
		// Recuperation de paramètres du fichier temoin
		int SEEDrecup[1];
		int NBTHETArecup[1];
		int NBPHIrecup[1];
		int NBSTOKESrecup[1];
		int DIOPTRErecup[1];
		int DIFFFrecup[1];
		int PROFILrecup[1];
		int SIMrecup[1];
		int SURrecup[1];
		float THSDEGrecup[1];
		float LAMBDArecup[1];
		float TAURAYrecup[1];
		float TAUAERrecup[1];
		float W0AERrecup[1];
		float HArecup[1];
		float HRrecup[1];
		float ZMINrecup[1];
		float ZMAXrecup[1];
		float WINDSPEEDrecup[1];
		float NH2Orecup[1];
		float CONPHYrecup[1];
		
		SDreadattr(sdsTab, SDfindattr(sdsTab, "SEED"), (VOIDP)SEEDrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NBTHETA"), (VOIDP)NBTHETArecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NBPHI"), (VOIDP)NBPHIrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NBSTOKES"), (VOIDP)NBSTOKESrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "DIOPTRE"), (VOIDP)DIOPTRErecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "DIFFF"), (VOIDP)DIFFFrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "PROFIL"), (VOIDP)PROFILrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "SIM"), (VOIDP)SIMrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "SUR"), (VOIDP)SURrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "THSDEG"), (VOIDP)THSDEGrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "LAMBDA"), (VOIDP)LAMBDArecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "TAURAY"), (VOIDP)TAURAYrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "TAUAER"), (VOIDP)TAUAERrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "W0AER"), (VOIDP)W0AERrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "HA"), (VOIDP)HArecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "HR"), (VOIDP)HRrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "ZMIN"), (VOIDP)ZMINrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "ZMAX"), (VOIDP)ZMAXrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "WINDSPEED"), (VOIDP)WINDSPEEDrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NH2O"), (VOIDP)NH2Orecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "CONPHY"), (VOIDP)CONPHYrecup);
		
		// Si les parametres sont les memes on recupere des informations pour poursuivre la simulation précédente
		if(NBTHETArecup[0] == NBTHETA
			&& NBPHIrecup[0] == NBPHI
			&& NBSTOKESrecup[0] == NBSTOKES
			&& DIOPTRErecup[0] == DIOPTRE
			&& DIFFFrecup[0] == DIFFF
			&& PROFILrecup[0] == PROFIL
			&& SIMrecup[0] == SIM
			&& SURrecup[0] == SUR
			&& THSDEGrecup[0] == THSDEG
			&& LAMBDArecup[0] == LAMBDA
			&& TAURAYrecup[0] == TAURAY
			&& TAUAERrecup[0] == TAUAER
			&& W0AERrecup[0] == W0AER
			&& HArecup[0] == HA
			&& HRrecup[0] == HR
			&& ZMINrecup[0] == ZMIN
			&& ZMAXrecup[0] == ZMAX
			&& WINDSPEEDrecup[0] == WINDSPEED
			&& NH2Orecup[0] == NH2O
			&& CONPHYrecup[0] == CONPHY)
		{
			printf("\nPOURSUITE D'UNE SIMULATION ANTERIEURE\n");
			if(SEEDrecup[0] == SEED) printf("ATTENTION: Nous recommandons SEED=-1 sinon les nombres aleatoires sont identiques a chaque lancement.\n");
			// Recuperation du nombre de photons traités et du nombre d'erreurs
			double nbPhotonsTotDouble[1]; //on récupère d'abord la variable en double
			unsigned long long nbPhotonsTotRecup[1]; //puis on la passera en unsigned long long
			int nbErreursPoidsRecup[1];
			int nbErreursThetaRecup[1];
			double tempsEcouleRecup[1];
	
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbPhotonsTot"), (VOIDP)nbPhotonsTotDouble);
			nbPhotonsTotRecup[0] = (unsigned long long)nbPhotonsTotDouble[0];
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbErreursPoids"), (VOIDP)nbErreursPoidsRecup);
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbErreursTheta"), (VOIDP)nbErreursThetaRecup);
			SDreadattr(sdsTab, SDfindattr(sdsTab, "tempsEcoule"), (VOIDP)tempsEcouleRecup);
	
			var_H->erreurpoids = nbErreursPoidsRecup[0];//nombre de photons ayant un poids anormalement élevé
			var_H->erreurtheta = nbErreursThetaRecup[0];//nombre de photons sortant dans la direction solaire
			cudaMemcpy(var_D, var_H, sizeof(Variables), cudaMemcpyHostToDevice);
			*(nbPhotonsTot) = nbPhotonsTotRecup[0];
			*(tempsEcoule) = tempsEcouleRecup[0];
	
			// Recuperation du tableau
			int nbDimsTab = 1; //nombre de dimensions du tableau
			int startTab[nbDimsTab], edgesTab[nbDimsTab]; //debut et fin de la lecture du tableau
			startTab[0] = 0;
			edgesTab[0] = NBTHETA * NBPHI * NBSTOKES;
			double tabPhotonsTotRecup[NBTHETA * NBPHI * NBSTOKES]; //tableau de récuperation en double
	
			int status = SDreaddata (sdsTab, startTab, NULL, edgesTab, (VOIDP)tabPhotonsTotRecup);
			// Vérification du bon fonctionnement de la lecture
			if(status)
			{
				printf("\nERREUR : read hdf temoin\n");
				exit(1);
			}
	
			// Conversion en unsigned long long
			for(int i = 0; i < NBTHETA * NBPHI * NBSTOKES; i++)
			{
				tabPhotonsTot[i] = (unsigned long long)tabPhotonsTotRecup[i]; //conversion en ull
			}
			
		}
		// Fermeture du tableau
		SDendaccess (sdsTab);
	}
	// Fermeture du fichier
	SDend (sdFichier);
}
	
// Fonction qui crée le fichier .hdf contenant le résultat final pour une demi-sphère
void creerHDFResultats(float* tabFinal, float* tabTh, float* tabPhi,
		unsigned long long nbPhotonsTot, Variables* var, double tempsPrec)
{
	// Création du fichier de sortie
	int sdFichier = SDstart(PATHRESULTATSHDF, DFACC_CREATE);
	// Ecriture des informations sur la simulation : paramètres, nbphotons, nbErreurs, tempsEcoule
	double NBPHOTONSdouble = (double)NBPHOTONS;
	double nbPhotonsTotdouble = (double)nbPhotonsTot;
	double tempsEcouledouble = tempsPrec + (double)(clock() / CLOCKS_PER_SEC);

	SDsetattr(sdFichier, "NBPHOTONS", DFNT_FLOAT64, 1, &NBPHOTONSdouble);
	SDsetattr(sdFichier, "NBLOOP", DFNT_UINT32, 1, &NBLOOP);
	SDsetattr(sdFichier, "SEED", DFNT_UINT32, 1, &SEED);
	SDsetattr(sdFichier, "XBLOCK", DFNT_INT32, 1, &XBLOCK);
	SDsetattr(sdFichier, "YBLOCK", DFNT_INT32, 1, &YBLOCK);
	SDsetattr(sdFichier, "XGRID", DFNT_INT32, 1, &XGRID);
	SDsetattr(sdFichier, "YGRID", DFNT_INT32, 1, &YGRID);
	SDsetattr(sdFichier, "NBTHETA", DFNT_INT32, 1, &NBTHETA);
	SDsetattr(sdFichier, "NBPHI", DFNT_INT32, 1, &NBPHI);
	SDsetattr(sdFichier, "NBSTOKES", DFNT_INT32, 1, &NBSTOKES);
	SDsetattr(sdFichier, "DIOPTRE", DFNT_INT32, 1, &DIOPTRE);
	SDsetattr(sdFichier, "DIFFF", DFNT_INT32, 1, &DIFFF);
	SDsetattr(sdFichier, "PROFIL", DFNT_INT32, 1, &PROFIL);
	SDsetattr(sdFichier, "SIM", DFNT_INT32, 1, &SIM);
	SDsetattr(sdFichier, "SUR", DFNT_INT32, 1, &SUR);
	SDsetattr(sdFichier, "THSDEG", DFNT_FLOAT32, 1, &THSDEG);
	SDsetattr(sdFichier, "LAMBDA", DFNT_FLOAT32, 1, &LAMBDA);
	SDsetattr(sdFichier, "TAURAY", DFNT_FLOAT32, 1, &TAURAY);
	SDsetattr(sdFichier, "TAUAER", DFNT_FLOAT32, 1, &TAUAER);
	
	SDsetattr(sdFichier, "LSAAER", DFNT_UINT32, 1, &LSAAER);
	SDsetattr(sdFichier, "NFAER", DFNT_UINT32, 1, &NFAER);
	
	SDsetattr(sdFichier, "W0AER", DFNT_FLOAT32, 1, &W0AER);
	SDsetattr(sdFichier, "HA", DFNT_FLOAT32, 1, &HA);
	SDsetattr(sdFichier, "HR", DFNT_FLOAT32, 1, &HR);
	SDsetattr(sdFichier, "ZMIN", DFNT_FLOAT32, 1, &ZMIN);
	SDsetattr(sdFichier, "ZMAX", DFNT_FLOAT32, 1, &ZMAX);
	SDsetattr(sdFichier, "WINDSPEED", DFNT_FLOAT32, 1, &WINDSPEED);
	SDsetattr(sdFichier, "NH2O", DFNT_FLOAT32, 1, &NH2O);
	SDsetattr(sdFichier, "CONPHY", DFNT_FLOAT32, 1, &CONPHY);
	SDsetattr(sdFichier, "PATHRESULTATSHDF", DFNT_CHAR8, strlen(PATHRESULTATSHDF), PATHRESULTATSHDF);
	SDsetattr(sdFichier, "PATHTEMOINHDF", DFNT_CHAR8, strlen(PATHTEMOINHDF), PATHTEMOINHDF);
	SDsetattr(sdFichier, "PATHDIFFAER", DFNT_CHAR8, strlen(PATHDIFFAER), PATHDIFFAER);
	SDsetattr(sdFichier, "PATHPROFILATM", DFNT_CHAR8, strlen(PATHPROFILATM), PATHPROFILATM);
	
	SDsetattr(sdFichier, "nbPhotonsTot", DFNT_FLOAT64, 1, &nbPhotonsTotdouble);
	SDsetattr(sdFichier, "nbErreursPoids", DFNT_INT32, 1, &(var->erreurpoids));
	SDsetattr(sdFichier, "nbErreursTheta", DFNT_INT32, 1, &(var->erreurtheta));
	SDsetattr(sdFichier, "tempsEcoule", DFNT_FLOAT64, 1, &tempsEcouledouble);
	
	// Pour chaque phi on ajoute au fichier le tableau représentant le résultat final en fonction de theta
	for(int iphi = 0; iphi < NBPHI; iphi++)
	{
		// Création et remplissage d'un tableau du fichier
		// tab[ith*2+0] est le résultat final en ith et tab[ith*2+1] est la valeur de theta
		float* tab;
		tab = (float*)malloc(2 * NBTHETA * sizeof(float));
		memset(tab, 0, 2 * NBTHETA * sizeof(float));
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			tab[ith*2+0] = tabFinal[ith*NBPHI+iphi];
			tab[ith*2+1] = tabTh[ith];
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
		if(status)
		{
			printf("\nERREUR : write hdf resultats\n");
			exit(1);
		}
		// Ecriture d'informations sur le tableau
		char description[20];
		sprintf(description, "%f", tabPhi[iphi]);
		SDsetattr(sdsTab, "phi", DFNT_CHAR8, strlen(description), description);
		
		// Fermeture du tableau
		SDendaccess(sdsTab);
		// Liberation du tableau
		free(tab);
	}
	// Fermeture du fichier
	SDend(sdFichier);
}

// Fonction qui libère les tableaux envoyés dans le kernel
void freeTableaux(Tableaux* tab_H, Tableaux* tab_D)
{
	#ifdef RANDMWC
	// Liberation des tableaux de generateurs du random MWC
	cudaFree(tab_D->etat);
	free(tab_H->etat);
	cudaFree(tab_D->config);
	free(tab_H->config);
	#endif
	#ifdef RANDCUDA
	// Liberation du tableau de generateurs du random Cuda
	cudaFree(tab_D->etat);
	#endif
	#ifdef RANDMT
	// Liberation des tableaux de generateurs du random Mersenen Twister
	cudaFree(tab_D->config);
	cudaFree(tab_D->etat);
	free(tab_H->config);
	#endif
	// Liberation du tableau du poids des photons
	cudaFree(tab_D->tabPhotons);
	free(tab_H->tabPhotons);
	
	// Libération du modèle de diffusion des aérosols
	cudaFree(tab_D->faer);
	free(tab_H->faer);
	
	// Libération du modèle atmosphérique
	cudaFree(tab_D->tauCouche);
	free(tab_H->tauCouche);
	cudaFree(tab_D->pMol);
	free(tab_H->pMol);
}

/********************/
// Ajouts Florent
/********************/

/* Calcul du modèle de diffusion des aérosol */
void calculFaer( const char* nomFichier, Tableaux tab_H, Tableaux tab_D ){
	
	FILE* fichier = fopen(nomFichier, "r");

	float *scum = (float*) malloc(LSAAER*sizeof(*scum));
	scum[0] = 0;
	int iang = 0, ipf = 0;
	float dtheta, pm1, pm2, sin1, sin2;
	float z, norm;

	/** Allocation de la mémoire des tableaux contenant les données **/
	float *ang;
	float *p1, *p2, *p3, *p4;
	ang = (float*) malloc(LSAAER*sizeof(float));
	p1 = (float*) malloc(LSAAER*sizeof(float));
	p2 = (float*) malloc(LSAAER*sizeof(float));
	p3 = (float*) malloc(LSAAER*sizeof(float));
	p4 = (float*) malloc(LSAAER*sizeof(float));
	
	/** Lecture des données sur le modèle de diffusion des aérosols **/
	if(fichier == NULL){
		printf("ERREUR : Ouverture impossible du fichier %s pour la diffusion d'aérosol", nomFichier );
		exit(1);
	}
	
	else{
		for(iang=0; iang<LSAAER; iang++){
			fscanf(fichier, "%f %f %f %f %f", ang+iang,p1+iang,p2+iang,p3+iang,p4+iang );
			// Conversion en radians
			ang[iang] = ang[iang]*DEG2RAD;
		}
	}
	
	if(fclose(fichier) == EOF){
		printf("ERREUR : Probleme de fermeture du fichier %s", nomFichier);
	}
		
	/** Calcul de scum **/
	for(iang=1; iang<LSAAER; iang++){
		
		dtheta = ang[iang] - ang[iang-1];
		pm1= p1[iang-1] + p2[iang-1];
		pm2= p1[iang] + p2[iang];
		sin1= sin(ang[iang-1]);
		sin2= sin(ang[iang]);
		
		scum[iang] = scum[iang-1] + dtheta*( (sin1*pm1+sin2*pm2)/3 + (sin1*pm2+sin2*pm1)/6 )*DEUXPI; 
	}
	
	// Normalisation
	for(iang=0; iang<LSAAER; iang++){
		scum[iang] = scum[iang]/scum[LSAAER-1];
	}
	
	/** Calcul des faer **/
	for(iang=0; iang<NFAER-1; iang++){
		z = float(iang)/float(NFAER);
		
		while( scum[ipf+1]<z )
			ipf++;
		
		tab_H.faer[iang*5+4] = ((scum[ipf+1]-z)*ang[ipf] + (z-scum[ipf])*ang[ipf+1])/(scum[ipf+1]-scum[ipf]);
		norm = p1[ipf]+p2[ipf];			// Angle
		tab_H.faer[iang*5+0] = p1[ipf]/norm;	// I paralèlle
		tab_H.faer[iang*5+1] = p2[ipf]/norm;	// I perpendiculaire
		tab_H.faer[iang*5+2] = p3[ipf]/norm;	// u
		tab_H.faer[iang*5+3] = 0.F;				// v, toujours nul
	}
	
	tab_H.faer[(NFAER-1)*5+4] = PI;
	tab_H.faer[(NFAER-1)*5+0] = 0.5F+00;
	tab_H.faer[(NFAER-1)*5+1] = 0.5F+00;
	tab_H.faer[(NFAER-1)*5+2] =p3[LSAAER-1]/(p1[LSAAER-1]+p2[LSAAER-1]);
	tab_H.faer[(NFAER-1)*5+3] = 0.F+00;
	
	free(scum);
	free(ang);
	free(p1);
	free(p2);
	free(p3);
	free(p4);
	
	/** Allocation des FAER dans la device memory **/		

	if( cudaMemcpy(tab_D.faer, tab_H.faer, 5*NFAER*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ){
		printf("ERREUR : Mauvaise copie mémoire FAERd\n");
		exit(1);
	}
	
}

/* Permet de vérifier que le modèle FAER généré est correct */
void verificationFAER( const char* nomFichier, Tableaux tab){

	FILE* fichier = fopen(nomFichier, "w");
	int i;
	
	fprintf( fichier, "angle\tI//\tIp\n" );
	
	for(i=0; i<NFAER; i++){
		fprintf(fichier, "%f\t%f\t%f\n", tab.faer[i*5+4],tab.faer[i*5+0], tab.faer[i*5+1]);
	}
	
	fclose(fichier);

}

/* Calcul du mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche */
void profilAtm( Tableaux tab_H, Tableaux tab_D ){

	/** Déclaration des variables **/
	/*NOTE: différence avec le code fortran: je n'utilise pas int ncouche */
	
	float z[NATM+1];		// Altitude à chaque couche
	float tauMol[NATM+1];	// Epaisseur optique des molécules à chaque couche
	float tauAer[NATM+1];	// Epaisseur optique des aérosols à chaque couche
	int i=0;
	float va=0, vr=0;	// Variables tampons
	
	/** Conditions aux limites au sommet de l'atmosphère **/
	z[0] = 100.0;
	tauMol[0] = 0.0;
	tauAer[0] = 0.0;
	tab_H.tauCouche[0] = 0.0;
	tab_H.pMol[0] = 0.0;	//Je n'utilise pas la proportion d'aérosols car on l'obtient par 1-PMOL

	/** Cas Particuliers **/
	
	// Épaisseur optique moléculaire très faible
	// On ne considère une seule sous-couche dans laquelle on trouve tous les aérosols
	if( TAURAY < 0.0001 ){
		tauMol[1] = 0;
		tauAer[1] = TAUAER;
		z[1] = 0;
		tab_H.tauCouche[1] = tauMol[1] + tauAer[1];
		tab_H.pMol[1] = 0;
		return;
	}
	
	// Épaisseur optique aérosol très faible OU Épaisseur optique moléculaire et aérosol très faible
	// On ne considère une seule sous-couche dans laquelle on trouve toutes les molécules
	if( (TAUAER < 0.0001) || ((TAUAER < 0.0001)&&(TAURAY < 0.0001)) ){
		tauMol[1] = TAURAY;
		tauAer[1] = 0;
		z[1] = 0;
		tab_H.tauCouche[1] = tauMol[1] + tauAer[1];
		tab_H.pMol[1] = 1.0;
		return;
	}
	
	/** Profil standard avec échelle de hauteur **/
	if( PROFIL == 0 ){
		
		/* Si HA << HR => pas de mélange dans les couches
		On considere alors une atmosphere divisee en deux sous-couches, la  couche superieure contenant toutes les molecules, la couche inferieure contenant tous les aerosols.
		*/
		if( HA < 0.0001 ){
			tauMol[1] = TAURAY;
			tauAer[1] = 0;
			z[1] = 0;
			tab_H.tauCouche[1] = tauMol[1] + tauAer[1];
			tab_H.pMol[1] = 1.0;
			
			tauMol[2] = 0;
			tauAer[2] = TAUAER;
			z[2] = 0.0;
			tab_H.tauCouche[2] = tab_H.tauCouche[1] + tauMol[2] + tauAer[2];
			tab_H.pMol[2] = 0.0;
		}
		
		/* Si HA >> HR => pas de mélange dans les couches
		On considere alors une atmosphere divisee en deux sous-couches, la  couche superieure contenant tous les aérosols, la couche inferieure contenant toutes les molécules.
		*/
		else if( HA > 499.99 ){
			tauMol[1] = 0.0;
			tauAer[1] = TAUAER;
			z[1] = 0.0;
			tab_H.tauCouche[1] = tauMol[1] + tauAer[1];
			tab_H.pMol[1] = 0.0;
			
			tauMol[2] = TAURAY;
			tauAer[2] = 0.0;
			z[2] = 0.0;
			tab_H.tauCouche[2] = tab_H.tauCouche[1] + tauMol[2] + tauAer[2];
			tab_H.pMol[2] = 1.0;
		}
		
		/* Cas Standard avec deux échelles */
		else{
			for( i=0; i<NATM+1; i++){
				if(i!=0)
					z[i] = 100.F - float(i)*(100.F/NATM);

				vr = TAURAY*exp( -(z[i]/HR) );
				va = TAUAER*exp( -(z[i]/HA) );
				
				tab_H.tauCouche[i] = va+vr;
				
				vr = vr/HR;
				va = va/HA;
				vr = vr/(va+vr);
				tab_H.pMol[i] = vr;
			}
			tab_H.tauCouche[0] = 0;
		}
	}
	
	/** Profil à 2 ou 3 couches **/
	else if( PROFIL == 3 ){

		float tauRay1;	// Epaisseur optique moleculaire de la couche 1
		float tauRay2;	// Epaisseur optique moleculaire de la couche 2
		
		tauRay1 = TAURAY*exp(-(ZMAX/HR));	// Epaisseur optique moleculaire de la couche la plus haute
		if( ZMIN < 0.0001 ){
			tauRay2 = TAURAY*(exp(-(ZMIN/HR))-exp(-(ZMAX/HR)));	// Epaisseur optique moleculaire de la couche la plus basse
		}
		
		else{
			tauRay2 = TAURAY*(exp(-(ZMIN/HR))-exp(-(ZMAX/HR)));	// Epaisseur optique moleculaire de la couche intermédiaire
		}
		
		/** Calcul des grandeurs utiles aux OS pour la couche la plus haute **/
		z[1] = -( HR*log(tauRay1/TAURAY) );                                      
		tauMol[1] = tauRay1;
		tauAer[1] = 0.F;                                    
		tab_H.tauCouche[1] = tauMol[1] + tauAer[1];
		tab_H.pMol[1] = 1.F;

		/** Calcul des grandeurs utiles aux OS pour la deuxieme couche   **/
		if( ZMAX == ZMIN ){ //Uniquement des aerosols dans la couche intermediaire
			z[2] = ZMAX; // ou zmin, puisque zmin=zmax
			tauMol[2] = tauRay1;                                                      
			tauAer[2] = TAUAER;
			tab_H.tauCouche[2] = tauMol[2] + tauAer[2];
			tab_H.pMol[2] = 0.F;                                                      
		}
		
		else{	// Melange homogene d'aerosol et de molecules dans la couche intermediaire
			z[2] = ZMIN;
			tauMol[2] = tauRay1+tauRay2;
			tauAer[2] = TAUAER;
			tab_H.tauCouche[2] = tauMol[2] + tauAer[2];
			tab_H.pMol[2] = 0.5F;
		}
		
		/** Calcul des grandeurs utiles aux OS pour la troisieme couche **/
		z[3] = 0.F;
		tauMol[3] = TAURAY;
		tauAer[3] = TAUAER;
		tab_H.tauCouche[3] = tauMol[3] + tauAer[3];
		tab_H.pMol[3] = 1.F;
	}
	
	else if( PROFIL == 2 ){
		
		//ATTENTION NOTE: pas encore vérifié
		
		// Profil utilisateur
		/* Format du fichier
		=> Ne pas mettre de ligne vide sur la première
		=> n	alt		tauMol		tauAer		tauCouche		pAer		pMol
		*/
		FILE* profil = fopen( PATHPROFILATM , "r" );
		float *garbage = NULL;
		int icouche=0;
	
		if(profil == NULL){
			printf("ERREUR : Ouverture impossible du fichier %s pour la diffusion d'aérosol", PATHPROFILATM );
			exit(1);
		}
	
		else{
			i=0;
			
			while((fscanf(profil, "%d %f %f %f %f %f %f", &icouche, garbage, garbage, garbage, tab_H.tauCouche+i, garbage, tab_H.pMol+i ) !=EOF) && icouche<=NATM ){
			i = icouche+1;
			}
	}
	
		if(fclose(profil) == EOF){
			printf("ERREUR : Probleme de fermeture du fichier %s", PATHPROFILATM);
		}
	}
	
	/* Vérification du modèle
	FILE* fichier = fopen("./test/modele_atm_cuda.txt", "w");
	
	fprintf( fichier, "couche\tpropMol\tTauCouche\n" );
	
	for(i=0; i<NATM+1; i++){
		fprintf(fichier, "%d\t%10.8f\t%10.8f\n",i,PMOL[i], TAUCOUCHE[i]);
	}
	
	fclose(fichier);
	*/
	
	/** Envoie des informations dans le device **/

	if( cudaMemcpy(tab_D.tauCouche, tab_H.tauCouche, (NATM+1)*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ){
		printf("ERREUR : Mauvaise copie mémoire TAUCOUCHEd\n");
		exit(1);
	}
	
	if( cudaMemcpy(tab_D.pMol, tab_H.pMol, (NATM+1)*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ){
		printf("ERREUR : Mauvaise copie mémoire PMOLd\n");
		exit(1);
	}
	
}

