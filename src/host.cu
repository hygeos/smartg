
/**********************************************************
*	> Includes
***********************************************************/

#include "communs.h"
#include "host.h"
#include "device.h"

// __constant__ float foce_c[5*10000000];
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


/* initRandMTConfig
* Fonction qui initialise en partie les generateurs du random Mersenen Twister
*/
void initRandMTConfig(ConfigMT* config_H, ConfigMT* config_D, int nbThreads)
{
	// Ouverture du fichier
	const char *fname = "auxdata/MT/MersenneTwister.dat";
	FILE* fd = fopen(fname, "rb");
	if(!fd)
	{
		printf("ERREUR: ouverture fichier MT");
		exit(0);
	}
	// Lecture et initialisation de la config pour chaque generateur (= pour chaque thread)
	for(int i = 0; i < nbThreads; i++)
	{
		/* Le fichier ne contient que 4096 configs, on reutilise donc les memes configs pour les threads en trop mais les nombres
		aléatoires restent independants car les etats des threads sont differents */
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
	
	cudaError_t erreur = cudaMemcpy(config_D, config_H, nbThreads * sizeof(ConfigMT), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie config_H dans initRandMTConfig\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
}


/**********************************************************
*	> Travail sur les fichiers
***********************************************************/

/* initConstantesHost
* Fonction qui récupère les valeurs des constantes dans le fichier paramètres et initialise les constantes du host
*/
void initConstantesHost(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("ERREUR : lecture argv\n");
		exit(1);
	}
	
	FILE* parametres = fopen( argv[1], "r" );
	
	if( parametres == NULL ){
		printf("ERREUR: Impossible d'ouvrir le fichier %s\n", argv[1] );
		exit(1);
	}
	
	char s[256];
    double dbl;

	strcpy(s,"");
	chercheConstante( parametres, "NBPHOTONS", s);
    dbl = atof(s);
    NBPHOTONS = (unsigned long long)dbl;
	
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
	chercheConstante(parametres, "SIM", s);
	SIM = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "SUR", s);
	SUR = atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "DIOPTRE", s);
	DIOPTRE= atoi(s);

	strcpy(s,"");
	chercheConstante(parametres, "ENV", s);
	ENV= atoi(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "THVDEG", s);
	THVDEG = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "LAMBDA", s);
	LAMBDA = atof(s);

    strcpy(s,"");
	chercheConstante(parametres, "ENV_SIZE", s);
	ENV_SIZE = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "X0", s);
	X0 = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "Y0", s);
	Y0 = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "DEPO", s);
	DEPO = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "NFAER", s);
	NFAER = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "NFOCE", s);
	NFOCE = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "WINDSPEED", s);
	WINDSPEED = atof(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "NH2O", s);
	NH2O = atof(s);
	
    #ifdef FLAGOCEAN
	chercheConstante( parametres, "PATHDIFFOCE", PATHDIFFOCE );
	chercheConstante( parametres, "PATHPROFILOCE", PATHPROFILOCE );

    #endif

	strcpy(s,"");
	chercheConstante(parametres, "OUTPUT_LAYERS", s);
	OUTPUT_LAYERS = atoi(s);
	
	chercheConstante(parametres, "PATHRESULTATSHDF", PATHRESULTATSHDF);
    
	chercheConstante( parametres, "PATHDIFFAER", PATHDIFFAER );
	
	chercheConstante( parametres, "PATHPROFILATM", PATHPROFILATM );

	chercheConstante( parametres, "PATHALB", PATHALB );

	chercheConstante( parametres, "DEVICE", s);
    DEVICE = atoi(s);

	fclose( parametres );
}


/* chercheConstante
* Fonction qui cherche nomConstante dans le fichier et met la valeur de la constante dans chaineValeur (en string)
*/
void chercheConstante(FILE* fichier, const char* nomConstante, char* chaineValeur)
{
	int longueur = strlen(nomConstante);
	char ligne[2048];
	int motTrouve = 0;
	
	// Tant que la constante n'est pas trouvee et qu'on n'est pas à la fin du fichier on lit la ligne
	while(fgets(ligne, 2048, fichier) && !motTrouve)
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
				printf("ERROR when reading keyword %s in parameter file. Line is:\n", nomConstante);
                printf("%s\n", ligne);
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
		printf("ERREUR : lecture fichier paramètres - Constante manquante: %s\n",nomConstante);
		exit(1);
	}
}



void init_profileATM(int *NATM, float *HATM, int *NLAM, char *PATHPROFILATM) {
    //
    // reads the number of layers NATM in the atmosphere profile, and the
    // height of the top layer
    // the profile file contains NATM+1 interfaces from 0 to NATM
    // Eventually reads the number of consecutive profiles NLAM

    printf("Read %s\n", PATHPROFILATM);

    FILE* fp;
    int c, i, read_first=1;
    float H;
    char buffer[2048];
    *HATM = -1;
    *NLAM = 1;

    fp = fopen(PATHPROFILATM, "r");

    if (fp == NULL) {
        printf("ERROR: Cannot open profile '%s'\n", PATHPROFILATM);
        exit(1);
    }

    // skip first line
    fgets(buffer, 2048, fp);
    *NATM = -1;

    // read first layer
    while(1) {
        if (fgets(buffer, 2048, fp) == NULL) break;
        if (buffer[0] == '#') {
            *NLAM += 1;
            read_first = 0;
            continue;
        }

        c = sscanf(buffer, "%d\t%f\t", &i, &H);
        if (c != 2) break;
        if (*HATM < 0) *HATM = H;
        if (read_first) *NATM += 1;
    }

    fclose(fp);
}

#ifdef FLAGOCEAN
void init_profileOCE(int *NOCE, int *NLAM, char *PATHPROFILOCE) {
    //
    // reads the number of layers NOCE in the ocean profile
    // the profile file contains NOCE+1 interfaces from 0 to NOCE
    // Eventually reads the number of consecutive profiles NLAM

    printf("Read %s\n", PATHPROFILOCE);

    FILE* fp;
    int c, i, read_first=1;
    float H;
    char buffer[2048];
    *NLAM = 1;

    fp = fopen(PATHPROFILOCE, "r");

    if (fp == NULL) {
        printf("ERROR: Cannot open profile '%s'\n", PATHPROFILOCE);
        exit(1);
    }

    // skip first line
    fgets(buffer, 2048, fp);
    *NOCE = -1;

    // read first layer
    while(1) {
        if (fgets(buffer, 2048, fp) == NULL) break;
        if (buffer[0] == '#') {
            *NLAM += 1;
            read_first = 0;
            continue;
        }

        c = sscanf(buffer, "%d\t%f\t", &i, &H);
        if (c != 2) break;
        if (read_first) *NOCE += 1;
    }

    fclose(fp);
}
#endif

int count_lines(char *PATHDIFF) {
    //
    // count the number of uncommented lines in a file
    //

    if (strcmp(PATHDIFF, "None") == 0) {
        return 0;
    }

    FILE *fp;
    int c = 0;
    char buffer[2048];
    fp = fopen(PATHDIFF, "r");
    if (fp == NULL) {
        printf("ERROR: cannot open file '%s'\n", PATHDIFF);
        exit(1);
    }
    while (1) {
        if (fgets(buffer, 2048, fp) == NULL) break;
        if (buffer[0] == '\n') break;
        if (buffer[0] == '#') continue;
        c++;
    }
    fclose(fp);

    return c;
}


/**********************************************************
*	> Initialisation des différentes structures
***********************************************************/

/* initVariables
* Fonction qui initialise les variables à envoyer dans le kernel.
*/
void initVariables(Variables** var_H, Variables** var_D)
{
	// 	Initialisation de la version host des variables
	*var_H = (Variables*)malloc(sizeof(Variables));
	if( var_H == NULL ){
		printf("#--------------------#\n");
		printf("ERREUR: Problème de malloc de var_H dans initVariables\n");
		printf("#--------------------#\n");
		exit(1);
	}
	memset(*var_H, 0, sizeof(Variables));
	
	//	Initialisation de la version device des variables
	if( cudaMalloc(var_D, sizeof(Variables)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de var_D dans initVariables\n");
		exit(1);
	}
	
	cudaError_t err = cudaMemset(*(var_D), 0, sizeof(Variables));
	if( err != cudaSuccess ){
		printf("#--------------------#\n");
		printf("# ERREUR: Problème de cudaMemset var_D dans initVariables\n");
		printf("# Nature de l'erreur: %s\n",cudaGetErrorString(err) );
		printf("#--------------------#\n");
		exit(1);
	}

}


/* reinitVariables
* Fonction qui réinitialise certaines variables avant chaque envoi dans le kernel
*/
void reinitVariables(Variables* var_H, Variables* var_D)
{
	// Le nombre de photons traités pour un appel du Kernel est remis à zéro
	var_H->nbPhotons = 0;
	#ifdef PROGRESSION
	// Le nombre de photons ressortis pour un appel du Kernel est remis à zéro
	var_H->nbPhotonsSor = 0;
	#endif
	// On copie le nouveau var_H dans var_D
	cudaError_t erreur = cudaMemcpy(var_D, var_H, sizeof(Variables), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf("#--------------------#\n");
		printf("# ERREUR: Problème de copie var_H dans reinitVariables\n");
		printf("# Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		printf("# sizeof(*var_D)=%d\tsizeof(*var_H)=%d\tsizeof(*Variables)=%d\n",sizeof(*var_D),sizeof(*var_H),sizeof(Variables));
		printf("# Adresse de var_D : %p\tAdresse de var_H : %p\n", var_H, var_D);
		printf("#--------------------#\n");
		exit(1);
	}
}


#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
/* initInit
* Initialisation de la structure Init contenant les paramètres initiaux du photon rentrant dans l'atmosphère.
* Ces paramètres sont utiles pour une atmosphère sphérique et sont calculés une seule fois dans le host, d'où cette fonction
* et la structure Init
*/
void initInit(Init** init_H, Init** init_D)
{
	// 	Initialisation de la version host des variables
	*init_H = (Init*)malloc(sizeof(Init));
	if( init_H == NULL ){
	printf("#--------------------#\n");
	printf("ERREUR: Problème de malloc de init_H dans initInit\n");
	printf("#--------------------#\n");
	exit(1);
	}
	memset(*init_H, 0, sizeof(Init));
   
	//	Initialisation de la version device des variables
	if( cudaMalloc(init_D, sizeof(Init)) == cudaErrorMemoryAllocation ){
	   printf("ERREUR: Problème de cudaMalloc de init_D dans initInit\n");
	   exit(1);
	}

	cudaError_t err = cudaMemset(*(init_D), 0, sizeof(Init));
	if( err != cudaSuccess ){
	   printf("#--------------------#\n");
	   printf("# ERREUR: Problème de cudaMemset init_D dans initInit\n");
	   printf("# Nature de l'erreur: %s\n",cudaGetErrorString(err) );
	   printf("#--------------------#\n");
	   exit(1);
	}

}
#endif


/* initTableaux
* Fonction qui initialise les tableaux à envoyer dans le kernel par allocation mémoire et memset
*/
void initTableaux(Tableaux* tab_H, Tableaux* tab_D)
{
	cudaError_t cudaErreur;	// Permet de tester les erreurs d'allocation mémoire
	
	#ifdef RANDMWC	
	// Création des tableaux de generateurs pour la fonction Random MWC
	tab_H->etat = (unsigned long long*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long));
	if( tab_H->etat == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->etat dans initTableaux\n");
		exit(1);
	}
	
	if( cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long)) == cudaErrorMemoryAllocation){
		printf("ERREUR: Problème de cudaMalloc de tab_D->etat dans initTableaux\n");
		exit(1);	
	}
	
	tab_H->config = (unsigned int*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int));
	if( tab_H->config == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->config dans initTableaux\n");
		exit(1);
	}
	
	if( cudaMalloc(&(tab_D->config), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int)) == cudaErrorMemoryAllocation){
		printf("ERREUR: Problème de cudaMalloc de tab_D->config dans initTableaux\n");
		exit(1);	
	}
	
	// Initialisation des tableaux host à l'aide du fichier et du seed
	initRandMWC(tab_H->etat, tab_H->config, XBLOCK * YBLOCK * XGRID * YGRID, "auxdata/MWC/MWC.txt", (unsigned long long)SEED);
	
	// Copie dans les tableaux device
	cudaErreur = cudaMemcpy(tab_D->etat, tab_H->etat, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->etat dans initTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	
	cudaErreur = cudaMemcpy(tab_D->config, tab_H->config, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->config dans initTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	#endif
	
        #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
	// Création du tableau de generateurs (=etat+config) pour la fonction Random Cuda
	if( cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(curandSTATE)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->etat dans initTableaux\n");
		exit(1);	
	}
	
	// Initialisation du tableau dans une fonction du kernel
        #if defined(RANDCUDA)
	initRandCUDA<<<XGRID * YGRID, XBLOCK * YBLOCK>>>(tab_D->etat, (unsigned long long)SEED);
        #else
        unsigned long long NbThreads = XGRID * YGRID * XBLOCK * YBLOCK;
        unsigned long long NbDimensions = NbThreads > 20000 ? 20000 : NbThreads;
        curandDirectionVectors32_t *d_qrngDirections = 0;
        cudaErreur = cudaMalloc((void **)&d_qrngDirections, NbDimensions*sizeof(curandDirectionVectors32_t));
        if( cudaErreur != cudaSuccess ){
            printf( "ERREUR: Problème d'allocation de d_qrngDirections\n");
            printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
            exit(1);
        }
        curandDirectionVectors32_t *h_rngDirections;
        curandGetDirectionVectors32(&h_rngDirections, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6);
        cudaErreur = cudaMemcpy(d_qrngDirections, h_rngDirections, NbDimensions*sizeof(curandDirectionVectors32_t), cudaMemcpyHostToDevice);
        if( cudaErreur != cudaSuccess ){
            printf( "ERREUR: Problème de copie h_rngDirections dans d_qrngDirections\n");
            printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
            exit(1);
        }
        initRandCUDANDQRNGs<<< dim3(XGRID,YGRID,1), dim3(XBLOCK,YBLOCK,1)>>>(tab_D->etat, d_qrngDirections);
        #endif
	#endif
	
	#ifdef RANDMT
	// Création des tableaux de generateurs pour la fonction Random Mersenne Twister
	if( cudaMalloc(&(tab_D->config), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(ConfigMT)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->config dans initTableaux\n");
		exit(1);	
	}
	
	if( cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(EtatMT)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->etat dans initTableaux\n");
		exit(1);	
	}
	
	tab_H->config = (ConfigMT*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(ConfigMT));
	if( tab_H->config == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->config dans initTableaux\n");
		exit(1);
	}
	
	tab_H->etat = (EtatMT*)malloc(XBLOCK * YBLOCK * XGRID * YGRID * sizeof(EtatMT));
	if( tab_H->etat == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->etat dans initTableaux\n");
		exit(1);
	}
		
	// Initialisation du tableau des configs à l'aide du fichier
	initRandMTConfig(tab_H->config, tab_D->config, XBLOCK * YBLOCK * XGRID * YGRID);
	// Initialisation du tableau des etats dans le kernel
	initRandMTEtat<<<XGRID * YGRID, XBLOCK * YBLOCK>>>(tab_D->etat, tab_D->config);
	#endif

        #ifdef RANDPHILOX4x32_7
        //Memset de deux valeurs pour la creation des generateurs philox
        unsigned int compteur_init = 0;
        unsigned int clef_utilisateur = (unsigned int) SEED; /*LDS: eventuellement la conversion ici change la graine mais c'est sans reelle importance il me semble*/
        tab_D->config = clef_utilisateur;
        cudaErreur = cudaMalloc((void**)&(tab_D->etat), sizeof(unsigned int)* XBLOCK * YBLOCK * XGRID * YGRID);
        if( cudaErreur != cudaSuccess){
            printf("ERREUR: Problème de cudaMalloc de tab_D->etat dans initTableaux\n");
            printf("\t->detail de l'erreur : %s\n", cudaGetErrorString(cudaErreur));
            exit(1);
        }
        initPhilox4x32_7Compteur<<<dim3(XGRID,YGRID,1), dim3(XBLOCK,YBLOCK,1)>>>(tab_D->etat, compteur_init);
        #endif
	
	// Tableau du poids des photons ressortis
	tab_H->tabPhotons = (float*)malloc(4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotons)));
	if( tab_H->tabPhotons == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotons dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotons,0,4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotons)) );
	
	if( cudaMalloc(&(tab_D->tabPhotons), 4 * NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotons))) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotons dans initTableaux\n");
		exit(1);	
	}

	cudaErreur = cudaMemset(tab_D->tabPhotons, 0, 4*NBTHETA * NBPHI * NLAM *  sizeof(*(tab_D->tabPhotons)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotons dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	
	// Tableau du nombre des photons injectes par interval NLAM
	tab_H->nbPhotonsInter = (unsigned long long*)malloc(NLAM * sizeof(*(tab_H->nbPhotonsInter)));
	if( tab_H->nbPhotonsInter == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->nbPhotonsInter dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->nbPhotonsInter,0,NLAM * sizeof(*(tab_H->nbPhotonsInter)) );
	
	if( cudaMalloc(&(tab_D->nbPhotonsInter), NLAM * sizeof(*(tab_D->nbPhotonsInter))) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tab_D->nbPhotonsInter dans initTableaux\n");
		exit(1);	
	}

	cudaErreur = cudaMemset(tab_D->nbPhotonsInter, 0,  NLAM * sizeof(*(tab_D->nbPhotonsInter)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.nbPhotonsInter dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	
	
	// Weight Table of the descending  photons above the surface
	tab_H->tabPhotonsDown0P = (float*)malloc(4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsDown0P)));
	if( tab_H->tabPhotonsDown0P == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotonsDown dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotonsDown0P,0,4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsDown0P)) );
	
	if( cudaMalloc(&(tab_D->tabPhotonsDown0P), 4 * NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsDown0P))) != cudaSuccess){

		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotonsDown dans initTableaux\n");
		exit(1);	
	}

	cudaErreur = cudaMemset(tab_D->tabPhotonsDown0P, 0, 4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsDown0P)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsDown0P dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	
	
	// Weight Table of the descending  photons below the surface
	tab_H->tabPhotonsDown0M = (float*)malloc(4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsDown0M)));
	if( tab_H->tabPhotonsDown0M == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotonsDown0M dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotonsDown0M,0,4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsDown0M)) );
	
	if( cudaMalloc(&(tab_D->tabPhotonsDown0M), 4 * NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsDown0M))) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotonsDown0M dans initTableaux\n");
		exit(1);	
	}
	
	cudaErreur = cudaMemset(tab_D->tabPhotonsDown0M, 0, 4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsDown0M)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsDown0M dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	

	// Weight Table of the ascending  photons above the surface
	tab_H->tabPhotonsUp0P = (float*)malloc(4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsUp0P)));
	if( tab_H->tabPhotonsUp0P == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotonsUp0P dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotonsUp0P,0,4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsUp0P)) );
	
	if( cudaMalloc(&(tab_D->tabPhotonsUp0P), 4 * NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsUp0P))) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotonsUp0P dans initTableaux\n");
		exit(1);	
	}
	
	cudaErreur = cudaMemset(tab_D->tabPhotonsUp0P, 0, 4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsUp0P)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsUp0P dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	

	// Weight Table of the ascending  photons below the surface
	tab_H->tabPhotonsUp0M = (float*)malloc(4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_H->tabPhotonsUp0M)));
	if( tab_H->tabPhotonsUp0M == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotonsUp0M dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotonsUp0M,0,4*NBTHETA * NBPHI * sizeof(*(tab_H->tabPhotonsUp0M)) );
	
	if( cudaMalloc(&(tab_D->tabPhotonsUp0M), 4 * NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsUp0M))) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotonsUp0M dans initTableaux\n");
		exit(1);	
	}
	
	cudaErreur = cudaMemset(tab_D->tabPhotonsUp0M, 0, 4*NBTHETA * NBPHI * NLAM * sizeof(*(tab_D->tabPhotonsUp0M)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsUp0M dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
	}
	

	/** Modèle de diffusion **/
	// Modèle de diffusion des aérosols
	tab_H->faer = (float*)malloc(5 * NFAER * sizeof(float));
	if( tab_H->faer == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->faer dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->faer,0,5 * NFAER*sizeof(float) );
	
	if( cudaMalloc(&(tab_D->faer), 5 * NFAER * sizeof(float)) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->faer dans initTableaux\n");
		exit(1);	
	}
	
	#ifdef FLAGOCEAN
	/** Modèle de l'ocean **/
	// Fonction de phase 
	tab_H->foce = (float*)malloc(5 * NFOCE * sizeof(float));
	if( tab_H->foce == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->foce dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->foce,0,5 * NFOCE*sizeof(float) );
	
	if( cudaMalloc(&(tab_D->foce), 5 * NFOCE * sizeof(float)) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->foce dans initTableaux\n");
		exit(1);	
	}

	// Epaisseur optique par couche
	tab_H->ho =  (float*)malloc((NOCE+1)*NLAM*sizeof(*(tab_H->ho)));
	if( tab_H->ho == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->ho dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->ho,0,(NOCE+1)*NLAM*sizeof(*(tab_H->ho)) );
	
	if( cudaMalloc( &(tab_D->ho), (NOCE+1)*NLAM*sizeof(*(tab_H->ho)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->ho dans initTableaux\n");
		exit(1);	
	}

    // SSA 
	tab_H->sso =  (float*)malloc((NOCE+1)*NLAM*sizeof(float));
	if( tab_H->sso == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->sso dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->sso,0,(NOCE+1)*NLAM*sizeof(float) );
	
	if( cudaMalloc( &(tab_D->sso), (NOCE+1)*NLAM*sizeof(float) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->sso dans initTableaux\n");
		exit(1);	
	}
	
	#endif
	
	
	/** Modèle de l'atmosphère **/
	// Epaisseur optique par couche
	tab_H->h =  (float*)malloc((NATM+1)*NLAM*sizeof(*(tab_H->h)));
	if( tab_H->h == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->h dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->h,0,(NATM+1)*NLAM*sizeof(*(tab_H->h)) );
	
	if( cudaMalloc( &(tab_D->h), (NATM+1)*NLAM*sizeof(*(tab_H->h)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->h dans initTableaux\n");
		exit(1);	
	}
	
	// Proportion moléculaire par couche
	tab_H->pMol =  (float*)malloc((NATM+1)*NLAM*sizeof(float));
	if( tab_H->pMol == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->pMol dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->pMol,0,(NATM+1)*NLAM*sizeof(float) );
	
	if( cudaMalloc( &(tab_D->pMol), (NATM+1)*NLAM*sizeof(float) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->pMol dans initTableaux\n");
		exit(1);	
	}

    //
	tab_H->abs =  (float*)malloc((NATM+1)*NLAM*sizeof(float));
	if( tab_H->abs == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->abs dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->abs,0,(NATM+1)*NLAM*sizeof(float) );
	
	if( cudaMalloc( &(tab_D->abs), (NATM+1)*NLAM*sizeof(float) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->abs dans initTableaux\n");
		exit(1);	
	}
	
    //
	tab_H->ssa =  (float*)malloc((NATM+1)*NLAM*sizeof(float));
	if( tab_H->ssa == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->ssa dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->ssa,0,(NATM+1)*NLAM*sizeof(float) );
	
	if( cudaMalloc( &(tab_D->ssa), (NATM+1)*NLAM*sizeof(float) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->ssa dans initTableaux\n");
		exit(1);	
	}
	
	
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	
	// Altitude des couches
	tab_H->z =  (float*)malloc((NATM+1)*sizeof(*(tab_H->z)));
	if( tab_H->z == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->z dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->z,0,(NATM+1)*sizeof(*(tab_H->z)) );
	
	if( cudaMalloc( &(tab_D->z), (NATM+1)*sizeof(*(tab_H->z)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->z dans initTableaux\n");
		exit(1);	
	}
	
	/** Profil initial vu par le photon **/
	tab_H->zph0 =  (float*)malloc((NATM+1)*sizeof(*(tab_H->zph0)));
	if( tab_H->zph0 == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->zph0 dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->zph0,0,(NATM+1)*sizeof(*(tab_H->zph0)) );
	
	if( cudaMalloc( &(tab_D->zph0), (NATM+1)*sizeof(*(tab_D->zph0)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->zph0 dans initTableaux\n");
		exit(1);	
	}
	
	tab_H->hph0 =  (float*)malloc((NATM+1)*NLAM*sizeof(*(tab_H->hph0)));
	if( tab_H->hph0 == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->hph0 dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->hph0,0,(NATM+1)*NLAM*sizeof(*(tab_H->hph0)) );
	
	if( cudaMalloc( &(tab_D->hph0), (NATM+1)*NLAM*sizeof(*(tab_D->hph0)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->hph0 dans initTableaux\n");
		exit(1);	
	}
	#endif

    // Spectral albedo
	tab_H->alb =  (float*)malloc(2*NLAM*sizeof(*(tab_H->alb)));
	if( tab_H->alb == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->alb dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->alb,0,2*NLAM*sizeof(*(tab_H->alb)) );
	
	if( cudaMalloc( &(tab_D->alb), 2*NLAM*sizeof(*(tab_D->alb)) ) != cudaSuccess ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->alb dans initTableaux\n");
		exit(1);	
	}
	
}


/* freeTableaux
* Fonction qui libère l'espace mémoire de tous les tableaux alloués
*/
void freeTableaux(Tableaux* tab_H, Tableaux* tab_D)
{
	
	cudaError_t erreur;	// Permet de tester le bon déroulement des cudaFree
	
	#ifdef RANDMWC
	// Liberation des tableaux de generateurs du random MWC
	erreur = cudaFree(tab_D->etat);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->etat dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->etat);
	
	erreur = cudaFree(tab_D->config);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->config dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->config);
	#endif
	
        #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
	// Liberation du tableau de generateurs du random Cuda
	erreur = cudaFree(tab_D->etat);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->etat dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	#endif
	
	#ifdef RANDMT
	// Liberation des tableaux de generateurs du random Mersenen Twister
	erreur = cudaFree(tab_D->config);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->config dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	erreur = cudaFree(tab_D->etat);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->etat dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->config);
	free(tab_H->etat);
	#endif
	
        #ifdef RANDPHILOX4x32_7
	// Liberation du tableaux de compteurs des philox
	erreur = cudaFree(tab_D->etat);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->config dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	#endif

	// Liberation du tableau du poids des photons
	erreur = cudaFree(tab_D->tabPhotons);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->tabPhotons dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	// 	cudaFreeHost(tab_H->tabPhotons);
	free(tab_H->tabPhotons);
	
	erreur = cudaFree(tab_D->nbPhotonsInter);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->nbPhotonsInter dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	// 	cudaFreeHost(tab_H->nbPhotonsInter);
	free(tab_H->nbPhotonsInter);
	
	/** Modèles de diffusion **/
	// Libération du modèle de diffusion des aérosols
	erreur = cudaFree(tab_D->faer);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->faer dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	free(tab_H->faer);
	
	#ifdef FLAGOCEAN
	// Libération du modèle ocean
	// Diffusion dans l'océan
	erreur = cudaFree(tab_D->foce);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->foce dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	free(tab_H->foce);

	erreur = cudaFree(tab_D->ho);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->ho dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->ho);
	
	erreur = cudaFree(tab_D->sso);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->sso dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->sso);
	#endif
	
	
	/** Profil amosphèrique **/	
	// Libération du modèle atmosphérique
	erreur = cudaFree(tab_D->h);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->h dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->h);
	
	//
	erreur = cudaFree(tab_D->pMol);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->pMol dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->pMol);
	
	//
	erreur = cudaFree(tab_D->abs);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->abs dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->abs);
	
	//
	erreur = cudaFree(tab_D->ssa);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->ssa dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->ssa);
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	
	// Altitude des couches
	erreur = cudaFree(tab_D->z);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->z dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->z);
	
	// Profil initial vu par la photon
	erreur = cudaFree(tab_D->zph0);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->zph0 dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->zph0);
	
	erreur = cudaFree(tab_D->hph0);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->hph0 dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->hph0);
	
	#endif
	
	//
	erreur = cudaFree(tab_D->alb);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->alb dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->alb);
}


/**********************************************************
*	> Calculs de profils
***********************************************************/

/* calculFaer
* Calcul de la fonction de phase des aérosols
*/
void calculF( const char* nomFichier, float* phase_H, float* phase_D , int lsa, int nf){

    // not necessary when file name is "None"
    if (strcmp(nomFichier, "None") == 0) {
        return;
    }

	FILE* fichier = fopen(nomFichier, "r");

	double *scum = (double*) malloc(lsa*sizeof(*scum));
	if( scum==NULL ){
		printf("ERREUR: Problème de malloc de scum dans calculF\n");
		exit(1);
	}
	
	scum[0] = 0;
	int iang = 0, ipf = 0;
	double dtheta, pm1, pm2, sin1, sin2;
	double z;
    char buffer[1024];
    char *ptr;

	/** Allocation de la mémoire des tableaux contenant les données **/
	double *ang;
	double *p1, *p2, *p3, *p4;
	ang = (double*) malloc(lsa*sizeof(*ang));
	p1 = (double*) malloc(lsa*sizeof(*p1));
	p2 = (double*) malloc(lsa*sizeof(*p2));
	p3 = (double*) malloc(lsa*sizeof(*p3));
	p4 = (double*) malloc(lsa*sizeof(*p4));
	if( ang==NULL || p1==NULL || p2==NULL || p3==NULL || p4==NULL ){
		printf("ERREUR: Problème de malloc de ang ou pi dans calculFaer\n");
		exit(1);
	}
	
	/** Lecture des données sur le modèle de diffusion des aérosols **/
	if(fichier == NULL){
		printf("ERREUR : Ouverture impossible du fichier %s pour la diffusion d'aérosol", nomFichier );
		exit(1);
	}
	
	else{
        char c = getc(fichier);
        while(c=='#') {
            while((c=getc(fichier))!='\n');
            c = getc(fichier);
        }
        fseek(fichier, -1, SEEK_CUR);
		for(iang=0; iang<lsa; iang++){
            fgets(buffer, 1024, fichier);

            // replace all occurences of 'D' by 'E'
            // (compatibility with fortran output)
            ptr = buffer;
            do {
                ptr = strchr(ptr, 'D');
                if (ptr != NULL) {
                    *ptr = 'E';
                }
            } while(ptr != NULL);

            ptr = buffer;
            while ((*ptr == ' ') || (*ptr == '\t')) ptr++;
            *(ang+iang) = atof(ptr);
            while ((*ptr != ' ') && (*ptr != '\t')) ptr++;
            while ((*ptr == ' ') || (*ptr == '\t')) ptr++;
            *(p2+iang) = atof(ptr);
            while ((*ptr != ' ') && (*ptr != '\t')) ptr++;
            while ((*ptr == ' ') || (*ptr == '\t')) ptr++;
            *(p1+iang) = atof(ptr);
            while ((*ptr != ' ') && (*ptr != '\t')) ptr++;
            while ((*ptr == ' ') || (*ptr == '\t')) ptr++;
            *(p3+iang) = atof(ptr);
            while ((*ptr != ' ') && (*ptr != '\t')) ptr++;
            while ((*ptr == ' ') || (*ptr == '\t')) ptr++;
            *(p4+iang) = atof(ptr);
            
			// Conversion en radians
			ang[iang] = ang[iang]*DEG2RAD;
		}
	}
	
	if(fclose(fichier) == EOF){
		printf("ERREUR : Probleme de fermeture du fichier %s", nomFichier);
	}
		
	/** Calcul de scum **/
	for(iang=1; iang<lsa; iang++){
		
		dtheta = ang[iang] - ang[iang-1];
		pm1= p1[iang-1] + p2[iang-1];
		pm2= p1[iang] + p2[iang];
		sin1= sin(ang[iang-1]);
		sin2= sin(ang[iang]);
		
		scum[iang] = scum[iang-1] + dtheta*( (sin1*pm1+sin2*pm2)/3 + (sin1*pm2+sin2*pm1)/6 )*DEUXPI; 
	}
	
	// Normalisation
	for(iang=0; iang<lsa; iang++){
		scum[iang] = scum[iang]/scum[lsa-1];
	}
	
	/** Calcul des faer **/
//	for(iang=0; iang<nf-1; iang++){
		for(iang=0; iang<nf; iang++){
		z = double(iang+1)/double(nf);
		while( (scum[ipf+1]<z) )
			ipf++;
		
			phase_H[iang*5+4] = float( ((scum[ipf+1]-z)*ang[ipf] + (z-scum[ipf])*ang[ipf+1])/(scum[ipf+1]-scum[ipf]) );
//		norm = p1[ipf]+p2[ipf];			// Angle
//		phase_H[iang*5+0] = float( p1[ipf]/norm );	// I paralèlle
//		phase_H[iang*5+1] = float( p2[ipf]/norm );	// I perpendiculaire
//		phase_H[iang*5+2] = float( p3[ipf]/norm );	// u
//		phase_H[iang*5+3] = 0.F;			// v, toujours nul
			phase_H[iang*5+0] = float( p1[ipf] );	// I paralèlle
			phase_H[iang*5+1] = float( p2[ipf] );	// I perpendiculaire
			phase_H[iang*5+2] = float( p3[ipf] );	// u
			phase_H[iang*5+3] = 0.F;	       	// v, toujours nul
	}
	
//	phase_H[(nf-1)*5+4] = PI;
//	phase_H[(nf-1)*5+0] = 0.5F+00;
//	phase_H[(nf-1)*5+1] = 0.5F+00;
//	phase_H[(nf-1)*5+2] = float( p3[lsa-1]/(p1[lsa-1]+p2[lsa-1]) );
//	phase_H[(nf-1)*5+3] = 0.F+00;
	
	free(scum);
	free(ang);
	free(p1);
	free(p2);
	free(p3);
	free(p4);
	
	/** Allocation des FAER dans la device memory **/		

	cudaError_t erreur = cudaMemcpy(phase_D, phase_H, 5*nf*sizeof(float), cudaMemcpyHostToDevice); 
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie phase_D dans calculF\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
}

/* Read spectral albedo (for surface ,seafloor and sea reflectance)*/
void profilAlb( Tableaux* tab_H, Tableaux* tab_D ){
    int ilam;
    // Profil utilisateur
    /* Format du fichier
    => alb_surface alb_sea(floor) 
    */
    FILE* profil = fopen( PATHALB , "r" );
    char ligne[1024];
	cudaError_t erreur;		// Permet de tester le bon déroulement des opérations mémoires

    if(profil == NULL){
        printf("ERREUR : Ouverture impossible du fichier %s pour le profil albedo\n", PATHALB );
        exit(1);
    }

    else{
        for( ilam=0; ilam<NLAM; ilam++){
           // skip comment line
           fgets(ligne,1024,profil);
           fscanf(profil, "%f\t%f\n", tab_H->alb+0+ilam*2,tab_H->alb+1+ilam*2);
        }
    }

	if(fclose(profil) == EOF){
		printf("ERREUR : Probleme de fermeture du fichier %s", PATHALB);
	}

	erreur = cudaMemcpy(tab_D->alb, tab_H->alb, 2*NLAM*sizeof(*(tab_H->alb)), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->alb dans profilAlb\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
}


#ifdef FLAGOCEAN
/* Read ocean extinction coefficient and single scattering albedo for ocean*/
void profilOce( Tableaux* tab_H, Tableaux* tab_D ){
    int ilam;
    int icouche=0;
    float garbage;
    int i;
    // Profil utilisateur
    /* Format du fichier
    => n	alt		ho		sso	
    */
    FILE* profil = fopen( PATHPROFILOCE , "r" );
    char ligne[1024];
	cudaError_t erreur;		// Permet de tester le bon déroulement des opérations mémoires

	for( ilam=0;ilam<NLAM;ilam++) {
        tab_H->ho[0 + ilam * (NOCE+1)] = 0.;
        tab_H->sso[0 + ilam * (NOCE+1)] = 1.;
    }

    if(profil == NULL){
        printf("ERREUR : Ouverture impossible du fichier %s pour le profil oceanique\n", PATHPROFILOCE );
        exit(1);
    }

    else{
        for( ilam=0; ilam<NLAM; ilam++){
           // skip comment line
           fgets(ligne,1024,profil);
           for( icouche=0; icouche<NOCE+1; icouche++ ){
              fscanf(profil, "%d\t%f\t%f\t%f\n", &i, &garbage, tab_H->ho+icouche+ilam*(NOCE+1), tab_H->sso+icouche+ilam*(NOCE+1));
              //printf("%d\t%f\t%f\t%f\n", i, garbage, tab_H->ho[icouche+ilam*(NOCE+1)], tab_H->sso[icouche+ilam*(NOCE+1)]);
           }
        }
    }

	if(fclose(profil) == EOF){
		printf("ERREUR : Probleme de fermeture du fichier %s", PATHPROFILOCE);
	}

	erreur = cudaMemcpy(tab_D->ho, tab_H->ho, (NOCE+1)*NLAM*sizeof(*(tab_H->ho)), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->ho dans profilOce\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	erreur = cudaMemcpy(tab_D->sso, tab_H->sso, (NOCE+1)*NLAM*sizeof(*(tab_H->sso)), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->sso dans profilOce\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
}
#endif



/* profilAtm
* Calcul du profil atmosphérique dans l'atmosphère en fonction de la couche
* Mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
*/
void profilAtm( Tableaux* tab_H, Tableaux* tab_D ){

	/** Déclaration des variables **/
	
	int i=0, ilam;
	cudaError_t erreur;		// Permet de tester le bon déroulement des opérations mémoires
	
	/** Conditions aux limites au sommet de l'atmosphère **/
	#ifdef SPHERIQUE
	tab_H->z[0] = HATM;
	#endif
	for( ilam=0;ilam<NLAM;ilam++) {
        tab_H->h[0 + ilam * (NATM+1)] = 0.;
        tab_H->pMol[0 + ilam * (NATM+1)] = 0.;
      }

    // Profil utilisateur
    /* Format du fichier
    => n	alt		tauMol		tauAer		h		pAer		pMol
    */
    FILE* profil = fopen( PATHPROFILATM , "r" );
    float garbage;
    
    int icouche=0;
    char ligne[1024];

    if(profil == NULL){
        printf("ERREUR : Ouverture impossible du fichier %s pour le profil atmosphérique\n", PATHPROFILATM );
        exit(1);
    }

		
		
    
    else{

        // Extraction des informations
        #if defined(SPHERIQUE) 
        for( ilam=0; ilam<NLAM; ilam++){
         // skip comment line
         fgets(ligne,1024,profil);
         for( icouche=0; icouche<NATM+1; icouche++ ){
            fscanf(profil, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", &i, tab_H->z+icouche, &garbage, &garbage, tab_H->h+icouche+ilam*(NATM+1),
                   &garbage,tab_H->pMol+icouche+ilam*(NATM+1), tab_H->ssa+icouche+ilam*(NATM+1), tab_H->abs+icouche+ilam*(NATM+1) );
         }
        }
        #endif
        #if !defined(SPHERIQUE) 
        for( ilam=0; ilam<NLAM; ilam++){
         // skip comment line
         fgets(ligne,1024,profil);
         for( icouche=0; icouche<NATM+1; icouche++ ){
            fscanf(profil, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", &i, &garbage, &garbage, &garbage, tab_H->h+icouche+ilam*(NATM+1),
                   &garbage,tab_H->pMol+icouche+ilam*(NATM+1), tab_H->ssa+icouche+ilam*(NATM+1), tab_H->abs+icouche+ilam*(NATM+1) );
         }
        }
        TAUATM = tab_H->h[NATM];
        #endif
    }
	
		if(fclose(profil) == EOF){
			printf("ERREUR : Probleme de fermeture du fichier %s", PATHPROFILATM);
		}
		
	
	
		/** Envoie des informations dans le device **/
		erreur = cudaMemcpy(tab_D->h, tab_H->h, (NATM+1)*NLAM*sizeof(*(tab_H->h)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->h dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
		
		erreur = cudaMemcpy(tab_D->pMol, tab_H->pMol, (NATM+1)*NLAM*sizeof(*(tab_H->pMol)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->pMol dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}	

        erreur = cudaMemcpy(tab_D->abs, tab_H->abs, (NATM+1)*NLAM*sizeof(*(tab_H->abs)), cudaMemcpyHostToDevice);
        if( erreur != cudaSuccess ){
            printf( "ERREUR: Problème de copie tab_D->abs dans profilAtm\n");
            printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
            exit(1);
        }

        erreur = cudaMemcpy(tab_D->ssa, tab_H->ssa, (NATM+1)*NLAM*sizeof(*(tab_H->ssa)), cudaMemcpyHostToDevice);
        if( erreur != cudaSuccess ){
            printf( "ERREUR: Problème de copie tab_D->ssa dans profilAtm\n");
            printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
            exit(1);
        }

		#ifdef SPHERIQUE
		erreur = cudaMemcpy(tab_D->z, tab_H->z, (NATM+1)*sizeof(*(tab_H->z)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->z dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
		#endif
	
}


/** Séparation du code pour atmosphère sphérique ou parallèle **/
#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */

/* impactInit
* Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
* Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init
*/
void impactInit(Init* init_H, Init* init_D, Tableaux* tab_H, Tableaux* tab_D){
	
	double thv, localh;
	double rdelta;
	double xphbis,yphbis,zphbis;	//Coordonnées intermédiaire du photon
	double rsolfi,rsol1,rsol2;
	
	// Correspond aux paramètres initiaux du photon
	double vx = -sin(THVDEG*DEG2RAD);
	double vy = 0.;
	double vz = -cos(THVDEG*DEG2RAD);
	
    int ilam;

	/** Calcul du point d'impact **/
	thv = THVDEG*DEG2RAD;
	
	rdelta = 4.*RTER*RTER + 4.*( tan(thv)*tan(thv)+1. )*( HATM*HATM + 2.*HATM*RTER );
	localh = ( -2.*RTER+sqrt(rdelta) )/( 2.*(tan(thv)*tan(thv)+1.) );
	
	init_H->x0 = (float) localh*tan(thv);
	init_H->y0 = 0.f;
	init_H->z0 = (float) RTER + localh;	
	
	tab_H->zph0[0] = 0.;
	for(ilam=0; ilam<NLAM; ilam++){
	   tab_H->hph0[0 + ilam*(NATM+1)] = 0.;
    }
	
	xphbis = init_H->x0;
	yphbis = init_H->y0;
	zphbis = init_H->z0;
	
	/** Création hphoton et zphoton, chemin optique entre sommet atmosphère et sol pour la direction d'incidence **/
	for(int icouche=1; icouche<NATM+1; icouche++){
		
		rdelta = 4.*(vx*xphbis + vy*yphbis + vz*zphbis)*(vx*xphbis + vy*yphbis + vz*zphbis)
			- 4.*(xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - ((double)tab_H->z[icouche]+RTER)*((double)tab_H->z[icouche]+RTER));
		rsol1 = 0.5*( -2.*(vx*xphbis + vy*yphbis + vz*zphbis) + sqrt(rdelta) );
		rsol2 = 0.5*( -2.*(vx*xphbis + vy*yphbis + vz*zphbis) - sqrt(rdelta) );
		
		// solution : la plus petite distance positive
		if(rsol1>0.){
			if( rsol2>0.)
				rsolfi = min(rsol1,rsol2);
			else
				rsolfi = rsol1;
		}
		else{
			if( rsol2>0. )
				rsolfi=rsol2;
		}
		
		tab_H->zph0[icouche] = tab_H->zph0[icouche-1] + (float)rsolfi;
	    for(ilam=0; ilam<NLAM; ilam++){
		    tab_H->hph0[icouche + ilam*(NATM+1)] = tab_H->hph0[icouche-1+ ilam*(NATM+1)] + 
				( abs( tab_H->h[icouche+ ilam*(NATM+1)] - tab_H->h[icouche-1+ ilam*(NATM+1)])*rsolfi )/( abs( tab_H->z[icouche-1] - tab_H->z[icouche]) );
        }
		
		xphbis+= vx*rsolfi;
		yphbis+= vy*rsolfi;
		zphbis+= vz*rsolfi;
		
	}

	//for(ilam=0; ilam<NLAM; ilam++){
	//    init_H->taumax0[ilam] = tab_H->hph0[NATM + ilam*(NATM+1)];
    //}
	//init_H->zintermax0 = tab_H->zph0[NATM];

	
	/** Envoie des données dans le device **/
	cudaError_t erreur = cudaMemcpy(init_D, init_H, sizeof(Init), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf("#--------------------#\n");
		printf("# ERREUR: Problème de copie init_H dans initInit\n");
		printf("# Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		printf("#--------------------#\n");
		exit(1);
	}
	
	erreur = cudaMemcpy(tab_D->hph0, tab_H->hph0, (NATM+1)*NLAM*sizeof(*(tab_H->hph0)), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->hph0 dans initInit\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	erreur = cudaMemcpy(tab_D->zph0, tab_H->zph0, (NATM+1)*sizeof(*(tab_H->zph0)), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->zph0 dans initInit\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
}
#endif


/**********************************************************
*	> Fonctions d'affichage
***********************************************************/

/* afficheParametres
* Affiche les paramètres de la simulation
*/
void afficheParametres()
{
	printf("\n#--------- Paramètres de simulation --------#\n");
	printf(" NBPHOTONS =\t%llu", NBPHOTONS);
	printf("\n");
	printf(" NBTHETA =\t%d", NBTHETA);
	printf("\n");
	printf(" NBPHI\t=\t%d", NBPHI);
	printf("\n");
	printf(" THVDEG\t=\t%f (degrés)", THVDEG);
	printf("\n");
	printf(" LAMBDA\t=\t%f", LAMBDA);
	printf("\n");
	printf(" NLAM\t=\t%d", NLAM);
	printf("\n");
	printf(" SIM\t=\t%d", SIM);
		if( SIM==-2 )
			printf("\t(Atmosphère seule)");
		if( SIM==-1 )
			printf("\t(Dioptre seul)");
		if( SIM==0 )
			printf("\t(Océan et Surface)");
		if( SIM==1 )
			printf("\t(Atmosphère et Surface)");
		if( SIM==2 )
			printf("\t(Atmosphère, Dioptre et Océan)");
		if( SIM==3 )
			printf("\t(Océan seul)");
		
	printf("\n");
	printf(" SEED\t=\t%d", SEED);
	printf("\n");
	
	printf("\n#------- Paramètres de performances --------#\n");
	printf(" NBLOOP\t=\t%u", NBLOOP);
	printf("\n");
	printf(" XBLOCK\t=\t%d", XBLOCK);
	printf("\n");
	printf(" YBLOCK\t=\t%d", YBLOCK);
	printf("\n");
	printf(" XGRID\t=\t%d", XGRID);
	printf("\n");
	printf(" YGRID\t=\t%d", YGRID);
	printf("\n");
	
	
	printf("\n#--------------- Atmosphère ----------------#\n");
	if( SIM==-2 || SIM==1 || SIM==2 ){
		#ifdef SPHERIQUE
		printf(" Géométrie de l'atmosphère: \tSphérique");
		printf("\n");
		#endif
		#ifndef SPHERIQUE
		printf(" Géométrie de l'atmosphère: \tParallèle");
		printf("\n");
		#endif
		
		printf(" LSAAER\t=\t%u", LSAAER);
		printf("\n");
		printf(" NFAER\t=\t%u", NFAER);
		printf("\n");
		printf(" NATM\t=\t%d", NATM);
		printf("\n");
		printf(" HATM\t=\t%f", HATM);
		printf("\n");
	}
	else{
		printf("\tPas de contribution de l'atmosphère\n");
	}
	
	
	printf("\n#--------- Contribution du dioptre ---------#\n");
	if( SIM==-1 || SIM==0 || SIM==1 || SIM==2 ){
		printf(" SUR\t=\t%d", SUR);
		printf("\n");
		printf(" DIOPTRE =\t%d", DIOPTRE);
		printf("\n");
		printf(" WINDSPEED =\t%f", WINDSPEED);
		printf("\n");
	}
	else{
		printf("\tPas de dioptre\n");
	}
   
    printf("\n#--------- Contribution de l'environnement -----#\n");
	if( ENV!=0){
		printf(" ENV_SIZE\t=\t%.1f (km)", ENV_SIZE);
		printf("\n");
		printf(" X0 =\t%.1f (km)", X0);
		printf(" Y0 =\t%.1f (km)", Y0);
		printf("\n");
	}
	else{
		printf("\tPas d'effet d'environnement\n");
	}

	#ifdef FLAGOCEAN
	printf("\n#----------------- Océan ------------------#\n");
	printf(" LSAOCE\t=\t%u", LSAOCE);
	printf("\n");
	printf(" NFOCE\t=\t%u", NFOCE);
	printf("\n");
	printf(" NH2O\t=\t%f", NH2O);
	printf("\n");
	printf(" NOCE\t=\t%d", NOCE);
	printf("\n");
	#endif
	
	printf("\n#----------- Chemin des fichiers -----------#\n");
	printf(" PATHRESULTATSHDF = %s", PATHRESULTATSHDF);
	printf("\n");
	printf(" PATHDIFFAER = %s", PATHDIFFAER);
	printf("\n");
	printf(" PATHPROFILATM = %s", PATHPROFILATM);
	printf("\n");
	printf(" PATHALB = %s", PATHALB);
	printf("\n");
    #ifdef FLAGOCEAN
    printf(" PATHDIFFOCE = %s\n", PATHDIFFOCE);
	printf(" PATHPROFILOCE = %s", PATHPROFILOCE);
	printf("\n");
    #endif
	
	// Calcul la date et l'heure courante
	time_t dateTime = time(NULL);
	struct tm* date = localtime(&dateTime);
	printf("\n  Date de début  : %02u/%02u/%04u %02u:%02u:%02u\n", date->tm_mday, date->tm_mon+1, 1900 + date->tm_year,
		   date->tm_hour, date->tm_min, date->tm_sec);

}


/* afficheProgress
* Affiche la progression de la simulation
*/
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
	printf("  Date actuelle  : %02u/%02u/%04u %02u:%02u:%02u\n", date->tm_mday, date->tm_mon+1, 1900 + date->tm_year, date->tm_hour,
		   date->tm_min, date->tm_sec);
	printf(" --------------------------------------\n");
   
	#ifdef PROGRESSION
	printf("%d%% - ", (int)(100*nbPhotonsTot/NBPHOTONS));
	printf("Temps: %d - ", tempsEcoule);
	printf("phot sortis: %lu - ", nbPhotonsSorTot);
	printf("phot traités: %lu - ", nbPhotonsTot);
	printf("erreur poids/theta/vxy/vy/case: %d/%d/%d/%d/%d", var->erreurpoids, var->erreurtheta, var->erreurvxy,
		var->erreurvy, var->erreurcase);
	printf("\n");
	#endif
}




/**********************************************************
*	> Calcul pour sauvegarde des résultats finaux
***********************************************************/

/* calculOmega
* Fonction qui calcule l'aire normalisée de chaque boite, son theta, et son psi, sous forme de 3 tableaux
*/
void calculOmega(double* tabTh, double* tabPhi, double* tabOmega)
{
	// Tableau contenant l'angle theta de chaque morceau de sphère
	memset(tabTh, 0, NBTHETA * sizeof(*tabPhi));
	double dth = DEMIPI / NBTHETA;
	tabTh[0] = dth/4;
	tabTh[1] = dth;
	for(int ith = 2; ith < NBTHETA; ith++){
		tabTh[ith] = tabTh[ith-1] + dth;
	}
	
	// Tableau contenant l'angle psi de chaque morceau de sphère
	memset(tabPhi, 0, NBPHI * sizeof(*tabPhi));
	double dphi = PI / NBPHI;
 	tabPhi[0] = dphi / 2;
	for(int iphi = 1; iphi < NBPHI; iphi++){ 
		tabPhi[iphi] = tabPhi[iphi-1] + dphi;
	}
	// Tableau contenant l'aire de chaque morceau de sphère
	double sumds = 0;
	double *tabds;
    tabds = (double*)malloc(NBTHETA * NBPHI * sizeof(double));
	memset(tabds, 0, NBTHETA * NBPHI * sizeof(double));
	for(int ith = 0; ith < NBTHETA; ith++)
	{
		if( ith==0 )
			dth = DEMIPI / (2*NBTHETA);
		else 
			dth = DEMIPI / NBTHETA;
			
		for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			tabds[ith * NBPHI + iphi] = sin(tabTh[ith]) * dth * dphi;
			sumds += tabds[ith * NBPHI + iphi];
		}
	}
	
	// La derniere demi boite 89.75->90
	for(int iphi = 0; iphi < NBPHI; iphi++)
		{
			sumds += sin( (DEMIPI+tabTh[NBTHETA-1])/2 ) * (dth/2) * dphi;
		}
	
	// Normalisation de l'aire de chaque morceau de sphère
	memset(tabOmega, 0, NBTHETA * NBPHI * sizeof(*tabOmega));
	for(int ith = 0; ith < NBTHETA; ith++)
		for(int iphi = 0; iphi < NBPHI; iphi++){
			tabOmega[ith * NBPHI + iphi] = tabds[ith * NBPHI + iphi] / sumds;
		}
    
    free(tabds);
}


/* calculTabFinal
* Fonction qui remplit le tabFinal correspondant à la reflectance (R), Q et U sur tous l'espace de sorti (dans chaque boite)
*/
void calculTabFinal(double* tabFinal, double* tabTh, double* tabPhi, double* tabPhotonsTot, unsigned long long nbPhotonsTot,
                   unsigned long long* nbPhotonsTotInter)
{
	
    double norm, normInter;
	double *tabOmega;
    tabOmega = (double*)malloc(NBTHETA * NBPHI * sizeof(double));
	// Remplissage des tableaux tabTh, tabPhi, et tabOmega
	calculOmega(tabTh, tabPhi, tabOmega);
	
	// Remplissage du tableau final
	for(int iphi = 0; iphi < NBPHI; iphi++)
	{
		for(int ith = 0; ith < NBTHETA; ith++)
		{
            norm = 2.0 * tabOmega[ith*NBPHI+iphi] * cos(tabTh[ith]);

            for(int i=0;i<NLAM;i++){
               normInter = norm * nbPhotonsTotInter[i];
			  // Reflectance
			          tabFinal[0*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] =
				(tabPhotonsTot[0*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi] +
				 tabPhotonsTot[1*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi]) / normInter;
			
			  // Q
			          tabFinal[1*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] =
				(tabPhotonsTot[0*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi] -
				 tabPhotonsTot[1*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi]) / normInter;
			
			  // U
			          tabFinal[2*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] =
                (tabPhotonsTot[2*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi]) / normInter;
				
			  // N
			          tabFinal[3*NBTHETA*NBPHI*NLAM + i*NBTHETA*NBPHI + iphi*NBTHETA + ith] =
			    (tabPhotonsTot[3*NBPHI*NBTHETA*NLAM + i*NBTHETA*NBPHI + ith*NBPHI    + iphi])  ;
            }
				
		}
	}
    free(tabOmega);
}



/* creerHDFResultats
* Fonction qui crée le fichier .hdf contenant le résultat final pour une demi-sphère
*/
void creerHDFResultats(double* tabFinal,double* tabFinalDown0P, double* tabFinalDown0M,double* tabFinalUp0P,double* tabFinalUp0M, double* tabTh, double* tabPhi,unsigned long long nbPhotonsTot, Variables* var,double
tempsPrec)
{
	// Tableau temporaire utile pour la suite
	double *tab;
    tab = (double*)malloc(NBPHI*NBTHETA*NLAM*sizeof(double));
    char nomTab[256];

	// Création du fichier de sortie
	int sdFichier = SDstart(PATHRESULTATSHDF, DFACC_CREATE);
	if (sdFichier == FAIL) {
		printf("ERREUR: création du fichier HDF : %s\n", PATHRESULTATSHDF);
		exit(1);
	}
	
	// Ecriture des informations sur la simulation : paramètres, nbphotons, nbErreurs, tempsEcoule
	double NBPHOTONSdouble = (double)NBPHOTONS;
	double nbPhotonsTotdouble = (double)nbPhotonsTot;
	double tempsEcouledouble = tempsPrec + (double)(clock() / CLOCKS_PER_SEC);

    #ifdef SPHERIQUE
	SDsetattr(sdFichier, "MODE", DFNT_CHAR8, 2, "SSA");
    #else
	SDsetattr(sdFichier, "MODE", DFNT_CHAR8, 2, "PPA");
    #endif
	SDsetattr(sdFichier, "NBPHOTONS", DFNT_FLOAT64, 1, &NBPHOTONSdouble);
	SDsetattr(sdFichier, "NBLOOP", DFNT_UINT32, 1, &NBLOOP);
	SDsetattr(sdFichier, "SEED", DFNT_UINT32, 1, &SEED);
	SDsetattr(sdFichier, "XBLOCK", DFNT_INT32, 1, &XBLOCK);
	SDsetattr(sdFichier, "YBLOCK", DFNT_INT32, 1, &YBLOCK);
	SDsetattr(sdFichier, "XGRID", DFNT_INT32, 1, &XGRID);
	SDsetattr(sdFichier, "YGRID", DFNT_INT32, 1, &YGRID);
	SDsetattr(sdFichier, "NBTHETA", DFNT_INT32, 1, &NBTHETA);
	SDsetattr(sdFichier, "NBPHI", DFNT_INT32, 1, &NBPHI);
	SDsetattr(sdFichier, "DIOPTRE", DFNT_INT32, 1, &DIOPTRE);
	SDsetattr(sdFichier, "ENV", DFNT_INT32, 1, &ENV);
	SDsetattr(sdFichier, "SIM", DFNT_INT32, 1, &SIM);
	SDsetattr(sdFichier, "SUR", DFNT_INT32, 1, &SUR);
	SDsetattr(sdFichier, "VZA (deg.)", DFNT_FLOAT32, 1, &THVDEG);
	SDsetattr(sdFichier, "LAMBDA", DFNT_FLOAT32, 1, &LAMBDA);
	SDsetattr(sdFichier, "NLAM", DFNT_INT32, 1, &NLAM);
	SDsetattr(sdFichier, "TAURAY", DFNT_FLOAT32, 1, &TAURAY);
	SDsetattr(sdFichier, "TAUAER", DFNT_FLOAT32, 1, &TAUAER);
	
	SDsetattr(sdFichier, "LSAAER", DFNT_UINT32, 1, &LSAAER);
	SDsetattr(sdFichier, "NFAER", DFNT_UINT32, 1, &NFAER);
	SDsetattr(sdFichier, "LSAOCE", DFNT_UINT32, 1, &LSAOCE);
	SDsetattr(sdFichier, "NFOCE", DFNT_UINT32, 1, &NFOCE);
	
	SDsetattr(sdFichier, "ENV_SIZE", DFNT_FLOAT32, 1, &ENV_SIZE);
	SDsetattr(sdFichier, "X0", DFNT_FLOAT32, 1, &X0);
	SDsetattr(sdFichier, "Y0", DFNT_FLOAT32, 1, &Y0);
	SDsetattr(sdFichier, "NATM", DFNT_INT32, 1, &NATM);
	SDsetattr(sdFichier, "HATM", DFNT_FLOAT32, 1, &HATM);
	SDsetattr(sdFichier, "WINDSPEED", DFNT_FLOAT32, 1, &WINDSPEED);
	SDsetattr(sdFichier, "NH2O", DFNT_FLOAT32, 1, &NH2O);
	SDsetattr(sdFichier, "TRANSDIR", DFNT_FLOAT32, 1, &TRANSDIR);
    #ifdef FLAGOCEAN
	SDsetattr(sdFichier, "NOCE", DFNT_INT32, 1, &NOCE);
    SDsetattr(sdFichier, "PATHDIFFOCE", DFNT_CHAR8, strlen(PATHDIFFOCE), PATHDIFFOCE);
    SDsetattr(sdFichier, "PATHPROFILOCE", DFNT_CHAR8, strlen(PATHPROFILOCE), PATHPROFILOCE);
    #endif
	SDsetattr(sdFichier, "PATHRESULTATSHDF", DFNT_CHAR8, strlen(PATHRESULTATSHDF), PATHRESULTATSHDF);
	SDsetattr(sdFichier, "PATHDIFFAER", DFNT_CHAR8, strlen(PATHDIFFAER), PATHDIFFAER);
	SDsetattr(sdFichier, "PATHPROFILATM", DFNT_CHAR8, strlen(PATHPROFILATM), PATHPROFILATM);
	SDsetattr(sdFichier, "PATHALB", DFNT_CHAR8, strlen(PATHALB), PATHALB);
	
	SDsetattr(sdFichier, "nbPhotonsTot", DFNT_FLOAT64, 1, &nbPhotonsTotdouble);
	SDsetattr(sdFichier, "nbErreursPoids", DFNT_INT32, 1, &(var->erreurpoids));
	SDsetattr(sdFichier, "nbErreursTheta", DFNT_INT32, 1, &(var->erreurtheta));
	SDsetattr(sdFichier, "tempsEcoule", DFNT_FLOAT64, 1, &tempsEcouledouble);
	
	/** 	Création du 1er tableau dans le fichier hdf
		Valeur de la reflectance pour phi et theta donnés		**/
//	char* nomTab="Valeurs de la reflectance (I)"; //nom du tableau
	//char* nomTab="I_up (TOA)"; //nom du tableau
    int nbDimsTab=3;
    sprintf(nomTab, "I_up (TOA)");
	int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
	valDimsTab[2] = NBTHETA;	//colonnes
	valDimsTab[1] = NBPHI;
	valDimsTab[0] = NLAM;
	int typeTab = DFNT_FLOAT64; //type des éléments du tableau
	
	// Création du tableau
	int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	int startTab[nbDimsTab]; //début de la lecture du tableau
	startTab[0]=0;
	startTab[1]=0;
    startTab[2]=0;
	// Ecriture du tableau dans le fichier
	int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabFinal);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats reflectance\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	
	/** 	Création du tableau Q dans le fichier hdf
		Valeur de Q pour phi et theta donnés		**/
    sprintf(nomTab, "Q_up (TOA)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinal+NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats Q\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	
	/** 	Création du tableau U dans le fichier hdf
	Valeur de U pour phi et theta donnés		**/
    sprintf(nomTab, "U_up (TOA)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinal+2*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

	/** 	Création du tableau de lumière polarisée dans le fichier hdf
	Valeur de la lumière polarisée pour phi et theta donnés		**/
    sprintf(nomTab, "LP_up (TOA)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	for(int i = 0; i < NBTHETA*NBPHI*NLAM; i++){
		tab[i] = sqrt( tabFinal[1*NBTHETA*NBPHI*NLAM+i]*tabFinal[1*NBTHETA*NBPHI*NLAM+i] +
						tabFinal[2*NBTHETA*NBPHI*NLAM+i]*tabFinal[2*NBTHETA*NBPHI*NLAM+i] );
	}
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) tab );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats lumiere polarisee\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

    // flux descendant 0+
    sprintf(nomTab, "I_down (0+)");
	nbDimsTab = 3; //nombre de dimensions du tableau
	valDimsTab[2] = NBTHETA;	//colonnes
	valDimsTab[1] = NBPHI;
	valDimsTab[0] = NLAM;
	typeTab = DFNT_FLOAT64; //type des éléments du tableau
	
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	startTab[0]=0;
	startTab[1]=0;
	startTab[2]=0;
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabFinalDown0P);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats reflectance\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	
    // flux descendant 0-
    sprintf(nomTab, "I_down (0-)");
	nbDimsTab = 3; //nombre de dimensions du tableau
	valDimsTab[2] = NBTHETA;	//colonnes
	valDimsTab[1] = NBPHI;
	valDimsTab[0] = NLAM;
	typeTab = DFNT_FLOAT64; //type des éléments du tableau
	
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	startTab[0]=0;
	startTab[1]=0;
	startTab[2]=0;
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabFinalDown0M);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats reflectance\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

    // flux ascendant 0+
    sprintf(nomTab, "I_up (0+)");
	nbDimsTab = 3; //nombre de dimensions du tableau
	valDimsTab[2] = NBTHETA;	//colonnes
	valDimsTab[1] = NBPHI;
	valDimsTab[0] = NLAM;
	typeTab = DFNT_FLOAT64; //type des éléments du tableau
	
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	startTab[0]=0;
	startTab[1]=0;
	startTab[2]=0;
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabFinalUp0P);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats reflectance\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	
    // flux asscendant 0-
    sprintf(nomTab, "I_up (0-)");
	nbDimsTab = 3; //nombre de dimensions du tableau
	valDimsTab[2] = NBTHETA;	//colonnes
	valDimsTab[1] = NBPHI;
	valDimsTab[0] = NLAM;
	typeTab = DFNT_FLOAT64; //type des éléments du tableau
	
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	startTab[0]=0;
	startTab[1]=0;
	startTab[2]=0;
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabFinalUp0M);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats reflectance\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);


	/** 	Création du tableau Q dans le fichier hdf
		Valeur de Q pour phi et theta donnés		**/
    sprintf(nomTab, "Q_down (0+)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalDown0P+NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats Q\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	

    sprintf(nomTab, "Q_down (0-)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalDown0M+NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats Q\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	
    sprintf(nomTab, "Q_up (0+)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalUp0P+NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats Q\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	

    sprintf(nomTab, "Q_up (0-)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalUp0M+NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats Q\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	

	/** 	Création du tableau U dans le fichier hdf
	Valeur de U pour phi et theta donnés		**/
    sprintf(nomTab, "U_down (0+)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalDown0P+2*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

    sprintf(nomTab, "U_down (0-)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalDown0M+2*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

    sprintf(nomTab, "U_up (0+)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalUp0P+2*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);

    sprintf(nomTab, "U_up (0-)");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinalUp0M+2*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);



	/** 	Création du tableau N dans le fichier hdf
	Valeur de N pour phi et theta donnés		**/
    sprintf(nomTab, "Numbers of photons");
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinal+3*NBPHI*NBTHETA*NLAM) );
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats U\n");
		exit(1);
	}
	
	// Fermeture du tableau
	SDendaccess(sdsTab);
	

	
	/** 	Création du tableau theta
		Valeurs de theta en fonction de l'indice	**/
	//conversion en degrès de theta pour une meilleure visualisation de la sortie
	float *tabThBis;
    tabThBis = (float*)malloc(NBTHETA*sizeof(float));
	for(int i=0; i<NBTHETA; i++)
		tabThBis[i] = tabTh[i]/DEG2RAD;
	
    sprintf(nomTab, "Zenith angles");
	nbDimsTab = 1;
	int valDimsTab2[nbDimsTab];
	valDimsTab2[0] = NBTHETA;
	typeTab = DFNT_FLOAT32;
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab2);
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab2, (VOIDP) tabThBis);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats - tab Theta\n");
		exit(1);
	}
			
	// Fermeture du tableau
    free(tabThBis);
	SDendaccess(sdsTab);
	
	/** 	Création du tableau phi
		Valeurs de phi en fonction de l'indice	**/
	float *tabPhiBis;
    tabPhiBis = (float*)malloc(NBPHI*sizeof(float));
	for(int i=0; i<NBPHI; i++)
		tabPhiBis[i] = tabPhi[i]/DEG2RAD;
	
    sprintf(nomTab, "Azimut angles");
	nbDimsTab = 1;
	int valDimsTab3[nbDimsTab];
	valDimsTab3[0] = NBPHI;
	typeTab = DFNT_FLOAT32;
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab3);
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab3, (VOIDP)tabPhiBis);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats - tab Phi\n");
		exit(1);
	}
	
	// Fermeture du tableau
    free(tabPhiBis);
	SDendaccess(sdsTab);
	
	/** 	Création du tableau lambda
		Valeurs de lambda en fonction de l'indice	**/
	
	double* tabLamBis;
    tabLamBis = (double *)malloc(NLAM * sizeof(*(tabLamBis)));
	for(int i=0; i<NLAM; i++)
		tabLamBis[i] = i;

	sprintf(nomTab, "Valeurs de Lambda");
	nbDimsTab = 1;

	int valDimsTab4[nbDimsTab];
	valDimsTab4[0] = NLAM;

	typeTab = DFNT_FLOAT64;
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab4);
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab4, (VOIDP) tabLamBis);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf resultats - tab Lam\n");
		exit(1);
	}
	// Fermeture du fichier
	SDend(sdFichier);
	
    free(tab);
}
