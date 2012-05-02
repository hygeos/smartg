
/**********************************************************
*	> Includes
***********************************************************/

#include "communs.h"
#include "host.h"
#include "device.h"


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
	

	if( SIM!=-2 ) DIFFF = 0;
	else{
		strcpy(s,"");
		chercheConstante(parametres, "DIFFF", s);
		DIFFF = atoi(s);
	}
	
	if( argc>2){ // Il est possible de rentrer theta à la main, utile pour debug et boucle shell
		THSDEG = atof(argv[2]);
	}
	else{
		strcpy(s,"");
		chercheConstante(parametres, "THSDEG", s);
		THSDEG = atof(s);
	}
	
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
	chercheConstante(parametres, "W0LAM", s);
	W0LAM = atof(s);
	
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
	chercheConstante(parametres, "NATM", s);
	NATM = atoi(s);
	
	strcpy(s,"");
	chercheConstante(parametres, "HATM", s);
	HATM = atoi(s);
	
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
	// Remplir automatiquement le nom complet du fichier
	definirNomFichier(s);
// 	chercheConstante(parametres, "PATHTEMOINHDF", PATHTEMOINHDF);

	chercheConstante( parametres, "PATHDIFFAER", PATHDIFFAER );
	
	chercheConstante( parametres, "PATHPROFILATM", PATHPROFILATM );
	
	fclose( parametres );
}


/* chercheConstante
* Fonction qui cherche nomConstante dans le fichier et met la valeur de la constante dans chaineValeur (en string)
*/
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


/* definirNomFichier
* Le nom du fichier de sorti est créé automatiquement en fonction du type de simulation
* Il est également stoké dans un dossier en fonction de la date est du type de simulation
* Le chemin indiqué dans le fichier paramètres est le préfixe du chemin créé ici
*/
void definirNomFichier( char* s ){
	
	// Type de simulation
	definirSimulation(s);
	sprintf(PATHRESULTATSHDF,"%s/%s",PATHRESULTATSHDF,s);
	
	// Date de la simulation
	time_t dateTime = time(NULL);
	struct tm* date = localtime(&dateTime);
	sprintf(PATHRESULTATSHDF,"%s/simulation_%02u%02u%04u/",PATHRESULTATSHDF,date->tm_mday, date->tm_mon+1, 1900 + date->tm_year);
	
	// Création du dossier de stockage du résultat
	strcpy(s,"mkdir -p ");
	strcat(s,PATHRESULTATSHDF);
	system(s);
	
	// Création du dossier de stockage du temoin
	strcpy(PATHTEMOINHDF,PATHRESULTATSHDF);
	strcat(PATHTEMOINHDF,"tmp/");
	strcpy(s,"mkdir -p ");
	strcat(s,PATHTEMOINHDF);
	system(s);
	strcat(PATHTEMOINHDF,"tmp_");

	// Nom du fichier de sortie le plus compréhensible et utile possible
	if( SIM==1 ){
		if( DIOPTRE==3 )
		sprintf(s,"out_CUDA_atmos_dioptre_lambertien_ths=%4.2f_tRay=%4.4f_tAer=%4.4f_ws=%3.2f.hdf",THSDEG,TAURAY,TAUAER,WINDSPEED);
		else if( DIOPTRE==2 || DIOPTRE==1 )
			sprintf(s,"out_CUDA_atmos_dioptre_agite_ths=%4.2f_tRay=%4.4f_tAer=%4.4f_ws=%3.2f.hdf",THSDEG,TAURAY,TAUAER,WINDSPEED);
		else
			sprintf(s,"out_CUDA_atmos_dioptre_plan_ths=%4.2f_tRay=%4.4f_tAer=%4.4f.hdf",THSDEG,TAURAY,TAUAER);
	}
	else if( SIM==-1 ){
		if( DIOPTRE==3 )
			sprintf(s,"out_CUDA_dioptre_lambertien_ths=%4.2f_ws=%3.2f.hdf",THSDEG,WINDSPEED);
		else if( DIOPTRE==2 || DIOPTRE==1 )
			sprintf(s,"out_CUDA_dioptre_agite_ths=%4.2f_ws=%3.2f.hdf",THSDEG,WINDSPEED);
		else
			sprintf(s,"out_CUDA_dioptre_plan_ths=%4.2f_ws=%3.2f.hdf",THSDEG,WINDSPEED);
	}
	else if( SIM==-2 )
		sprintf(s,"out_CUDA_atmos_ths=%4.2f_tRay=%4.4f_tAer=%4.4f.hdf",THSDEG,TAURAY,TAUAER);
	else
		sprintf(s,"out_CUDA_ths=%4.2f_tRay=%4.4f_tAer=%4.4f_ws=%3.2f_sim=%d.hdf",THSDEG,TAURAY,TAUAER,WINDSPEED,SIM);
	
	strcat( PATHTEMOINHDF, s);
	strcat(PATHRESULTATSHDF,s);
	
}


/* definirSimulation
* Défini le type de simulation pour la création du chemin et le nom du fichier résultat
*/
void definirSimulation( char* s){

	if( SIM==1 ){
		if( TAUAER==0 && TAURAY!=0){
			if( DIOPTRE==3 )
				sprintf(s,"molecules_dioptre_lambertien");
			else if( DIOPTRE==2 || DIOPTRE==1 )
				sprintf(s,"molecules_dioptre_agite");
			else
				sprintf(s,"molecules_dioptre_plan");
		}
		else if( TAUAER!=0 && TAURAY==0){
			if( DIOPTRE==3 )
				sprintf(s,"aerosols_dioptre_lambertien");
			else if( DIOPTRE==2 || DIOPTRE==1 )
				sprintf(s,"aerosols_dioptre_agite");
			else
				sprintf(s,"aerosols_dioptre_plan");
		}
		else{
			if( DIOPTRE==3 )
				sprintf(s,"atmos_dioptre_lambertien");
			else if( DIOPTRE==2 || DIOPTRE==1 )
				sprintf(s,"atmos_dioptre_agite");
			else
				sprintf(s,"atmos_dioptre_plan");
		}
	}
	
	else if( SIM==-1 ){
		if( DIOPTRE==3 )
			sprintf(s,"dioptre_lambertien");
		else if( DIOPTRE==2 || DIOPTRE==1 )
			sprintf(s,"dioptre_agite");
		else
			sprintf(s,"dioptre_plan");
	}
	
	else if( SIM==-2 ){
		if( TAUAER==0 && TAURAY!=0)
			sprintf(s,"molecules_seules");
		else if( TAURAY==0 && TAUAER!=0)
			sprintf(s,"aerosols_seuls");
		else
			sprintf(s,"atmos_seule");
	}
	
	else
		sprintf(s,"SIM=%d",SIM);
}


/* verifierFichier
* Fonction qui vérifie l'état des fichiers temoin et résultats
* Demande à l'utilisateur s'il veut les supprimer ou non
*/
void verifierFichier(){
	char command[256];
	char res_supp;
	// S'il existe déjà un fichier nommé NOMRESULTATSHDF (Parametres.txt) on arrête le programme
	FILE* fic;
	fic = fopen(PATHTEMOINHDF, "rb");
	if ( fic != NULL)
	{
		printf("ATTENTION: Le fichier temoin %s existe deja.\n",PATHTEMOINHDF);
		printf("Voulez-vous le supprimer? [y/n]\n");
		res_supp=getchar();
		if( res_supp=='y' ){
			sprintf(command,"rm %s",PATHTEMOINHDF);
			system(command);
		}
		getchar();
		fclose(fic);
	}
	
	
	fic = fopen(PATHRESULTATSHDF, "rb");
	if ( fic != NULL)
	{
		printf("ATTENTION: Le fichier resultat %s existe deja.\n",PATHRESULTATSHDF);
		printf("Voulez-vous le supprimer pour continuer? [y/n]\n");
		// 		res_supp=getchar();
		// 		if( res_supp=='y' ){
//    sprintf(command,"rm %s",PATHRESULTATSHDF);
//    system(command);
   // 		}
	   fclose(fic);
	}
	
	
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
	
	// Initialisation de la version device des variables
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
	
	
	// var_H est une variable page-locked accessible par le device
// 	if( cudaHostAlloc( var_H, sizeof(Variables), cudaHostAllocPortable ) != cudaSuccess ){
// 		printf("#--------------------#\n");
// 		printf("ERREUR: Problème d'allocation de var_H dans initVariables\n");
// 		printf("#--------------------#\n");
// 		exit(1);
// 	}
	
	// Un pointeur est associé pour travailler sur le device
// 	err = cudaHostGetDevicePointer( var_D, *(var_H) ,0); 
// 	if( err != cudaSuccess ){
// 		printf("#--------------------#\n");
// 		printf("ERREUR: Problème de mappage de var_D dans initVariables\n");
// 		printf("#--------------------#\n");
// 		exit(1);
// 	}

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
// 		Initialisation de la version host des variables
	*init_H = (Init*)malloc(sizeof(Init));
	if( init_H == NULL ){
	printf("#--------------------#\n");
	printf("ERREUR: Problème de malloc de init_H dans initInit\n");
	printf("#--------------------#\n");
	exit(1);
   	}
   	memset(*init_H, 0, sizeof(Init));
   
   // Initialisation de la version device des variables
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
   
   
   
//    // var_H est une variable page-locked accessible par le device
//    if( cudaHostAlloc( var_H, sizeof(Variables), cudaHostAllocPortable ) != cudaSuccess ){
// 	   printf("#--------------------#\n");
// 	   printf("ERREUR: Problème d'allocation de var_H dans initVariables\n");
// 	   printf("#--------------------#\n");
// 	   exit(1);
//    }
   
   // Un pointeur est associé pour travailler sur le device
   // 	err = cudaHostGetDevicePointer( var_D, *(var_H) ,0); 
   // 	if( err != cudaSuccess ){
   // 		printf("#--------------------#\n");
   // 		printf("ERREUR: Problème de mappage de var_D dans initVariables\n");
   // 		printf("#--------------------#\n");
   // 		exit(1);
   // 	}
   
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
	initRandMWC(tab_H->etat, tab_H->config, XBLOCK * YBLOCK * XGRID * YGRID, "MWC.txt", (unsigned long long)SEED);
	
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
	
	#ifdef RANDCUDA
	// Création du tableau de generateurs (=etat+config) pour la fonction Random Cuda
	if( cudaMalloc(&(tab_D->etat), XBLOCK * YBLOCK * XGRID * YGRID * sizeof(curandState_t)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->etat dans initTableaux\n");
		exit(1);	
	}
	
	// Initialisation du tableau dans une fonction du kernel
	initRandCUDA<<<XGRID * YGRID, XBLOCK * YBLOCK>>>(tab_D->etat, (unsigned long long)SEED);
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
	
	//if( cudaHostAlloc( &(tab_H->tabPhotons), 4*NBTHETA*NBPHI*sizeof(*(tab_H->tabPhotons)), cudaHostAllocPortable ) !=cudaSuccess){
	// 		printf("#--------------------#\n");
	// 		printf("ERREUR: Problème d'allocation de tab_H->tabPhotons dans initTableaux\n");
	// 		printf("#--------------------#\n");
	// 		exit(1);
	// 	}
	
	// Tableau du poids des photons ressortis
	tab_H->tabPhotons = (float*)malloc(4*NBTHETA * NBPHI * sizeof(*(tab_H->tabPhotons)));
	if( tab_H->tabPhotons == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->tabPhotons dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->tabPhotons,0,4*NBTHETA * NBPHI * sizeof(*(tab_H->tabPhotons)) );
	
	if( cudaMalloc(&(tab_D->tabPhotons), 4 * NBTHETA * NBPHI * sizeof(*(tab_D->tabPhotons))) == cudaErrorMemoryAllocation){
		printf("ERREUR: Problème de cudaMalloc de tab_D->tabPhotons dans initTableaux\n");
		exit(1);	
	}
	
	cudaErreur = cudaMemset(tab_D->tabPhotons, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D->tabPhotons)));
	if( cudaErreur != cudaSuccess ){
	printf("#--------------------#\n");
	printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotons dans le initTableaux\n");
	printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
	printf("#--------------------#\n");
	exit(1);
}
	
	
	// Modèle de diffusion des aérosols
	tab_H->faer = (float*)malloc(5 * NFAER * sizeof(float));
	if( tab_H->faer == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->faer dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->faer,0,5 * NFAER*sizeof(float) );
	
	if( cudaMalloc(&(tab_D->faer), 5 * NFAER * sizeof(float)) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->faer dans initTableaux\n");
		exit(1);	
	}
	
	/** Modèle de l'atmosphère **/
	// Epaisseur optique par couche
	tab_H->h =  (float*)malloc((NATM+1)*sizeof(*(tab_H->h)));
	if( tab_H->h == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->h dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->h,0,(NATM+1)*sizeof(*(tab_H->h)) );
	
	if( cudaMalloc( &(tab_D->h), (NATM+1)*sizeof(*(tab_H->h)) ) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->h dans initTableaux\n");
		exit(1);	
	}
	
	// Proportion moléculaire par couche
	tab_H->pMol =  (float*)malloc((NATM+1)*sizeof(float));
	if( tab_H->pMol == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->pMol dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->pMol,0,(NATM+1)*sizeof(float) );
	
	if( cudaMalloc( &(tab_D->pMol), (NATM+1)*sizeof(float) ) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->pMol dans initTableaux\n");
		exit(1);	
	}
	
	// Altitude des couches
	tab_H->z =  (float*)malloc((NATM+1)*sizeof(*(tab_H->z)));
	if( tab_H->z == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->z dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->z,0,(NATM+1)*sizeof(*(tab_H->z)) );
	
	if( cudaMalloc( &(tab_D->z), (NATM+1)*sizeof(*(tab_H->z)) ) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->z dans initTableaux\n");
		exit(1);	
	}
	
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	/** Profil initial vu par le photon **/
	tab_H->zph0 =  (float*)malloc((NATM+1)*sizeof(*(tab_H->zph0)));
	if( tab_H->zph0 == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->zph0 dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->zph0,0,(NATM+1)*sizeof(*(tab_H->zph0)) );
	
	if( cudaMalloc( &(tab_D->zph0), (NATM+1)*sizeof(*(tab_D->zph0)) ) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->zph0 dans initTableaux\n");
		exit(1);	
	}
	
	tab_H->hph0 =  (float*)malloc((NATM+1)*sizeof(*(tab_H->hph0)));
	if( tab_H->hph0 == NULL ){
		printf("ERREUR: Problème de malloc de tab_H->hph0 dans initTableaux\n");
		exit(1);
	}
	memset(tab_H->hph0,0,(NATM+1)*sizeof(*(tab_H->hph0)) );
	
	if( cudaMalloc( &(tab_D->hph0), (NATM+1)*sizeof(*(tab_D->hph0)) ) == cudaErrorMemoryAllocation ){
		printf("ERREUR: Problème de cudaMalloc de tab_D->hph0 dans initTableaux\n");
		exit(1);	
	}
	#endif
	
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
	
	#ifdef RANDCUDA
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
	
	// Liberation du tableau du poids des photons
	erreur = cudaFree(tab_D->tabPhotons);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->tabPhotons dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	// 	cudaFreeHost(tab_H->tabPhotons);
	free(tab_H->tabPhotons);
	
	// Libération du modèle de diffusion des aérosols
	erreur = cudaFree(tab_D->faer);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->faer dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	free(tab_H->faer);
	
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
	erreur = cudaFree(tab_D->z);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de cudaFree de tab_D->z dans freeTableaux\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
	free(tab_H->z);
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
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
	
}


/**********************************************************
*	> Calculs de profils
***********************************************************/

/* calculFaer
* Calcul de la fonction de phase des aérosols
*/
void calculFaer( const char* nomFichier, Tableaux* tab_H, Tableaux* tab_D ){
	
	FILE* fichier = fopen(nomFichier, "r");

	double *scum = (double*) malloc(LSAAER*sizeof(*scum));
	if( scum==NULL ){
		printf("ERREUR: Problème de malloc de scum dans calculFaer\n");
		exit(1);
	}
	
	scum[0] = 0;
	int iang = 0, ipf = 0;
	double dtheta, pm1, pm2, sin1, sin2;
	double z, norm;

	/** Allocation de la mémoire des tableaux contenant les données **/
	double *ang;
	double *p1, *p2, *p3, *p4;
	ang = (double*) malloc(LSAAER*sizeof(*ang));
	p1 = (double*) malloc(LSAAER*sizeof(*p1));
	p2 = (double*) malloc(LSAAER*sizeof(*p2));
	p3 = (double*) malloc(LSAAER*sizeof(*p3));
	p4 = (double*) malloc(LSAAER*sizeof(*p4));
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
		for(iang=0; iang<LSAAER; iang++){
			fscanf(fichier, "%lf\t%lf\t%lf\t%lf\t%lf", ang+iang,p1+iang,p2+iang,p3+iang,p4+iang );
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
// 		printf("scum[%d]=%10.10lf\n",iang,scum[iang] );
// 		if( scum[iang] == 1 )
// 			printf("Egal 1, iang=%d\n",iang);
	}
	
	/** Calcul des faer **/
	for(iang=0; iang<NFAER-1; iang++){
		z = double(iang+1)/double(NFAER);
// 		ipf=0;	// NOTE: Surement inutile
		while( (scum[ipf+1]<z) && ipf<(LSAAER-1) )
			ipf++;
		
		tab_H->faer[iang*5+4] = float( ((scum[ipf+1]-z)*ang[ipf] + (z-scum[ipf])*ang[ipf+1])/(scum[ipf+1]-scum[ipf]) );
		norm = p1[ipf]+p2[ipf];			// Angle
		tab_H->faer[iang*5+0] = float( p1[ipf]/norm );	// I paralèlle
		tab_H->faer[iang*5+1] = float( p2[ipf]/norm );	// I perpendiculaire
		tab_H->faer[iang*5+2] = float( p3[ipf]/norm );	// u
		tab_H->faer[iang*5+3] = 0.F;			// v, toujours nul
	}
	
	tab_H->faer[(NFAER-1)*5+4] = PI;
	tab_H->faer[(NFAER-1)*5+0] = 0.5F+00;
	tab_H->faer[(NFAER-1)*5+1] = 0.5F+00;
	tab_H->faer[(NFAER-1)*5+2] = float( p3[LSAAER-1]/(p1[LSAAER-1]+p2[LSAAER-1]) );
	tab_H->faer[(NFAER-1)*5+3] = 0.F+00;
	
	free(scum);
	free(ang);
	free(p1);
	free(p2);
	free(p3);
	free(p4);
	
	/** Allocation des FAER dans la device memory **/		

	cudaError_t erreur = cudaMemcpy(tab_D->faer, tab_H->faer, 5*NFAER*sizeof(*(tab_H->faer)), cudaMemcpyHostToDevice); 
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_D->faer dans calculFaer\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
	
}


/* verificationFAER
* Sauvegarde la fonction de phase des aérosols calculée dans un fichier
* Permet de valider le bon calcul de la fonction de phase
*/
void verificationFAER( const char* nomFichier, Tableaux tab){

	FILE* fichier = fopen(nomFichier, "w");
	int i;
	
	fprintf( fichier, "angle\tI//\tIp\n" );
	
	for(i=0; i<NFAER; i++){
		fprintf(fichier, "%f\t%f\t%f\n", tab.faer[i*5+4],tab.faer[i*5+0], tab.faer[i*5+1]);
	}
	
	fclose(fichier);

}


/* profilAtm
* Calcul du profil atmosphérique dans l'atmosphère en fonction de la couche
* Mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
*/
void profilAtm( Tableaux* tab_H, Tableaux* tab_D ){

	/** Déclaration des variables **/
	/*NOTE: différence avec le code fortran: je n'utilise pas int ncouche */
	
	float tauMol[NATM+1];	// Epaisseur optique des molécules à chaque couche
	float tauAer[NATM+1];	// Epaisseur optique des aérosols à chaque couche
	int i=0;
	float va=0, vr=0;	// Variables tampons
	cudaError_t erreur;	// Permet de tester le bon déroulement des opérations mémoires
	
	/** Conditions aux limites au sommet de l'atmosphère **/
	tab_H->z[0] = HATM;
	tauMol[0] = 0.0;
	tauAer[0] = 0.0;
	tab_H->h[0] = 0.0;
	tab_H->pMol[0] = 0.0;	//Je n'utilise pas la proportion d'aérosols car on l'obtient par 1-PMOL

	/** Cas Particuliers **/
	// Épaisseur optique aérosol très faible OU Épaisseur optique moléculaire et aérosol très faible
	// On ne considère une seule sous-couche dans laquelle on trouve toutes les molécules
	if( /*(TAUAER < 0.0001) ||*/ ((TAUAER < 0.0001)&&(TAURAY < 0.0001)) ){
		tauMol[1] = TAURAY;
		tauAer[1] = 0;
		tab_H->z[1]=0;
		tab_H->h[1] = tauMol[1] + tauAer[1];
		tab_H->pMol[1] = 1.0;
		
		/** Envoie des informations dans le device **/
		erreur = cudaMemcpy(tab_D->h, tab_H->h, (NATM+1)*sizeof(*(tab_H->h)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->h dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
		
		erreur = cudaMemcpy(tab_D->pMol, tab_H->pMol, (NATM+1)*sizeof(*(tab_H->pMol)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->pMol dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}		
		
		erreur = cudaMemcpy(tab_D->z, tab_H->z, (NATM+1)*sizeof(*(tab_H->z)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->z dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
		return;
	}
	
	/** Profil standard avec échelle de hauteur **/
	if( PROFIL == 0 ){
		
		/* Si HA << HR => pas de mélange dans les couches
		On considere alors une atmosphere divisee en deux sous-couches, la  couche superieure contenant toutes les molecules, la
couche inferieure contenant tous les aerosols.
		*/
		if( HA < 0.0001 ){
			tauMol[1] = TAURAY;
			tauAer[1] = 0;
			tab_H->z[1]=0.f;
			tab_H->h[1] = tauMol[1] + tauAer[1];
			tab_H->pMol[1] = 1.0;
			
			tauMol[2] = 0;
			tauAer[2] = TAUAER;
			tab_H->z[2]=0.f;
			tab_H->h[2] = tab_H->h[1] + tauMol[2] + tauAer[2];
			tab_H->pMol[2] = 0.0;
		}
		
		/* Si HA >> HR => pas de mélange dans les couches
		On considere alors une atmosphere divisee en deux sous-couches, la  couche superieure contenant tous les aérosols, la couche
inferieure contenant toutes les molécules.
		*/
		else if( HA > 499.99 ){
			tauMol[1] = 0.0;
			tauAer[1] = TAUAER;
			tab_H->z[1]=0.f;
			tab_H->h[1] = tauMol[1] + tauAer[1];
			tab_H->pMol[1] = 0.0;
			
			tauMol[2] = TAURAY;
			tauAer[2] = 0.0;
			tab_H->z[2]=0.f;
			tab_H->h[2] = tab_H->h[1] + tauMol[2] + tauAer[2];
			tab_H->pMol[2] = 1.0;
		}
		
		/* Cas Standard avec deux échelles */
		else{
			for( i=0; i<NATM+1; i++){
				if(i!=0){
					tab_H->z[i]=100.F - float(i)*(100.F/NATM);
				}
				vr = TAURAY*exp( -(tab_H->z[i]/HR) );
				va = TAUAER*exp( -(tab_H->z[i]/HA) );
				
				tab_H->h[i] = va+vr;
				
				vr = vr/HR;
				va = va/HA;
				vr = vr/(va+vr);
				tab_H->pMol[i] = vr;
			}
			tab_H->h[0] = 0;
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
		tab_H->z[1]=-( HR*log(tauRay1/TAURAY) );
		tauMol[1] = tauRay1;
		tauAer[1] = 0.F;                                    
		tab_H->h[1] = tauMol[1] + tauAer[1];
		tab_H->pMol[1] = 1.F;

		/** Calcul des grandeurs utiles aux OS pour la deuxieme couche   **/
		if( ZMAX == ZMIN ){ //Uniquement des aerosols dans la couche intermediaire
			tab_H->z[2]=ZMAX;
			tauMol[2] = tauRay1;                                                      
			tauAer[2] = TAUAER;
			tab_H->h[2] = tauMol[2] + tauAer[2];
			tab_H->pMol[2] = 0.F;                                                      
		}
		
		else{	// Melange homogene d'aerosol et de molecules dans la couche intermediaire
			tab_H->z[2]=ZMIN;
			tauMol[2] = tauRay1+tauRay2;
			tauAer[2] = TAUAER;
			tab_H->h[2] = tauMol[2] + tauAer[2];
			tab_H->pMol[2] = 0.5F;
		}
		
		/** Calcul des grandeurs utiles aux OS pour la troisieme couche **/
		tab_H->z[3]=0.f;
		tauMol[3] = TAURAY;
		tauAer[3] = TAUAER;
		tab_H->h[3] = tauMol[3] + tauAer[3];
		tab_H->pMol[3] = 1.F;
	}
	
	else if( PROFIL == 2 ){
		// Profil utilisateur
		/* Format du fichier
		=> Ne pas mettre de ligne vide sur la première
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
			// Passage de la premiere ligne
			fgets(ligne,1024,profil);

			// Extraction des informations
			for( icouche=0; icouche<NATM+1; icouche++ ){
				fscanf(profil, "%d\t%f\t%f\t%f\t%f\t%f\t%f", &i, tab_H->z+icouche, &garbage, &garbage, tab_H->h+icouche,
&garbage,tab_H->pMol+icouche );
			
			}
		}
	
		if(fclose(profil) == EOF){
			printf("ERREUR : Probleme de fermeture du fichier %s", PATHPROFILATM);
		}
		
	}
	
	
		/** Envoie des informations dans le device **/
		erreur = cudaMemcpy(tab_D->h, tab_H->h, (NATM+1)*sizeof(*(tab_H->h)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->h dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
		
		erreur = cudaMemcpy(tab_D->pMol, tab_H->pMol, (NATM+1)*sizeof(*(tab_H->pMol)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->pMol dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}	
		
		erreur = cudaMemcpy(tab_D->z, tab_H->z, (NATM+1)*sizeof(*(tab_H->z)), cudaMemcpyHostToDevice);
		if( erreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_D->z dans profilAtm\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
			exit(1);
		}
	
}


/* verificationAtm
* Sauvegarde du profil atmosphérique dans un fichier
* Permet de valider le bon calcul
*/
void verificationAtm( Tableaux tab_H ){
	
	// Vérification du modèle
	FILE* fichier = fopen("./test/modele_atm_cuda.txt", "w+");
	
	fprintf( fichier, "couche\tz\tpropMol\th\n" );
	
	for( int i=0; i<NATM+1; i++){
		fprintf(fichier, "%d\t%10.8f\t%10.8f\t%10.8f\n",i,tab_H.z[i],tab_H.pMol[i], tab_H.h[i]);
	}
	
	fprintf( fichier, "couche\tz\tpropMol\th\n" );
	
	fclose(fichier);
}

/** Séparation du code pour atmosphère sphérique ou parallèle **/
#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */

/* impactInit
* Calcul du profil que le photon va rencontrer lors de son premier passage dans l'atmosphère
* Sauvegarde de ce profil dans tab et sauvegarde des coordonnées initiales du photon dans init
*/
void impactInit(Init* init_H, Init* init_D, Tableaux* tab_H, Tableaux* tab_D){
	
	double thss, localh;
	double rdelta;
	double xphbis,yphbis,zphbis;	//Coordonnées intermédiaire du photon
	double rsolfi,rsol1,rsol2;
	
	// Correspond aux paramètres initiaux du photon
	double vx = -sin(THSDEG*DEG2RAD);
	double vy = 0.;
	double vz = -cos(THSDEG*DEG2RAD);
	
	/** Calcul du point d'impact **/
	// 	thss = abs(acosf(abs(vz)));
	thss = THSDEG*DEG2RAD;
	
	rdelta = 4.*RTER*RTER + 4.*( tan(thss)*tan(thss)+1. )*( HATM*HATM + 2.*HATM*RTER );
	localh = ( -2.*RTER+sqrt(rdelta) )/( 2.*(tan(thss)*tan(thss)+1.) );
	
	init_H->x0 = localh*tan(thss);
	init_H->y0 = 0.;
	init_H->z0 = RTER + localh;	
	
	tab_H->zph0[0] = 0.;
	tab_H->hph0[0] = 0.;
	
	xphbis = init_H->x0;
	yphbis = init_H->y0;
	zphbis = init_H->z0;
	
	/** Création hphoton et zphoton, chemin optique entre sommet atmosphère et sol pour la direction d'incidence **/
	for(int icouche=1; icouche<NATM+1; icouche++){
		
		rdelta = 4.*(vx*xphbis + vy*yphbis + vz*zphbis)*(vx*xphbis + vy*yphbis + vz*zphbis)
			- 4.*(xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - ((double)tab_H->z[icouche]+RTER)*((double)tab_H->z[icouche]+RTER));
		rsol1 = 0.5*( -2.*(vx*xphbis + vy*yphbis + vz*zphbis) + sqrt(rdelta) );
		rsol2 = 0.5*( -2.*(vx*xphbis + vy*yphbis + vz*zphbis) - sqrt(rdelta) );
		
		// Il faut choisir la plus petite distance en faisant attention qu'elle soit positive
		if(rsol1>0.){
			if( rsol2>0.)
				rsolfi = min(rsol1,rsol2);
			else
				rsolfi = rsol1;
		}
		else{
			if( rsol2>0. )
				rsolfi=rsol1;
		}
		
		tab_H->zph0[icouche] = tab_H->zph0[icouche-1] + (float)rsolfi;
		tab_H->hph0[icouche] = (float)tab_H->hph0[icouche-1] + 
				(float)( abs( tab_H->h[icouche] - tab_H->h[icouche-1])*rsolfi )/
				(float)( abs( tab_H->z[icouche-1] - tab_H->z[icouche]) );
		
		xphbis+= vx*rsolfi;
		yphbis+= vy*rsolfi;
		zphbis+= vz*rsolfi;
		
	}

	init_H->taumax0 = tab_H->hph0[NATM];
	init_H->zintermax0 = tab_H->zph0[NATM];

	
	/** Envoie des données dans le device **/
	cudaError_t erreur = cudaMemcpy(init_D, init_H, sizeof(Init), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf("#--------------------#\n");
		printf("# ERREUR: Problème de copie init_H dans initInit\n");
		printf("# Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		printf("#--------------------#\n");
		exit(1);
	}
	
	erreur = cudaMemcpy(tab_D->hph0, tab_H->hph0, (NATM+1)*sizeof(*(tab_H->hph0)), cudaMemcpyHostToDevice);
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
	printf(" THSDEG\t=\t%f (degrés)", THSDEG);
	printf("\n");
	printf(" LAMBDA\t=\t%f", LAMBDA);
	printf("\n");
	printf(" CONPHY\t=\t%f", CONPHY);
	printf("\n");
	printf(" DIFFF\t=\t%d", DIFFF);
	printf("\n");
	printf(" SIM\t=\t%d", SIM);
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
	
	#ifdef SPHERIQUE
	printf(" Géométrie de l'atmosphère: \tSphérique");
	printf("\n");
	#endif
	#ifndef SPHERIQUE
	printf(" Géométrie de l'atmosphère: \tParallèle");
	printf("\n");
	#endif
	
	printf(" TAURAY\t=\t%f", TAURAY);
	printf("\n");
	printf(" TAUAER\t=\t%f", TAUAER);
	printf("\n");
	printf(" W0AER\t=\t%f", W0AER);
	printf("\n");
	printf(" LSAAER\t=\t%u", LSAAER);
	printf("\n");
	printf(" NFAER\t=\t%u", NFAER);
	printf("\n");
	printf(" PROFIL\t=\t%d", PROFIL);
	printf("\n");
	printf(" HA\t=\t%f", HA);
	printf("\n");
	printf(" HR\t=\t%f", HR);
	printf("\n");
	printf(" ZMIN\t=\t%f", ZMIN);
	printf("\n");
	printf(" ZMAX\t=\t%f", ZMAX);
	printf("\n");
	printf(" NATM\t=\t%d", NATM);
	printf("\n");
	printf(" HATM\t=\t%d", HATM);
	printf("\n");
	
	printf("\n#--------- Contribution du dioptre ---------#\n");
	printf(" SUR\t=\t%d", SUR);
	printf("\n");
	printf(" DIOPTRE =\t%d", DIOPTRE);
	printf("\n");
	printf(" W0LAM\t=\t%f", W0LAM);
	printf("\n");
	printf(" WINDSPEED =\t%f", WINDSPEED);
	printf("\n");
	printf(" NH2O\t=\t%f", NH2O);
	printf("\n");
	
	printf("\n#----------- Chemin des fichiers -----------#\n");
	printf(" PATHRESULTATSHDF = %s", PATHRESULTATSHDF);
	printf("\n");
	printf(" PATHTEMOINHDF = %s", PATHTEMOINHDF);
	printf("\n");
	printf(" PATHDIFFAER = %s", PATHDIFFAER);
	printf("\n");
	printf(" PATHPROFILATM = %s", PATHPROFILATM);
	printf("\n");
	
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
var->erreurvy,		   var->erreurcase);
		   printf("\n");
		   #endif
}


#ifdef TRAJET
/* initEvnt
* Initialisation des variables à envoyer dans le kernel pour récupérer le trajet d'un photon
*/
void initEvnt(Evnt* evnt_H, Evnt* evnt_D)
{
	for(int i = 0; i < NBTRAJET; i++) evnt_H[i].action = 0;
	cudaError_t erreur = cudaMemcpy(evnt_D, evnt_H, NBTRAJET * sizeof(Evnt), cudaMemcpyHostToDevice);
	if( erreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie evnt_H dans initEvnt\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
		exit(1);
	}
}


/* afficheTrajet
* Fonction qui affiche le début du trajet du premier thread
*/
void afficheTrajet(Evnt* evnt_H)
{
	printf("\nTrajet d'un thread :\n");
	for(int i = 0; i < NBTRAJET; i++)
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
			printf("\nERREUR : host afficheTrajet: Aucun trajet a afficher\n");
			return;
		}
		else printf("exit : ");
		printf("tau=%10.9f ", evnt_H[i].tau);
		printf("poids=%10.9f", evnt_H[i].poids);
		printf("\n");
	}
}
#endif


/**********************************************************
*	> Calcul pour sauvegarde des résultats finaux
***********************************************************/

/* calculOmega
* Fonction qui calcule l'aire normalisée de chaque boite, son theta, et son psi, sous forme de 3 tableaux
*/
void calculOmega(float* tabTh, float* tabPhi, float* tabOmega)
{
	// Tableau contenant l'angle theta de chaque morceau de sphère
	memset(tabTh, 0, NBTHETA * sizeof(*tabPhi));
	float dth = DEMIPI / NBTHETA;
	tabTh[0] = dth/4;
	tabTh[1] = dth;
	for(int ith = 2; ith < NBTHETA; ith++){
		tabTh[ith] = tabTh[ith-1] + dth;
	}
	
	// Tableau contenant l'angle psi de chaque morceau de sphère
	memset(tabPhi, 0, NBPHI * sizeof(*tabPhi));
	float dphi = PI / NBPHI;
 	tabPhi[0] = dphi / 2;
	for(int iphi = 1; iphi < NBPHI; iphi++){ 
		tabPhi[iphi] = tabPhi[iphi-1] + dphi;
	}
	// Tableau contenant l'aire de chaque morceau de sphère
	float sumds = 0;
	float tabds[NBTHETA * NBPHI];
	memset(tabds, 0, NBTHETA * NBPHI * sizeof(*tabds));
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
}


/* calculTabFinal
* Fonction qui remplit le tabFinal correspondant à la reflectance (R), Q et U sur tous l'espace de sorti (dans chaque boite)
*/
void calculTabFinal(float* tabFinal, float* tabTh, float* tabPhi, float* tabPhotonsTot, unsigned long long nbPhotonsTot)
{
	
	float tabOmega[NBTHETA * NBPHI]; //tableau contenant l'aire de chaque morceau de sphère
	// Remplissage des tableaux tabTh, tabPhi, et tabOmega
	calculOmega(tabTh, tabPhi, tabOmega);
	
	// Remplissage du tableau final
	for(int iphi = 0; iphi < NBPHI; iphi++)
	{
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			// Reflectance
			tabFinal[0*NBTHETA*NBPHI + iphi*NBTHETA+ith] =
				(tabPhotonsTot[0*NBPHI*NBTHETA+ith*NBPHI+iphi] + tabPhotonsTot[1*NBPHI*NBTHETA+ith*NBPHI+iphi]) / 
				(2* nbPhotonsTot * tabOmega[ith*NBPHI+iphi]* cosf(tabTh[ith]));
			
			// Q
			tabFinal[1*NBTHETA*NBPHI + iphi*NBTHETA+ith] =
				(tabPhotonsTot[0*NBPHI*NBTHETA+ith*NBPHI+iphi] - tabPhotonsTot[1*NBPHI*NBTHETA+ith*NBPHI+iphi]) / 
				(2* nbPhotonsTot * tabOmega[ith*NBPHI+iphi] * cosf(tabTh[ith]));
			
			// U
			tabFinal[2*NBTHETA*NBPHI + iphi*NBTHETA+ith] = (tabPhotonsTot[2*NBPHI*NBTHETA+ith*NBPHI+iphi]) / 
				(2* nbPhotonsTot * tabOmega[ith*NBPHI+iphi] * cosf(tabTh[ith]));
				
		}
	}
}


/**********************************************************
*	> Fichier hdf (lecture/écriture témoin, écriture résultats)
***********************************************************/

/* creerHDFTemoin
* Fonction qui crée un fichier .hdf contenant les informations nécessaires à la reprise du programme
* //TODO: 	écrire moins régulièrement le témoin (non pas une écriture par appel de kernel)
*			changer le format (écrire un .bin par exemple) pour éventuellement gagner du temps (calculer le gain éventuel)
*/
void creerHDFTemoin(float* tabPhotonsTot, unsigned long long nbPhotonsTot, Variables* var, double tempsPrec)
{
	// Création du fichier de sortie
	int sdFichier = SDstart(PATHTEMOINHDF, DFACC_CREATE);
	
	char nomTab[20]; //nom du tableau
	sprintf(nomTab,"Temoin (%d%%)", (int)(100 * nbPhotonsTot / NBPHOTONS));
	int nbDimsTab = 1; //nombre de dimensions du tableau
	int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
	valDimsTab[0] = 4 * NBTHETA * NBPHI;
	int typeTab = DFNT_FLOAT32 ; //type des éléments du tableau
	// Création du tableau
	int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	int startTab[nbDimsTab]; //début de la lecture du tableau
	startTab[0]=0;
	// Ecriture du tableau dans le fichier
	int status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP)tabPhotonsTot);
	// Vérification du bon fonctionnement de l'écriture
	if(status)
	{
		printf("\nERREUR : write hdf temoin\n");
		exit(1);
	}
	
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
	SDsetattr(sdsTab, "W0LAM", DFNT_FLOAT32, 1, &W0LAM);
	
	SDsetattr(sdsTab, "LSAAER", DFNT_UINT32, 1, &LSAAER);
	SDsetattr(sdsTab, "NFAER", DFNT_UINT32, 1, &NFAER);
	
	SDsetattr(sdsTab, "HA", DFNT_FLOAT32, 1, &HA);
	SDsetattr(sdsTab, "HR", DFNT_FLOAT32, 1, &HR);
	SDsetattr(sdsTab, "ZMIN", DFNT_FLOAT32, 1, &ZMIN);
	SDsetattr(sdsTab, "ZMAX", DFNT_FLOAT32, 1, &ZMAX);
	SDsetattr(sdsTab, "NATM", DFNT_INT32, 1, &NATM);
	SDsetattr(sdsTab, "HATM", DFNT_INT32, 1, &HATM);
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
	
	#ifdef PROGRESSION
	SDsetattr(sdsTab, "nbThreads", DFNT_UINT32, 1, &(var->nbThreads));
	SDsetattr(sdsTab, "nbPhotonsSor", DFNT_UINT32, 1, &(var->nbPhotonsSor));
	SDsetattr(sdsTab, "erreurvxy", DFNT_INT32, 1, &(var->erreurvxy));
	SDsetattr(sdsTab, "erreurvy", DFNT_INT32, 1, &(var->erreurvy));
	SDsetattr(sdsTab, "erreurcase", DFNT_INT32, 1, &(var->erreurcase));
	#endif

	// Fermeture du tableau
	SDendaccess(sdsTab);
	// Fermeture du fichier
	SDend(sdFichier);
}


/* lireHDFTemoin
* Si un fichier temoin existe et que les paramètres correspondent à la simulation en cours, cette simulation se poursuit à
* partir de celle sauvée dans le fichier témoin.
*/
void lireHDFTemoin(Variables* var_H, Variables* var_D,
		unsigned long long* nbPhotonsTot, float* tabPhotonsTot, double* tempsEcoule)
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
		float W0LAMrecup[1];
		float HArecup[1];
		float HRrecup[1];
		float ZMINrecup[1];
		float ZMAXrecup[1];
		int NATMrecup[1];
		int HATMrecup[1];
		float WINDSPEEDrecup[1];
		float NH2Orecup[1];
		float CONPHYrecup[1];
		
		SDreadattr(sdsTab, SDfindattr(sdsTab, "SEED"), (VOIDP)SEEDrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NBTHETA"), (VOIDP)NBTHETArecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NBPHI"), (VOIDP)NBPHIrecup);
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
		SDreadattr(sdsTab, SDfindattr(sdsTab, "W0LAM"), (VOIDP)W0LAMrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "HA"), (VOIDP)HArecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "HR"), (VOIDP)HRrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "ZMIN"), (VOIDP)ZMINrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "ZMAX"), (VOIDP)ZMAXrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NATM"), (VOIDP)NATMrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "HATM"), (VOIDP)HATMrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "WINDSPEED"), (VOIDP)WINDSPEEDrecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "NH2O"), (VOIDP)NH2Orecup);
		SDreadattr(sdsTab, SDfindattr(sdsTab, "CONPHY"), (VOIDP)CONPHYrecup);
		
		// Si les parametres sont les memes on recupere des informations pour poursuivre la simulation précédente
		if(NBTHETArecup[0] == NBTHETA
			&& NBPHIrecup[0] == NBPHI
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
			&& W0LAMrecup[0] == W0LAM
			&& HArecup[0] == HA
			&& HRrecup[0] == HR
			&& ZMINrecup[0] == ZMIN
			&& ZMAXrecup[0] == ZMAX
			&& NATMrecup[0] == NATM
			&& HATMrecup[0] == HATM
			&& WINDSPEEDrecup[0] == WINDSPEED
			&& NH2Orecup[0] == NH2O
			&& CONPHYrecup[0] == CONPHY)
		{
			
			// Recuperation du nombre de photons traités et du nombre d'erreurs
			double nbPhotonsTotDouble[1]; //on récupère d'abord la variable en double
			unsigned long long nbPhotonsTotRecup[1]; //puis on la passera en unsigned long long
			int nbErreursPoidsRecup[1];
			int nbErreursThetaRecup[1];
			double tempsEcouleRecup[1];
	
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbPhotonsTot"), (VOIDP)nbPhotonsTotDouble);
			nbPhotonsTotRecup[0] = (unsigned long long)nbPhotonsTotDouble[0];
			
			printf("\nPOURSUITE D'UNE SIMULATION ANTERIEURE AU PHOTON %e\n",nbPhotonsTotDouble[0]);
			if(SEEDrecup[0] == SEED) printf("ATTENTION: Nous recommandons SEED=-1 sinon les nombres aleatoires sont\
identiques a chaque lancement.\n");
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbErreursPoids"), (VOIDP)nbErreursPoidsRecup);
			SDreadattr(sdsTab, SDfindattr(sdsTab, "nbErreursTheta"), (VOIDP)nbErreursThetaRecup);
			SDreadattr(sdsTab, SDfindattr(sdsTab, "tempsEcoule"), (VOIDP)tempsEcouleRecup);
	
			var_H->erreurpoids = nbErreursPoidsRecup[0];//nombre de photons ayant un poids anormalement élevé
			var_H->erreurtheta = nbErreursThetaRecup[0];//nombre de photons sortant dans la direction solaire
			
			#ifdef PROGRESSION
			unsigned long long nbThreadsRecup[1]; //nombre total de threads lancés
			unsigned long long nbPhotonsSorRecup[1]; //nombre de photons ressortis pour un appel du Kernel
			int erreurvxyRecup[1]; //nombre de photons sortant au zénith et donc difficiles à classer
			int erreurvyRecup[1]; //nombre de photons sortant à phi=0 ou phi=PI et donc difficiles à classer
			int erreurcaseRecup[1]; // nombre de photons rangé dans une case inexistante
			
			SDreadattr(sdsTab, SDfindattr(sdsTab,"nbThreads"), (VOIDP)nbThreadsRecup );
			SDreadattr(sdsTab, SDfindattr(sdsTab,"nbPhotonsSor"), (VOIDP)nbPhotonsSorRecup );
			SDreadattr(sdsTab, SDfindattr(sdsTab,"erreurvxy"), (VOIDP)erreurvxyRecup );
			SDreadattr(sdsTab, SDfindattr(sdsTab,"erreurvy"), (VOIDP)erreurvyRecup );
			SDreadattr(sdsTab, SDfindattr(sdsTab,"erreurcase"), (VOIDP)erreurcaseRecup );
			
			var_H->nbThreads = nbThreadsRecup[0];
			var_H->nbPhotonsSor = nbPhotonsSorRecup[0];
			var_H->erreurvxy = erreurvxyRecup[0];
			var_H->erreurvy = erreurvyRecup[0];
			var_H->erreurcase = erreurcaseRecup[0];
			
			#endif
			
			cudaError_t erreur = cudaMemcpy(var_D, var_H, sizeof(Variables), cudaMemcpyHostToDevice);
			if( erreur != cudaSuccess ){
				printf( "ERREUR: Problème de copie var_H dans lireHDFTemoin\n");
				printf( "Nature de l'erreur: %s\n",cudaGetErrorString(erreur) );
				exit(1);
			}
			*(nbPhotonsTot) = nbPhotonsTotRecup[0];
			*(tempsEcoule) = tempsEcouleRecup[0];
	
			// Recuperation du tableau
			int nbDimsTab = 1; //nombre de dimensions du tableau
			int startTab[nbDimsTab], edgesTab[nbDimsTab]; //debut et fin de la lecture du tableau
			startTab[0] = 0;
			edgesTab[0] = 4*NBTHETA * NBPHI;
// 			float tabPhotonsTotRecup[4*NBTHETA * NBPHI]; //tableau de récuperation en float
	
			int status = SDreaddata (sdsTab, startTab, NULL, edgesTab, (VOIDP)tabPhotonsTot/*Recup*/);
			// Vérification du bon fonctionnement de la lecture
			if(status)
			{
				printf("\nERREUR : read hdf temoin\n");
				exit(1);
			}
	
			// Sauvegarde
// 			for(int i = 0; i < 4*NBTHETA * NBPHI; i++)
// 			{
// 				tabPhotonsTot[i] = tabPhotonsTotRecup[i];
// 			}
			
		}
		// Fermeture du tableau
		SDendaccess (sdsTab);
	}
	// Fermeture du fichier
	SDend (sdFichier);
}


/* creerHDFResultats
* Fonction qui crée le fichier .hdf contenant le résultat final pour une demi-sphère
*/
void creerHDFResultats(float* tabFinal, float* tabTh, float* tabPhi,unsigned long long nbPhotonsTot, Variables* var,double tempsPrec)
{
	// Tableau temporaire utile pour la suite
	float tab[NBPHI*NBTHETA];
	// Création du fichier de sortie
	int sdFichier = SDstart(PATHRESULTATSHDF, DFACC_CREATE);
	if( sdFichier == FAIL ){
		printf("ERREUR: Erreur d'ouverture du fichier HDF : %s\nFin du programme\n", PATHRESULTATSHDF);
		exit(1);
	}
	
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
	SDsetattr(sdFichier, "W0LAM", DFNT_FLOAT32, 1, &W0LAM);
	SDsetattr(sdFichier, "HA", DFNT_FLOAT32, 1, &HA);
	SDsetattr(sdFichier, "HR", DFNT_FLOAT32, 1, &HR);
	SDsetattr(sdFichier, "ZMIN", DFNT_FLOAT32, 1, &ZMIN);
	SDsetattr(sdFichier, "ZMAX", DFNT_FLOAT32, 1, &ZMAX);
	SDsetattr(sdFichier, "NATM", DFNT_INT32, 1, &NATM);
	SDsetattr(sdFichier, "HATM", DFNT_INT32, 1, &HATM);
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
	
	/** 	Création du 1er tableau dans le fichier hdf
		Valeur de la reflectance pour phi et theta donnés		**/
	char* nomTab="Valeurs de la reflectance (I)"; //nom du tableau
	int nbDimsTab = 2; //nombre de dimensions du tableau
	int valDimsTab[nbDimsTab]; //valeurs des dimensions du tableau
	valDimsTab[1] = NBTHETA;	//colonnes
	valDimsTab[0] = NBPHI;
	int typeTab = DFNT_FLOAT32; //type des éléments du tableau
	
	// Création du tableau
	int sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	int startTab[nbDimsTab]; //début de la lecture du tableau
	startTab[0]=0;
	startTab[1]=0;
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
	nomTab="Valeurs de Q"; //nom du tableau
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinal+NBPHI*NBTHETA) );
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
	nomTab="Valeurs de U"; //nom du tableau
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	// Création du tableau
	sdsTab = SDcreate(sdFichier, nomTab, typeTab, nbDimsTab, valDimsTab);
	// Ecriture du tableau dans le fichier
	status = SDwritedata(sdsTab, startTab, NULL, valDimsTab, (VOIDP) (tabFinal+2*NBPHI*NBTHETA) );
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
	nomTab="Valeurs de la lumiere polarisee (LP)"; //nom du tableau
	// La plupart des paramètres restent les mêmes, pas besoin de les réinitialiser
	
	for(int i = 0; i < NBTHETA*NBPHI; i++){
		tab[i] = sqrtf( tabFinal[1*NBTHETA*NBPHI+i]*tabFinal[1*NBTHETA*NBPHI+i] +
						tabFinal[2*NBTHETA*NBPHI+i]*tabFinal[2*NBTHETA*NBPHI+i] );
		// 	printf("[%d]\tQ= %17.17f\tU= %17.17f\tLP=%17.17f\n",i,tabFinal[1*NBTHETA*NBPHI+i],tabFinal[2*NBTHETA*NBPHI+i],tab[i]);
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
	
	
	/** 	Création du tableau theta
		Valeurs de theta en fonction de l'indice	**/
	//conversion en degrès de theta pour une meilleure visualisation de la sortie
	float tabThBis[NBTHETA];
	for(int i=0; i<NBTHETA; i++)
		tabThBis[i] = tabTh[i]/DEG2RAD;
	
	nomTab = "Valeurs de theta echantillonnees";
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
	SDendaccess(sdsTab);
	
	/** 	Création du tableau phi
		Valeurs de phi en fonction de l'indice	**/
	float tabPhiBis[NBPHI];
	for(int i=0; i<NBPHI; i++)
		tabPhiBis[i] = tabPhi[i]/DEG2RAD;
	
	nomTab = "Valeurs de phi echantillonnees";
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
	SDendaccess(sdsTab);
	
	// Fermeture du fichier
	SDend(sdFichier);
	
}
