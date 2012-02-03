
	  //////////////
	 // INCLUDES //
	//////////////

#include "main.h"
#include "communs.h"
#include "device.h"
#include "host.h"

	  ///////////////////
	 // FONCTION MAIN //
	///////////////////

// Fonction principale
int main (int argc, char *argv[])
{
	cudaDeviceReset();

	/** Variables du main **/
	cudaError_t cudaErreur;	// Permet de vérifier les allocations mémoires

	/** Initialisation des constantes du host (en partie recuperees dans le fichier Parametres.txt) **/
	initConstantesHost(argc, argv);

	/** Initialisation des constantes du device à partir des constantes du host **/
	initConstantesDevice();

	// S'il existe déjà un fichier nommé NOMRESULTATSHDF (Parametres.txt) on arrête le programme
	if (fopen(PATHRESULTATSHDF, "rb") != NULL)
	{
		printf("ERREUR: Le fichier %s existe deja.\n",PATHRESULTATSHDF);
		exit(1);
	}

	#ifdef SORTIEINT
	// Fichiers de sortie pour le débuggage

	float seuilPds = 0;
	char detail[256];

	// Fichier où seront stockés les photons avec un poids supérieur à un seuil
	sprintf(detail,"out_prog/sortie_int/poids/poids_tauRay=%f_tauAer=%f_difff=%d_ths=%f_sim=%d.txt",TAURAY,TAUAER,DIFFF,
 THSDEG,SIM);
	FILE* fic_poids = fopen(detail,"w");
	if( fic_poids == NULL){
		printf("ERREUR: Impossible d'ouvrir le fichier %s\n", detail);
		exit(1);
	}

	// Le nombre total de photons y sera sauvé
	sprintf(detail,"out_prog/sortie_int/nbre_photons/photons_tauRay=%f_tauAer=%f_difff=%d_ths=%f_sim=%d.txt",TAURAY,
TAUAER, DIFFF,THSDEG,SIM);
	FILE* fic_nbre_ph = fopen(detail,"w");
	if( fic_nbre_ph == NULL){
		printf("ERREUR: Impossible d'ouvrir le fichier %s\n",detail);
		exit(1);
	}
	
	int tabNbBoucleTot[NBLOOP];
	for( int i=0; i<NBLOOP; i++)
		tabNbBoucleTot[i]=0;
	sprintf(detail,"out_prog/sortie_int/nbre_boucle/nbBoucle_tauRay=%f_tauAer=%f_difff=%d_ths=%f_sim=%d.txt",TAURAY,
TAUAER, DIFFF,THSDEG,SIM);
	FILE* fic_nbre_boucle = fopen(detail,"w");
	if( fic_nbre_boucle == NULL){
		printf("ERREUR: Impossible d'ouvrir le fichier %s\n",detail);
		exit(1);
	}
	#endif

	// DEBUG : Affichage basique des parametres de la simulation
	printf("\n%lu - %u - %d - %d - %d - %d - %d - %d\n", NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI);

	/** Regroupement et initialisation des variables a envoyer dans le kernel (structure de variables) **/
	Variables* var_H; //variables version host
	Variables* var_D; //variables version device
	initVariables(&var_H, &var_D);

	/** Regroupement et initialisation des tableaux a envoyer dans le kernel (structure de pointeurs) **/
	Tableaux tab_H; //tableaux version host
	Tableaux tab_D; //tableaux version device
	initTableaux(&tab_H, &tab_D);


	/** Variables et tableaux qui restent dans le host et se remplissent petit à petit **/
	unsigned long long nbPhotonsTot = 0; //nombre total de photons traités

	#ifdef PROGRESSION
	unsigned long long nbPhotonsSorTot = 0; //nombre total de photons ressortis
	#endif

	unsigned long long* tabPhotonsTot; //tableau du poids total des photons sortis
	tabPhotonsTot = (unsigned long long*)malloc(NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
	if( tabPhotonsTot == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTot dans le main\n");
		exit(1);
	}

	#ifdef TABRAND
	// DEBUG Recuperations des nombres aleatoires du random en place
	float tableauRand_H[100] = {0};
	float* tableauRand_D;
	if( cudaMalloc(&tableauRand_D, 100 * sizeof(float) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tableauRand_D dans le main\n");
		exit(1);
	}
	cudaMemset(tableauRand_D, 0, 100 * sizeof(float));
	#endif

	/** Calcul des modèles utiles à l'algorithme **/
	// Calcul de faer, modèle de diffusion des aérosols
	if( TAUAER > 0.0001 ){
		calculFaer( PATHDIFFAER, &tab_H, &tab_D );
// 		verificationFAER( "./test/FAER_test.txt", tab_H );

	}

	// Calcul du mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
	profilAtm( &tab_H, &tab_D );
// 	verificationAtm( tab_H );

	/** Fonction qui permet de poursuivre la simulation précédente si elle n'est pas terminee **/
	double tempsPrec = 0.; //temps ecoule de la simulation precedente
	lireHDFTemoin(var_H, var_D, &nbPhotonsTot, tabPhotonsTot, &tempsPrec);
	
	#ifdef TRAJET
	// DEBUG : Variables permettant de récupérer le début du trajet d'un photon
	Evnt evnt_H[20];
	Evnt* evnt_D;
	if( cudaMalloc(&evnt_D, 20 * sizeof(Evnt)) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de evnt_D dans le main\n");
		exit(1);
	}

	initEvnt(evnt_H, evnt_D);
	#endif
	
	/** Organisation des threads en blocks de threads et en grids de blocks **/
	dim3 blockSize(XBLOCK,YBLOCK);
	dim3 gridSize(XGRID,YGRID);
	
	// Affichage des paramètres de la simulation
	#ifdef PARAMETRES
	afficheParametres();
	#endif


	// Variable permettant de savoir si on est passé dans la boucle ou non
	bool passageBoucle = false;
	if(nbPhotonsTot < NBPHOTONS) 
		passageBoucle = true;

	// Tant qu'il n'y a pas assez de photons traités on relance le kernel
	while(nbPhotonsTot < NBPHOTONS)
	{
		/** Remise à zéro de certaines variables et certains tableaux **/
		reinitVariables(var_H, var_D);
		cudaErreur = cudaMemset(tab_D.tabPhotons, 0, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
		
		#ifdef SORTIEINT
		cudaErreur = cudaMemset(tab_D.nbBoucle, 0, NBLOOP*sizeof(*(tab_D.nbBoucle)) );
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D->nbBoucle dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
		#endif
		
		/** Lancement du kernel **/
// 		switch(SIM){
// 
// 		case -2:
			lancementKernel<<<gridSize, blockSize>>>(var_D, tab_D			
					#ifdef TABRAND
					, tableauRand_D
					#endif
					#ifdef TRAJET
					, evnt_D //récupération d'un trajet de photons
					#endif
								);
			// Attend que tous les threads aient fini avant de faire autre chose
			cudaThreadSynchronize();
	// 		break;
// 
// 		default:
// 			break;
// 
// 		
// 		} // Fin du switch
		
		/** Récupération des variables et d'un tableau envoyés dans le kernel **/
		cudaErreur = cudaMemcpy(var_H, var_D, sizeof(Variables), cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de copie var_D dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
// 			printf("# sizeof(*var_D)=%d\tsizeof(*var_H)=%d\tsizeof(*Variables)=%d\n", 
// sizeof(*var_D),sizeof(*var_H),sizeof(Variables));
// 			printf("# Adresse pointée par var_D : %p\tAdresse pointée par var_H : %p\n", var_H, var_D);
			printf("#--------------------#\n");
			exit(1);
		}

		cudaErreur = cudaMemcpy(tab_H.tabPhotons, tab_D.tabPhotons, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotons dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}

		#ifdef SORTIEINT
		cudaErreur = cudaMemcpy(tab_H.poids, tab_D.poids, NBLOOP * sizeof(float), cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.poids dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}
		for(int i=0; i<NBLOOP; i++){
			if(abs(tab_H.poids[i])>seuilPds)
				fprintf(fic_poids,"%f\n",tab_H.poids[i]);
		}
		
		cudaErreur = cudaMemcpy(tab_H.nbBoucle, tab_D.nbBoucle, NBLOOP * sizeof(*(tab_D.nbBoucle)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.nbBoucle dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}
		
		for(int i = 0; i <NBLOOP; i++)
			tabNbBoucleTot[i] += tab_H.nbBoucle[i];
		#endif
				
		// On remplit les variables et tableau qui restent dans le host
		nbPhotonsTot += var_H->nbPhotons;
		#ifdef PROGRESSION
		nbPhotonsSorTot += var_H->nbPhotonsSor;
		#endif
		for(int i = 0; i < NBTHETA * NBPHI * NBSTOKES; i++)
			tabPhotonsTot[i] += tab_H.tabPhotons[i];
		
		/** Creation d'un fichier témoin pour pouvoir reprendre la simulation en cas d'arrêt **/
		creerHDFTemoin(tabPhotonsTot, nbPhotonsTot,var_H, tempsPrec);
		
		/** Affichage de l'avancement de la simulation **/
		afficheProgress(nbPhotonsTot, var_H, tempsPrec
			#ifdef PROGRESSION
			, nbPhotonsSorTot
			#endif
			       );
	}
	
	// Si on n'est pas passé dans la boucle on affiche quand-même l'avancement de la simulation
	if(!passageBoucle) afficheProgress(nbPhotonsTot, var_H, tempsPrec
					#ifdef PROGRESSION
					, nbPhotonsSorTot
					#endif
					  );
			       
	#ifdef TABRAND
	// DEBUG Recuperations et affichage des nombres aleatoires du random
	cudaErreur = cudaMemcpy(tableauRand_H, tableauRand_D, 100 * sizeof(float), cudaMemcpyDeviceToHost);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tableauRand_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}


	printf("\n=====RAND========================\n");
	for(int i = 0; i < 10; i++)
	{
		printf("thread%d : ", i%5);
		for(int j = 0; j < 10; j++)
		{
			printf("%f - ", tableauRand_H[i*10+j]);
		}
		printf("\n");
	}
	printf("==================================\n");
	#endif
	
	#ifdef TRAJET
	// DEBUG Récupération des variables envoyées dans le kernel
	cudaErreur = cudaMemcpy(evnt_H, evnt_D, 20 * sizeof(Evnt), cudaMemcpyDeviceToHost);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie evnt_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}

	// Affichage du trajet du premier thread
	afficheTrajet(evnt_H);
	#endif

	#ifdef SORTIEINT
	// Sauvegarde du nombre de photons par boite
	for(int iphi = 0; iphi < NBPHI; iphi++)
	{
		for(int ith = 0; ith < NBTHETA; ith++)
		{
			fprintf(fic_nbre_ph,"%llu\t",(tabPhotonsTot[0*NBPHI*NBTHETA+ith*NBPHI+iphi] + tabPhotonsTot[1*NBPHI*NBTHETA+ith*NBPHI+iphi])/2);
		}
		fprintf( fic_nbre_ph, "\n" );
	}
	
// 	cudaErreur = cudaMemcpy(tab_H.nbBoucle, tab_D.nbBoucle, NBLOOP * sizeof(float), cudaMemcpyDeviceToHost);
// 	if( cudaErreur != cudaSuccess ){
// 		printf( "ERREUR: Problème de copie tab_H.nbBoucle dans le main\n");
// 		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
// 		exit(1);
// 	}
		
	for( int i = 0; i<NBLOOP; i++ ){
		fprintf( fic_nbre_boucle, "%d\t%d\n", i+1,tabNbBoucleTot[i] );
		
	}
	
	fclose(fic_poids);
	fclose(fic_nbre_ph);
	fclose(fic_nbre_boucle);
	#endif

	/** Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de surface) **/
	float tabFinal[NBTHETA * NBPHI]; //tableau final
	float tabTh[NBTHETA]; //tableau contenant l'angle theta de chaque morceau de sphère
	float tabPhi[NBPHI]; //tableau contenant l'angle psi de chaque morceau de sphère
	// Remplissage des 3 tableaux
	calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, nbPhotonsTot);

	/** Fonction qui crée le fichier .hdf contenant le résultat final sur la demi-sphère **/
	creerHDFResultats(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	printf(" Fin de l'execution du programme. Resultats stockes dans %s\n",PATHRESULTATSHDF);

	/** Libération de la mémoire allouée **/
	// Libération du groupe de variables envoyé dans le kernel
// 	cudaErreur = cudaFree(var_D);
// 	if( cudaErreur != cudaSuccess ){
// 		printf( "ERREUR: Problème de free de var_D dans le main\n");
// 		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
// 		exit(1);
// 	}
// 
// 	free(var_H);
	cudaFreeHost(var_H);

	// Libération des tableaux envoyés dans le kernel
	freeTableaux(&tab_H, &tab_D);
	// Libération du tableau du host
	free(tabPhotonsTot);


	// Libération des variables qui récupèrent le trajet d'un photon
	#ifdef TRAJET
	cudaErreur = cudaFree(evnt_D);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de free de evnt_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	#endif
	
	#ifdef TABRAND
	//DEBUG random
	cudaErreur = cudaFree(tableauRand_D);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tableauRand_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	#endif

	cudaDeviceReset();

	return 0;

}
