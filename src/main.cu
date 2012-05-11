
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
	/** Initialisation de la carte graphique **/
	cudaError_t cudaErreur;	// Permet de vérifier les allocations mémoire
	
	//
	cudaDeviceReset();
	
	// Préférer utiliser plus de mémoire cache que de shared memory
	cudaErreur = cudaFuncSetCacheConfig (lancementKernel,  cudaFuncCachePreferL1);
	if( cudaErreur != cudaSuccess ){
		printf("#--------------------#\n");
		printf("# ERREUR: Problème cuFuncSetCacheConfig dans le main\n");
		printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		printf("#--------------------#\n");
		exit(1);
	}
	
	
	/** Initialisation des constantes du host (en partie recuperees dans le fichier Parametres.txt) **/
	initConstantesHost(argc, argv);
	
	
	/** Variables du main **/
	
	double tempsPrec = 0.; 	//temps ecoule de la simulation precedente
	
	// Regroupement et initialisation des variables a envoyer dans le kernel (structure de variables)
	Variables* var_H; //variables version host
	Variables* var_D; //variables version device
	initVariables(&var_H, &var_D);
	
	// Regroupement et initialisation des tableaux a envoyer dans le kernel (structure de pointeurs)
	Tableaux tab_H; //tableaux version host
	Tableaux tab_D; //tableaux version device
	initTableaux(&tab_H, &tab_D);
	
	// Variables et tableaux qui restent dans le host et se remplissent petit à petit
	unsigned long long nbPhotonsTot = 0; //nombre total de photons traités
	
	#ifdef PROGRESSION
	unsigned long long nbPhotonsSorTot = 0; //nombre total de photons ressortis
	#endif
	
	float* tabPhotonsTot; //tableau du poids total des photons sortis
	tabPhotonsTot = (float*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTot)));
	if( tabPhotonsTot == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTot dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTot,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTot)));
	
	// Variables permettant le calcul du résultat final
	float tabFinal[3*NBTHETA * NBPHI];	 /* tableau final: 3 dimensions pour R=stokes1+stokes2(dim0) , Q=stokes1-stokes2(dim1) et
											U=stokes3(dim2) */
	float tabTh[NBTHETA]; //tableau contenant l'angle theta de chaque morceau de sphère
	float tabPhi[NBPHI]; //tableau contenant l'angle psi de chaque morceau de sphère
	
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	// Définition et initialisation des constantes initiales du photon
	Init* init_H;
	Init* init_D;
	initInit(&init_H, &init_D);
	#endif
	
	#ifdef TABRAND
	// DEBUG Recuperations des nombres aleatoires du random en place
	float tableauRand_H[100] = {0};
	float* tableauRand_D;
	if( cudaMalloc(&tableauRand_D, 100 * sizeof(float)) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de tableauRand_D dans le main\n");
		exit(1);
	}
	cudaMemset(tableauRand_D, 0, 100 * sizeof(float));
	#endif
	
	
	/** Initialisation des constantes du device à partir des constantes du host **/
	initConstantesDevice();
	

	#ifdef TEMOIN
	/** Vérification de l'existence ou non d'un fichier témoin **/
	verifierFichier();
	#endif
	
	
	#ifdef PARAMETRES
	/** Affichage des paramètres de la simulation **/
	afficheParametres();
	#endif
	
	
	/** Calcul des modèles utiles à l'algorithme **/
	// Calcul de faer, modèle de diffusion des aérosols
	if( TAUAER > 0 ){
		calculFaer( PATHDIFFAER, &tab_H, &tab_D );
// 		verificationFAER( "./test/FAER_test.txt", tab_H );
	}

	// Calcul du mélange Molécule/Aérosol dans l'atmosphère en fonction de la couche
	profilAtm( &tab_H, &tab_D );
// 	verificationAtm( tab_H );
	
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	// Calcul du point d'impact du photon
	impactInit(init_H, init_D, &tab_H, &tab_D);

	#ifdef DEBUG
	printf("Paramètres initiaux du photon: taumax0=%lf - zintermax=%lf - (%lf,%lf,%lf)\n",\
		   init_H->taumax0, init_H->zintermax0, init_H->x0, init_H->y0, init_H->z0 );
// 	for(int i=0; i<NATM+1; i++)
// 		printf("zph0[%d]=%10.7f - hph0[%d]=%10.7f\n",i, tab_H.zph0[i], i ,tab_H.hph0[i] );
	#endif
	
	#endif /* Fin de la spécification atmosphère sphérique */
	
	
	/** Fonction qui permet de poursuivre la simulation précédente si elle n'est pas terminee **/
	#ifdef TEMOIN
	lireHDFTemoin(var_H, var_D, &nbPhotonsTot, tabPhotonsTot, &tempsPrec);
	#endif
	
	
	#ifdef TRAJET
	// DEBUG : Variables permettant de récupérer le début du trajet d'un photon
	Evnt evnt_H[NBTRAJET];
	Evnt* evnt_D;
	if( cudaMalloc(&evnt_D, NBTRAJET* sizeof(Evnt)) != cudaSuccess){
		printf("ERREUR: Problème de cudaMalloc de evnt_D dans le main\n");
		exit(1);
	}

	initEvnt(evnt_H, evnt_D);
	#endif
	
	/** Organisation des threads en blocks de threads et en grids de blocks **/
	dim3 blockSize(XBLOCK,YBLOCK);
	dim3 gridSize(XGRID,YGRID);


	// Variable permettant de savoir si on est passé dans la boucle ou non
	bool passageBoucle = false;
	if(nbPhotonsTot < NBPHOTONS) 
		passageBoucle = true;
	
	
	// Tant qu'il n'y a pas assez de photons traités on relance le kernel
	while(nbPhotonsTot < NBPHOTONS)
	{
		/** Remise à zéro de certaines variables et certains tableaux **/
// 		reinitVariables(var_H, var_D);
		cudaErreur = cudaMemset(&(var_D->nbPhotons), 0, sizeof(var_D->nbPhotons));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset var_D.nbPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
		
		#ifdef PROGRESSION
		cudaErreur = cudaMemset(&(var_D->nbPhotonsSor), 0, sizeof(var_D->nbPhotonsSor));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset var_D.nbPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
		#endif
		
		
		#ifdef TEMOIN
		/** Réinitialisation des données de la simulation **/
		
		cudaErreur = cudaMemset(tab_D.tabPhotons, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotons)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
		#endif
		
		
		/** Lancement du kernel **/
		lancementKernel<<<gridSize, blockSize>>>(var_D, tab_D
				#ifdef SPHERIQUE
				, init_D
				#endif
				#ifdef TABRAND
				, tableauRand_D
				#endif
				#ifdef TRAJET
				, evnt_D //récupération d'un trajet de photons
				#endif
							);
		// Attend que tous les threads aient fini avant de faire autre chose
// 		cudaThreadSynchronize();
		
		/** Récupération des variables et d'un tableau envoyés dans le kernel **/
		cudaErreur = cudaMemcpy(var_H, var_D, sizeof(Variables), cudaMemcpyDeviceToHost);
// 		cudaErreur = cudaMemcpyAsync( var_H, var_D, sizeof(Variables),cudaMemcpyDeviceToHost , stream1 );
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de copie var_D dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
// 			printf("#	sizeof(*var_D)=%d\tsizeof(*var_H)=%d\tsizeof(*Variables)=%d\n",
// 					sizeof(*var_D),sizeof(*var_H),sizeof(Variables));
// 			printf("# Adresse pointée par var_D : %p\tAdresse pointée par var_H : %p\n", var_H, var_D);
			printf("#--------------------#\n");
			exit(1);
		}

		#ifdef TEMOIN
		cudaErreur = cudaMemcpy(tab_H.tabPhotons, tab_D.tabPhotons, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotons)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotons dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}
		
		/** Creation d'un fichier témoin pour pouvoir reprendre la simulation en cas d'arrêt **/
		for(int i = 0; i < 4*NBTHETA*NBPHI; i++)
			tabPhotonsTot[i] += tab_H.tabPhotons[i];
		
		creerHDFTemoin(tabPhotonsTot, nbPhotonsTot,var_H, tempsPrec);
		#endif


		// On remplit les variables et tableau qui restent dans le host
		nbPhotonsTot += var_H->nbPhotons;
		
		#ifdef PROGRESSION
		nbPhotonsSorTot += var_H->nbPhotonsSor;
		#endif
		
		
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
	
	
	#ifndef TEMOIN
	/** Récupération des données du device vers le host **/
	cudaErreur = cudaMemcpy(tab_H.tabPhotons, tab_D.tabPhotons, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotons)),
cudaMemcpyDeviceToHost);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie tab_H.tabPhotons dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	
	for(int i = 0; i < 4*NBTHETA*NBPHI; i++)
		tabPhotonsTot[i] += tab_H.tabPhotons[i];
	#endif
	
	
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
	
	
	/** Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de
surface) **/	
	// Remplissage des 3 tableaux
	calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, nbPhotonsTot);

	
	/** Fonction qui crée le fichier .hdf contenant le résultat final sur la demi-sphère **/
	creerHDFResultats(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	printf(" Fin de l'execution du programme. Resultats stockes dans %s\n",PATHRESULTATSHDF);

	
	#ifdef TRAJET
	/** Affichage du trajet du photon **/
	// Récupération des variables envoyées dans le kernel
	cudaErreur = cudaMemcpy(evnt_H, evnt_D, NBTRAJET * sizeof(Evnt), cudaMemcpyDeviceToHost);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de copie evnt_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	
	// Affichage du trajet du premier thread
	afficheTrajet(evnt_H);
	#endif
	
	/** Libération de la mémoire allouée **/
	// Libération du groupe de variables envoyé dans le kernel
	cudaErreur = cudaFree(var_D);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de free de var_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}

	free(var_H);
// 	cudaFreeHost(var_H);

	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	cudaErreur = cudaFree(init_D);
	if( cudaErreur != cudaSuccess ){
		printf( "ERREUR: Problème de free de init_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}

	free(init_H);
	#endif
	
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
		printf( "ERREUR: Problème de free tableauRand_D dans le main\n");
		printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
		exit(1);
	}
	#endif

	return 0;

}
