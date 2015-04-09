
/**********************************************************
*	> Includes
***********************************************************/

#include "main.h"
#include "communs.h"
#include "device.h"
#include "host.h"
#include "checkGPUcontext.h"

#ifdef RANDPHILOX4x32_7
//Cette partie est necessitee par les generateurs "1,2,3" pour leur version 64bits.
//Le Philox ne nous interesse pas en 64bits mais a moins de revoir tous les includes fournis avec le Philox (!),
//on est plutot contraint de rajouter ces trois lignes...
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#endif

#ifdef _PERF
#include "perfo.h"
static SPerf* perfPrint;
static SPerf* perfInitG;
static SPerf* perfKernel;
static SPerf* perfMemcpyH2DVar;
static SPerf* perfMemcpyH2DTab;
static SPerf* perfMemcpyD2HVar;
static SPerf* perfMemcpyD2HTab;
static SPerf* perfCreateWitness;
static SPerf* perfFree;
static SPerf* perfCreateFinalTab;
#endif


/**********************************************************
*	> Fonction main
***********************************************************/

// Fonction principale
int main (int argc, char *argv[])
{
        /** Initialisation des timers **/
        long int time_current;
        long int time_lastwrite = 0;

#ifdef _PERF
        perfPrint = NULL;
        perfInitG = NULL;
        perfKernel = NULL;
        perfMemcpyH2DVar = NULL;
        perfMemcpyH2DTab = NULL;
        perfMemcpyD2HVar = NULL;
        perfMemcpyD2HTab = NULL;
        perfCreateWitness = NULL;
        perfCreateFinalTab = NULL;
        perfFree = NULL;
//         perfPrint = CreateSPerf("Affichage");
        perfInitG = CreateSPerf("Init Totale");
        perfKernel = CreateSPerf("Kernel");
        perfMemcpyH2DVar = CreateSPerf("Copie Host vers Device (variables)");
        perfMemcpyH2DTab = CreateSPerf("Copie Host vers Device (tableaux)");
        perfMemcpyD2HVar = CreateSPerf("Copie Device vers Host (variables)");
        perfMemcpyD2HTab = CreateSPerf("Copie Device vers Host (tableaux)");
        perfCreateWitness = CreateSPerf("Creation du fichier temoin (calcul + ecriture)");
        perfCreateFinalTab = CreateSPerf("Creation du fichier final (calcul + ecriture)");
        perfFree = CreateSPerf("Liberation de la memoire");
        StartProcessing(perfInitG);
#endif

	/** Initialisation des constantes du host (en partie recuperees dans le fichier Parametres.txt) **/
	initConstantesHost(argc, argv);


    // read NATM and HATM in profile
    // (if simulation includes atmosphere)
    if ((SIM == -2) || (SIM == 1) || (SIM == 2)) {
        init_profile(&NATM, &HATM, PATHPROFILATM);
    } else {
        HATM = 0;
        NATM = 0;
        TRANSDIR=1.;
    }


    // read LSAAER and LSAOCE
    LSAAER = count_lines(PATHDIFFAER);
    LSAOCE = count_lines(PATHDIFFOCE);
	

	/** Initialisation de la carte graphique **/
	cudaError_t cudaErreur;	// Permet de vérifier les allocations mémoire
	
        // Verification de l'environnement GPU
        DEVICE = CheckGPUContext(DEVICE);
        if (DEVICE < 0){
            printf("\n!!MCCUDA Erreur!! main : erreur au sein de CheckGPUContext()\n");
            exit(1);
        }

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
	
	/** Vérification que le code compilé est compatible avec la simulation demandée **/
	#ifndef FLAGOCEAN
	if( SIM==0 || SIM==2 || SIM==3 ){
		printf("Veuillez compiler avec le flag FLAGOCEAN afin d'utiliser ce milieu \n");
		exit(1);
	}
	#endif
	
	
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
	
	double* tabPhotonsTot; //tableau du poids total des photons sortis
	tabPhotonsTot = (double*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTot)));
	if( tabPhotonsTot == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTot dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTot,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTot)));
	
	double* tabPhotonsTotDown0P; //tableau du poids total des photons sortis
	tabPhotonsTotDown0P = (double*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTotDown0P)));
	if( tabPhotonsTotDown0P == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTotDown0P dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTotDown0P,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTotDown0P)));
	
	double* tabPhotonsTotDown0M; //tableau du poids total des photons sortis
	tabPhotonsTotDown0M = (double*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTotDown0M)));
	if( tabPhotonsTotDown0M == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTotDown0M dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTotDown0M,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTotDown0M)));
	
	double* tabPhotonsTotUp0M; //tableau du poids total des photons sortis
	tabPhotonsTotUp0M = (double*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTotUp0M)));
	if( tabPhotonsTotUp0M == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTotUp0M dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTotUp0M,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTotUp0M)));
	
	double* tabPhotonsTotUp0P; //tableau du poids total des photons sortis
	tabPhotonsTotUp0P = (double*)malloc(4*NBTHETA * NBPHI * sizeof(*(tabPhotonsTotUp0P)));
	if( tabPhotonsTotUp0P == NULL ){
		printf("ERREUR: Problème de malloc de tabPhotonsTotUp0P dans le main\n");
		exit(1);
	}
	
	memset(tabPhotonsTotUp0P,0,4*NBTHETA*NBPHI*sizeof(*(tabPhotonsTotUp0P)));
	
	// Variables permettant le calcul du résultat final
    double *tabFinal;   // tableau final: 4 dimensions pour
                        // R=stokes1+stokes2(dim0) , Q=stokes1-stokes2(dim1),
                        // U=stokes3(dim2)  et Nbphoton(dim4)
    tabFinal = (double*)malloc(4*NBTHETA*NBPHI*sizeof(double));

    double *tabFinalDown0P;   // tableau final: 4 dimensions pour
                        // R=stokes1+stokes2(dim0) , Q=stokes1-stokes2(dim1),
                        // U=stokes3(dim2)  et Nbphoton(dim4)
    tabFinalDown0P = (double*)malloc(4*NBTHETA*NBPHI*sizeof(double));

    double *tabFinalDown0M; 
    tabFinalDown0M = (double*)malloc(4*NBTHETA*NBPHI*sizeof(double));
    double *tabFinalUp0P; 
    tabFinalUp0P = (double*)malloc(4*NBTHETA*NBPHI*sizeof(double));
    double *tabFinalUp0M; 
    tabFinalUp0M = (double*)malloc(4*NBTHETA*NBPHI*sizeof(double));

	double *tabTh;
    tabTh = (double*)malloc(NBTHETA*sizeof(double));
	double *tabPhi;
    tabPhi = (double*)malloc(NBPHI*sizeof(double));
	
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	// Définition et initialisation des constantes initiales du photon
	Init* init_H;
	Init* init_D;
	initInit(&init_H, &init_D);
	#endif
	
#ifdef _PERF
    StopProcessing(perfInitG);
    GetElapsedTime(perfInitG);
#endif

	
	/** Vérification de l'existence ou non d'un fichier témoin **/
    verifierFichier();
	
	#ifdef PARAMETRES
	/** Affichage des paramètres de la simulation **/
	afficheParametres();
	#endif
	
#ifdef _PERF
        StartProcessing(perfInitG);
#endif
	/** Calcul des modèles utiles à l'algorithme **/
	// Calcul de faer, modèle de diffusion des aérosols
    calculF( PATHDIFFAER, tab_H.faer, tab_D.faer, LSAAER, NFAER);
#ifdef _PERF
                StopProcessing(perfInitG);
                GetElapsedTime(perfInitG);
#endif
	
	// Calcul de foce, modèle de diffusion dans l'océan
	#ifdef FLAGOCEAN
	if( SIM==0 || SIM==2 || SIM==3 ){
		calculF( PATHDIFFOCE, tab_H.foce, tab_D.foce, LSAOCE, NFOCE);
        float extoce = ATOT + BTOT;
        W0OCE = BTOT/extoce;
	}
	#endif

#ifdef _PERF
        StartProcessing(perfInitG);
#endif
    if ((SIM == -2) || (SIM == 1) || (SIM == 2)) {
        // Read atmospheric profile
        profilAtm(&tab_H, &tab_D);
        TRANSDIR = exp(-TAUATM/cos(THVDEG*PI/180.));
    }
#ifdef _PERF
        StopProcessing(perfInitG);
        GetElapsedTime(perfInitG);
#endif
	
	
	/** Initialisation des constantes du device à partir des constantes du host **/
#ifdef _PERF
        StartProcessing(perfInitG);
#endif
	initConstantesDevice();

	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	// Calcul du point d'impact du photon
	impactInit(init_H, init_D, &tab_H, &tab_D);
    TRANSDIR = exp(-init_H->taumax0);

	#ifdef DEBUG
	printf("Paramètres initiaux du photon: taumax0=%lf - zintermax=%lf - (%lf,%lf,%lf)\n",\
		   init_H->taumax0, init_H->zintermax0, init_H->x0, init_H->y0, init_H->z0 );
// 	for(int i=0; i<NATM+1; i++)
// 		printf("zph0[%d]=%10.7f - hph0[%d]=%10.7f\n",i, tab_H.zph0[i], i ,tab_H.hph0[i] );
	#endif
	
	#endif /* Fin de la spécification atmosphère sphérique */
	
	
#ifdef _PERF
    StopProcessing(perfInitG);
    GetElapsedTime(perfInitG);
#endif

	/** Fonction qui permet de poursuivre la simulation précédente si elle n'est pas terminee **/
	//lireHDFTemoin(var_H, var_D, &nbPhotonsTot, tabPhotonsTot, tabPhotonsTotDown0P, &tempsPrec);
	
	
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
#ifdef _PERF
        StartProcessing(perfMemcpyH2DVar);
#endif
// 		reinitVariables(var_H, var_D);
		cudaErreur = cudaMemset(&(var_D->nbPhotons), 0, sizeof(var_D->nbPhotons));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset var_D.nbPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
		}
#ifdef _PERF
        StopProcessing(perfMemcpyH2DVar);
        GetElapsedTime(perfMemcpyH2DVar);
#endif
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
		
		
		/** Réinitialisation des données de la simulation **/
#ifdef _PERF
                StartProcessing(perfMemcpyH2DTab);
#endif
		cudaErreur = cudaMemset(tab_D.tabPhotons, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotons)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotons dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }

		cudaErreur = cudaMemset(tab_D.tabPhotonsDown0P, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotonsDown0P)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsDown0P dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }

		cudaErreur = cudaMemset(tab_D.tabPhotonsDown0M, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotonsDown0M)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsDown0M dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }

		cudaErreur = cudaMemset(tab_D.tabPhotonsUp0P, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotonsUp0P)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsUp0P dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }

		cudaErreur = cudaMemset(tab_D.tabPhotonsUp0M, 0, 4*NBTHETA * NBPHI * sizeof(*(tab_D.tabPhotonsUp0M)));
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de cudaMemset tab_D.tabPhotonsUp0M dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }
#ifdef _PERF
                StopProcessing(perfMemcpyH2DTab);
                GetElapsedTime(perfMemcpyH2DTab);
#endif
		
#ifdef _PERF
                StartProcessing(perfKernel);
#endif
		/** Lancement du kernel **/
		lancementKernel<<<gridSize, blockSize>>>(var_D, tab_D
				#ifdef SPHERIQUE
				, init_D
				#endif
							);
		// Attend que tous les threads aient fini avant de faire autre chose
// 		cudaThreadSynchronize();
#ifdef _PERF
                cudaThreadSynchronize();
                StopProcessing(perfKernel);
                GetElapsedTime(perfKernel);
#endif
		
#ifdef _PERF
                StartProcessing(perfMemcpyD2HVar);
#endif
		/** Récupération des variables et d'un tableau envoyés dans le kernel **/
		cudaErreur = cudaMemcpy(var_H, var_D, sizeof(Variables), cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf("#--------------------#\n");
			printf("# ERREUR: Problème de copie var_D dans le main\n");
			printf("# Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			printf("#--------------------#\n");
			exit(1);
                }
#ifdef _PERF
                        StopProcessing(perfMemcpyD2HVar);
                        GetElapsedTime(perfMemcpyD2HVar);
#endif

		// On remplit les variables et tableau qui restent dans le host
		nbPhotonsTot += var_H->nbPhotons;

#ifdef _PERF
                StartProcessing(perfMemcpyD2HTab);
#endif
		/** Copie des informations du device pour la création du fichier témoin **/
		/* Il a été remarqué que sans cette copie et remise à zéro du tableau tab_D.tabPhotons, des erreurs apparaissent si les
valeurs stockées sont élevées. Ceci doit venir du fait que l'on somme une grosse valeur à une plus faible */
		cudaErreur = cudaMemcpy(tab_H.tabPhotons, tab_D.tabPhotons, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotons)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotons dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}

		cudaErreur = cudaMemcpy(tab_H.tabPhotonsDown0P, tab_D.tabPhotonsDown0P, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotonsDown0P)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotonsDown0P dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}

		cudaErreur = cudaMemcpy(tab_H.tabPhotonsDown0M, tab_D.tabPhotonsDown0M, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotonsDown0M)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotonsDown0M dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}

		cudaErreur = cudaMemcpy(tab_H.tabPhotonsUp0P, tab_D.tabPhotonsUp0P, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotonsUp0P)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotonsUp0P dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
		}

		cudaErreur = cudaMemcpy(tab_H.tabPhotonsUp0M, tab_D.tabPhotonsUp0M, 4*NBTHETA * NBPHI * sizeof(*(tab_H.tabPhotonsUp0M)),
cudaMemcpyDeviceToHost);
		if( cudaErreur != cudaSuccess ){
			printf( "ERREUR: Problème de copie tab_H.tabPhotonsUp0M dans le main\n");
			printf( "Nature de l'erreur: %s\n",cudaGetErrorString(cudaErreur) );
			exit(1);
        }
#ifdef _PERF
                        StopProcessing(perfMemcpyD2HTab);
                        GetElapsedTime(perfMemcpyD2HTab);
#endif

#ifdef _PERF
                StartProcessing(perfCreateWitness);
#endif
		/** Creation d'un fichier témoin pour pouvoir reprendre la simulation en cas d'arrêt **/
		for(int i = 0; i < 4*NBTHETA*NBPHI; i++) {
			tabPhotonsTot[i] += (double) tab_H.tabPhotons[i];
			tabPhotonsTotDown0P[i] += (double) tab_H.tabPhotonsDown0P[i];
			tabPhotonsTotDown0M[i] += (double) tab_H.tabPhotonsDown0M[i];
			tabPhotonsTotUp0P[i] += (double) tab_H.tabPhotonsUp0P[i];
			tabPhotonsTotUp0M[i] += (double) tab_H.tabPhotonsUp0M[i];
        }
		
        time_current = clock() / CLOCKS_PER_SEC;
		if ((WRITE_PERIOD > 0) && ((time_current - time_lastwrite > 60*WRITE_PERIOD) || (time_lastwrite == 0))) {
            printf("== WRITING WITNESS FILE ==\n"); // FIXME
            //creerHDFTemoin(tabPhotonsTot, tabPhotonsTotDown0P, nbPhotonsTot,var_H, tempsPrec);
            time_lastwrite = time_current;
        }
#ifdef _PERF
                        StopProcessing(perfCreateWitness);
                        GetElapsedTime(perfCreateWitness);
#endif
		
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
	
	
#ifdef _PERF
        StartProcessing(perfCreateFinalTab);
#endif
	/** Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère,
	 * par unité de surface) 
	**/	
	// Remplissage des 3 tableaux
	calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, nbPhotonsTot);
	calculTabFinal(tabFinalDown0P, tabTh, tabPhi, tabPhotonsTotDown0P, nbPhotonsTot);
	calculTabFinal(tabFinalDown0M, tabTh, tabPhi, tabPhotonsTotDown0M, nbPhotonsTot);
	calculTabFinal(tabFinalUp0P, tabTh, tabPhi, tabPhotonsTotUp0P, nbPhotonsTot);
	calculTabFinal(tabFinalUp0M, tabTh, tabPhi, tabPhotonsTotUp0M, nbPhotonsTot);

	
	/** Fonction qui crée le fichier .hdf contenant le résultat final sur la demi-sphère **/
//	creerHDFResultats(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
//	creerHDFResultats(tabFinal, tabFinalDown0P, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	creerHDFResultats(tabFinal, tabFinalDown0P, tabFinalDown0M, tabFinalUp0P, tabFinalUp0M, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
#ifdef _PERF
        StopProcessing(perfCreateFinalTab);
        GetElapsedTime(perfCreateFinalTab);
#endif
	printf(" Fin de l'execution du programme. Resultats stockes dans %s\n",PATHRESULTATSHDF);
	

#ifdef _PERF
        StartProcessing(perfFree);
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
    free(tabFinal);
    free(tabFinalDown0P);
    free(tabFinalDown0M);
    free(tabFinalUp0P);
    free(tabFinalUp0M);
    free(tabPhi);
    free(tabTh);

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
	free( tabPhotonsTot );
	free( tabPhotonsTotDown0P );
	free( tabPhotonsTotDown0M );
	free( tabPhotonsTotUp0P );
	free( tabPhotonsTotUp0M );

	
#ifdef _PERF
        StopProcessing(perfFree);
        GetElapsedTime(perfFree);
#endif

#ifdef _PERF
        if ( perfPrint != NULL )
            printf("\n %s...%u us", GetName(perfPrint), GetTotalElapsedTime(perfPrint));
        if ( perfInitG != NULL )
            printf("\n %s...%u us", GetName(perfInitG), GetTotalElapsedTime(perfInitG));
        if ( perfKernel != NULL )
            printf("\n %s...%u us", GetName(perfKernel), GetTotalElapsedTime(perfKernel));
        if ( perfMemcpyH2DVar != NULL )
            printf("\n %s...%u us", GetName(perfMemcpyH2DVar), GetTotalElapsedTime(perfMemcpyH2DVar));
        if ( perfMemcpyH2DTab != NULL )
            printf("\n %s...%u us", GetName(perfMemcpyH2DTab), GetTotalElapsedTime(perfMemcpyH2DTab));
        if ( perfMemcpyD2HVar != NULL )
            printf("\n %s...%u us", GetName(perfMemcpyD2HVar), GetTotalElapsedTime(perfMemcpyD2HVar));
        if ( perfMemcpyD2HTab != NULL )
            printf("\n %s...%u us", GetName(perfMemcpyD2HTab), GetTotalElapsedTime(perfMemcpyD2HTab));
        if ( perfCreateWitness != NULL )
            printf("\n %s...%u us", GetName(perfCreateWitness), GetTotalElapsedTime(perfCreateWitness));
        if ( perfCreateFinalTab != NULL )
            printf("\n %s...%u us", GetName(perfCreateFinalTab), GetTotalElapsedTime(perfCreateFinalTab));
        if ( perfFree != NULL )
            printf("\n %s...%u us", GetName(perfFree), GetTotalElapsedTime(perfFree));
        DeleteSPerf(perfPrint);
        DeleteSPerf(perfInitG);
        DeleteSPerf(perfKernel);
        DeleteSPerf(perfMemcpyH2DVar);
        DeleteSPerf(perfMemcpyH2DTab);
        DeleteSPerf(perfMemcpyD2HVar);
        DeleteSPerf(perfMemcpyD2HTab);
        DeleteSPerf(perfCreateWitness);
        DeleteSPerf(perfCreateFinalTab);
        DeleteSPerf(perfFree);
        printf("\n");
#endif

    message_end(DEVICE);

        //
        cudaDeviceReset();
	return 0;

}
