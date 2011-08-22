
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
	// Suppression des anciens fichiers hdf pour éviter les confusions
	system("rm -f out_prog/*");
	
	// Initialisation des constantes du host (en partie recuperees dans le fichier Parametres.txt)
	initConstantesHost(argc, argv);
	// Initialisation des constantes du device à partir des constantes du host
	initConstantesDevice();
	
	// DEBUG : Affichage basique des parametres de la simulation
	printf("\n%lu - %u - %d - %d - %d - %d - %d - %d\n", NBPHOTONS,NBLOOP,XBLOCK,YBLOCK,XGRID,YGRID,NBTHETA,NBPHI);

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
	unsigned long long* tabPhotonsTot; //tableau du poids total des photons sortis
	tabPhotonsTot = (unsigned long long*)malloc(NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));

	#ifdef TABRAND
	// DEBUG Recuperations des nombres aleatoires du random en place
	float tableauRand_H[100] = {0};
	float* tableauRand_D;
	cudaMalloc(&tableauRand_D, 100 * sizeof(float));
	cudaMemset(tableauRand_D, 0, 100 * sizeof(float));
	#endif
	
	// Fonction qui permet de poursuivre la simulation précédente si elle n'est pas terminee
	double tempsPrec = 0.; //temps ecoule de la simulation precedente
//lireHDFTemoin(var_H, var_D, &nbPhotonsTot, tabPhotonsTot, &tempsPrec);
	
	#ifdef TRAJET
	// DEBUG : Variables permettant de récupérer le début du trajet d'un photon
	Evnt evnt_H[20];
	Evnt* evnt_D;
	cudaMalloc(&evnt_D, 20 * sizeof(Evnt));
	initEvnt(evnt_H, evnt_D);
	#endif
	
	// Organisation des threads en blocks de threads et en grids de blocks
	dim3 blockSize(XBLOCK,YBLOCK);
	dim3 gridSize(XGRID,YGRID);
	
	// Affichage des paramètres de la simulation
	#ifdef PARAMETRES
	afficheParametres();
	#endif
	
	// Tant qu'il n'y a pas assez de photons traités on relance le kernel
	while(nbPhotonsTot < NBPHOTONS)
	{
		// Remise à zéro de certaines variables et certains tableaux
		reinitVariables(var_H, var_D);
		cudaMemset(tab_D.tabPhotons, 0, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
		
		// Lancement du kernel
		lancementKernel<<<gridSize, blockSize>>>(var_D, tab_D			
				#ifdef TABRAND
				, tableauRand_D
				#endif
				#ifdef TRAJET
				, evnt_D //récupération d'un trajet de photons
				#endif
							);
		// Attend que tous les threads avant de faire autre chose
		cudaThreadSynchronize();
		
		// Récupération des variables et d'un tableau envoyés dans le kernel
		cudaMemcpy(var_H, var_D, sizeof(Variables), cudaMemcpyDeviceToHost);
		cudaMemcpy(tab_H.tabPhotons, tab_D.tabPhotons, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
				
		// On remplit les variables et tableau qui restent dans le host
		nbPhotonsTot += var_H->nbPhotons;
		#ifdef PROGRESSION
		nbPhotonsSorTot += var_H->nbPhotonsSor;
		#endif
		for(int i = 0; i < NBTHETA * NBPHI * NBSTOKES; i++)
			tabPhotonsTot[i] += tab_H.tabPhotons[i];
		
		// Creation d'un fichier témoin pour pouvoir reprendre la simulation en cas d'arrêt
		creerHDFTemoin(tabPhotonsTot, nbPhotonsTot, var_H, tempsPrec);
		
		// Affichage de l'avancement de la simulation
		afficheProgress(nbPhotonsTot, var_H, tempsPrec
			#ifdef PROGRESSION
			, nbPhotonsSorTot
			#endif
			       );
	}
	
	#ifdef TABRAND
	// DEBUG Recuperations et affichage des nombres aleatoires du random
	cudaMemcpy(tableauRand_H, tableauRand_D, 100 * sizeof(float), cudaMemcpyDeviceToHost);
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
	cudaMemcpy(evnt_H, evnt_D, 20 * sizeof(Evnt), cudaMemcpyDeviceToHost);
	// Affichage du trajet du premier thread
	afficheTrajet(evnt_H);
	#endif
	
	#ifdef TABSTOKES
	// Affichage des tableaux "finaux" pour chaque nombre de Stokes
	afficheTabStokes(tabPhotonsTot);
	#endif

	// Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de surface)
	float tabFinal[NBTHETA * NBPHI]; //tableau final
	float tabTh[NBTHETA]; //tableau contenant l'angle theta de chaque morceau de sphère
	float tabPhi[NBPHI]; //tableau contenant l'angle psi de chaque morceau de sphère
	// Remplissage des 3 tableaux
	calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, nbPhotonsTot);

	#ifdef TABFINAL
	// DEBUG Affichage du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de surface)
	afficheTabFinal(tabFinal);
	#endif

	// Fonction qui crée le fichier .hdf contenant le résultat final reporté sur un quart de sphère
	#ifdef QUART
	creerHDFResultatsQuartsphere(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	#endif
	// DEBUG Fonction qui crée le fichier .hdf permettant de comparer les 2 quarts de sphère
	#ifdef COMPARAISON
	creerHDFComparaison(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	#endif
	// Fonction qui crée le fichier .hdf contenant le résultat final sur la demi-sphère
	creerHDFResultats(tabFinal, tabTh, tabPhi, nbPhotonsTot, var_H, tempsPrec);
	// Suppression du fichier Temoin.hdf
	remove("tmp/Temoin.hdf");

	// Libération du groupe de variables envoyé dans le kernel
	cudaFree(var_D);
	free(var_H);
	// Libération des tableaux envoyés dans le kernel
	freeTableaux(&tab_H, &tab_D);
	// Libération du tableau du host
	free(tabPhotonsTot);
	// Libération des variables qui récupèrent le trajet d'un photon
	#ifdef TRAJET
	cudaFree(evnt_D);
	#endif
	
	#ifdef TABRAND
	//DEBUG random
	cudaFree(tableauRand_D);
	#endif
}
