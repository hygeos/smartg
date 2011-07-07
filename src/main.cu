
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "main.h"
#include "device.h"
#include "host.h"

	  ///////////////////
	 // FONCTION MAIN //
	///////////////////

// Fonction principale
int main()
{
	#ifdef TEMPS
	// Récupération du temps initial
	clock_t start, finish;
	double duration;
	start = clock();
	#endif
	
// Organisation des threads en blocks de threads et en grids de blocks
	dim3 blockSize(XBLOCK,YBLOCK);
	dim3 gridSize(XGRID,YGRID);
	
// Fonction aléatoire : Allocate memory for RNG's
	Random random_H[XBLOCK * YBLOCK * XGRID * YGRID];
	Random* random_D;
	cudaMalloc(&random_D, XBLOCK * YBLOCK * XGRID * YGRID * sizeof(Random));
	initRandom(random_H, random_D);
	
// Constantes à envoyer dans le kernel
	Constantes constantes_H;
	Constantes* constantes_D;
	cudaMalloc(&(constantes_D), sizeof(Constantes));
	initConstantes(&constantes_H, constantes_D);

// Variables de récupération d'information (avancement)
	unsigned long long nbPhotonsTot = 0; //nombre total de photons traités
	#ifdef PROGRESSION
	unsigned long long nbPhotonsSorTot = 0; //nombre total de photons ressortis
	#endif
	Progress progress_H;
	Progress* progress_D;
	cudaMalloc(&progress_D, sizeof(Progress));
	initProgress(&progress_H, progress_D);
	
// Variables de récupération d'informations (tableaux)
	unsigned long long tabPhotonsTot[NBTHETA * NBPHI * NBSTOKES] = {0}; //tableau regroupant le poids de tous les photons ressortis sur une demi-sphère pour chaque Stokes
	unsigned long long* tabPhotons_D; //tableau des poids du photons ressortis pour un appel du Kernel (Device)
	cudaMalloc(&tabPhotons_D, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
	unsigned long long tabPhotons_H[NBTHETA * NBPHI * NBSTOKES]; //tableau du poids des photons ressortis pour un appel du Kernel (Host)

	#ifdef TABNBPHOTONS
	unsigned long long tabNbPhotonsTot[NBTHETA * NBPHI] = {0}; //tableau regroupant le nombre total de photons ressortis sur une demi-sphère
	unsigned long long* tabNbPhotons_D; //tableau du nombre de photons ressortis pour un appel du Kernel (Device)
	cudaMalloc(&tabNbPhotons_D, NBTHETA * NBPHI * sizeof(unsigned long long));
	unsigned long long tabNbPhotons_H[NBTHETA * NBPHI]; //tableau du nombre de photons ressortis pour un appel du Kernel (Host)
	#endif

// Variables de récupération d'informations (trajet)
	#ifdef TRAJET
	Evnt evnt_H[20];
	Evnt* evnt_D;
	cudaMalloc(&evnt_D, 20 * sizeof(Evnt));
	initEvnt(evnt_H, evnt_D);
	#endif
	
	// Affichage des paramètres de la simulation
	#ifdef PARAMETRES
	afficheParametres();
	#endif
	// Affichage de l'avancement de la simulation
	#ifdef PROGRESSION
	printf("\n");
	#endif

	// Tant qu'il n'y a pas assez de photons traités on relance le kernel
	while(nbPhotonsTot < NBPHOTONS)
	{
		reinitProgress(&progress_H, progress_D);

		// Le tableau du poids des photons ressortis pour un appel du Kernel est remis à zéro
		cudaMemset(tabPhotons_D, 0, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));
		memset(tabPhotons_H, 0, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long));

		#ifdef TABNBPHOTONS
		// Le tableau du nombre de photons ressortis pour un appel du Kernel est remis à zéro
		cudaMemset(tabNbPhotons_D, 0, NBTHETA * NBPHI * sizeof(unsigned long long));
		memset(tabNbPhotons_H, 0, NBTHETA * NBPHI * sizeof(unsigned long long));
		#endif

		// Lancement du kernel
		lancementKernel<<<gridSize, blockSize>>>(random_D, //variables fonction aléatoire
				constantes_D, //constantes
				progress_D //récupération d'informations
				, tabPhotons_D //récupération d'informations
				
				#ifdef TABNBPHOTONS
				, tabNbPhotons_D //récupération d'informations
				#endif
				
				#ifdef TRAJET
				, evnt_D //récupération d'informations
				#endif
							); 

		// Récupération des variables envoyées dans le kernel
		cudaMemcpy(&progress_H, progress_D, sizeof(Progress), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tabPhotons_H, tabPhotons_D, NBTHETA * NBPHI * NBSTOKES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		
		#ifdef TABNBPHOTONS
		cudaMemcpy(&tabNbPhotons_H, tabNbPhotons_D, NBTHETA * NBPHI * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		#endif
		
		// Remplissage des variables "total"
		nbPhotonsTot  += progress_H.nbPhotons;
		
		#ifdef PROGRESSION
		nbPhotonsSorTot += progress_H.nbPhotonsSor;
		#endif
		
		for(int i = 0; i < NBTHETA * NBPHI * NBSTOKES; i++)
		{
			#ifdef CONTROLE
			if(tabPhotons_H[i] + tabPhotonsTot[i] < tabPhotonsTot[i])
			{
				printf("\nERREUR : main tabPhotonsTot\n");
				return 1;
			}
			#endif
			tabPhotonsTot[i] += tabPhotons_H[i];
		}

		#ifdef TABNBPHOTONS
		for(int i = 0; i < NBTHETA * NBPHI; i++)  tabNbPhotonsTot[i] += tabNbPhotons_H[i];
		#endif
		
		// Affichage de l'avancement de la simulation
		#ifdef PROGRESSION
		afficheProgress(nbPhotonsTot, nbPhotonsSorTot, &progress_H);
		#endif
	}

	#ifdef TRAJET
	// Récupération des variables envoyées dans le kernel
	cudaMemcpy(evnt_H, evnt_D, 20 * sizeof(Evnt), cudaMemcpyDeviceToHost);
	// Affichage du trajet du premier thread
	afficheTrajet(evnt_H);
	#endif

	#ifdef TABSTOKES
	// Affichage des tableaux "finaux" pour chaque nombre de Stokes
	afficheTabStokes(tabPhotonsTot);
	#endif

	#ifdef TABNBPHOTONS
	// Affichage du tableau regroupant le nombre de photons ressortis sur une demi-sphère
	afficheTabNbPhotons(tabNbPhotonsTot);
	#endif

	// Création et calcul du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de surface)
	float tabFinal[NBTHETA * NBPHI]; //tableau final
	float tabTh[NBTHETA]; //tableau contenant l'angle theta de chaque morceau de sphère
	float tabPhi[NBPHI]; //tableau contenant l'angle psi de chaque morceau de sphère
	// Remplissage des 3 tableaux
	calculTabFinal(tabFinal, tabTh, tabPhi, tabPhotonsTot, &progress_H, nbPhotonsTot);

	#ifdef TABFINAL
	// Calcul et affichage du tableau final (regroupant le poids de tous les photons ressortis sur une demi-sphère, par unité de surface)
	afficheTabFinal(tabFinal);
	#endif

	// Fonction qui crée le fichier .hdf contenant le résultat final sur la demi-sphère
	creerHDFResultats(tabFinal, tabTh, tabPhi);

	#ifdef QUART
	// Fonction qui crée le fichier .hdf contenant le résultat final reporté sur un quart de sphère
	creerHDFResultatsQuartsphere(tabFinal, tabTh, tabPhi);
	#endif

	#ifdef COMPARAISON
	// Fonction qui crée le fichier .hdf permettant de comparer les 2 quarts de sphère
	creerHDFComparaison(tabFinal, tabTh, tabPhi);
	#endif

/////////////////////////////////////////////////////////////////////////////////////
	/*
	comp_info c_info;
	comp_coder_t comp_type = COMP_CODE_DEFLATE; // Gzip
	c_info.deflate.level = 9;
	SDsetcompress(sdsSymetrie, comp_type, &c_info);
	
	if (hdf_datatype == DFNT_FLOAT32) {
	float fillvalue_float = NaN;
	SDsetfillvalue(sds_id, &fillvalue_float);
}
	*/
	/*
	FILE* fichier;
	char nomFichier[30] = "Resultats";
	
	char texte[30] = "blablabla";
	int i = 12;
	unsigned long long j = 123456789123456789;
	float f = 2.014;

	fichier = fopen(nomFichier, "w");
	
	fprintf(fichier, "texte : %s\n", texte);
	fprintf(fichier, "entier : %d\n", i);
	fprintf(fichier, "ull : %lu\n", j);
	fprintf(fichier, "float : %f\n", f);
	
	fclose(fichier);
	
	//Deuxième partie : Lire et afficher le contenu du fichier
	P_FICHIER = fopen(NOM_FICHIER, "r");
	C = 0;
	while (!feof(P_FICHIER))
	{
	fscanf(P_FICHIER, "%s\n", NOM_PERS);
	printf("NOM : %s\n", NOM_PERS);
	C++;
}
	fclose(P_FICHIER);
	*/

	/*
	DFNT_CHAR8 (4) 8-bit character type
	DFNT_UCHAR8 (3) 8-bit unsigned character type
	DFNT_INT8 (20) 8-bit integer type
	DFNT_UINT8 (21) 8-bit unsigned integer type
	DFNT_INT16 (22) 16-bit integer type
	DFNT_UINT16 (23) 16-bit unsigned integer type
	DFNT_INT32 (24) 32-bit integer type
	DFNT_UINT32 (25) 32-bit unsigned integer type
	DFNT_FLOAT32 (5) 32-bit floating-point type
	DFNT_FLOAT64 (6) 64-bit floating-point type
	*/

	
/////////////////////////////////////////////////////////////////////////////////////
	
	//Libération de la mémoire
	cudaFree(random_D);
	cudaFree(constantes_D);
	cudaFree(progress_D);
	cudaFree(tabPhotons_D);

	#ifdef TABNBPHOTONS
	cudaFree(tabNbPhotons_D);
	#endif

	#ifdef TRAJET
	cudaFree(evnt_D);
	#endif

	#ifdef TEMPS
	// Récupération du temps final et affichage du temps total
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\n%.1fs\n", duration);
	#endif
}
