
/**********************************************************
*
*			device.cu
*
*	> Kernel
*	> Modélisation phénomènes physiques
*	> Initialisation de données dans le device
*	> Fonctions liées au générateur aléatoire
*
***********************************************************/


/**********************************************************
*	> Includes
***********************************************************/

#include "communs.h"
#include "device.h"


/**********************************************************
*	> Kernel
***********************************************************/

/* lancementKernel
* Kernel de lancement et gestion de la simulation
* Les fonctions de plus bas niveau sont appelées en fonction de la localisation du photon
* Il peut être important de rappeler que le kernel lance tous les threads mais effectue des calculs similaires. La boucle de la
* fonction va donc être effectuée pour chaque thread du block de la grille
*/
__global__ void lancementKernel(Variables* var, Tableaux tab
		#ifdef SPHERIQUE
		, Init* init
		#endif
		#ifdef TABRAND
		, float* tableauRand
		#endif
		#ifdef TRAJET
		, Evnt* evnt
		#endif
			       )
{
	// idx est l'indice du thread considéré
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);

	// Paramètres de la fonction random en mémoire locale
	#ifdef RANDMWC
	unsigned long long etatThr;
	unsigned int configThr;
	configThr = tab.config[idx];
	etatThr = tab.etat[idx];
	#endif
	#ifdef RANDCUDA
	curandState_t etatThr;
	etatThr = tab.etat[idx];
	#endif
	#ifdef RANDMT
	ConfigMT configThr;
	EtatMT etatThr;
	configThr = tab.config[idx];
	etatThr = tab.etat[idx];
	#endif

	#ifdef TABRAND
	// DEBUG Recuperation des nombres aleatoires generes par la fonction random utilisée
	if(idx < 5)
		if(tableauRand[50] == 0.f)
		{
			int k = 0;
			if(tableauRand[0] != 0.f) k = 50;
			for(int j = 0; j < 10; j++) tableauRand[k+idx*10+j] = RANDDEBUG;
		}
	#endif
	
	// Création de variable propres à chaque thread
	unsigned long long nbPhotonsThr = 0; 	// Nombre de photons traités par le thread
	
	#ifndef SPHERIQUE
	int flagDiff = DIFFFd;
	#endif
	
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; 		// Nombre de photons traités par le thread et ressortis dans l'espace
	#endif
	
	Photon ph; 		// On associe une structure de photon au thread
	ph.loc = NONE;	// Initialement le photon n'est nulle part, il doit être initialisé
	
	/** Mesure du temps d'execution **/
	#ifdef TEMPS
	clock_t start, stop;
	float time;
	#endif
	
	
	/** Boucle de calcul **/   
	// Dans cette boucle on simule le parcours du photon, puis on le réinitialise,... Le thread lance plusieurs photons
	for(unsigned int iloop= 0; iloop < NBLOOPd; iloop++)
	{
		// Si le photon est à NONE on l'initialise et on le met à ATMOS
		if(ph.loc == NONE){
			
			#ifdef TEMPS
			if(idx==0){
				start = clock();
			}
			
			#endif
			
			initPhoton(&ph/*, hph0_s, zph0_s*/
				#ifdef SPHERIQUE
				, tab, init
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
					);

			#ifdef TEMPS
			if(idx==0){
				stop = clock();
				time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
				printf("(1) Temps de initPhoton: %f\n", time);
			}
			#endif
			
			#ifndef SPHERIQUE
			flagDiff = DIFFFd;
			#endif
			
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		
		// Si le photon est à ATMOS on le fait avancer jusqu'à SURFACE, ou SPACE, ou ATMOS s'il subit une diffusion
		if( (ph.loc == ATMOS) /*&& (SIMd==-2 || SIMd==1 || SIMd==2)*/ ){
			
			#ifdef TEMPS
			if(idx==0){
				start = clock();
			}
			#endif
			
			move(&ph
				#ifndef SPHERIQUE
				,flagDiff
				#endif
				#ifdef SPHERIQUE
				, tab, init
				#endif
				#ifdef DEBUG
				, var
				#endif
				, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
						);
						
			#ifdef TEMPS
			if(idx==0){
				stop = clock();
				time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
				printf("(2) Temps de move: %f\n", time);
			}
			#endif
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est encore à ATMOS il subit une diffusion et reste dans ATMOS
		if( (ph.loc == ATMOS) /*&& (SIMd==-2 || SIMd==1 || SIMd==2)*/){
	
			#ifdef TEMPS
			if(idx==0){
				start = clock();
			}
			#endif
			
			// Diffusion
			scatter( &ph, tab, &etatThr
			#if defined(RANDMWC) || defined(RANDMT)
			, &configThr
			#endif
			#ifdef TRAJET
			, idx, evnt
			#endif
				);
				
			#ifdef TEMPS
			if(idx==0){
				stop = clock();
				time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
				printf("(3) Temps de scatter: %f\n", time);
			}
			#endif

		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est à SURFACE
		if(ph.loc == SURFACE) {
			
			if( DIOPTREd!=3 )surfaceAgitee(&ph, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
						);
						
			else
				surfaceLambertienne(&ph, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
				);
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est dans SPACE ou ABSORBED on récupère ses infos et on le remet à NONE
		
		// Gain de tps
		if(ph.loc == ABSORBED){
			ph.loc = NONE;
			nbPhotonsThr++;

		}
		syncthreads();
		
		if(ph.loc == SPACE){
			exit(&ph, var, tab, &nbPhotonsThr
						#ifdef PROGRESSION
						, &nbPhotonsSorThr
						#endif
						#ifdef TRAJET
						, idx, evnt
						#endif
						);
		}
		syncthreads();
		

		#ifndef SPHERIQUE	/* Code spécifique à une atmosphère parallèle */
		//Mise à jour du poids suite à la 1ère diffusion forcée
		if(flagDiff==1 ){
			ph.weight *= (1.F - __expf(-TAUMAXd));
			flagDiff=0;
		}
		syncthreads();
		#endif

	}// Fin boucle for
	

	// Après la boucle on rassemble les nombres de photons traités par chaque thread
	atomicAdd(&(var->nbPhotons), nbPhotonsThr);
	
	#ifdef PROGRESSION
	// On rassemble les nombres de photons traités et sortis de chaque thread
	atomicAdd(&(var->nbPhotonsSor), nbPhotonsSorThr);

	// On incrémente avncement qui compte le nombre d'appels du Kernel
	atomicAdd(&(var->nbThreads), 1);
	#endif
	
	// Sauvegarde de l'état du random pour que les nombres ne soient pas identiques à chaque appel du kernel
	tab.etat[idx] = etatThr;
}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph/*, float* hph0_s, float* zph0_s*/
		#ifdef SPHERIQUE
		, Tableaux tab, Init* init
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		    )
{	
	// Initialisation du vecteur vitesse
	ph->vx = - STHSd;
	ph->vy = 0.F;
	ph->vz = - CTHSd;
	
	// Initialisation du vecteur orthogonal au vecteur vitesse
	ph->ux = -ph->vz;
	ph->uy = 0.F;
	ph->uz = ph->vx;
	
	// Le photon est initialement dans l'atmosphère, et tau peut être vu comme sa hauteur par rapport au sol
	if( SIMd!=-1)
		ph->loc = ATMOS;
	else
		ph->loc = SURFACE;
	
	ph->weight = WEIGHTINIT;
	// Initialisation des paramètres de stokes du photon
	ph->stokes1 = 0.5F;
	ph->stokes2 = 0.5F;
	ph->stokes3 = 0.F;
// 	ph->stokes4 = 0.F;


	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	ph->locPrec=NONE;
	
// 	Paramètres initiaux calculés dans impactInit - host.cu
	ph->x = init->x0;
	ph->y = init->y0;
	ph->z = init->z0;
	ph->couche=0;	// Sommet de l'atmosphère
	ph->rayon = sqrt(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z );
	#endif
	
	#ifndef SPHERIQUE	/* Code spécifique à une atmosphère en plan parallèle */
	ph->tau = TAUATMd;
	#endif
	
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
			// "1"représente l'événement "initialisation" du photon
			evnt[i].action = 1;
			// On récupère le tau et le poids du photon
			evnt[i].poids = ph->weight;
	}
	#endif
}


/* move
* Effectue le déplacement du photon dans l'atmosphère
* Pour l'atmosphère sphèrique, l'algorithme est basé sur la formule de pythagore généralisée.
* Modification des coordonnées position du photon
*/
__device__ void move(Photon* ph
		#ifndef SPHERIQUE
		,int flagDiff
		#endif
		#ifdef SPHERIQUE
		, Tableaux tab, Init* init
		#endif
		#ifdef DEBUG
		, Variables* var
		#endif
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		    )
{
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	float rsolfi=666;
	float delta;
	
	float tauRdm;	// Epaisseur optique aléatoire tirée
	
	// Permet la sauvegarde du profil parcouru par le photon
	float zph = 0.f;
	float zph_p = 0.f;
	float hph = 0.f;
	float hph_p = 0.f;
	
	float vzn;				// projeté de vz sur l'axe defini par le photon et le centre de la terre
	float sinth;			// Sinus de l'angle entre z' et Vz'
	float costh;			// Cosinus de l'angle entre z' et Vz'
	float ztangentielle;
	
	
	float rayon2;
	float rayon;
	int icouchefi = 0;
	int icompteur = 0;
	int coucheTangentielle = 666;
	int sens;			// Sens=1 si le photon monte, sens=-1 si il descend. Il est initialisé à 1 pour gagner du temps
	int icoucheTemp; 		// Couche suivante que le photon va toucher
	int flagSortie = 0;		// Indique si le photon va sortir sans interaction dans l'atmosphère
	
	float rra, rdist;
	
	#ifdef DEBUG
	double rsol1,rsol2;
	#endif
	
	/** Mesure du temps d'execution **/
	#ifdef TEMPS
	clock_t start, stop;
	float time;
	#endif
	
	/** Tirage au sort de la profondeur optique à parcourir **/
	/*  Tirage effectué lors de chaque appel de la fonction */
	// Pas de diffusion forcée en sphérique
	
	tauRdm = -logf(1.F-RAND);
// 	tauRdm=0.5;
	
	if( tauRdm == 0. ){
		/* Le photon ne bouge pas mais il faut tout de même considéré le fait qu'il a subit un déplacement "nul"
		*/
		ph->locPrec = ATMOS;
		return;
	}


	/** Calcul puis déduction de la couche du photon **/
	#ifdef TEMPS
	if(idx==0){
		start = clock();
	}
	#endif
	
	if( ph->locPrec==NONE ){
		/* Le photon vient de l'espace et rentre pour la première fois dans l'atmosphère
		*/
		
		// Le photon descend forcement car il vient du sommet de l'atmosphère
		sens = -1;
		
		// Si tauRdm est plus élevé que Taumax, le photon va directement heurter la surface
		if( tauRdm >= (init->taumax0) ){
			flagSortie = 1;
			zph=tab.zph0[NATMd]; /* Pour retrouver le zintermax ensuite. Cette valeur signifie que le photon a traversé toute
									l'atmosphère */
		}


		while( (hph < tauRdm) && (flagSortie!=1) ){
		/* Le photon vient du sommet de l'atmosphère - On parcourt le profil initial calculé dans impact.
		*/

			#ifdef DEBUG
			if( icompteur==(NATMd+1) ){
				printf("icompteur = NATMd+1 pour premier calcul de position - tauRdm=%f - taumax=%f - hph_p=%f - hph=%f\n",\
					tauRdm, ph->taumax, hph_p, hph);
					flagSortie = 1;
					break;
			}
			#endif
			
			// Sauvegarde du calcul de la couche précédente
			hph_p = hph;
			zph_p = zph;
			
			hph = tab.hph0[icompteur];
			zph = tab.zph0[icompteur];
			
			icompteur++;
		}
		
		#ifdef TEMPS
		if(idx==0){
			stop = clock();
			time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
			printf("(2.1) Temps de move pour une 1ère intéraction: %f\n", time);
		}
		#endif
	}


	else if( ((ph->locPrec==ATMOS)||(ph->locPrec==SURFACE)) ){
		/* Le photon vient de l'atmosphère ou de la surface
		* Son profil est calculé jusqu'à arriver à la position voulue, c'est à dire que le photon parcourt l'épaisseur optique TauRdm
		* zph est la distance parcourue par le photon, hph est l'épaisseur optique parcourue.
		*/


		/** Test initial pour le photon venant de la surface **/
		/* Il faut abandonner le photon si ce test est positif
		* Il signifie que sur une surface agitée, le photon peut être réfléchi mais toujours se diriger vers la surface
		*/
// 		if( (ph->locPrec==SURFACE)&&
// 			( sqrtf( (ph->x+ph->vx)*(ph->x+ph->vx) + (ph->y+ph->vy)*(ph->y+ph->vy) + (ph->z+ph->vz)*(ph->z+ph->vz))<RTER ) ){
// //    			printf("pb#1 - Z=(%lf, %lf, %lf) - V=(%lf, %lf, %lf)\n",\
// 			ph->x, ph->y, ph->z, ph->vx, ph->vy, ph->vz);
// 			ph->loc=ABSORBED;
// 			return;
// 		}
		
		/** Changement de repère **/
		/* calcul du nouveau z', axe passant par le centre de la terre et le photon
		* Cette axe permet de définir une projection de vz donnant la direction locale du photon
		*/
		
		rayon = ph->rayon;
		rayon2 = rayon*rayon;

		// Calcul du Vz' par rapport à z'. Son signe donne la direction du photon
		// Vz'=vzn= V.Z/|Z|		
		vzn = __fdividef( ph->vx*ph->x + ph->vy*ph->y + ph->vz*ph->z ,rayon);
		
		if((vzn<0.f)&&(ph->locPrec==SURFACE)){
			// Problème du à des imprécisions de calculs je pense ,également à la surface agitée
// 			printf("pb#2 vzn= %lf - Z=(%lf, %lf, %lf) - V=(%lf, %lf, %lf) - test=%d\n",\
			vzn, ph->x, ph->y, ph->z, vx_d, vy_d, vz_d, flagTest);
			ph->loc=ABSORBED;
			return;
		}


		/* Calcul costh= Vz'/|Vz| */
		costh = vzn;
		
		if( abs(costh)>1.f ){
			#ifdef DEBUG
			printf("PROBLEME costh (%20.19lf)\n",costh);
			#endif
			ph->loc=NONE;
			return;
		}
		
		sinth = sqrtf(1.f-costh*costh);	// Le signe n'importe pas car on prend le carré ou la valeur absolu pour ztangentielle


		/** Initialisation des paramètres du photon : couche et sens de propagation dans l'atmosphère **/
		
		/* Couche dans laquel se trouve le photon
		* La valeur de cette couche va évoluer au fur et à mesure des calculs, représentant un photon virtuel passant de couches
		* en couches
		*/
		icoucheTemp = ph->couche;
		
		// On choisit que par défaut le photon monte, ceci pour éviter un test
		sens = 1;
		
		if(vzn<0.f){
		// Le photon descend
			sens = -1;
			
			/* Calcul de la distance tangentielle, distance minimale au centre de la terre sur le parcours du photon
			* Cette couche sera la couche dans laquelle le photon va changer de direction de propagation (montant ou descandant 
			* dans l'atmosphère)
			*/
			ztangentielle = rayon*sinth;
			
			// Puis recherche de la couche correspondante
			if( ztangentielle>RTER ){
				coucheTangentielle = 0;
				while( (RTER+tab.z[coucheTangentielle])>ztangentielle){ 
					coucheTangentielle++;
					#ifdef DEBUG
					if( coucheTangentielle==(NATMd+1) ){
						printf("Arret de calcul couche ztangentielle (%lf)\n", ztangentielle);
						ph->loc = NONE;
						return;
					}
					#endif
				}
			}
			// Sinon le photon va forcement passer par la surface, on ne change pas coucheTangentielle
		}


		/** Recherche des couches à parcourir **/
		/* Le premier cas est un peu différent car le photon se trouve entre 2 couches.
		* On le positionne virtuellement sur une couche (cela ne change rien pour les calculs).
		* Le calcul est basé sur la formule de pythagore généralisée. Elle calcule la distance parcourue par le photon jusqu'à
		* une couche donnée.
		*/
		
		// Si le photon arrive dans la couche tangentielle, il change de sens de propagation
		if( (icoucheTemp==coucheTangentielle)&&(sens==-1) ){
			// Le photon va remonter dans les couches
			sens=1;
		}


		// icouchefi est la couche suivante que va toucher le photon "virtuel", elle dépend du sens
		// L'équivalent est icouchefi = icoucheTemp; if( sens== +1) icouchefi--;
		bool test_sens = (sens==1);
		icouchefi = icoucheTemp - test_sens;
		
		if( icouchefi<0 ){
			/* Cela signifie que le photon est à la limite de l'atmosphère mais qu'il va sortir car il remonte l'atmosphère
			* On le force donc à sortir en le mettant dans l'espace
			*/
			ph->loc=SPACE;
			return;
		}

		#ifdef DEBUG
		if( (icouchefi<0)||(icouchefi>NATMd) ){
			printf("OUPS#1: icouchefi=%d  sens=%d  icoucheTemp=%d  vzn=%lf  locPrec=%d  rayon=%25.20lf\n\t\
(%20.16lf , %20.16lf , %20.16lf )\n",\
			icouchefi,sens,icoucheTemp, vzn, ph->locPrec, rayon, ph->x, ph->y, ph->z);
			ph->loc=NONE;
			return;
		}
		#endif
		
		
		/** Premier calcul **/
		// Déterminant de l'équation déduite du pythagore généralisé
		delta = 4.f*( (tab.z[icouchefi]+RTER)*(tab.z[icouchefi]+RTER) - rayon2*sinth*sinth);
		
		
		if(delta<0){
			#ifdef DEBUG
				printf("OUPS rdelta #1=%lf - icoucheTemp=%d - tab.z[icoucheTemp]= %16.13lf - rayon= %17.13lf - rayon2=%20.16lf\n\t\
sinth= %20.19lf - sens=%d\n",\
				delta, icoucheTemp, tab.z[icoucheTemp], rayon, rayon2, sinth, sens);
			atomicAdd(&(var->erreurpoids), 1);
			#endif
			ph->loc=NONE;
			return;
		}
		

		/* Calcul de la solution avec astuce
		* Sens indique si le photon monte ou descends, et la solution se déduit de ce signe. Un dessin permet de s'en convaincre
		* Si le photon monte, costh>0, il faut prendre la plus petite solution
		* Si il descend dans les couches, costh<0, il faut prendre la plus petite solution
		* Si il monte dans les couches avec costh<0, il faut prendre la plus grande solution
		*/
		rsolfi= 0.5f*( -2.f*rayon*costh + sens*sqrtf(delta));
		
		if( abs(rsolfi) < 5e-3f ){
			rsolfi=0.f;
		}
		
		if( rsolfi<0.f ){
			#ifdef DEBUG
			rsol1=0.5*( -2*rayon*costh + sqrt(delta));
			rsol2=0.5*( -2*rayon*costh - sqrt(delta));
			printf("OUPS: rsolfi #1=%lf, (%lf,%lf) - vzn=%lf - sens=%d - locPrec=%d\n\t\
costh= %16.15lf - rayon= %16.12lf - delta= %16.10lf - icoucheTemp=%d - icouchefi=%d\n",\
			rsolfi,rsol1,rsol2, vzn, sens, ph->locPrec, costh, rayon, delta, icoucheTemp, icouchefi);
			atomicAdd(&(var->erreurpoids), 1);
			#endif
			
			ph->loc=NONE;
			return;
		}


		// Calcul des paramètres du profil du photon au cours de son parcours
		if( icouchefi!=icoucheTemp ){
			hph = __fdividef( abs(tab.h[icoucheTemp] - tab.h[icouchefi])*rsolfi, abs(tab.z[icouchefi] - tab.z[icoucheTemp]) );
		}
		else{
			if( icouchefi==0 ){
				hph = __fdividef( abs(tab.h[1] - tab.h[0])*rsolfi, abs(tab.z[1] - tab.z[0]) );
			}
			else{
				hph = __fdividef( abs(tab.h[icouchefi-1] - tab.h[icouchefi])*rsolfi, abs(tab.z[icouchefi-1] - tab.z[icouchefi]) );
			}
		}

		zph=rsolfi;

		#ifdef DEBUG
// 		if( idx==0 ){
// 			printf("(%d)Profil: rsolfi=%lf - rdelta=%lf - sens=%d - icoucheTemp=%d - icouchefi=%d - icoucheTan=%d\n\t\
// rayon=%lf - vzn=%lf - hph=%lf - zph=%lf - tauRdm=%f - taumax0=%f - locPrec=%d\n",\
// 			icompteur, rsolfi, delta, sens, icoucheTemp, icouchefi, coucheTangentielle,\
// 			rayon, vzn, hph, zph, tauRdm, init->taumax0, ph->locPrec);
// 		}
		#endif

		icoucheTemp = icouchefi;


		/** Calcul du profil total **/
		// Calcul jusqu'à sortir ou intéragir
		
		while( (hph < tauRdm) ){
			
			icompteur++;
			
			/* Vérification si le photon est sorti de l'atmosphère
			* La variable sens permettra ensuite de savoir si le photon sort vers la surface ou l'espace
			*/
			if( (icoucheTemp==0)||(icoucheTemp==NATMd) ) {
				flagSortie=1;
				break;
			}

			// Mise à jour de la couche que va toucher le photon (c'est icouchefi)
			icouchefi = icoucheTemp - sens;
			
			if( (icouchefi==coucheTangentielle)&&(sens==-1) ){
				// Le photon va remonter dans les couches
				sens=1;
				icouchefi = icoucheTemp;
			}


			// Solution de l'équation issue de pythagore généralisé
			delta = 4.f*( (tab.z[icouchefi]+RTER)*(tab.z[icouchefi]+RTER) - rayon2*sinth*sinth);
			
			
			if(delta<0){
				#ifdef DEBUG
				printf("OUPS delta #2=%lf - icouchefi=%d - tab.z[icouchefi]= %16.13lf - rayon= %17.13lf - rayon2= %20.16lf\n\t\
sinth= %20.19lf - sens=%d\n",\
				delta, icouchefi, tab.z[icouchefi], rayon, rayon2, sinth, sens);
				atomicAdd(&(var->erreurpoids), 1);
				#endif
				ph->loc=NONE;
				return;
			}
			
			
			// Calcul de la solution avec astuce
			rsolfi= 0.5f*( -2.f*rayon*costh + sens*sqrtf(delta));
			
			if( rsolfi<0.f ){
				// Problème, c'est un rayon
				#ifdef DEBUG
				rsol1=0.5*( -2*rayon*costh + sqrt(delta));
				rsol2=0.5*( -2*rayon*costh - sqrt(delta));
				printf("OUPS: rsolfi #2=%lf, (%lf,%lf) - vzn=%lf - sens=%d - costh=%lf - rayon=%lf - delta=%lf\n",\
				rsolfi,rsol1,rsol2, vzn, sens, costh, rayon, delta);
				atomicAdd(&(var->erreurpoids), 1);
				#endif
				ph->loc=NONE;
				return;
			}
			
			// Calcul des grandeurs du profil
			hph_p = hph;
			zph_p = zph;
			
			#ifdef DEBUG
			if( icouchefi<0 ){
				printf("OUPS: icouchefi #1 = %d - rayon=%lf - icouchePhoton=%d - icoucheTemp=%d\n",\
				icouchefi, rayon, ph->couche, icoucheTemp);
				ph->loc=NONE;
				return;
			}
			if(icouchefi>NATMd){
				printf("OUPS: icouchefi #1 = %d\n",icouchefi);
				ph->loc=NONE;
				return;
			}
			#endif
			
			// Valeur de la couche actuelle
			if( icouchefi!=icoucheTemp ){
				hph += __fdividef( 	abs(tab.h[icoucheTemp] - tab.h[icouchefi])*(rsolfi-zph_p), 
									abs(tab.z[icouchefi] - tab.z[icoucheTemp]) );
			}
			else{
				if( icouchefi==0 ){
					hph += __fdividef( 	abs(tab.h[1] - tab.h[0])*(rsolfi-zph_p) , abs(tab.z[1]- tab.z[0]) );
				}
				else{
					hph += __fdividef( 	abs(tab.h[icouchefi-1] - tab.h[icouchefi])*(rsolfi-zph_p),
										abs(tab.z[icouchefi-1] - tab.z[icouchefi]) );
				}
			}

			zph=rsolfi;

			#ifdef DEBUG
// 			if( idx==0 ){
// 				rsol1=0.5*( -2*rayon*costh + sqrt(delta));
// 				rsol2=0.5*( -2*rayon*costh - sqrt(delta));
// 				printf("(%d)Profil: rsolfi=%lf - rdelta=%lf - sens=%d - icoucheTemp=%d - icouchefi=%d - icoucheTan=%d\n\t\
// rayon= %lf - vzn= %lf - hph=%lf - zph=%lf - tauRdm=%f - locPrec=%d\n",\
// 				icompteur, rsolfi, delta, sens, icoucheTemp, icouchefi, coucheTangentielle,\
// 				rayon, vzn, hph, zph, tauRdm, ph->locPrec);
// 			}
			#endif

			icoucheTemp = icouchefi;
			
			#ifdef DEBUG
			// Compteur de débordement
			if(icompteur==(2*NATMd+2)){
				printf("icouche = 2(NATMd+1) - (%lf,%lf,%lf) - icouchefi=%d - flagSortie=%d\n\t\
	ph->vz=%f - rsolfi=%f - tauRdm=%f - hph=%f\n",\
				ph->x,ph->y,ph->z, icouchefi,flagSortie, ph->vz,rsolfi,tauRdm,hph);
				ph->loc=NONE;
				return;
			}
			#endif
		
		}// Fin while
		
		#ifdef TEMPS
		if(idx==0){
			stop = clock();
			time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
			printf("(2.2) Temps de move pour une intéraction quelconque: %f\n", time);
		}
		#endif

	}// Fin de si photon provenant de l'atmosphere ou la surface

	
	/** Actualisation des coordonnées du photon **/
	/* Calcul des nouvelles coordonnées (x,y,z) du photon
	* Si le photon va intéragir dans l'atmosphère:
	* 		Interpolation linéaire entre les bornes de la couche car le trajet au sein de la couche est rectiligne
	* 		la 2ème ligne peut être remplacée pour la compréhension par rrb = zph - hph*rra; rdist = rra*tauRdm + rrb; mais les 
	* 		performances sont légérement réduites.
	* Sinon
	*		La distance parcourue correspond à la traversée de l'atmosphère, représentée par zph
	*/
	
	if( flagSortie==0 ){
	rra = __fdividef( zph_p - zph , hph_p - hph );
	rdist = rra*( tauRdm-hph ) + zph;
	}
	else{
		rdist = zph;
	}
	
	// Test de réduction du temps de calcul > Inefficace
	// rdist = (1-flagSortie)*( __fdividef( zph_p - zph , hph_p - hph )*( tauRdm-hph ) + zph ) + flagSortie*zph ;
	// rdist = (1-flagSortie)*( rra*( tauRdm-hph ) + zph ) + flagSortie*zph ;
	
	ph->x = ph->x + ph->vx*rdist;
	ph->y = ph->y + ph->vy*rdist;
	ph->z = ph->z + ph->vz*rdist;
	

	/** Sortie sans intéraction **/
	if( flagSortie==1 ){
		// Il n'y a pas eu d'intéraction avec l'atmosphère
		
		if(sens==-1){
			ph->loc = SURFACE;
			ph->couche = NATMd;
			ph->rayon = RTER;
		}
		else{
			ph->loc = SPACE;
		}
		
		#ifdef DEBUG
// 		if(idx==0){
// 		printf("Sortie sans inter: rayon=%20.16lf, (%16.12lf,%16.12lf,%20.16lf), hph=%lf, zph=%lf, locPrec=%d\n",\
// 		sqrt(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z),ph->x,ph->y,ph->z,hph, zph, ph->locPrec);
// 		}
		#endif

		return;
	}


	/** Sorti avec intéraction **/
	// Calcul du rayon
	rayon2 = ph->x*ph->x + ph->y*ph->y + ph->z*ph->z;
	rayon = sqrtf(rayon2);

	
	if((rayon < RTER)/* || (rayon>RTER+HATMd)*/){
		if( (rayon-RTER)<1.e-4f ){
			// Ce test apparait lorsqu'il y a la surface. Le rayon n'est pas égal à RTER, surement à cause d'erreur de calcul du GPU
			rayon=RTER;
			ph->loc=SURFACE;
			#ifdef DEBUG
			printf("MetaProblème #2: Correction du rayon\n");
			#endif
		}
		else{
			#ifdef DEBUG
			printf("MetaProblème #2: rayon=%20.16lf - (%lf,%lf,%lf) - icouchefi=%d - icompteur=%d -locPrec=%d\n\t\
rsolfi=%15.12lf - tauRdm= %lf - hph_p= %15.12lf - hph= %15.12lf - zph_p= %15.12lf - zph= %15.12lf\n",\
			rayon,ph->x, ph->y,ph->z, icouchefi, icompteur,ph->locPrec,\
			rsolfi,tauRdm, hph_p, hph, zph_p, zph);
			#endif
			ph->loc = NONE;
			return;
		}
	}

	// Boucle pour définir entre quels rayons est le photon
	icoucheTemp = 0;
	while((RTER+tab.z[icoucheTemp])>rayon){
		icoucheTemp++;
		#ifdef DEBUG
		if (icoucheTemp==NATMd+1){
			printf("Arret de calcul couche #2 (rayon=%f)\n", rayon);
			ph->loc=NONE;
			return;
		}
		#endif
	}
	
	#ifdef DEBUG
// 	if( idx==0 ){
// 		printf("Sortie avec interaction: rayon=%f, Z=(%lf, %lf, %lf), icouche=%d, hph=%lf, zph=%lf, locPrec=%d\n",\
// 		rayon, ph->x, ph->y, ph->z, icoucheTemp,hph,zph,ph->locPrec );
// 	}
	#endif

	ph->couche = icoucheTemp;
	ph->rayon = rayon;
	ph->locPrec=ATMOS;
	
	// On sort maintenant de la fonction et comme le photon reste dans ATMOS, le kernel appelle scatter()

	#endif	/* Fin de la partie atmosphère sphérique */


	#ifndef SPHERIQUE	/* Code spécifique à une atmosphère parallèle */
	ph->tau += -__logf( flagDiff + RAND*(1.F +(__expf(-TAUMAXd)-2.f)*flagDiff))*ph->vz;
	
	
	// Si tau<0 le photon atteint la surface
	if(ph->tau < 0.F){
		ph->loc = SURFACE;
		ph->tau = 0.F;
	}
	// Si tau>TAURAY le photon atteint l'espace
	else if(ph->tau > TAUATMd) ph->loc = SPACE;
	// Sinon on ne fait rien car il reste dans l'atmosphère, et va être traité par scatter
	#endif
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		
		// "2"représente l'événement "move" du photon
		evnt[i].action = 2;
		// On récupère le tau et le poids du photon
		evnt[i].poids = ph->weight;
	}
	#endif
}


/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter(Photon* ph, Tableaux tab
			#ifdef RANDMWC
			, unsigned long long* etatThr, unsigned int* configThr
			#endif
			#ifdef RANDCUDA
			, curandState_t* etatThr
			#endif
			#ifdef RANDMT
			, EtatMT* etatThr, ConfigMT* configThr
			#endif
			#ifdef TRAJET
			, int idx, Evnt* evnt
			#endif
			){
	
	float cTh=0.f, sTh, psi, cPsi, sPsi;
	float wx, wy, wz, vx, vy, vz;
	
	
	psi = RAND * DEUXPI; //psiPhoton
	cPsi = __cosf(psi); //cosPsiPhoton
	sPsi = __sinf(psi); //sinPsiPhoton
	
	// Modification des nombres de Stokes
	modifStokes(ph, psi, cPsi, sPsi, 1);
	
	calculDiffScatter( ph, &cTh, tab
			#ifdef RANDMWC
			, etatThr, configThr
				#endif
			#ifdef RANDCUDA
			, etatThr
			#endif
			#ifdef RANDMT
			, etatThr, configThr
			#endif
			);
	
	sTh = sqrtf(1.F - cTh*cTh);	// sinThetaPhoton
	
	/** Création de 2 vecteurs provisoires w et v **/
	// w est le rotationnel entre l'ancien vecteur u et l'ancien vecteur v du photon
	wx = ph->uy * ph->vz - ph->uz * ph->vy;
	wy = ph->uz * ph->vx - ph->ux * ph->vz;
	wz = ph->ux * ph->vy - ph->uy * ph->vx;
	
	// v est le nouveau vecteur v du photon
	vx = cTh * ph->vx + sTh * ( cPsi * ph->ux + sPsi * wx );
	vy = cTh * ph->vy + sTh * ( cPsi * ph->uy + sPsi * wy );
	vz = cTh * ph->vz + sTh * ( cPsi * ph->uz + sPsi * wz );
	
	// Changement du vecteur u (orthogonal au vecteur vitesse du photon)
	ph->ux = __fdividef(cTh * vx - ph->vx, sTh);
	ph->uy = __fdividef(cTh * vy - ph->vy, sTh);
	ph->uz = __fdividef(cTh * vz - ph->vz, sTh);
	
	// Changement du vecteur v (vitesse du photon)
	ph->vx = vx;
	ph->vy = vy;
	ph->vz = vz;


	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
	   int i = 0;
	   // On cherche la première action vide du tableau
	   while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
	   // Et on remplit la première case vide des tableaux (tableaux de 20 cases)
	   // 		if(i <20 )
	   // 		{
		   // "3"représente l'événement "scatter" du photon
		   evnt[i].action = 3;
		   // On récupère le tau et le poids du photon
		   evnt[i].poids = ph->weight;
		   // 		}
	}
	#endif
}


/* calculDiffScatter
* Regroupe l'ensemble des calculs propre à la diffusion moléculaire ou par les aérosols.
* Pour l'optimisation du programme, il est possible d'effectuer un travail de réduction au maximum de cette fonction. L'idée est
* de calculer et d'utiliser la fonction de phase moléculaire
*/
__device__ void calculDiffScatter( Photon* ph, float* cTh, Tableaux tab
			#ifdef RANDMWC
			, unsigned long long* etatThr, unsigned int* configThr
			#endif
			#ifdef RANDCUDA
			, curandState_t* etatThr
			#endif
			#ifdef RANDMT
			, EtatMT* etatThr, ConfigMT* configThr
			#endif
			){

	float zang=0.f, theta=0.f;
	int iang;
	float stokes1, stokes2;
	float cTh2;
	int icouche;
	float prop_aer;
	float rra, tau;
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	icouche = ph->couche;
	#endif
	
	#ifndef SPHERIQUE	/* Code spécifique à une atmosphère en plan parallèle */
	float tauBis = TAUATMd-ph->tau;
	icouche = 1;
	
	while( (tab.h[icouche] < (tauBis)) && icouche<NATMd ){
		icouche++;
	}
	#endif
	
	/** Interpolation linéaire pour connaitre la proportion d'aérosols à l'endroit où se situe le photon **/
// 	if(icouche==0){
// 		printf("icouche=0 pour le calcul de proportion d'aérosols\n");
// 		prop_aer = 1.f - tab.pMol[icouche];
// 	}
// 	else{
// 		rra = __fdividef( tab.pMol[icouche] - tab.pMol[icouche-1] , tab.h[icouche] -tab.h[icouche-1] );
// 		#ifdef SPHERIQUE
// 		prop_aer = 1.f - ( rra*(ph->z - tab.z[icouche]) + tab.pMol[icouche] );
// 		#endif
// 		#ifndef SPHERIQUE
// 		prop_aer = 1.f - ( rra*(ph->tau - tab.h[icouche]) + tab.pMol[icouche] );
// 		#endif
// 	}

	prop_aer = 1.f - tab.pMol[icouche];
	
	
	stokes1 = ph->stokes1;
	stokes2 = ph->stokes2;
	
	if( prop_aer<RAND ){
		// Theta calculé pour la diffusion moléculaire
		*cTh =  2.F * RAND - 1.F; // cosThetaPhoton
		cTh2 = (*cTh)*(*cTh);

		// Calcul du poids après diffusion
		ph->weight *= __fdividef(1.5F * ((1.F+GAMAd)*stokes2+((1.F-GAMAd)*cTh2+2.F*GAMAd)*stokes1), (1.F+2.F*GAMAd) *
		(stokes1+stokes2));
		// Calcul des parametres de Stokes du photon apres diffusion
		ph->stokes2 += GAMAd * stokes1;
		ph->stokes1 = ( (1.F - GAMAd) * cTh2 + GAMAd) * stokes1 + GAMAd * ph->stokes2;
		ph->stokes3 *= (1.F - GAMAd) * (*cTh);
		// 		photon->stokes4 = 0.F /*(1.F - 3.F * GAMAd) * (*cTh) * photon->stokes4*/;
	}
	else{
		// Aérosols
		zang = RAND*(NFAERd-1);
		iang= __float2int_rd(zang);
		
		zang = zang - iang;
		/* L'accès à faer[x][y] se fait par faer[y*5+x]*/
		theta = tab.faer[iang*5+4]+ zang*( tab.faer[(iang+1)*5+4]-tab.faer[iang*5+4] );
		
		*cTh = __cosf(theta);
		
		/** Changement du poids et des nombres de stokes du photon **/
		float faer1 = tab.faer[iang*5+0];
		float faer2 = tab.faer[iang*5+1];
		
		// Calcul du poids après diffusion
		ph->weight *= __fdividef( 2.0F*(stokes1*faer1+stokes2*faer2) , stokes1+stokes2)*W0AERd;
		
		// Calcul des parametres de Stokes du photon apres diffusion
		ph->stokes1 *= 2.0F*faer1;
		ph->stokes2 *= 2.0F*faer2;
		ph->stokes3 *= tab.faer[iang*5+2];
		// 		photon->stokes4 = 0.F;
	}
	
}


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
* //TODO: transmission vers l'océan et/ou reflexion totale
*/
__device__ void surfaceAgitee(Photon* ph
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#ifdef RANDCUDA
		, curandState_t* etatThr
		#endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
			){
	
	if( SIMd == -2){ // Atmosphère ou océan seuls, la surface absorbe tous les photons
		ph->loc = ABSORBED;
		
		#ifdef TRAJET
		// Récupération d'informations sur le premier photon traité
		if(idx == 0)
		{
			int i = 0;
			// On cherche la première action vide du tableau
			while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
			// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
			// 		if(i <20 )
			// 		{
				// "4"représente l'événement "surface" du photon
				evnt[i].action = 4;
				// On récupère le tau et le poids du photon
				evnt[i].poids = ph->weight;
				// 		}
		}
		#endif
		
		return;
	}
	
	// Réflexion sur le dioptre agité
	float theta;	// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float cTh, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	
	float sig = 0.F;
	float beta = 0.F;// Angle par rapport à la verticale du vecteur normal à une facette de vagues 
	float sBeta;
	float cBeta;
	
	float alpha = DEUXPI*RAND; //Angle azimutal du vecteur normal a une facette de vagues
	
	float nind;
	float temp;
	
	float nx, ny, nz;	// Coordonnées du vecteur normal à une facette de vague
	float s1, s2, s3;
	
	float rpar,rper;	// Coefficient de reflexion parallèle et perpendiculaire
	float rpar2;		// Coefficient de reflexion parallèle au carré
	float rper2;		// Coefficient de reflexion perpendiculaire au carré
	float rat;		// Rapport des coefficients de reflexion perpendiculaire et parallèle
	// 		float ReflTot;	// Flag pour la réflexion totale sur le dioptre
	float cot;		// Cosinus de l'angle de réfraction du photon
	float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
	// float tpar, tper;	//
	
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	
	/** Calcul du theta impact et phi impact **/
	//NOTE: Dans le code Fortran, ce calcul est effectué dans atmos
	float icp, isp, ict, ist;	// Sinus et cosinus de l'angle d'impact
	float thetaimp, phiimp;
	float vxn, vyn, vzn, uxn, uyn, uzn;
	
	double temp_d;
	
	ph->locPrec=ph->loc;
	
	
	/** Calcul de l'angle entre l'axe z et la normale au point d'impact **/
	//NOTE: le float pour les calculs suivant fait une erreur de 2.3% par exemple (theta_float=0.001196 / theta_double=0.0011691
	if( ph->z > 0. ){
		// 		temp = __fdividef(ph->z,RTER);
		temp_d = ph->z/RTER;
		// 		if(temp>1.f){
   if(temp_d>1){
	   thetaimp= 0.f;
   }
   else{
	   // 			thetaimp= acosf( temp );
	   thetaimp= acos( temp_d );
   }
   
   if(ph->x >= 0.) thetaimp = -thetaimp;
   
   // 	printf("Test : sqrt(x**2+y**2)=%16.14f\n",sqrtf(ph->x*ph->x + ph->y*ph->y));
   if( sqrtf(ph->x*ph->x + ph->y*ph->y)<1.e-6 ){/*NOTE En fortran ce test est à 1.e-8, relativement au double
	   utilisés, peut peut être être supprimer ici*/
	   // 			printf("Test : sqrt(x**2+y**2)=%f\n",sqrtf(ph->x*ph->x + ph->y*ph->y));
	   phiimp = 0.f;
	   }
	   else{
		   // printf("calcul phiimp ^_^ : double: %20.17lf , %20.17lf - float:%20.17f , %20.17f\n",\
		   // ph->x/sqrt(ph->x*ph->x + ph->y*ph->y), acos(ph->x/sqrt(ph->x*ph->x + ph->y*ph->y)),\
		   // __fdividef(ph->x, sqrtf(ph->x*ph->x + ph->y*ph->y)), acosf( __fdividef(ph->x, sqrtf(ph->x*ph->x +ph->y*ph->y))) );
		   // phiimp = acosf( __fdividef(ph->x, sqrtf(ph->x*ph->x + ph->y*ph->y)) );
		   phiimp = acos( ph->x/sqrt(ph->x*ph->x + ph->y*ph->y) );
		   if( ph->y < 0.f ) phiimp = -phiimp;
	   }
	}
	else{
		// Photon considéré comme perdu
		ph->loc = ABSORBED;	// Correspondant au weight=0 en Fortran
		return;
	}
	
	// 	if(idx==0)
	// 		printf("thetaimp=%16.14f - phiimp=%16.14f - (%lf,%lf,%lf)\n",\
	// 		thetaimp, phiimp, ph->x, ph->y, ph->z);
	
	
	/** Il faut exprimer Vx,y,z et Ux,y,z dans le repère de la normale au point d'impact **/
	icp = cosf(phiimp);
	isp = sinf(phiimp);
	ict = cosf(thetaimp);
	ist = sinf(thetaimp);
	
	vxn= ict*icp*ph->vx - ict*isp*ph->vy + ist*ph->vz;
	vyn= isp*ph->vx + icp*ph->vy;
	vzn= -icp*ist*ph->vx + ist*isp*ph->vy + ict*ph->vz;
	
	uxn= ict*icp*ph->ux - ict*isp*ph->uy + ist*ph->uz;
	uyn= isp*ph->ux + icp*ph->uy;
	uzn= -icp*ist*ph->ux + ist*isp*ph->uy + ict*ph->uz;
	
	ph->vx = vxn;
	ph->vy = vyn;
	ph->vz = vzn;
	ph->ux = uxn;
	ph->uy = uyn;
	ph->uz = uzn;
	#endif	/* Fin de la séparation atmosphère sphérique */
	
	/** **/
	
	if( DIOPTREd !=0 ){
		sig = sqrtf(0.003F + 0.00512f *WINDSPEEDd);
		beta = atanf( sig*sqrtf(-__logf(RAND)) );
	}
	sBeta = __sinf( beta );
	cBeta = __cosf( beta );
	
	nx = sBeta*__cosf( alpha );
	ny = sBeta*__sinf( alpha );
	
	// Projection de la surface apparente de la facette sur le plan horizontal		
	float signvz = __fdividef(abs( ph->vz), ph->vz);
	bool testn = ( ph->vz > 0.F );
	nind = testn*__fdividef(1.F,NH2Od) + !testn*NH2Od;
	nz = -signvz*cBeta;
	ph->weight *= -__fdividef(abs(nx*ph->vx + ny*ph->vy + nz*ph->vz),ph->vz*nz);

	temp = -(nx*ph->vx + ny*ph->vy + nz*ph->vz);
	
// 		if (abs(temp) > 1.01F){
// 			printf("ERREUR dans la fonction surface: variable temp supérieure à 1.01\n");
// 			printf(" temp = %f\n", temp);
// 			printf("nx=%f, ny=%f, nz=%f\tvx=%f, vy=%f, vz=%f\n", nx,ny,nz,ph->vx,ph->vy,ph->vz);
// 		}
	
	theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), temp ) ));

	if(theta >= DEMIPI){
		nx = -nx;
		ny = -ny;
		theta = acosf( -(nx*ph->vx + ny*ph->vy + nz*ph->vz) );
	}
	
	cTh = __cosf(theta);
	sTh = __sinf(theta);
	
	// Rotation des paramètres de Stokes
	s1 = ph->stokes1;
	s2 = ph->stokes2;
	s3 = ph->stokes3;
	
	if( (s1!=s2) || (s3!=0.F) ){
		
		temp = __fdividef(nx*ph->ux + ny*ph->uy + nz*ph->uz,sTh);
		psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	
		
		if( (nx*(ph->uy*ph->vz-ph->uz*ph->vy) + ny*(ph->uz*ph->vx-ph->ux*ph->vz) + nz*(ph->ux*ph->vy-ph->uy*ph->vx) ) <0 )
		{
			psi = -psi;
		}
		
	/*psi est l'angle entre le plan de diffusion et le plan de diffusion precedent. Rotation des
	parametres de Stoke du photon d'apres cet angle.*/
	modifStokes(ph, psi, __cosf(psi), __sinf(psi), 0 );
	
	}
	/**  **/
// 	bool test_s = (sTh<=nind);
// 	temp = __fdividef(sTh,nind);
// 	cot = sqrtf( 1.0F - temp*temp )*test_s;
// 	ncot = nind*cot;
// 	ncTh = nind*cTh;
// 	rpar = __fdividef(cot - ncTh,cot + ncTh)*test_s + 1.F*(!test_s);
// 	rpar2 = rpar*rpar;
// 	rper = __fdividef(cTh - ncot,cTh + ncot)*test_s + 1.F*(!test_s);
// 	rper2 = rper*rper;
// // 		ReflTot = !(test_s);
// Rapport de l'intensite speculaire reflechie
// rat = __fdividef(photon->stokes1*rper2 + photon->stokes2*rpar2,photon->stokes1+photon->stokes2)*test_s;

	/*****************/
	if( sTh<=nind){
		temp = __fdividef(sTh,nind);
		cot = sqrtf( 1.0F - temp*temp );
		ncTh = nind*cTh;
		ncot = nind*cot;
		rpar = __fdividef(cot - ncTh,cot + ncTh);
		rper = __fdividef(cTh - ncot,cTh + ncot);
		rpar2 = rpar*rpar;
		rper2 = rper*rper;
		rat = __fdividef(ph->stokes2*rper2 + ph->stokes1*rpar2,ph->stokes1+ph->stokes2);
	}
	else{
		cot = 0.f;
		rpar = 1.f;
		rper = 1.f;
		ncTh = nind*cTh;
		ncot = nind*cot;
		rat=0;
		rpar2 = rpar*rpar;
		rper2 = rper*rper;
	}
/*	
	if( isnan(cot)!=0 ){
		printf("cot = %f - temp=%f - nind=%f - sTh=%f - cTh=%f - theta=%f\n",\
		cot,temp, nind, sTh,cTh, theta);
		ph->loc=NONE;
		return;
	}*/
	
	// Rapport de l'intensite speculaire reflechie
// 	rat = __fdividef(ph->stokes2*rper2 + ph->stokes1*rpar2,ph->stokes1+ph->stokes2)*test_s;
	//if (SURd==1){ /*On pondere le poids du photon par le coefficient de reflexion dans le cas 
	// d'une reflexion speculaire sur le dioptre (mirroir parfait)*/
	ph->weight *= rat;
	// 			}
	
// 		float coeffper, coeffpar;
	
// 		if( (ReflTot==1) || (SURd==1) || ( (SURd==3)&&(RAND<rat) ) ){
		//Nouveau parametre pour le photon apres reflexion speculaire
		ph->stokes2 *= rper2;
		ph->stokes1 *= rpar2;
// 		ph->stokes4 *= -rpar*rper;
		ph->stokes3 *= -rpar*rper;

// 			coeffper = rper;
// 			coeffpar = rpar;
		
		if( (isnan(ph->stokes1)!=0)||(isnan(ph->stokes2)!=0)){
			printf("Problème NaN#1.2 - s1=%f - s2=%f\n", ph->stokes1, ph->stokes2);
			ph->loc=NONE;
			return;
		}
		
		ph->vx += 2.F*cTh*nx;
		ph->vy += 2.F*cTh*ny;
		ph->vz += 2.F*cTh*nz;
		ph->ux = __fdividef( nx-cTh*ph->vx,sTh );
		ph->uy = __fdividef( ny-cTh*ph->vy,sTh );
		ph->uz = __fdividef( nz-cTh*ph->vz,sTh );
		
		// Le photon est renvoyé dans l'atmosphère
		ph->loc = ATMOS;
		if((ph->vz<0) && (DIOPTREd==2)){
			// Suppression des reflexions multiples
			ph->loc = ABSORBED;
		}
// 			bool cond = ((ph->vz<0) && (DIOPTREd==2));
// 			ph->loc = ABSORBED*cond + ATMOS*(!cond);

		
// 			if( abs( 1.F - sqrtf(ph->ux*ph->ux+ph->uy*ph->uy+ph->uz*ph->uz) )>1.E-05){
// 				ph->weight = 0;
// 				ph->loc = ABSORBED;
// 				printf("suppression du photon\n");
// 				if(RAND<0.1){
// 				printf("valeur a pb:%10.8f - ux=%10.8f - uy=%10.8f - uz=%10.8f\n",
// 					   sqrt(ph->ux*ph->ux + ph->uy*ph->uy+ph->uz*ph->uz),ph->ux ,ph->uy, ph->uz);
// 					   printf("ux2=%10.8f - uy2=%10.8f-uy2=%10.8f\n",
// 							  ph->ux*ph->ux,ph->uy*ph->uy,ph->uz*ph->uz);
		
// 				}
// 				return;
// 			}
		
		
// 		}
	
// 		else{	// Transmission par le dioptre	//NOTE: Inutile pour le moment
// 			
// // 			tpar = __fdividef( 2*cTh,ncTh+ cot);
// // 			tper = __fdividef( 2*cTh,cTh+ ncot);
// // 			
// // 			ph->stokes2 *= tper*tper;
// // 			ph->stokes1 *= tpar*tpar;
// // 			ph->stokes3 *= -tpar*tper;
// // 			ph->stokes4 *= -tpar*tper;
// 			
// 			coeffpar = __fdividef( 2*cTh,ncTh+ cot);
// 			coeffper = __fdividef( 2*cTh,cTh+ ncot);
// 			
// 			alpha = __fdividef(cTh,nind) - cot;
// 			ph->vx = __fdividef(ph->vx,nind) + alpha*nx;
// 			ph->vy = __fdividef(ph->vy,nind) + alpha*ny;
// 			ph->vz = __fdividef(ph->vz,nind) + alpha*nz;
// 			ph->ux = __fdividef( nx+cot*ph->vx,sTh )*nind;
// 			ph->uy = __fdividef( ny+cot*ph->vy,sTh )*nind;
// 			ph->uz = __fdividef( nz+cot*ph->vz,sTh )*nind;
// 			
// 			// Le photon est renvoyé dans l'atmosphère
// // 			ph->loc = ;
// 			
// 			/* On pondere le poids du photon par le coefficient de transmission dans le cas d'une reflexion
// 			speculaire sur le dioptre plan (ocean diffusant) */
// 			if( SURd == 2)
// 				ph->weight *= (1-rat);
// 			
// 		}
// 		
// 		// Calcul commun sortis de la boucle pour gain de temps
// 		ph->stokes2 *= coeffper*coeffper;
// 		ph->stokes1 *= coeffpar*coeffpar;
// 		ph->stokes4 *= -coeffpar*coeffper;
// 		ph->stokes3 *= -coeffpar*coeffper;
	
	if( SIMd == -1) // Dioptre seul
		ph->loc=SPACE;
	
	
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	/** Retour dans le repère d'origine **/
	
	icp = cosf(-phiimp);
	isp = sinf(-phiimp);
	ict = cosf(-thetaimp);
	ist = sinf(-thetaimp);
	
	vxn= ict*icp*ph->vx - ict*isp*ph->vy + ist*ph->vz;
	vyn= isp*ph->vx + icp*ph->vy;
	vzn= -icp*ist*ph->vx + ist*isp*ph->vy + ict*ph->vz;
	
	uxn= ict*icp*ph->ux - ict*isp*ph->uy + ist*ph->uz;
	uyn= isp*ph->ux + icp*ph->uy;
	uzn= -icp*ist*ph->ux + ist*isp*ph->uy + ict*ph->uz;
	
	ph->vx = vxn;
	ph->vy = vyn;
	ph->vz = vzn;
	ph->ux = uxn;
	ph->uy = uyn;
	ph->uz = uzn;
	#endif

	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
// 		if(i <20 )
// 		{
			// "4"représente l'événement "surface" du photon
			evnt[i].action = 4;
			// On récupère le tau et le poids du photon
			evnt[i].poids = ph->weight;
// 		}
	}
	#endif
}


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* ph
						#ifdef RANDMWC
						, unsigned long long* etatThr, unsigned int* configThr
						#endif
						#ifdef RANDCUDA
						, curandState_t* etatThr
						#endif
						#ifdef RANDMT
						, EtatMT* etatThr, ConfigMT* configThr
						#endif
						#ifdef TRAJET
						, int idx, Evnt* evnt
						#endif
						){
	
	if( SIMd == -2){ // Atmosphère ou océan seuls, la surface absorbe tous les photons
		ph->loc = ABSORBED;
	}
	
	else{
	float thetab;	// angle de diffusion (entre le vecteur avt et après reflexion)
	float uxn,vxn,uyn,vyn,uzn,vzn;	// Vecteur du photon après reflexion
	float cTh2 = RAND;
	float cTh = sqrtf( cTh2 );
	float sTh = sqrtf( 1.0F - cTh2 );
	
	float phi = RAND*DEUXPI;	//angle azimutal
	float cPhi = __cosf(phi);
	float sPhi = __sinf(phi);
	
	float psi;
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	ph->locPrec=ph->loc;
	#endif
	
	
	/** calcul u,v new **/
	vxn = cPhi*sTh;
	vyn = sPhi*sTh;
	vzn = cTh;
	
	uxn = cPhi*cTh;
	uyn = sPhi*cTh;
	uzn = -sTh;
	
	/** Calcul angle Psi **/
	float temp;
	// Calcul du produit scalaire V.Vnew
	temp = ph->vx*vxn + ph->vy*vyn + ph->vz*vzn;
	thetab = acosf( fmin( fmax(-1.f,temp),1.f ) );
	if( thetab==0){
		ph->loc=SPACE;
		printf("theta nul\n");
		return;
	}
	
	// (Produit scalaire V.Unew)/sin(theta)
	temp = __fdividef( ph->vx*uxn + ph->vy*uyn + ph->vz*uzn, __sinf(thetab) );
	psi = acosf( fmin( fmax(-1.f,temp),1.f ) );	// angle entre le plan (u,v)old et (u,v)new
	
	if( (ph->vx*(uyn*vzn-uzn*vyn) + ph->vy*(uzn*vxn-uxn*vzn) + ph->vz*(uxn*vyn-uyn*vxn) ) <0 )
	{	// test du signe de v.(unew^vnew) (scalaire et vectoriel)
	psi = -psi;
	}
	
	modifStokes(ph, psi, __cosf(psi) , __sinf(psi), 0 );
	
	ph->vx = vxn;
	ph->vy = vyn;
	ph->vz = vzn;
	ph->ux = uxn;
	ph->uy = uyn;
	ph->uz = uzn;
	
	// Aucun photon n'est absorbés mais on pondère le poids par l'albedo de diffusion de la surface lambertienne.
	ph->weight *= W0LAMd;

	// Si le dioptre est seul, le photon est mis dans l'espace
	bool test_s = ( SIMd == -1);
	ph->loc = SPACE*test_s + ATMOS*(!test_s);
	
	}
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		// 		if(i <20 )
		// 		{
			// "4"représente l'événement "surface" du photon
			evnt[i].action = 4;
			// On récupère le tau et le poids du photon
			evnt[i].poids = ph->weight;
			// 		}
	}
	#endif
	
}


/* exit
* Sauve les paramètres des photons sortis dans l'espace dans la boite correspondant à la direction de sortie
*/
__device__ void exit(Photon* ph, Variables* var, Tableaux tab, unsigned long long* nbPhotonsThr
		#ifdef PROGRESSION
		, unsigned int* nbPhotonsSorThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		    )
{
	// Remise à zéro de la localisation du photon
	ph->loc = NONE;
	

// si son poids est anormalement élevé on le compte comme une erreur. Test effectué uniquement en présence de dioptre
	if( (ph->weight > WEIGHTMAX) && (SIMd!=-2)){
		// printf("Erreur poids trop eleve\n");
		atomicAdd(&(var->erreurpoids), 1);
		return;
	}
	
	
	// Sinon on traite le photon et on l'ajoute dans le tableau tabPhotons de ce thread
	// Incrémentation du nombre de photons traités par le thread
	(*nbPhotonsThr)++;
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	if( ph->vz<=0.f ){
		return;
	}
	#endif
	
	// Création d'un float theta qui sert à modifier les nombres de Stokes
	float theta = acosf(fmin(1.F, fmax(-1.F, STHSd * ph->vx + CTHSd * ph->vz)) );
	
	// Si theta = 0 on l'ignore (cas où le photon repart dans la direction solaire)
	if(theta == 0.F)
	{
// 			printf("Erreur theta nul\n");
		atomicAdd(&(var->erreurtheta), 1);
		// Incrémentation du nombre de photons traités par le thread
		return;
	}

	// Création d'un angle psi qui sert à modifier les nombres de Stokes
	float psi;
	int ith=0, iphi=0;
	// Initialisation de psi
	calculPsi(ph, &psi, theta);
	
	// Modification des nombres de Stokes
	float cPsi = __cosf(psi);
	float sPsi = __sinf(psi);
	modifStokes(ph, psi, cPsi, sPsi, 0);
	
	// Calcul de la case dans laquelle le photon sort
	calculCase(&ith, &iphi, ph, var);
	
	#ifndef SPHERIQUE
	/** A garder ou pas ? **/
	// Modification de U
	if( ph->vy<0.f )
		ph->stokes3 = -ph->stokes3;
	#endif
	
	// On modifie ensuite le poids du photon
	ph->weight = __fdividef(ph->weight, ph->stokes1 + ph->stokes2);
	

	// Rangement du photon dans sa case, et incrémentation de variables
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)))
	{
		// Rangement dans le tableau des paramètres pondérés du photon

		atomicAdd(tab.tabPhotons+(0 * NBTHETAd * NBPHId + ith * NBPHId + iphi), ph->weight * ph->stokes1);

		atomicAdd(tab.tabPhotons+(1 * NBTHETAd * NBPHId + ith * NBPHId + iphi), ph->weight * ph->stokes2);

		atomicAdd(tab.tabPhotons+(2 * NBTHETAd * NBPHId + ith * NBPHId + iphi), ph->weight * ph->stokes3);
				
// 		atomicAdd(tab.tabPhotons+(3 * NBTHETAd * NBPHId + ith * NBPHId + iphi), ph->weight * ph->stokes4);

		#ifdef PROGRESSION
		// Incrémentation du nombre de photons sortis dans l'espace pour ce thread
		(*nbPhotonsSorThr)++;
		#endif

	}
	else
	{
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurcase), 1);
		#endif
	}

	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<NBTRAJET-1) i++;
			// Et on remplit la première case vide des tableaux (tableaux de NBTRAJET cases)
// 		if(i <NBTRAJET )
// 		{
				// "5"représente l'événement "exit" du photon
			evnt[i].action = 5;
				// On récupère le tau et le poids du photon
			evnt[i].poids = ph->weight;
// 		}
	}
	#endif
}


/* modifStokes
* Modifie les paramètres de stokes
* Flag permet de tester (si flag=1) ou non la valeur des paramètres avant modification
*/
__device__ void modifStokes(Photon* photon, float psi, float cPsi, float sPsi, int flag)
{
	// On modifie les nombres de Stokes grâce à psi
	if( ((photon->stokes1 != photon->stokes2) || (photon->stokes3 != 0.F) ) || (flag==0))
	{
		float cPsi2 = cPsi * cPsi;
		float sPsi2 = sPsi * sPsi;
		float psi2 = 2.F*psi;
		float stokes1, stokes2, stokes3;
		float a, s2Psi;
		stokes1 = photon->stokes1;
		stokes2 = photon->stokes2;
		stokes3 = photon->stokes3;
		s2Psi = __sinf(psi2);
		a = 0.5f*s2Psi*stokes3;
		photon->stokes1 = cPsi2 * stokes1 + sPsi2 * stokes2 + a;
		photon->stokes2 = sPsi2 * stokes1 + cPsi2 * stokes2 - a;
		photon->stokes3 = s2Psi * (stokes2 - stokes1) + __cosf(psi2) * stokes3;
	}
}


/* calculPsi
* Calcul du psi pour la direction de sortie du photon
*/
__device__ void calculPsi(Photon* photon, float* psi, float theta)
{
	float sign;
	if (theta >= 0.05F)
	{
		*psi = acosf(fmin(1.F, fmax(-1.F, __fdividef(STHSd * photon->ux + CTHSd * photon->uz, __sinf(theta)))));
	}
	else
	{
		*psi = acosf(fmin(1.F - VALMIN, fmax(-(1.F - VALMIN), - CTHSd * photon->ux + STHSd * photon->uz)));
	}
	
	sign = STHSd * (photon->uy * photon->vz - photon->uz * photon->vy) + CTHSd * (photon->ux * photon->vy - photon->uy * photon->vx);
	if (sign < 0.F) *psi = -(*psi);
}


/* calculCase
* Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void calculCase(int* ith, int* iphi, Photon* photon, Variables* var)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->vx * photon->vx + photon->vy * photon->vy);

	// Calcul de la valeur de ithv
	// _rn correspond à round to the nearest integer
	*ith = __float2int_rn(__fdividef(acosf(photon->vz) * NBTHETAd, DEMIPI));

	// Si le photon ressort très près du zénith on ne peut plus calculer iphi,
	// on est à l'intersection de toutes les cases du haut
	
	if(vxy >= VALMIN)
	{	//on calcule iphi
	
		// On place d'abord le photon dans un demi-cercle
		float cPhiP = __fdividef(photon->vx, vxy); //cosPhiPhoton
		// Cas limite où phi est très proche de 0, la formule générale ne marche pas
		if(cPhiP >= 1.F) *iphi = 0;
		// Cas limite où phi est très proche de PI, la formule générale ne marche pas
		else if(cPhiP <= -1.F) *iphi = (NBPHId) - 1;
		// Cas général
		else *iphi = __float2int_rd(__fdividef(acosf(cPhiP) * NBPHId, PI));
		// Puis on place le photon dans l'autre demi-cercle selon vy
// 		if(photon->vy < 0.F) *iphi = NBPHId - 1 - *iphi;
		#ifdef PROGRESSION
		// Lorsque vy=0 on décide par défaut que le photon reste du côté vy>0
		if(photon->vy == 0.F) atomicAdd(&(var->erreurvy), 1);
		#endif
	}
	
	else
	{	// Photon très près du zenith
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurvxy), 1);
		#endif
// 		/*if(photon->vy < 0.F) *iphi = NBPHId - 1;
// 		else*/ *iphi = 0;
		if(photon->vy >= 0.F)  *iphi = 0;
		else *iphi = NBPHId - 1;
// 		return;
	}
	
}


/**********************************************************
*	> Initialisation de données dans le device
***********************************************************/

/* initConstantesDevice
* Fonction qui initialise les constantes du device calculés dans le host
* Elle doit rester dans ce fichier
*/
void initConstantesDevice()
{
	cudaMemcpyToSymbol(NBPHOTONSd, &NBPHOTONS, sizeof(unsigned long long));
	cudaMemcpyToSymbol(NBLOOPd, &NBLOOP, sizeof(unsigned int));
	cudaMemcpyToSymbol(THSDEGd, &THSDEG, sizeof(float));
	cudaMemcpyToSymbol(LAMBDAd, &LAMBDA, sizeof(float));
	cudaMemcpyToSymbol(TAURAYd, &TAURAY, sizeof(float));
	cudaMemcpyToSymbol(TAUAERd, &TAUAER, sizeof(float));
	cudaMemcpyToSymbol(W0AERd, &W0AER, sizeof(float));
	cudaMemcpyToSymbol(W0LAMd, &W0LAM, sizeof(float));
	cudaMemcpyToSymbol(HAd, &HA, sizeof(float));
	cudaMemcpyToSymbol(HRd, &HR, sizeof(float));
	cudaMemcpyToSymbol(ZMINd, &ZMIN, sizeof(float));
	cudaMemcpyToSymbol(ZMAXd, &ZMAX, sizeof(float));
	cudaMemcpyToSymbol(NATMd, &NATM, sizeof(int));
	cudaMemcpyToSymbol(HATMd, &HATM, sizeof(int));
	
	cudaMemcpyToSymbol(WINDSPEEDd, &WINDSPEED, sizeof(float));
	cudaMemcpyToSymbol(NH2Od, &NH2O, sizeof(float));
	cudaMemcpyToSymbol(CONPHYd, &CONPHY, sizeof(float));
	cudaMemcpyToSymbol(XBLOCKd, &XBLOCK, sizeof(int));
	cudaMemcpyToSymbol(YBLOCKd, &YBLOCK, sizeof(int));
	cudaMemcpyToSymbol(XGRIDd, &XGRID, sizeof(int));
	cudaMemcpyToSymbol(YGRIDd, &YGRID, sizeof(int));
	cudaMemcpyToSymbol(NBTHETAd, &NBTHETA, sizeof(int));
	cudaMemcpyToSymbol(NBPHId, &NBPHI, sizeof(int));
	cudaMemcpyToSymbol(PROFILd, &PROFIL, sizeof(int));
	cudaMemcpyToSymbol(SIMd, &SIM, sizeof(int));
	cudaMemcpyToSymbol(SURd, &SUR, sizeof(int));
	cudaMemcpyToSymbol(DIOPTREd, &DIOPTRE, sizeof(int));
	cudaMemcpyToSymbol(DIFFFd, &DIFFF, sizeof(int));
	
	cudaMemcpyToSymbol(NFAERd, &NFAER, sizeof(unsigned int));
	
	float THSbis = THSDEG*DEG2RAD; //thetaSolaire en radians
	cudaMemcpyToSymbol(THSd, &THSbis, sizeof(float));
	
	float CTHSbis = cos(THSbis); //cosThetaSolaire
	cudaMemcpyToSymbol(CTHSd, &CTHSbis, sizeof(float));
	
	float STHSbis = sin(THSbis); //sinThetaSolaire
	cudaMemcpyToSymbol(STHSd, &STHSbis, sizeof(float));
	
	float GAMAbis = DEPO / (2.F-DEPO);
	cudaMemcpyToSymbol(GAMAd, &GAMAbis, sizeof(float));
	
	#ifndef SPHERIQUE
	float TAUATM = TAURAY+TAUAER;
	cudaMemcpyToSymbol(TAUATMd, &TAUATM, sizeof(float));
	
	float TAUMAX = TAUATM / CTHSbis; //tau initial du photon
	cudaMemcpyToSymbol(TAUMAXd, &TAUMAX, sizeof(float));
	#endif
}


/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/

/* initRandCUDA
* Fonction qui initialise les generateurs du random cuda
*/
__global__ void initRandCUDA(curandState_t* etat, unsigned long long seed)
{
	// Pour chaque thread on initialise son generateur avec le meme seed mais un idx different
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
	curand_init(seed, idx, 0, etat+idx);
}


/* initRandMTEtat
* Fonction qui initialise l'etat des generateurs du random Mersenne Twister (generateur = etat + config)
*/
__global__ void initRandMTEtat(EtatMT* etat, ConfigMT* config)
{
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
	// Initialisation de l'etat du MT de chaque thread avec un seed different et aleatoire
	etat[idx].mt[0] = config[idx].seed;
	for (int i = 1; i < MT_NN; i++)
		etat[idx].mt[i] = (1812433253U * (etat[idx].mt[i - 1] ^ (etat[idx].mt[i - 1] >> 30)) + i) & MT_WMASK;
	etat[idx].iState = 0;
	etat[idx].mti1 = etat[idx].mt[0];
}


/* randomMWCfloat
* Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
*/
__device__ float randomMWCfloat(unsigned long long* x,unsigned int* a)
{
	//Generate a random number (0,1]
	*x=(*x&0xffffffffull)*(*a)+(*x>>32);
	return __fdividef(__uint2float_rz((unsigned int)(*x)) + 1.0f,(float)0x100000000);
}


/* randomMTfloat
* Fonction random Mersenne Twister qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
*/
__device__ float randomMTfloat(EtatMT* etat, ConfigMT* config)
{
	//Convert to (0, 1] float
	return __fdividef(__uint2float_rz(randomMTuint(etat, config)) + 1.0f, 4294967296.0f);
}


/* randomMTuint
* Fonction random Mersenne Twister qui renvoit un uint à partir d'un generateur (etat+config)
*/
__device__ unsigned int randomMTuint(EtatMT* etat, ConfigMT* config)
{
	unsigned int mti;
	unsigned int mtiM;
	unsigned int x;
	int iState1;
	int iStateM;
	iState1 = etat->iState + 1;
	iStateM = etat->iState + MT_MM;
	if(iState1 >= MT_NN) iState1 -= MT_NN;
	if(iStateM >= MT_NN) iStateM -= MT_NN;
	mti  = etat->mti1;
	etat->mti1 = etat->mt[iState1];
	mtiM = etat->mt[iStateM];
	
	// MT recurrence
	x = (mti & MT_UMASK) | (etat->mti1 & MT_LMASK);
	x = mtiM ^ (x >> 1) ^ ((x & 1) ? config->matrix_a : 0);
	
	etat->mt[etat->iState] = x;
	etat->iState = iState1;
	
	//Tempering transformation
	x ^= (x >> MT_SHIFT0);
	x ^= (x << MT_SHIFTB) & config->mask_b;
	x ^= (x << MT_SHIFTC) & config->mask_c;
	x ^= (x >> MT_SHIFT1);
	return x;
}

