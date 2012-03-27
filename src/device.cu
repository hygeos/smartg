
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "device.h"

// extern texture<float, cudaTextureType1D, cudaReadModeElementType> tex_faer;

	  //////////////////////
	 // FONCTIONS DEVICE //
	//////////////////////

// Fonction device principale qui lance tous les threads, leur associe des photons, et les fait évoluer
__global__ void lancementKernel(Variables* var, Tableaux tab, Init* init
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
	unsigned long long nbPhotonsThr = 0; // nombre de photons taités par le thread
	int flagDiff = DIFFFd;
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; // nombre de photons taités par le thread et resortis dans l'espace
	#endif
	
	Photon ph; // On associe une structure de photon au thread
	ph.loc = NONE; // Initialement le photon n'est nulle part, il doit être initialisé
	
	/** Utilisation shared memory **/
	__shared__ float hph0_s[NATM+1];
	__shared__ float zph0_s[NATM+1];
	
	for( int i=0; i<NATM+1; i++){
		hph0_s[i] = tab.hph0[i];
		zph0_s[i] = tab.zph0[i];
	}
	
	
	/** Allocation des tableaux nécessaire au photon **/
	/* La libération de mémoire s'effectue après la boucle */
	
	ph.hph = (float*) malloc((2*NATM+2)*sizeof(float));
	if( ph.hph == NULL ){
		printf("ERREUR: Problème de malloc de ph->hph dans init idx(%d)\n",idx);
	}
	
	ph.zph = (float*) malloc((2*NATM+2)*sizeof(float));
	if( ph.zph == NULL ){
		printf("ERREUR: Problème de malloc de ph->zph dans init idx(%d)\n",idx);
	}
   
	// Dans cette boucle on simule le parcours du photon, puis on le réinitialise,... Le thread lance plusieurs photons
	for(unsigned int iloop= 0; iloop < NBLOOPd; iloop++)
	{
		// Si le photon est à NONE on l'initialise et on le met à ATMOS
		if(ph.loc == NONE){initPhoton(&ph, tab, init, hph0_s, zph0_s
				#ifdef TRAJET
				, idx, evnt
				#endif
				#ifdef SORTIEINT
				, iloop
				#endif
					);
// 			flagDiff = DIFFFd;
			
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		
		// Si le photon est à ATMOS on le fait avancer jusqu'à SURFACE, ou SPACE, ou ATMOS s'il subit une diffusion
		if( (ph.loc == ATMOS) /*&& (SIMd==-2 || SIMd==1 || SIMd==2)*/ ){ move(&ph, tab, flagDiff, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
						);
		//NOTE IMPORTANT: scatter est appelé dans move pour respecter le code Fortran!!!!
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est encore à ATMOS il subit une diffusion et reste dans ATMOS
// 		if( (ph.loc == ATMOS) /*&& (SIMd==-2 || SIMd==1 || SIMd==2)*/){
// 			scatterTot(&ph, tab, &etatThr
// 			#if defined(RANDMWC) || defined(RANDMT)
// 			, &configThr
// 			#endif
// 			#ifdef TRAJET
// 			, idx, evnt
// 			#endif
// 				);
				
// 			scatter( &ph, tab, tabCouche_s, &etatThr
// 			#if defined(RANDMWC) || defined(RANDMT)
// 			, &configThr
// 			#endif
// 			#ifdef TRAJET
// 			, idx, evnt
// 			#endif
// 				);
// 
// 		}
		
		// Chaque block attend tous ses threads avant de continuer
// 		syncthreads();
		
		// Si le photon est à SURFACE on le met à ABSORBED
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
			ph.locPrec = ABSORBED;
			ph.loc = NONE;
			nbPhotonsThr++;
		}
// 		syncthreads();
		
		if(ph.loc == SPACE){
			exit(&ph, var, tab, &nbPhotonsThr
						#ifdef PROGRESSION
						, &nbPhotonsSorThr
						#endif
						#ifdef TRAJET
						, idx, evnt
						#endif
						#ifdef SORTIEINT
						, iloop
						#endif
						);
		}
		syncthreads();
		

		//Mise à jour du poids suite à la 1ère diffusion forcée
// 		if(flagDiff==1 ){
// 			ph.weight *= (1.F - __expf(-ph.taumax0));
// 			flagDiff=0;
// 		}
// 		syncthreads();

	}// Fin boucle for
	
	/** Libération de la mémoire **/
	free(ph.hph);
	free(ph.zph);
	
	#ifdef SORTIEINT
	if( ph.numBoucle <NBLOOPd/2 ){
		atomicAdd(tab.nbBoucle + (NBLOOPd-1-ph.numBoucle),1);
// 		printf("numBoucle après sorti : %d\n",ph.numBoucle);
	}
// 	else
// 		printf("numBoucle dernier : %d\n",ph.numBoucle);
	
	#endif

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

// Fonction qui initialise les generateurs du random cuda
__global__ void initRandCUDA(curandState_t* etat, unsigned long long seed)
{
	// Pour chaque thread on initialise son generateur avec le meme seed mais un idx different
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
	curand_init(seed, idx, 0, etat+idx);
}

// Fonction qui initialise l'etat des generateurs du random Mersenne Twister (generateur = etat + config)
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

// Fonction device qui initialise le photon associé à un thread
__device__ void initPhoton(Photon* ph, Tableaux tab, Init* init, float* hph0_s, float* zph0_s
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		#ifdef SORTIEINT
		, unsigned int iloop
		#endif
		    )
{
// 	if( idx==0 )
// 		printf("Entree dans init - locPrec= %d\n",ph->locPrec);
	
	ph->locPrec=ph->loc;
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
	
// 	ph->tau = TAUATMd;
	ph->weight = WEIGHTINIT;
	// Initialisation des paramètres de stokes du photon
	ph->stokes1 = 0.5F;
	ph->stokes2 = 0.5F;
	ph->stokes3 = 0.F;
// 	ph->stokes4 = 0.F;
	
// 	impact( ph, tab
// 			#ifdef TRAJET
// 			, idx, evnt
// 			#endif
// 			);

// 	Paramètres initiaux calculés dans impactInit - host.cu
	ph->x = init->x0;
	ph->y = init->y0;
	ph->z = init->z0;
	ph->taumax = init->taumax0;
	ph->zintermax = init->zintermax0;
	
	for( int i=0; i<NATM+1; i++){
		ph->hph[i]=hph0_s[i];
		ph->zph[i]=zph0_s[i];
// 		ph->hph[i] = tab.hph0[i];
// 		ph->zph[i] = tab.zph0[i];
	}
	
	
// 	if( idx==0 )
// 		printf("Sortie de init -tomax0=%f\tzintermax0=%f\t(x0,y0,z0)=(%f,%f,%f)\n", ph->taumax,ph->zintermax,ph->x,ph->y,ph->z);
	
	#ifdef SORTIEINT
	ph->numBoucle = iloop;
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
			// "1"représente l'événement "initialisation" du photon
			evnt[i].action = 1;
			// On récupère le tau et le poids du photon
			evnt[i].tau = ph->tau;
			evnt[i].poids = ph->weight;
// 		}
	}
	#endif
}

// calcul du trajet optique-géométrique devant le photon
__device__ void calculProfil(Photon* ph, Tableaux tab
		#ifdef TRAJET
		, int idx
		#endif
		)
{
	float rdelta1, rdelta2, rdelta3;
	float rsolfi,rsol1,rsol2,rsol3,rsol4, rsol5,rsol6;
	float rcoeffA, rcoeffB, rcoeffC, rcoeffCbis, rcoeffCter;	// Variable pour optimisation du calcul trajet après diffusion
	float rsolA, rsolB, rsolC;
	int icouche, icouchefi, icouchefibis;
	int iray; 	// A dégager, utiliser aussi icouche
	
	int icompteur;
	float tointer, zinter;	// Variable à dégager proprement par la suite proche
	
	float rayon, rmoins, rplus, regal;
	float xphbis, yphbis, zphbis;
	
	/** Calcul des premiers termes **/
	/** Le photon a été diffusé: calcul du trajet optique-géométrique devant lui **/
	icouche = ph->couche;
	xphbis = ph->x;
	yphbis = ph->y;
	zphbis = ph->z;
	
	rmoins = RTER + tab.z[icouche-1];
	regal = RTER + tab.z[icouche];
	
	// 		if(idx==0)
	// 			printf("coordonées phbis après scatter: (%f,%f,%f)\n",xphbis, yphbis, zphbis);
	
	//Initialisation des solution
	rsol1 = -800.e+6f;
	rsol2 = -800.e+6f;
	rsol3 = -800.e+6f;
	rsol4 = -800.e+6f;
	
	rcoeffA= 1.f;
	rcoeffB= 2.f*(ph->vx*xphbis + ph->vy*yphbis + ph->vz*zphbis);
	rcoeffC= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - regal*regal;
	rcoeffCbis= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - rmoins*rmoins;
	
	rdelta1 = rcoeffB*rcoeffB - 4.f*rcoeffA*rcoeffC;
	rdelta2 = rcoeffB*rcoeffB - 4.f*rcoeffA*rcoeffCbis;
	
	if( rdelta1 == 0 ){
		rsol1= __fdividef(-rcoeffB,2.f*rcoeffA);
	}
	
	if( rdelta2==0 ){
		rsol3= __fdividef(-rcoeffB,2.f*rcoeffA);
	}
	
	if( rdelta1 > 0.f ){
		rsol1= __fdividef(-rcoeffB-sqrtf(rdelta1),2.f*rcoeffA);
		if( rsol1<=0 ) rsol2= __fdividef(-rcoeffB+sqrtf(rdelta1),2.f*rcoeffA);
	}
	
	if( rdelta2 > 0.f ){
		rsol3= __fdividef(-rcoeffB-sqrtf(rdelta2),2.f*rcoeffA);
		if( rsol3<=0 ) rsol4= __fdividef(-rcoeffB+sqrtf(rdelta2),2.f*rcoeffA);
	}
	
	/* Tous les cas de figure ont été abrordé pour résoudre l'équation du second degré. rsolfi sera la solution positive et
	plus petite possible */
	if( rsol1<0.f ){
		if( rsol2>0.f ) rsolA= rsol2;
		else 			rsolA= -800.e6f;
	}
	else{
		if( rsol2>0.f ) rsolA= fmin(rsol1,rsol2);
		else 			rsolA= rsol1;
	}
	
	if( rsol3<0.f ){
		if( rsol4>0.f ) rsolB= rsol4;
		else 			rsolB= -800.e6f;
	}
	else{
		if( rsol4>0.f ) rsolB= fmin(rsol3,rsol4);
		else 			rsolB= rsol3;
	}
	
	if( rsolA>0.f ){
		if( rsolB>0.f ){
			if( rsolA<rsolB ){
				rsolfi= rsolA;
				icouchefi= icouche;
			}
			else{
				rsolfi= rsolB;
				icouchefi= icouche-1;
			}
		}
		else{
			rsolfi= rsolA;
			icouchefi= icouche;
		}
	}
	else{
		rsolfi=rsolB;
		icouchefi= icouche-1;
	}
	
	// 	if( rsolfi == -800.e6f ){
   // 		printf("Problème rsolfi #1 = -800e6\n");
   // 	}
   // 		if(idx==0)
   // 			printf("rsolfi après scatter=%f\n", rsolfi);
   
   xphbis+= ph->vx*rsolfi;
   yphbis+= ph->vy*rsolfi;
   zphbis+= ph->vz*rsolfi;
   ph->hph[0] = 0.f;
   ph->zph[0] = 0.f;
   
   if( icouchefi!=icouche ){
	   ph->hph[1] = __fdividef(abs(tab.h[icouche]-tab.h[icouchefi])*rsolfi,
							   abs(tab.z[icouchefi]-tab.z[icouche]));
   }
   else{
	   if( icouchefi==0 ){
		   ph->hph[1] = __fdividef(abs(tab.h[icouchefi+1]-tab.h[icouchefi])*rsolfi,
								   abs(tab.z[icouchefi+1]-tab.z[icouchefi]));
	   }
	   else{
		   ph->hph[1] = __fdividef(abs(tab.h[icouchefi-1]-tab.h[icouchefi])*rsolfi,
								   abs(tab.z[icouchefi-1]-tab.z[icouchefi]));
	   }
   }
   
   ph->zph[1]=rsolfi;
   
   /* On est à l'interface d'une couche.
   Calcul du chemin optique à parcourir avant de ressortir ou de heurter le sol 
   Il y a 3 possibilités: on est à la couche i, on peut rencontrer i-1, i+1 ou i */
   // 	icompteur = 1;
   	tointer = ph->hph[1];
   	zinter = ph->zph[1];
 
	
	/** Calcul des autres termes: Boucle while **/
	// Initialisation
	icompteur = 1;
	
	// Calcul du nouveau profil vu par le photon
	// Test sur icouchefi
	while( (icouchefi!=0) && (icouchefi!=NATM)){
		
// 		if(idx==0){
   // 		// Affichage d'informations - Débugage
   // 		printf("%d:Profil atm: h=%f - z=%f\n",icouchefi,ph->hph[icompteur],ph->zph[icompteur]);
   // 	}
   
		if( icompteur==(2*NATM+1) ){
			// 				if(idx==0){
		printf("Il faut arreter, icompteur trop important - (icouchefi=%d)\n", icouchefi);
		// 				}
		ph->loc = NONE;
		return;
		
		}
		// On continue uniquement si icouchefi!=0 || !=NATM
		
		//Test sur les trois couches possible
		rayon= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis;
		if( sqrtf(rayon) < RTER ){
// 				if(idx==0){
			printf("Problème métaphysique #2 dans device.cu, rayon<RTER (%f)\n",sqrtf(rayon));
// 			printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
// 				}
// 				ph->isurface=1;
			ph->loc=NONE;	// NONE ou ABSORBED ???
			return;
		}
		
		// Boucle pour définir les rayons que l'on peut toucher
		iray=0;
													// 0.001 ds Fortran
		while( abs(RTER + tab.z[iray] - sqrtf(rayon))>=0.001f ){
			iray++;
			if( iray==NATM+1 ){
// 				printf("Arret couche #3 (rayon=%f - (%f,%f,%f) - calcul:%f - calculTotal=%f)\n", 
// 					   sqrtf(rayon), xphbis,yphbis,zphbis,RTER-sqrtf(rayon), abs(RTER + tab.z[iray-1]- sqrtf(rayon)) );
				ph->loc=NONE;
				return;
			}
		}
		
		// 			if( iray==0){
		// 				printf("Problème #2 iray=0 / move\n");
		// 			}
		// 			else if( iray==NATM+1){
		// 				printf("Problème #2 iray=NATM+1 / move\n");
		// 				ph->loc = NONE;
		// 				return;
		// 			}
		
		rmoins = RTER + tab.z[iray-1];
		regal = RTER + tab.z[iray];
		rplus = RTER + tab.z[iray+1];
		icouchefi=iray;
		
		// 			if( idx==0){
		// 				printf("iray=%d, z[iray]=%f, rayon=%f\n",iray,tab.z[iray],sqrtf(rayon));
		// 			}
		
		//Initialisation des solution
		rsol1 = -800.e+6f;
		rsol2 = -800.e+6f;
		rsol3 = -800.e+6f;
		rsol4 = -800.e+6f;
		rsol5 = -800.e+6f;
		rsol6 = -800.e+6f;
		
		rcoeffA= 1.f;
		rcoeffB= 2.f*(ph->vx*xphbis + ph->vy*yphbis + ph->vz*zphbis);
		rcoeffC= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - rplus*rplus;
		rcoeffCbis= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - rmoins*rmoins;
		rcoeffCter= xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - regal*regal;
		
		rdelta1 = rcoeffB*rcoeffB - 4.f*rcoeffA*rcoeffC;
		rdelta2 = rcoeffB*rcoeffB - 4.f*rcoeffA*rcoeffCbis;
		rdelta3 = rcoeffB*rcoeffB - 4.f*rcoeffA*rcoeffCter;
		
		if( rdelta1 == 0.f ){
			rsol1= __fdividef(-rcoeffB,2.f*rcoeffA);
		}
		if( rdelta2==0.f ){
			rsol3= __fdividef(-rcoeffB,2.f*rcoeffA);
		}
		if( rdelta3==0.f ){
			rsol5= __fdividef(-rcoeffB,2.f*rcoeffA);
			rsol5= -800.e+6f; //TODO: Logique? Non?
		}
		
		if( rdelta1 > 0.f ){
			rsol1= __fdividef(-rcoeffB-sqrtf(rdelta1),2.f*rcoeffA);
			if( rsol1<=0.f ) rsol2= __fdividef(-rcoeffB+sqrtf(rdelta1),2.f*rcoeffA);
		}
		
		if( rdelta2 > 0.f ){
			rsol3= __fdividef(-rcoeffB-sqrtf(rdelta2),2.f*rcoeffA);
			if( rsol3<=0.f ) rsol4= __fdividef(-rcoeffB+sqrtf(rdelta2),2.f*rcoeffA);
		}
		
		if( rdelta3 > 0.f ){
			rsol5= __fdividef(-rcoeffB-sqrtf(rdelta3),2.f*rcoeffA);
			if( rsol5<=0.f ) rsol6= __fdividef(-rcoeffB+sqrtf(rdelta3),2.f*rcoeffA);
			
			if( abs(rsol5)<1e-6 ){
				rsol5= -800.e+6f;
				// 					printf("|rsol5|<1e-6 lors du calcul de solution après diffusion\n");
			}
			if( abs(rsol6)<1e-6 ){
				rsol6= -800.e+6f;
				// 					printf("|rsol6|<1e-6 lors du calcul de solution après diffusion\n");
			}
		}
		
		/** Calcul des solutions et recherche de la solution **/
		/* Tous les cas de figure ont été abrordé pour résoudre l'équation du second degré. rsolfi sera la solution positive et
		plus petite possible */
		if( rsol1<0.f ){
			if( rsol2>0.f ) rsolA= rsol2;
			else 			rsolA= -800.e6f;
		}
		else{
			if( rsol2>0.f ) rsolA= fmin(rsol1,rsol2);
			else 			rsolA= rsol1;
		}
		
		if( rsol3<0.f ){
			if( rsol4>0.f ) rsolB= rsol4;
			else 			rsolB= -800.e6f;
		}
		else{
			if( rsol4>0.f ) rsolB= fmin(rsol3,rsol4);
			else 			rsolB= rsol3;
		}
		
		if( rsol5<0.f ){
			if( rsol6>0.f ) rsolC= rsol6;
			else 			rsolC= -800.e6f;
		}
		else{
			if( rsol6>0.f ) rsolC= fmin(rsol5,rsol6);
			else 			rsolC= rsol5;
		}
		
		// Recherche de la solution minimale
		if( rsolA>0.f ){
			if( rsolB>0.f ){
				if( rsolA<rsolB ){
					rsolfi= rsolA;
					icouchefibis= icouchefi+1;
				}
				else{
					rsolfi= rsolB;
					icouchefibis= icouchefi-1;
				}
			}
			else{
				rsolfi= rsolA;
				icouchefibis= icouchefi+1;
			}
		}
		else{
			if( rsolB>0.f ){
				rsolfi=rsolB;
				icouchefibis= icouchefi-1;
			}
			else{
				rsolfi= -800.e6f;
			}
		}
		
		if( rsolC>0.f ){
			if( rsolfi>0.f ){
				if( rsolC<rsolfi ){
					rsolfi= rsolC;
					icouchefibis= icouchefi;
				}
			}
		}
		
// 		if( rsolfi == -800.e6f ){
// 			printf("Problème rsolfi #2 = -800e6\n");
// 		}
		
		// 			if(idx==0)
		// 				printf("rsolfi boucle while :%f - icouchefi=%d\n",rsolfi, icouchefi );
		
		xphbis= xphbis + ph->vx*rsolfi;
		yphbis= yphbis + ph->vy*rsolfi;
		zphbis= zphbis + ph->vz*rsolfi;
		
		icompteur++;
		if( icouchefi != icouchefibis ){
			ph->hph[icompteur] = __fdividef(abs(tab.h[icouchefi]-tab.h[icouchefibis])*rsolfi,
											abs(tab.z[icouchefi]-tab.z[icouchefibis])) + ph->hph[icompteur-1];
		}
		else{
				// 				if( idx==0 ){
			// 					printf("icouchefi = icouchefibis\n");
			// 				}
			if( icouchefibis!=0 ){
				ph->hph[icompteur] = __fdividef(abs(tab.h[icouchefibis]-tab.h[icouchefibis-1])*rsolfi,
												abs(tab.z[icouchefibis]-tab.z[icouchefibis-1])) + ph->hph[icompteur-1];
			}
			else{
				ph->hph[icompteur] = __fdividef(abs(tab.h[icouchefibis]-tab.h[icouchefibis+1])*rsolfi,
												abs(tab.z[icouchefibis]-tab.z[icouchefibis+1])) + ph->hph[icompteur-1];
			}
		}
		
		ph->zph[icompteur]= ph->zph[icompteur-1]+rsolfi;
		tointer = ph->hph[icompteur];
		zinter = ph->zph[icompteur];
		icouchefi=icouchefibis;
   
	}// Fin boucle while
	
	/** Sauvegarde constantes **/
	// On a maintenant icouchefi = 0 ou NATM
// 	taumaxph= tointer;//ph->hph[icompteur];
// 	zintermax= zinter; //ph->zph[icompteur];

// 	if( zinter>1000.f )
// 		printf("zinter: %f, rsolfi:%f\n", zinter, rsolfi);
	
	ph->zintermax= zinter; 
	ph->taumax= tointer;
	ph->couche = icouchefi;
	
// 		if(idx==0){
// 			printf("\nParamètres du ph sortie du move avec interaction\n");
// 			printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
// 			
// 		}
	
}

// Fonction device qui traite les photons dans l'atmosphère en les faisant avancer
__device__ void move(Photon* ph, Tableaux tab, int flagDiff
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
	float rdelta1/*, rdelta2*//*, rdelta3*/;
	float xphbis,yphbis,zphbis;	//Coordonnées intermédiaire du photon
	float rsolfi,rsol1,rsol2/*,rsol3,rsol4*//*, rsol5,rsol6*/;
// 	float rcoeffA, rcoeffB, rcoeffC, rcoeffCbis/*, rcoeffCter*/;	// Variable pour optimisation du calcul trajet après diffusion
// 	float rsolA, rsolB/*, rsolC*/;
	int icouchefi/*, icouchefibis*/;
	int iray; 	// A dégager, utiliser aussi icouche
	
	int icouche/*, icompteur*/;
	float taumaxph,zintermax;
// 	float tointer, zinter;	// Variable à dégager proprement par la suite proche
	
	float tauRdm;	// Epaisseur optique aléatoire tirée
	float rra, rrb, rdist;
	float rayon/*, rmoins,*/ /*rplus,*/ /*regal*/;
	
	float rxn, ryn, rzn;	// Essayer de les dégager ensuite
	
// 	if( idx==0 )
// 		printf("Entree dans move - locPrec= %d\n",ph->locPrec);
	
	if(ph->locPrec==SURFACE){
	/** Le photon entre dans l'atmosphere par la base **/
// 		if( idx==0 )
// 			printf("On vient de SURFACE avant le move\n");
	
		ph->zph[0] = 0.f;
		ph->hph[0] = 0.f;
		
		xphbis = ph->x;
		yphbis = ph->y;
		zphbis = ph->z;
		
		//NOTE: En Fortran, a quoi sert ce test
		if( sqrtf( (ph->x+ph->vx)*(ph->x+ph->vx) + (ph->y+ph->vy)*(ph->y+ph->vy) + (ph->z+ph->vz)*(ph->z+ph->vz) ) < RTER ){
			ph->isurface=1;
			taumaxph=0.f;
			zintermax = 0.f;
			ph->weight = 0.f; 	//NOTE: Est-ce absorbé?
			ph->loc = NONE;
			return;
		}
		else{
			// Calcul du profil vu depuis la surface jusqu'au sommet de l'atmosphere
			for(icouche = 1; icouche<NATM+1; icouche++){
				ph->zph[icouche] = 0.f;
				ph->hph[icouche] = 0.f;
			}
			
			/* Calcul d'intersection entre droite et cercle. Fait pour l'ensemble des cercles concentriques délimitant les couches
atmosphériques */
			for(icouche=NATM+1; icouche>0; icouche--){
				rdelta1 =4.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)
					- 4.f*(xphbis*xphbis + yphbis*yphbis + zphbis*zphbis ) - (tab.z[icouche-1]+RTER)*(tab.z[icouche-1]+RTER);
				rsol1 = 0.5f*(-2.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)+sqrtf(rdelta1));
				rsol2 = 0.5f*(-2.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)-sqrtf(rdelta1));
				
				// Il faut choisir la plus petite distance en faisant attention qu'elle soit positive
				if(rsol1>0.f){
					if( rsol2>0.f)
						rsolfi = fmin(rsol1,rsol2);
					else
						rsolfi = rsol1;
				}
				else{
					if( rsol2>0.f )
						rsolfi=rsol1;
					else
						printf("Pas de solution correcte - ph venant de surface\n");
				}
				
				//NOTE attention au NATM+1 ou NATM
				ph->zph[NATM+1-icouche] = ph->zph[NATM+1-(icouche+1)] + rsolfi;
				ph->hph[NATM+1-icouche] = ph->hph[NATM+1-(icouche+1)] + 
				__fdividef(abs(tab.h[icouche]-tab.h[icouche-1])*rsolfi,abs(tab.z[icouche-1]-tab.z[icouche]));
				
				xphbis+=ph->vx*rsolfi;
				yphbis+=ph->vy*rsolfi;
				zphbis+=ph->vz*rsolfi;
				
				zintermax = ph->zph[NATM+1-icouche];
			}
			taumaxph = ph->hph[NATM];
			ph->isurface=-1;
		}
	}
	
	else if(ph->locPrec==NONE){
	/** Le photon entre dans l'atmosphere par le sommet 
		Calcul du profil atmosphérique vu par le photon **/
// 		if( idx==0 )
// 			printf("On vient de NONE avant le move\n");
	
		taumaxph = ph->taumax;
		zintermax = ph->zintermax;
		//NOTE: dans le Fortran initialisation des coordonnées cartésiennes du photon, je pense inutile
		// Beaucoup de redondance il me semble!!!!!! A vérifier
// 		for(icouche=0; icouche<NATM+1; icouche++){
// 			ph->hph[icouche] = ph->hphz[icouche];
// 			ph->zph[icouche] = ph->zphz[icouche];
// 		}
		
		ph->isurface = 1;
	}
	else{
// 		if( idx==0 )
// 			printf("On vient de ailleurs avant le move\n");
		// Définir zintermax et taumaxphoton quand meme
		//L'utilisation des valeurs initiales est une astuce pour sauver des résultats intermédiaires
		zintermax = ph->zintermax;
		taumaxph = ph->taumax;
	}
	
// 	if(idx==0){
// 		printf("\nParamètres du ph dans le move\n");
// 		printf("taumaxph=%f - zintermax=%f - (%f,%f,%f), locPrec=%d\n",taumaxph,zintermax,ph->x,ph->y,ph->z, ph->locPrec);
// 	}
	
	// Sauvegarde de zintermax et taumax dans ph->zintermax0 et ph->taumax0 pour le prochain appel de move
	ph->zintermax= zintermax; 
	ph->taumax= taumaxph;
	
	// Une fois les calculs effectués, on sauve la position précédente du photon
// 	ph->locPrec=ph->loc;
	
	/** Tirage au sort de la profondeur optique à parcourir **/
	/*  Tirage effectué lors de chaque appel de la fonction */
	// Pas de diffusion forcée en sphérique
	
	// 55
	
	tauRdm = -__logf(1.F-RAND);
// 	tauRdm = 0.2f;
// 	if( idx==0 )
// 		printf("tauRdm=%f\n",tauRdm);

	if( (tauRdm<=0.f) || (tauRdm>=taumaxph) ){
		//Il n'y a pas d'intéraction avec l'atmosphère
		ph->locPrec=ph->loc;
		
// 		if(idx==0){
//			printf("\nParamètres du ph sortie du move sans interaction\n");
//			printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
//    	}

		ph->x+= ph->vx*zintermax;
		ph->y+= ph->vy*zintermax;
		ph->z+= ph->vz*zintermax;
		
		if(ph->isurface<0.f){
			ph->loc = SPACE;
			// 			if( idx==0 )
			// 				printf("Sortie dans l'espace\n");
		}
		else{
			ph->loc = SURFACE;
			// 			if( idx==0 )
			// 				printf("Va heurter la surface\n");
		}
		return;
	}
   
   // Sinon il y a intéraction
		
// Le photon intéragit dans l'atmosphère
// 		if( idx==0 ){
// 			printf("Intéraction dans l'atmosphère - tauRdm=%f\n",tauRdm);
// 			printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
// 		}
	icouche =0;
	while( (ph->hph[icouche]<tauRdm) ){
		icouche++;
		if( icouche==2*NATM+2 ){
			printf("Arret de calcul couche #1\n");
			ph->loc=NONE;
			return;
		}
	}
// 		if( icouche==0){
// 			printf("Problème icouche=0 / device.cu\n");
// 		}
// 		else if( icouche==NATM+1){
// 			printf("Problème icouche=2*NATM+1 / device.cu\n");
// 		}
	/** Calcul des coordonnées (x,y,z) du photon **/
	rra = __fdividef(ph->zph[icouche-1]-ph->zph[icouche],ph->hph[icouche-1]-ph->hph[icouche] );
	rrb = ph->zph[icouche] - ph->hph[icouche]*rra;
	rdist = rra*tauRdm + rrb;
	
	rxn = ph->x + ph->vx*rdist;
	ryn = ph->y + ph->vy*rdist;
	rzn = ph->z + ph->vz*rdist;
	
	ph->x = rxn;
	ph->y = ryn;
	ph->z = rzn;
	
	// Calcul du rayon
	rayon = ph->x*ph->x + ph->y*ph->y + ph->z*ph->z;

	if( sqrtf(rayon) < RTER ){
// 			if(idx==0){	//NOTE: on a tjrs locPrec = ATMOS, souvent icouche=1
// 		printf("Problème métaphysique dans move, rayon<RTER (%f) - locPrec=%d -icouche=%d -hph[icouche]=%f - tauRdm=%f\n",
// 			   sqrtf(rayon),ph->locPrec,icouche,ph->hph[icouche],tauRdm);
// 		printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
// 			}
		ph->isurface=1;
		ph->loc=SPACE;
		ph->weight=0.f;
		return;
	}
	
	
	/** Boucle pour définir entre quels rayons est le photon **/
	// Utilisation de icouche pour ne pas définir d'autre variable
	iray = 0;
	while( ((RTER+tab.z[iray])*(RTER+tab.z[iray]))>rayon ){ 
		iray++;
		if (iray==NATM+1){
			printf("Arret de calcul couche #2 (rayon=%f)\n", sqrtf(rayon));
			ph->loc = NONE;	// ABSORBED ou NONE ???
			return;
		}
	}
// 		if( icouche==0){
// 			printf("Problème iray=0 / device.cu\n");
// 		}
// 		else if( icouche==2*NATM+1){
// 			printf("Problème icouche=2*NATM+1 / device.cu\n");
// 		}
	
	ph->couche = iray;
// 	icouche=iray;
	scatter( ph, tab, etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
					);
					
	
	/** Le photon a été diffusé: calcul du trajet optique-géométrique devant lui **/
	calculProfil(ph, tab
		#ifdef TRAJET
		, idx
		#endif
		);
	ph->locPrec=ph->loc;
	
// 		if(idx==0){
// 			printf("\nParamètres du ph sortie du move avec interaction\n");
// 			printf("taumaxph=%f - zintermax=%f - (%f,%f,%f)\n",taumaxph,zintermax,ph->x,ph->y,ph->z);
// 			
// 		}
	icouchefi = ph->couche;
	if( icouchefi==0 ){
		// Le photon vient de la surface du coup
		// NOTE: à éventuellement modifier
		ph->isurface= -1.f;
// 		return;
	}
	else if( icouchefi==NATM ){
		// NOTE: à eventuellement modifier
		ph->isurface= 1.f;
// 		return;
	}
	else{
		// Il y a eu un problème avant, on sort et on abandonne le photon
// 		printf("Problème de calcul de icouchefi (%d)\n", icouchefi);
// 		return;
	}
	
// 	printf("Sortie usuelle: loc=%d\n",ph->loc); // AFFICHE ATMOS, ok

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
		evnt[i].tau = ph->tau;
		evnt[i].poids = ph->weight;
	}
	#endif
}


// Fonction device qui traite les photons atteignant le sol
__device__ void surfaceAgitee(Photon* photon
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
// 	if( idx==0 )
// 		printf("Entree dans surfaceAgitee - locPrec= %d\n",photon->locPrec);
	
	photon->locPrec=photon->loc;
	
	if( SIMd == -2){ // Atmosphère ou océan seuls, la surface absorbe tous les photons
		photon->loc = ABSORBED;
		
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
				evnt[i].tau = photon->tau;
				evnt[i].poids = photon->weight;
				// 		}
		}
		#endif
		
		return;
	}
	
	//Calcul du theta impact et phi impact
	//NOTE: Dans le code Fortran, ce calcul est effectué dans atmos
	float thetaimp, phiimp;
	if( photon->z > 0.f ){
		if(__fdividef(photon->z,RTER)>1.f){
			thetaimp= 0.f;
		}
		else
			thetaimp= acosf( __fdividef(photon->z,RTER) );
	
		if(photon->x >= 0.f) thetaimp = -thetaimp;
	
		if( sqrtf(photon->x*photon->x + photon->y*photon->y)<1.e-6 ){/*NOTE En fortran ce test est à 1.e-8, relativement au double
utilisés, peut peut être être supprimer ici*/
			phiimp = 0.f;
		}
		else{
			phiimp = acosf( __fdividef(photon->x, sqrtf(photon->x*photon->x + photon->y*photon->y)) );
			if( photon->y < 0.f ) phiimp = -phiimp;
		}
	}
	else{
		// Photon considéré comme perdu
		photon->loc = ABSORBED;
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
// 		float tpar, tper;	// 
	
	if( DIOPTREd !=0 ){
		sig = sqrtf(0.003F + 0.00512f *WINDSPEEDd);
		beta = atanf( sig*sqrtf(-__logf(RAND)) );
	}
	sBeta = __sinf( beta );
	cBeta = __cosf( beta );
	
	nx = sBeta*__cosf( alpha );
	ny = sBeta*__sinf( alpha );
	
	// Projection de la surface apparente de la facette sur le plan horizontal		
	float signvz = __fdividef(abs( photon->vz), photon->vz);
	bool testn = ( photon->vz > 0.F );
	nind = testn*__fdividef(1.F,NH2Od) + !testn*NH2Od;
	nz = -signvz*cBeta;
	photon->weight *= -__fdividef(abs(nx*photon->vx + ny*photon->vy + nz*photon->vz),photon->vz*nz);

	temp = -(nx*photon->vx + ny*photon->vy + nz*photon->vz);
	
// 		if (abs(temp) > 1.01F){
// 			printf("ERREUR dans la fonction surface: variable temp supérieure à 1.01\n");
// 			printf(" temp = %f\n", temp);
// 			printf("nx=%f, ny=%f, nz=%f\tvx=%f, vy=%f, vz=%f\n", nx,ny,nz,photon->vx,photon->vy,photon->vz);
// 		}
	
	theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), temp ) ));

	if(theta >= DEMIPI){
		nx = -nx;
		ny = -ny;
		theta = acosf( -(nx*photon->vx + ny*photon->vy + nz*photon->vz) );
	}
	
	cTh = __cosf(theta);
	sTh = __sinf(theta);
	
	// Rotation des paramètres de Stokes
	s1 = photon->stokes1;
	s2 = photon->stokes2;
	s3 = photon->stokes3;
	
	if( (s1!=s2) || (s3!=0.F) ){
		
		temp = __fdividef(nx*photon->ux + ny*photon->uy + nz*photon->uz,sTh);
		psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	
		
		if( (nx*(photon->uy*photon->vz-photon->uz*photon->vy)+ \
			ny*(photon->uz*photon->vx-photon->ux*photon->vz) + \
			nz*(photon->ux*photon->vy-photon->uy*photon->vx) ) <0 )
		{
			psi = -psi;
		}
		
	/*psi est l'angle entre le plan de diffusion et le plan de diffusion precedent. Rotation des
	parametres de Stoke du photon d'apres cet angle.*/
	modifStokes(photon, psi, __cosf(psi), __sinf(psi), 0 );
	
	}

	bool test_s = (sTh<=nind);
	temp = __fdividef(sTh,nind);
	cot = sqrtf( 1.0F - temp*temp )*test_s;
	ncot = nind*cot;
	ncTh = nind*cTh;
	rpar = __fdividef(cot - ncTh,cot + ncTh)*test_s + 1.F*(!test_s);
	rpar2 = rpar*rpar;
	rper = __fdividef(cTh - ncot,cTh + ncot)*test_s + 1.F*(!test_s);
	rper2 = rper*rper;
// 		ReflTot = !(test_s);
	
	// Rapport de l'intensite speculaire reflechie
	rat = __fdividef(photon->stokes2*rper2 + photon->stokes1*rpar2,photon->stokes1+photon->stokes2)*test_s;
	//if (SURd==1){ /*On pondere le poids du photon par le coefficient de reflexion dans le cas 
	// d'une reflexion speculaire sur le dioptre (mirroir parfait)*/
	photon->weight *= rat;
	// 			}
	
// 		float coeffper, coeffpar;
	
// 		if( (ReflTot==1) || (SURd==1) || ( (SURd==3)&&(RAND<rat) ) ){
		//Nouveau parametre pour le photon apres reflexion speculaire
		photon->stokes2 *= rper2;
		photon->stokes1 *= rpar2;
// 		photon->stokes4 *= -rpar*rper;
		photon->stokes3 *= -rpar*rper;

// 			coeffper = rper;
// 			coeffpar = rpar;
		
		photon->vx += 2.F*cTh*nx;
		photon->vy += 2.F*cTh*ny;
		photon->vz += 2.F*cTh*nz;
		photon->ux = __fdividef( nx-cTh*photon->vx,sTh );
		photon->uy = __fdividef( ny-cTh*photon->vy,sTh );
		photon->uz = __fdividef( nz-cTh*photon->vz,sTh );
		
		// Le photon est renvoyé dans l'atmosphère
		photon->loc = ATMOS;
		if((photon->vz<0) && DIOPTREd==2){
			// Suppression des reflexions multiples
			photon->loc = ABSORBED;
		}
// 			bool cond = ((photon->vz<0) && (DIOPTREd==2));
// 			photon->loc = ABSORBED*cond + ATMOS*(!cond);

		
// 			if( abs( 1.F - sqrtf(photon->ux*photon->ux+photon->uy*photon->uy+photon->uz*photon->uz) )>1.E-05){
// 				photon->weight = 0;
// 				photon->loc = ABSORBED;
// 				printf("suppression du photon\n");
// 				if(RAND<0.1){
// 				printf("valeur a pb:%10.8f - ux=%10.8f - uy=%10.8f - uz=%10.8f\n",
// 					   sqrt(photon->ux*photon->ux + photon->uy*photon->uy+photon->uz*photon->uz),photon->ux ,photon->uy, photon->uz);
// 					   printf("ux2=%10.8f - uy2=%10.8f-uy2=%10.8f\n",
// 							  photon->ux*photon->ux,photon->uy*photon->uy,photon->uz*photon->uz);
		
// 				}
// 				return;
// 			}
		
		
// 		}
	
// 		else{	// Transmission par le dioptre	//NOTE: Inutile pour le moment
// 			
// // 			tpar = __fdividef( 2*cTh,ncTh+ cot);
// // 			tper = __fdividef( 2*cTh,cTh+ ncot);
// // 			
// // 			photon->stokes2 *= tper*tper;
// // 			photon->stokes1 *= tpar*tpar;
// // 			photon->stokes3 *= -tpar*tper;
// // 			photon->stokes4 *= -tpar*tper;
// 			
// 			coeffpar = __fdividef( 2*cTh,ncTh+ cot);
// 			coeffper = __fdividef( 2*cTh,cTh+ ncot);
// 			
// 			alpha = __fdividef(cTh,nind) - cot;
// 			photon->vx = __fdividef(photon->vx,nind) + alpha*nx;
// 			photon->vy = __fdividef(photon->vy,nind) + alpha*ny;
// 			photon->vz = __fdividef(photon->vz,nind) + alpha*nz;
// 			photon->ux = __fdividef( nx+cot*photon->vx,sTh )*nind;
// 			photon->uy = __fdividef( ny+cot*photon->vy,sTh )*nind;
// 			photon->uz = __fdividef( nz+cot*photon->vz,sTh )*nind;
// 			
// 			// Le photon est renvoyé dans l'atmosphère
// // 			photon->loc = ;
// 			
// 			/* On pondere le poids du photon par le coefficient de transmission dans le cas d'une reflexion
// 			speculaire sur le dioptre plan (ocean diffusant) */
// 			if( SURd == 2)
// 				photon->weight *= (1-rat);
// 			
// 		}
// 		
// 		// Calcul commun sortis de la boucle pour gain de temps
// 		photon->stokes2 *= coeffper*coeffper;
// 		photon->stokes1 *= coeffpar*coeffpar;
// 		photon->stokes4 *= -coeffpar*coeffper;
// 		photon->stokes3 *= -coeffpar*coeffper;
	
	if( SIMd == -1) // Dioptre seul
		photon->loc=SPACE;

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
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
// 		}
	}
	#endif
}

__device__ void surfaceLambertienne(Photon* photon
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
	
	photon->locPrec=photon->loc;
	
	if( SIMd == -2){ // Atmosphère ou océan seuls, la surface absorbe tous les photons
		photon->loc = ABSORBED;
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
	temp = photon->vx*vxn + photon->vy*vyn + photon->vz*vzn;
	thetab = acosf( fmin( fmax(-1.f,temp),1.f ) );
	if( thetab==0){
		photon->loc=SPACE;
		printf("theta nul\n");
		return;
	}
	
	// (Produit scalaire V.Unew)/sin(theta)
	temp = __fdividef( photon->vx*uxn + photon->vy*uyn + photon->vz*uzn, __sinf(thetab) );
	psi = acosf( fmin( fmax(-1.f,temp),1.f ) );	// angle entre le plan (u,v)old et (u,v)new
	
	if( (photon->vx*(uyn*vzn-uzn*vyn) + photon->vy*(uzn*vxn-uxn*vzn) + photon->vz*(uxn*vyn-uyn*vxn) ) <0 )
	{	// test du signe de v.(unew^vnew) (scalaire et vectoriel)
	psi = -psi;
	}
	
	modifStokes(photon, psi, __cosf(psi) , __sinf(psi), 0 );
	
	photon->vx = vxn;
	photon->vy = vyn;
	photon->vz = vzn;
	photon->ux = uxn;
	photon->uy = uyn;
	photon->uz = uzn;
	
	// Aucun photon n'est absorbés mais on pondère le poids par l'albedo de diffusion de la surface lambertienne.
	photon->weight *= W0LAMd;

	// Si le dioptre est seul, le photon est mis dans l'espace
	bool test_s = ( SIMd == -1);
	photon->loc = SPACE*test_s + ATMOS*(!test_s);
	
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
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
			// 		}
	}
	#endif
	
}

// Fonction device qui traite les photons absorbés ou atteignant l'espace
__device__ void exit(Photon* ph, Variables* var, Tableaux tab, unsigned long long* nbPhotonsThr
		#ifdef PROGRESSION
		, unsigned int* nbPhotonsSorThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		#ifdef SORTIEINT
		, unsigned int iloop
		#endif
		    )
{
// 	if( idx==0 )
// 		printf("Entree dans exit - locPrec= %d - loc=%d\n",ph->locPrec, ph->loc);
	
	ph->locPrec=ph->loc;
	// Remise à zéro de la localisation du photon
	ph->loc = NONE;
	
	#ifdef SORTIEINT
	/* On compte le nombre de boucles qu'a effectué un photon pour vérifier qu'il n'y a pas de bug de
	stationnarité.*/
	if( (iloop-ph->numBoucle)<NBLOOPd ){
	atomicAdd(tab.nbBoucle + (iloop-ph->numBoucle),1);
	}
	else{
		printf("Problème dans le comptage des boucles, débordement tableau\n");
	}
	#endif

// si son poids est anormalement élevé on le compte comme une erreur. Test effectué uniquement en présence de dioptre
	if( (ph->weight > WEIGHTMAX) && (SIMd!=-2)){
		// printf("Erreur poids trop eleve\n");
		atomicAdd(&(var->erreurpoids), 1);
		return;
	}
	
	if( ph->vz<=0.f ){
		(*nbPhotonsThr)++;
		return;
	}
	
	// Sinon on traite le photon et on l'ajoute dans le tableau tabPhotons de ce thread
	#ifdef SORTIEINT
	//Sauvegarde du poids pour debug
	tab.poids[*nbPhotonsThr] = ph->weight;
	#endif

	// Incrémentation du nombre de photons traités par le thread
	(*nbPhotonsThr)++;
	
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
	
	// On modifie ensuite le poids du photon
	ph->weight = __fdividef(ph->weight, ph->stokes1 + ph->stokes2);

	// Calcul de la case dans laquelle le photon sort
	calculCase(&ith, &iphi, ph, var);
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
			evnt[i].tau = ph->tau;
			evnt[i].poids = ph->weight;
// 		}
	}
	#endif
}

// Calcul du psi pour la nouvelle direction du photon
__device__ void calculPsi(Photon* photon, float* psi, float theta)
{
	float sign;
	if (theta < 0.05F)
	{
		*psi = acosf(fmin(1.F - VALMIN, fmax(-(1.F - VALMIN), - CTHSd * photon->ux + STHSd * photon->uz)));
	}
	else
	{
		*psi = acosf(fmin(1.F, fmax(-1.F, __fdividef(STHSd * photon->ux + CTHSd * photon->uz, __sinf(theta)))));
	}
	
	sign = STHSd * (photon->uy * photon->vz - photon->uz * photon->vy) + CTHSd * (photon->ux * photon->vy - photon->uy * photon->vx);
	if (sign < 0.F) *psi = -(*psi);
}

// Fonction qui traite les photons sortants dans l'espace: changement de Stokes
/* Modifie les paramètres de stokes
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
		a = __fdividef(s2Psi * stokes3, 2.F);
		photon->stokes1 = cPsi2 * stokes1 + sPsi2 * stokes2 + a;
		photon->stokes2 = sPsi2 * stokes1 + cPsi2 * stokes2 - a;
		photon->stokes3 = s2Psi * (stokes2 - stokes1) + __cosf(psi2) * stokes3;
// 		printf("Modif stokes:s1=%f - s2=%f - (s1-s2)=%f\n",stokes1,stokes2,stokes1-stokes2);
// 		printf("ph->loc=%d - s2Psi=%f - s3=%f - s2Psi*(s1-s2)=%f\n",photon->loc,s2Psi,photon->stokes3,s2Psi * (stokes1-stokes2));
	}
}

// Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
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
		/*if(photon->vy < 0.F) *iphi = NBPHId - 1;
		else*/ *iphi = 0;
// 		return;
	}
	
}

// Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
__device__ float randomMWCfloat(unsigned long long* x,unsigned int* a)
{
	//Generate a random number (0,1]
	*x=(*x&0xffffffffull)*(*a)+(*x>>32);
	return __fdividef(__uint2float_rz((unsigned int)(*x)) + 1.0f,(float)0x100000000);
}

// Fonction random Mersenne Twister qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
__device__ float randomMTfloat(EtatMT* etat, ConfigMT* config)
{
	//Convert to (0, 1] float
	return __fdividef(__uint2float_rz(randomMTuint(etat, config)) + 1.0f, 4294967296.0f);
}

// Fonction random Mersenne Twister qui renvoit un uint à partir d'un generateur (etat+config)
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

// Fonction permettant d'utiliser atomicAdd pour un unsigned long long
__device__ void atomicAddULL(unsigned long long* address, unsigned int add)
{
	if(atomicAdd((unsigned int*)address,add)+add<add)
		atomicAdd(((unsigned int*)address)+1,1u);
}

// Fonction qui initialise les constantes du device, elle doit rester dans ce fichier
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

	float CTHSbis = cosf(THSbis); //cosThetaSolaire
	cudaMemcpyToSymbol(CTHSd, &CTHSbis, sizeof(float));

	float STHSbis = sinf(THSbis); //sinThetaSolaire
	cudaMemcpyToSymbol(STHSd, &STHSbis, sizeof(float));
	
	float GAMAbis = DEPO / (2.F-DEPO);
	cudaMemcpyToSymbol(GAMAd, &GAMAbis, sizeof(float));

	float TAUATM = TAURAY+TAUAER;
	cudaMemcpyToSymbol(TAUATMd, &TAUATM, sizeof(float));

}


__device__ void calculDiffScatter( Photon* photon, float* cTh, Tableaux tab
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
	float prop_aer= 1.f-tab.pMol[photon->couche];

	stokes1 = photon->stokes1;
	stokes2 = photon->stokes2;
	
	if( prop_aer<RAND ){	// Theta calculé pour la diffusion moléculaire
		*cTh =  2.F * RAND - 1.F; //cosThetaPhoton
		cTh2 = (*cTh)*(*cTh);
		// Calcul du poids après diffusion
		photon->weight *= __fdividef(1.5F * ((1.F+GAMAd)*stokes2+((1.F-GAMAd)*cTh2+2.F*GAMAd)*stokes1), (1.F+2.F*GAMAd) *
		(stokes1+stokes2));
		// Calcul des parametres de Stokes du photon apres diffusion
		photon->stokes2 += GAMAd * stokes1;
		photon->stokes1 = ( (1.F - GAMAd) * cTh2 + GAMAd) * stokes1 + GAMAd * photon->stokes2;
		photon->stokes3 *= (1.F - GAMAd) * (*cTh);
// 		photon->stokes4 = 0.F /*(1.F - 3.F * GAMAd) * (*cTh) * photon->stokes4*/;
	}
	
	else{	// Aérosols
		zang = RAND*(NFAERd-1);
		iang= __float2int_rd(zang);
		
		zang = zang - iang;
		theta = tab.faer[iang*5+4]+ zang*( tab.faer[(iang+1)*5+4]-tab.faer[iang*5+4] );/* L'accès à faer[x][y] se fait
par faer[y*5+x]*/
		
		*cTh = __cosf(theta);
		
		/** Changement du poids et des nombres de stokes du photon **/
		float faer1 = tab.faer[iang*5+0];
		float faer2 = tab.faer[iang*5+1];
		
		// Calcul du poids après diffusion
		photon->weight *= __fdividef( 2.0F*(stokes1*faer1+stokes2*faer2) , stokes1+stokes2)*W0AERd;
		
		// Calcul des parametres de Stokes du photon apres diffusion
		photon->stokes1 *= 2.0F*faer1;
		photon->stokes2 *= 2.0F*faer2;
		photon->stokes3 *= tab.faer[iang*5+2];
// 		photon->stokes4 = 0.F;
	}
	
}

__device__ void scatter(Photon* photon, Tableaux tab
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

// 	float tauBis = TAUATMd-photon->tau;
// 	int icouche=1;
	float cTh=0, sTh, psi, cPsi, sPsi;
	float wx, wy, wz, vx, vy, vz;
	
	photon->locPrec=photon->loc;
// 	while( (tab.h[icouche] < (tauBis)) && icouche<NATM )
// 	while( (tabCouche_s[icouche] < (tauBis)) && icouche<NATM )
// 			icouche++;
	
	psi = RAND * DEUXPI; //psiPhoton
	cPsi = __cosf(psi); //cosPsiPhoton
	sPsi = __sinf(psi); //sinPsiPhoton
	
	// Modification des nombres de Stokes
	modifStokes(photon, psi, cPsi, sPsi, 1);
	
	calculDiffScatter( photon, &cTh, tab
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
						
	sTh = sqrtf(1.F - cTh*cTh);//sinThetaPhoton
	
	/** Création de 2 vecteurs provisoires w et v **/
	// w est le rotationnel entre l'ancien vecteur u et l'ancien vecteur v du photon
	wx = photon->uy * photon->vz - photon->uz * photon->vy;
	wy = photon->uz * photon->vx - photon->ux * photon->vz;
	wz = photon->ux * photon->vy - photon->uy * photon->vx;
	
	// v est le nouveau vecteur v du photon
	vx = cTh * photon->vx + sTh * ( cPsi * photon->ux + sPsi * wx );
	vy = cTh * photon->vy + sTh * ( cPsi * photon->uy + sPsi * wy );
	vz = cTh * photon->vz + sTh * ( cPsi * photon->uz + sPsi * wz );
	
	// Changement du vecteur u (orthogonal au vecteur vitesse du photon)
	photon->ux = __fdividef(cTh * vx - photon->vx, sTh);
	photon->uy = __fdividef(cTh * vy - photon->vy, sTh);
	photon->uz = __fdividef(cTh * vz - photon->vz, sTh);
	
	// Changement du vecteur v (vitesse du photon)
	photon->vx = vx;
	photon->vy = vy;
	photon->vz = vz;
	
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
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
			// 		}
	}
	#endif
}


// Calcul du point d'impact dans l'entrée de l'atmosphère
__device__ void impact(Photon* ph, Tableaux tab
			#ifdef TRAJET
			, int idx, Evnt* evnt
			#endif
			){

	float thss, localh;
	float rdelta;
	float xphbis,yphbis,zphbis;	//Coordonnées intermédiaire du photon
	float rsolfi,rsol1,rsol2;
	
	/** Calcul du point d'impact **/
// 	thss = abs(acosf(abs(ph->vz)));
	thss = THSDEGd*DEG2RAD;
	
	rdelta = 4.f*RTER*RTER + 4.f*(__tanf(thss)*__tanf(thss)+1.f)*(HATM*HATM+2.f*HATM*RTER);
	localh = __fdividef(-2.f*RTER+sqrtf(rdelta),2.f*(__tanf(thss)*__tanf(thss)+1.f));
	
	ph->x=localh*__tanf(thss);
	ph->y = 0.f;
	ph->z = RTER+localh;	
	
// 	ph->zphz[0] = 0.f;
// 	ph->hphz[0] = 0.f;
	ph->zph[0] = 0.f;
	ph->hph[0] = 0.f;
	
	xphbis = ph->x;
	yphbis = ph->y;
	zphbis = ph->z;
	
	/** Création hphoton et zphoton, chemin optique entre sommet atmosphère et sol pour la direction d'incidence **/
	for(int icouche=1; icouche<NATM+1; icouche++){
		
		rdelta = 4.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)
				- 4.f*(xphbis*xphbis + yphbis*yphbis + zphbis*zphbis - (tab.z[icouche]+RTER)*(tab.z[icouche]+RTER));
		rsol1 = 0.5f*(-2.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)+sqrtf(rdelta));
		rsol2 = 0.5f*(-2.f*(ph->vx*xphbis+ph->vy*yphbis+ph->vz*zphbis)-sqrtf(rdelta));
		
		// Il faut choisir la plus petite distance en faisant attention qu'elle soit positive
		if(rsol1>0.f){
			if( rsol2>0.f)
				rsolfi = fmin(rsol1,rsol2);
			else
				rsolfi = rsol1;
		}
		else{
			if( rsol2>0.f )
				rsolfi=rsol1;
		}
		
// 		ph->zphz[icouche] = ph->zphz[icouche-1] + rsolfi;
// 		ph->hphz[icouche] = ph->hphz[icouche-1] + 
// 					__fdividef(abs(tab.h[icouche]-tab.h[icouche-1])*rsolfi,abs(tab.z[icouche-1]-tab.z[icouche]));
// 		
		ph->zph[icouche] = ph->zph[icouche-1] + rsolfi;
		ph->hph[icouche] = ph->hph[icouche-1] + 
					__fdividef(abs(tab.h[icouche]-tab.h[icouche-1])*rsolfi,abs(tab.z[icouche-1]-tab.z[icouche]));	
	
		xphbis+= ph->vx*rsolfi;
		yphbis+= ph->vy*rsolfi;
		zphbis+= ph->vz*rsolfi;
		
	}
	
		ph->taumax = ph->hph[NATM];
		ph->zintermax = ph->zph[NATM];
		
}

