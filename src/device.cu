
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
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; // nombre de photons taités par le thread et resortis dans l'espace
	#endif
	
	Photon ph; // On associe une structure de photon au thread
	ph.loc = NONE; // Initialement le photon n'est nulle part, il doit être initialisé
	
	/** Mesure du temps d'execution **/
	#ifdef TEMPS
	clock_t start, stop;
	float time;
	#endif
	
	/** Utilisation shared memory **/
// 	__shared__ float hph0_s[NATMd+1];
// 	__shared__ float zph0_s[NATMd+1];
// 	
// 	for( int i=0; i<NATMd+1; i++){
// 		hph0_s[i] = tab.hph0[i];
// 		zph0_s[i] = tab.zph0[i];
// 	}

// 	__shared__ float* h_s;
// 	__shared__ float* z_s;
// 	
// 	h_s = (float*) malloc((NATMd+1)*sizeof(float));
// 		if( h_s == NULL ){
//    		printf("ERREUR: Problème de malloc de h_s dans le kernel idx(%d)\n",idx);
// 		return;
//    	}
//    	
//    	z_s = (float*) malloc((NATMd+1)*sizeof(float));
//    	if( z_s == NULL ){
//    		printf("ERREUR: Problème de malloc de z_s dans le kernel idx(%d)\n",idx);
// 		return;
//    	}
//    	
// 	for( int i=0; i<NATMd+1; i++){
// 		h_s[i] = tab.h[i];
// 		z_s[i] = tab.z[i];
// 	}
	
	
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
			
			initPhoton(&ph, tab, init/*, hph0_s, zph0_s*/
				#ifdef TRAJET
				, idx, evnt
				#endif
				#ifdef SORTIEINT
				, iloop
				#endif
					);

			#ifdef TEMPS
			if(idx==0){
				stop = clock();
				time = __fdividef((float) (stop-start),__int2float_rn(CLOCKS_PER_SEC));
				printf("(1) Temps de initPhoton: %f\n", time);
			}
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
			
			move(&ph, tab, /* h_s, z_s,*/ &etatThr
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
// 			ph.locPrec = ABSORBED;
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
// 	free( h_s );
// 	free( z_s );
	
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
__device__ void initPhoton(Photon* ph, Tableaux tab, Init* init/*, float* hph0_s, float* zph0_s*/
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
	
	ph->weight = WEIGHTINIT;
	// Initialisation des paramètres de stokes du photon
	ph->stokes1 = 0.5F;
	ph->stokes2 = 0.5F;
	ph->stokes3 = 0.F;
// 	ph->stokes4 = 0.F;

// 	Paramètres initiaux calculés dans impactInit - host.cu
	ph->x = init->x0;
	ph->y = init->y0;
	ph->z = init->z0;
	ph->taumax = init->taumax0;
// 	ph->zintermax = init->zintermax0;
	ph->couche=0;	// Sommet de l'atmosphère
	ph->isurface = 1;	
	
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
			evnt[i].poids = ph->weight;
// 		}
	}
	#endif
}

// Fonction device qui traite les photons dans l'atmosphère en les faisant avancer
__device__ void move(Photon* ph, Tableaux tab/* , float* h_s, float* z_s*/
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
	double vx_d, vy_d, vz_d;
	double rsolfi=666;
	
	double delta;
	double rsol1,rsol2/*,rsol3,rsol4, rsol5,rsol6*/;
	
	int icouchefi=0;
	int icouche=0, icompteur;
	double zintermax;
	
	double tauRdm;	// Epaisseur optique aléatoire tirée
	double rra, rrb, rdist;
	double rayon2;
	double rayon;
	
	double rxn, ryn, rzn;	// Essayer de les dégager ensuite
	
	double zph, zph_p, hph, hph_p;	// Permet la sauvegarde du profil parcouru par le photon
	
	int flagSortie = 0;
	
	double vzn;	// projeté de vz sur l'axe defini par le photon et le centre de la terre
	double sinth;	// Sinus de l'angle entre z' et Vz'
	double costh;
	double ztangentielle;
	int coucheTangentielle=666;
	int sens;
	int icoucheTemp; // Couche suivante que le photon va toucher
// 	if( idx==0 )
// 		printf("Entree dans move - locPrec= %d\n",ph->locPrec);
	
	/** Tirage au sort de la profondeur optique à parcourir **/
	/*  Tirage effectué lors de chaque appel de la fonction */
	// Pas de diffusion forcée en sphérique
	
	tauRdm = -log(1.F-RAND);
// 	tauRdm=0.1;
	
	if( tauRdm ==0. ){
// 		printf("TauRdm=0\n");
		flagSortie =1;
		if(ph->locPrec==NONE)
			zintermax = tab.zph0[NATMd];
		else if (ph->locPrec==ATMOS)
			zintermax = 0;	// Le photon ne va pas bouger
	}

	/** Calcul puis déduction de la couche du photon **/	
	icompteur = 0;

	// Initialisation
	hph_p = 0.;
	zph_p = 0.;
	hph=0.;
	zph=0.;
	
	if( (ph->locPrec==NONE) && (flagSortie!=1) ){
		
		// Déjà fait dans initPhoton
		// hph = tab.hph0[0];
		// zph = tab.zph0[0];
		// taumaxph = ph->taumax;
		// zintermax = ph->zintermax;
		// ph->isurface = 1;		
/*		
		if(idx==0){
			printf("TauRdm=%lf - taumax=%lf\n", tauRdm, ph->taumax);
		}*/
		
		if( tauRdm >= (ph->taumax) ){
			flagSortie = 1;
			zph=tab.zph0[NATMd]; // Pour retrouver le zintermax ensuite
		}
		
		
		while( (hph < tauRdm) && (flagSortie!=1) ){
		// Le photon vient du sommet de l'atmosphère - On va utiliser les calculs de impact.
		
			if( icompteur==(NATMd+1) ){
				printf("icompteur = NATMd+1 pour premier calcul de position - tauRdm=%f - taumax=%f - hph_p=%f - hph=%f\n",\
					tauRdm, ph->taumax, hph_p, hph);
					flagSortie = 1;
					break;
			}
			
			hph_p = hph;	// Sauvegarde du calcul de la couche précédente
			hph = tab.hph0[icompteur];
			zph_p = zph;	// Sauvegarde du calcul de la couche précédente
			zph = tab.zph0[icompteur];
			icompteur++;
			
			// Si on dépasse le taumax, on arrete, et pas d'interaction avec l'atmosphere

		}
		
	}
	
// 	else if ( (ph->locPrec==SURFACE) && (flagSortie!=1) ){
// // 		Le photon vient de la surface
// 		while( hph < tauRdm){
// 		}
// 	}
	
	else if( (ph->locPrec==ATMOS) && (flagSortie!=1) ){
		// Le photon vient de l'atmosphère

		/** Changement de repère **/
		// calcul du nouveau z', axe passant par le centre de la terre et le photon
		rayon2= ph->x*ph->x + ph->y*ph->y + ph->z*ph->z;
		rayon = sqrt(rayon2);
		
		
		if( (rayon < RTER) || ( rayon>RTER+HATM ) ){
				// if(idx==0){
				printf("MetaProblème #1.1:rayon=%20.16lf - icouche=%d - icouchefi=%d\n"
				,rayon,icouche, icouchefi);
				// }
				ph->loc = NONE;
				return;
		}
		
		// Calcul du Vz' par rapport à z'. Son signe donne la direction du photon
		// Vz'=vzn= V.Z/|Z|
		vx_d = (double)ph->vx;
		vy_d = (double)ph->vy;
		vz_d = (double)ph->vz;
		
		vzn = (vx_d*ph->x + vy_d*ph->y + vz_d*ph->z )/rayon;		
		
// 		if(idx==0){
// 			printf("vzn= %lf - Z=(%lf, %lf, %lf) - V=(%lf, %lf, %lf)\n",\
			vzn, ph->x, ph->y, ph->z, vx_d, vy_d, vz_d);
// 		}
		

		// Calcul costh= Vz'/|Vz|
		costh = vzn;
		
		#ifdef DEBUG
		if( abs(costh)>1 ){
			printf("PROBLEME costh (%20.19lf)\n",costh);
			ph->loc=NONE;
			return;
		}
		#endif
		
		sinth = sqrt(1-costh*costh);	// Le signe n'importe pas car on prend le carré ou la valeur absolu pour ztangentielle
		
		icoucheTemp = ph->couche;
		
		if( vzn>=0 ){
		// Le photon monte, il sortira forcement par le sommet. Il n'y a donc pas de couche tangentielle.
		// On la met à 666 pour le bien de l'algorithme ensuite
			coucheTangentielle=666;
			sens = +1;
		}
		else{
		// Le photon descend
			sens = -1;
			/** Calcul couche tangentielle **/
			// ht = rayon*sin(th) où th est l'angle entre z' et Vz'
			ztangentielle = rayon*sinth;
			
			/** Recherche de la couche correspondante **/
			if( ztangentielle<=RTER ){
				// Le photon va passer par la surface
				coucheTangentielle = 666;
			}
			else{
				coucheTangentielle = 0;
				while( (RTER+(double)tab.z[coucheTangentielle])>ztangentielle){ 
					coucheTangentielle++;
					if (coucheTangentielle==NATMd+1){
						printf("Arret de calcul couche ztangentielle (%lf)\n", ztangentielle);
						ph->loc = NONE;
						return;
					}
				}
			}
		}
		
		// Le premier cas est un peu différent car le photon se trouve entre 2 couches.
		// On le positionne virtuellement sur une couche (cela ne change rien pour les calculs).
		
		//icoucheTemp est la couche dans laquelle se trouve le photon "virtuel"
		icouchefi=icoucheTemp;	// icouchefi est la couche suivante que va toucher le photon "virtuel"
		// icouche devient la couche que le photon va atteindre
		if( (icoucheTemp==coucheTangentielle)&&(sens==-1) ){
			// Le photon va remonter dans les couche
			sens=1;
		}
		
		if( sens== +1) icouchefi--;
		
		// Le photon va etre vu comme étant sur la borne inférieure de la couche dans laquelle il se trouve rééllement
		
// 		if(idx==0){
// 			printf("Couches initiales: rayon=%lf - sens=%d - icouche=%d - coucheTang=%d - tab.z[icouche]= %16.13lf\n",\
			rayon, sens, icouche, coucheTangentielle, tab.z[icouche]);
// 			if( (icouche>NATMd)||(icouche<0) )
// 			printf("OUPS: icouche=%d\n", icouche);
// 		}
		
		/** Premier calcul **/
		// Distance entre le photon et la couche qu'il va toucher (en calculant sur toutes les couches)
		delta = 4*( ((double)tab.z[icouchefi]+RTER)*((double)tab.z[icouchefi]+RTER) - rayon2*sinth*sinth);
		
		#ifdef DEBUG
		if(delta<0){
// 			if(idx==0){
				printf("OUPS rdelta #1=%lf - icoucheTemp=%d - tab.z[icoucheTemp]= %16.13lf - rayon= %17.13lf - rayon2=%20.16lf\n\t\
sinth= %20.19lf - sens=%d\n",\
				delta, icoucheTemp, tab.z[icoucheTemp], rayon, rayon2, sinth, sens);
// 			}
			ph->loc=NONE;
			return;
		}
		#endif

		// Calcul de la solution avec astuce de sioux
		// sens indique si le photon monte ou descends, et la solution se déduit de ce signe
		rsolfi= 0.5*( -2*rayon*costh + sens*sqrt(delta));
		
		#ifdef DEBUG
		if( rsolfi<0 ){
			rsol1=0.5*( -2*rayon*costh + sqrt(delta));
			rsol2=0.5*( -2*rayon*costh - sqrt(delta));
			printf("OUPS: rsolfi #1=%lf, (%lf,%lf) - vzn=%lf - sens=%d\n\t\
costh= %16.15lf - rayon= %16.12lf - delta= %16.10lf\n",\
			rsolfi,rsol1,rsol2, vzn, sens, costh, rayon, delta);
			ph->loc=NONE;
			return;
		}
		#endif
		
		// Calcul des grandeurs du profil
		hph_p = hph;
		zph_p = zph;
		
		// Petite vérification
		#ifdef DEBUG
		if( icouchefi<0 ){
			printf("OUPS: icouchefi #1 = %d - rayon=%lf - icouchePhoton=%d - icoucheTemp=%d\n",\
			icouchefi, rayon, ph->couche, icoucheTemp);
			ph->loc=NONE;
			return;
		}
		if( icouchefi>NATMd ){
			printf("OUPS: icouchefi #1 = %d - rayon=%lf - icouchePhoton=%d - icoucheTemp=%d\n",\
			icouchefi, rayon, ph->couche, icoucheTemp);
			ph->loc=NONE;
			return;
		}
		#endif
		
		// Valeur de la couche actuelle
		if( icouchefi!=icoucheTemp ){
			hph = /*__fdividef*/( abs( /*(double)*/tab.h[icoucheTemp] - /*(double)*/tab.h[icouchefi] )*rsolfi)/
									( abs( /*(double)*/tab.z[icouchefi] -/* (double)*/tab.z[icoucheTemp]) );
		}
		else{
			if( icouchefi==0 ){
				if( icouchefi==NATMd ){
					printf("OUPS: icouchefi=NATM => débordement tableau");
					ph->loc=NONE;
					return;
				}
				hph =/*__fdividef*/ ( abs( /*(double)*/tab.h[icouchefi+1] - /*(double)*/tab.h[icouchefi])*rsolfi )/
				( abs( /*(double)*/tab.z[icouchefi+1] - /*(double)*/tab.z[icouchefi]) );
			}
			else{
				hph = /*__fdividef*/( abs(/* (double)*/tab.h[icouchefi-1] - /*(double)*/tab.h[icouchefi])*rsolfi)/
				( abs( /*(double)*/tab.z[icouchefi-1] - /*(double)*/tab.z[icouchefi]) );
			}
		}
		
		zph=rsolfi/* + zph_p*/;
// 		zintermax = zph;

		#ifdef DEBUG
// 		if( idx==0 ){
// 			rsol1=0.5*( -2*rayon*costh + sqrt(delta));
// 			rsol2=0.5*( -2*rayon*costh - sqrt(delta));
// 			printf("(%d)Profil: rsolfi=%lf (%lf, %lf) - rdelta=%lf - sens=%d - icoucheTemp=%d - icouchefi=%d - icoucheTan=%d\n\t\
// rayon=%lf - vzn=%lf - hph=%lf - zph=%lf\n",\
// 			icompteur, rsolfi, rsol1, rsol2, delta, sens, icoucheTemp, icouchefi, coucheTangentielle,\
// 			rayon, vzn, hph, zph);
// 		}
		#endif

		icoucheTemp = icouchefi;
		
		// Le photon est à présent virtuellement sur la bonne couche en fonction de s'il monte ou descend, on continue les calculs
		// du profil

		/** Calcul du profil **/
		// Calcul jusqu'à sortir ou intéragir
		while( (hph < tauRdm) && (flagSortie!=1)){
			
			// Vérification que le photon soit toujours dans l'atmosphère
			if( icoucheTemp==0 ) {
				ph->isurface=-1;
				flagSortie=1;
				break;
			}
			else if( icoucheTemp==(NATMd) ){
				ph->isurface=1;
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
			delta = 4*( ((double)tab.z[icouchefi]+RTER)*((double)tab.z[icouchefi]+RTER) - rayon2*sinth*sinth);
			
			if(delta<0){
// 				if(idx==0){
				printf("OUPS delta #2=%lf - icouche=%d - tab.z[icouche]= %16.13lf - rayon= %17.13lf - rayon2= %20.16lf\n\t\
sinth= %20.19lf - sens=%d\n",\
				delta, icouche, tab.z[icouche], rayon, rayon2, sinth, sens);
// 				}
				ph->loc=NONE;
				return;
			}
			
			// Calcul de la solution avec astuce de sioux
			rsolfi= 0.5*( -2*rayon*costh + sens*sqrt(delta));
			
			#ifdef DEBUG
			if( rsolfi<0 ){
				// Problème, c'est un rayon
				rsol1=0.5*( -2*rayon*costh + sqrt(delta));
				rsol2=0.5*( -2*rayon*costh - sqrt(delta));
				printf("OUPS: rsolfi #2=%lf, (%lf,%lf) - vzn=%lf - sens=%d - costh=%lf - rayon=%lf - delta=%lf\n",\
				rsolfi,rsol1,rsol2, vzn, sens, costh, rayon, delta);
				ph->loc=NONE;
				return;
			}
			#endif
			
			// Calcul des grandeurs du profil
			hph_p = hph;
			zph_p = zph;
			
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
			
			// Valeur de la couche actuelle
			if( icouchefi!=icoucheTemp ){
				hph = /*__fdividef*/( abs( /*(double)*/tab.h[icoucheTemp] - /*(double)*/tab.h[icouchefi] )*(rsolfi-zph_p))/
										( abs( /*(double)*/tab.z[icouchefi] -/* (double)*/tab.z[icoucheTemp]) ) + hph_p;
			}
			else{
				if( icouchefi==0 ){
					hph =/*__fdividef*/ ( abs(/*(double)*/tab.h[icouchefi+1] -/*(double)*/tab.h[icouchefi])*(rsolfi-zph_p))/
											( abs( /*(double)*/tab.z[icouchefi+1] - /*(double)*/tab.z[icouchefi]) ) + hph_p;
				}
				else{
					hph = /*__fdividef*/( abs(/*(double)*/tab.h[icouchefi-1] -/*(double)*/tab.h[icouchefi])*(rsolfi-zph_p))/
											( abs( /*(double)*/tab.z[icouchefi-1] - /*(double)*/tab.z[icouchefi]) ) + hph_p;
				}
			}
			
			zph=rsolfi/* + zph_p*/;
// 			zintermax = zph;

			#ifdef DEBUG
// 			if( idx==0 ){
// 				rsol1=0.5*( -2*rayon*costh + sqrt(delta));
// 				rsol2=0.5*( -2*rayon*costh - sqrt(delta));
// 				printf("(%d)Profil: rsolfi=%lf (%lf, %lf) - rdelta=%lf - sens=%d - icoucheTemp=%d - icouchefi=%d - icoucheTan=%d\n\t\
// rayon= %lf - vzn= %lf - hph=%lf - zph=%lf\n",\
// 				icompteur, rsolfi, rsol1, rsol2, delta, sens, icoucheTemp, icouchefi, coucheTangentielle,\
// 				rayon, vzn, hph, zph);
// 			}
			#endif

			icoucheTemp = icouchefi;
			
			// Compteur de débordement
			icompteur++;
			if(icompteur==(2*NATMd+2)){
				printf("icouche = 2(NATMd+1) - (%lf,%lf,%lf) - icouchefi=%d - flagSortie=%d\n\t\
	ph->vz=%f - rsolfi=%f - tauRdm=%f - hph=%f\n",\
				ph->x,ph->y,ph->z, icouchefi,flagSortie, ph->vz,rsolfi,tauRdm,hph);
				ph->loc=NONE;
				return;
			}
		
		}// Fin while
		
// 		ph->loc=NONE;
// 		return;
		
	}// Fin de si photon dans l'atmosphere

	/** Sauvegarde de quelques paramètres **/
	zintermax= zph;
// 	ph->taumax= hph;
	
	
// 	if( idx==0 ){
// 		printf("Paramètres du photon après calcul: zinter = %f - taumax = %f - hph = %f - icouche=%d - tauRdm=%f\n",
// 			   ph->zintermax,ph->taumax, hph, icouche, tauRdm);
// 	}
	/** Sortie **/
	vx_d = (double)ph->vx;
	vy_d = (double)ph->vy;
	vz_d = (double)ph->vz;

	if( flagSortie==1 ){
			//Il n'y a pas eu d'intéraction avec l'atmosphère
			
// 			if(idx==0){
// 				printf("Sortie du move sans interaction\n");
// 				printf("taumaxph=%f - zintermax=%f - (%f,%f,%f) - tauRdm=%f\n",taumaxph,zintermax,ph->x,ph->y,ph->z, tauRdm);
// 	   	}
		
		ph->x+= vx_d*zintermax;
		ph->y+= vy_d*zintermax;
		ph->z+= vz_d*zintermax;
		
		if(ph->isurface<0.f){
			ph->loc = SPACE;
// 				if( idx==0 )
// 					printf("Sortie dans l'espace\n");
		}
		else{
			ph->loc = SURFACE;
			// 	if( idx==0 )
			// 		printf("Va heurter la surface\n");
		}
		
// 		if(idx==0){
// 		printf("Sortie sans inter: rayon=%lf, (%lf,%lf,%lf), hph=%lf, zph=%lf, zintermax=%lf, locPrec=%d\n",\
// 		sqrt(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z),ph->x,ph->y,ph->z,hph, zph, zintermax, ph->locPrec);
// 		}
		
		return;
	}
	
// 	if( flagSortie==2 ){
// 	// Le photon reste dans l'atmosphère après intéraction
// 		
// 	
// 	return;
// 	}
	
	// Il y a intéraction dans l'atmosphère
	/** Calcul des coordonnées (x,y,z) du photon **/
	// Interpolation linéaire entre les bornes de la couche car le trajet au sein de la couche est rectiligne
	rra = /*__fdividef(*/ ( zph_p - zph )/( hph_p - hph );
	rrb = zph - hph*rra;
	rdist = rra*tauRdm + rrb;
	
	rxn = ph->x + vx_d*rdist;
	ryn = ph->y + vy_d*rdist;
	rzn = ph->z + vz_d*rdist;
	
	ph->x = rxn;
	ph->y = ryn;
	ph->z = rzn;
	
	// Calcul du rayon
	rayon2 = ph->x*ph->x + ph->y*ph->y + ph->z*ph->z;
	rayon = sqrt(rayon2);
// 	if(idx==0) printf("rayon final= %20.16lf\n", rayon );

	if( (rayon < RTER)||(rayon>RTER+HATM) ){
		
// 		if( (rayon-RTER)<1.e-4 ){
// 			rayon=RTER;
// 			ph->z = sqrt( RTER*RTER - ph->x*ph->x - ph->y*ph->y );
// 			ph->loc=SURFACE;
// 			return;
// 		}
// 		else{
// 			if(idx==0){
			printf("MetaProblème #2: rayon=%20.16lf - (%lf,%lf,%lf) - icouchefi=%d - icompteur=%d -locPrec=%d\n\t\
rsolfi=%15.12lf - tauRdm= %lf - hph_p= %15.12lf - hph= %15.12lf - zph_p= %15.12lf - zph= %15.12lf\n",\
rayon,ph->x, ph->y,ph->z, icouchefi, icompteur,ph->locPrec,\
rsolfi,tauRdm, hph_p, hph, zph_p, zph);	
   // 			}
			ph->loc = NONE;
			return;
// 		}
	}
	
	/*********************************************************************************/
	/** Boucle pour définir entre quels rayons est le photon **/
	// Utilisation de icouche pour ne pas définir d'autre variable
	icouche = 0;
	while( (RTER+(double)tab.z[icouche])>rayon ){ 
		icouche++;
		if (icouche==NATMd+1){
			printf("Arret de calcul couche #2 (rayon=%f)\n", rayon);
			ph->loc = NONE;	// ABSORBED ou NONE ???
			return;
		}
	}
	
// 	if( idx==0 ){
// 		printf("Sortie avec interaction: rayon=%f, Z=(%lf, %lf, %lf), icouche=%d, hph=%lf, zph=%lf, locPrec=%d\n",\
// 		rayon, ph->x, ph->y, ph->z, icouche,hph,zph,ph->locPrec );
// 	}

	ph->couche = icouche;
	ph->locPrec=ATMOS;
	
	// On sort maintenant de la fonction et comme le photon reste dans ATMOS, le kernel appelle scatter()

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
	float icp, isp, ict, ist;	// Sinus et cosinus de l'angle d'impact
	
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
	
	float vxn, vyn, vzn, uxn, uyn, uzn;
	
	/** Il faut exprimer Vx,y,z et Ux,y,z dans le repère de la normale au point d'impact **/
	icp = cosf(phiimp);
	isp = sinf(phiimp);
	ict = cosf(thetaimp);
	ist = sinf(thetaimp);

	vxn= ict*icp*ph->vx - ict*isp*ph->vy + ist*ph->vz
	vyn= isp*ph->vx + icp*ph->vy
	vzn= -icp*ist*ph->vx + ist*isp*ph->vy + ict*ph->vz
	
	uxn= ict*icp*ph->ux - ict*isp*ph->uy + ist*ph->uz
	uyn= isp*ux + icp*uy
	uzn= -icp*ist*ph->ux + ist*isp*ph->uy + ict*ph->uz
	
	ph->vx = vxn
	ph->vy = vyn
	ph->vz = vzn
	ph->ux = uxn
	ph->uy = uyn
	ph->uz = uzn
	
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
// 		printf("Entree dans exit - weight=%f - vz= %f\n",ph->weight, ph->vz);
	
// 	ph->locPrec=ph->loc;
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
	cudaMemcpyToSymbol(NATMd, &NATM, sizeof(int));
	
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
// 		*cTh = 0.1;
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

	float cTh=0, sTh, psi, cPsi, sPsi;
	float wx, wy, wz, vx, vy, vz;
	
	ph->locPrec=ph->loc;
	
// 	if(idx==0){
// 	printf("Couche dans scatter = %d - v_avt=(%f, %f, %f )\n",ph->couche,ph->vx, ph->vy, ph->vz);
// 	}
	
	psi = RAND * DEUXPI; //psiPhoton
// 	psi = PI/2;
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
						
	sTh = sqrtf(1.F - cTh*cTh);//sinThetaPhoton
	
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
	
// 	if(idx==0){
// 		printf("Sortie scatter - v_après=(%f, %f, %f )\n",ph->vx, ph->vy, ph->vz);
// 	}
	
	
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

