
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "device.h"

	  //////////////////////
	 // FONCTIONS DEVICE //
	//////////////////////

// Fonction device principale qui lance tous les threads, leur associe des photons, et les fait évoluer
__global__ void lancementKernel(Variables* var, Tableaux tab
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
	
	Photon photon; // On associe une structure de photon au thread
	photon.loc = NONE; // Initialement le photon n'est nulle part, il doit être initialisé

	// Dans cette boucle on simule le parcours du photon, puis on le réinitialise,... Le thread lance plusieurs photons
	for(unsigned int iloop= 0; iloop < NBLOOPd; iloop++)
	{
		// Si le photon est à NONE on l'initialise et on le met à ATMOS
		if(photon.loc == NONE){ init(&photon
				#ifdef TRAJET
				, idx, evnt
				#endif
				#ifdef SORTIEINT
				, iloop
				#endif
					);
			flagDiff = DIFFFd;
			
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		
		// Si le photon est à ATMOS on le fait avancer jusqu'à SURFACE, ou SPACE, ou ATMOS s'il subit une diffusion
		if( (photon.loc == ATMOS) && (SIMd==-2 || SIMd==1 || SIMd==2) ) move(&photon, flagDiff, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
						);
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est encore à ATMOS il subit une diffusion et reste dans ATMOS
		if( (photon.loc == ATMOS) && (SIMd==-2 || SIMd==1 || SIMd==2)){
// 			scatterTot(&photon, tab, &etatThr
// 			#if defined(RANDMWC) || defined(RANDMT)
// 			, &configThr
// 			#endif
// 			#ifdef TRAJET
// 			, idx, evnt
// 			#endif
// 				);
				
			scatter( &photon, tab, &etatThr
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
		
		// Si le photon est à SURFACE on le met à ABSORBED
		if(photon.loc == SURFACE) surfac(&photon, &etatThr
				#if defined(RANDMWC) || defined(RANDMT)
				, &configThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
						);
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();
		
		// Si le photon est dans SPACE ou ABSORBED on récupère ses infos et on le remet à NONE
		if(photon.loc == ABSORBED || photon.loc == SPACE){
			exit(&photon, var, tab, &nbPhotonsThr
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
									
			#ifdef SORTIEINT
			/* On compte le nombre de boucles qu'a effectué un photon pour vérifier qu'il n'y a pas de bug de
			stationnarité.*/
			atomicAdd(tab.nbBoucle + (iloop-photon.numBoucle),1);
			#endif
		}
		// Chaque block attend tous ses threads avant de continuer
		syncthreads();

		//Mise à jour du poids suite à la 1ère diffusion forcée
		if(flagDiff==1 ){
			photon.weight *= (1.F - __expf(-TAUMAXd));
			flagDiff=0;
		}
		syncthreads();

	}// Fin boucle for
	
	#ifdef SORTIEINT
	if( photon.numBoucle <NBLOOPd/2 ){
		atomicAdd(tab.nbBoucle + (NBLOOPd-1-photon.numBoucle),1);
// 		printf("numBoucle après sorti : %d\n",photon.numBoucle);
	}
// 	else
// 		printf("numBoucle dernier : %d\n",photon.numBoucle);
	
	#endif

	// Après la boucle on rassemble les nombres de photons traités par chaque thread
	// NOTE: ULL retiré
	atomicAdd(&(var->nbPhotons), nbPhotonsThr);
	
	#ifdef PROGRESSION
	// On rassemble les nombres de photons traités et sortis de chaque thread
	// NOTE: ULL retiré
	atomicAdd(&(var->nbPhotonsSor), nbPhotonsSorThr);
	// On incrémente avncement qui compte le nombre d'appels du Kernel
	// NOTE: ULL retiré
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
__device__ void init(Photon* photon
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		#ifdef SORTIEINT
		, unsigned int iloop
		#endif
		    )
{
	// Initialisation du vecteur vitesse
	photon->vx = - STHSd;
	photon->vy = 0.F;
	photon->vz = - CTHSd;
	// Initialisation du vecteur orthogonal au vecteur vitesse
	photon->ux = -photon->vz;
	photon->uy = 0.F;
	photon->uz = photon->vx;
	// Le photon est initialement dans l'atmosphère, et tau peut être vu comme sa hauteur par rapport au sol
	photon->loc = ATMOS;
	if( SIMd == -1 )
		photon->loc = SURFACE;
	
	photon->tau = TAUATMd;
	photon->weight = WEIGHTINIT;
	// Initialisation des paramètres de stokes du photon
	photon->stokes1 = 0.5F;
	photon->stokes2 = 0.5F;
	photon->stokes3 = 0.F;
	photon->stokes4 = 0.F;
	
	#ifdef SORTIEINT
	photon->numBoucle = iloop;
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
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
// 		}
	}
	#endif
}

// Fonction device qui traite les photons dans l'atmosphère en les faisant avancer
__device__ void move(Photon* photon, int flagDiff
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

	// tirage épaisseur optique avec diffusion forcée
	if( DIFFFd == 1 && flagDiff== 1 ){
		photon->tau += -__logf(1.F-RAND*(1.F-__expf(-TAUMAXd))) * photon->vz;
	}
	
	else	// Tirage de la nouvelle épaisseur optique du photon sans diffusion forcée
		photon->tau += -__logf(RAND) * photon->vz;


	// Si tau<0 le photon atteint la surface
	if(photon->tau < 0.F){
		photon->loc = SURFACE;
		photon->tau = 0.F;
	}
	// Si tau>TAURAY le photon atteint l'espace
	else if(photon->tau > TAUATMd) photon->loc = SPACE;
	// Sinon il a rencontré une molécule, on ne fait rien car il reste dans l'atmosphère, et va être traité par scatter
	
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
			// "2"représente l'événement "move" du photon
			evnt[i].action = 2;
			// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
// 		}
	}
	#endif
}

// Fonction device qui traite les photons qui sont encore dans l'atmosphère après "move" : rencontre avec une molécule
__device__ void scatterMol(Photon* photon
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
	// On détermine la nouvelle direction du photon
	float cTh, cTh2, sTh, psi, cPsi, sPsi;

	// On prend un cosinus aléatoirement entre -1 et 1, il s'agit du cosinus de l'angle que fait la nouvelle direction avec l'ancienne, on prend le cos aléatoirement pour l'équiprobabilité dans toutes les directons
	cTh = 2.F * RAND - 1.F; //cosThetaPhoton
	cTh2 = cTh * cTh; //cosThetaPhotonCarré
	sTh = sqrtf(1.F - cTh2);//sinThetaPhoton

	// On prend un angle phi aléatoirement entre 0 et 2.PI, il s'agit de l'angle entre le nouveau vecteur projeté dans le plan orthogonal à v et le vecteur u
	psi = RAND * DEUXPI; //psiPhoton
	cPsi = __cosf(psi); //cosPsiPhoton
	sPsi = __sinf(psi); //sinPsiPhoton

	// Modification des nombres de Stokes
	modifStokes(photon, psi, cPsi, sPsi, 1);
	
	// Création de 2 vecteurs provisoires w et v
	float wx, wy, wz, vx, vy, vz;

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
	
	// Changement du poids et des nombres de stokes du photon
	float stokes1 = photon->stokes1;
	float stokes2 = photon->stokes2;
	// Calcul du poids après diffusion
	photon->weight *=__fdividef(1.5F * ((1.F+GAMAd)*stokes2+((1.F-GAMAd)*cTh2+2.F*GAMAd)*stokes1),(1.F+2.F*GAMAd)*(stokes1+stokes2));
	// Calcul des parametres de Stokes du photon apres diffusion
	photon->stokes2 = stokes2 + GAMAd * stokes1;
	photon->stokes1 = ( (1.F - GAMAd) * cTh2 + GAMAd) * stokes1 + GAMAd * photon->stokes2;
	photon->stokes3 = (1.F - GAMAd) * cTh * photon->stokes3;
	photon->stokes4 = (1.F - 3.F * GAMAd) * cTh * photon->stokes4;
	
	
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

// Fonction device qui traite les photons atteignant le sol
__device__ void surfac(Photon* photon
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
	float theta;		// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float phi;
	float cTh=0, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	float cPhi, sPhi;
	
	if( SIMd == -2){ // Atmosphère ou océan seuls, la surface absorbe tous les photons
		photon->loc = ABSORBED;
	}
	
	else if( DIOPTREd !=3 ){	// Réflexion sur le dioptre agité
		float sig = 0.F;
		float beta = 0.F;// Angle par rapport à la verticale du vecteur normal à une facette de vagues 
		float sBeta = 0.F;
		float cBeta = 1;

		float alpha = DEUXPI*RAND; //Angle azimutal du vecteur normal a une facette de vagues
	
		float nind=0;
		float temp=0;
	
		float nx, ny, nz;	// Coordonnées du vecteur normal à une facette de vague
		float s1, s2, s3;
		
		float rpar=1.F, rper=1.F;	// Coefficient de reflexion parallèle et perpendiculaire
		float rpar2=1.F;		// Coefficient de reflexion parallèle au carré
		float rper2=1.F;		// Coefficient de reflexion perpendiculaire au carré
		float rat=0.F;		// Rapport des coefficients de reflexion perpendiculaire et parallèle
		float ReflTot = 1;	// Flag pour la réflexion totale sur le dioptre
		float cot=0.F;		// Cosinus de l'angle de réfraction du photon
		float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
		float tpar, tper;	// 
		
		if( DIOPTREd !=0 ){
			sig = sqrtf(0.003F + 0.00512*WINDSPEEDd);
			beta = atanf( sig*sqrtf(-__logf(RAND)) );
			sBeta = __sinf( beta );
			cBeta = __cosf( beta );
		}

		
		nx = sBeta*__cosf( alpha );
		ny = sBeta*__sinf( alpha );
		
		// Projection de la surface apparente de la facette sur le plan horizontal
		
		if( photon->vz > 0.F ){
			nind = __fdividef(1.F,NH2Od);
			nz = -cBeta;
			// Correction bug surface apparente facettes face/dos au photon
			photon->weight *= __fdividef( abs(nx*photon->vx + ny*photon->vy+ nz*photon->vz),photon->vz) ;
		}

		else{
			nind = NH2Od;
			nz = cBeta;
			// Correction bug surface apparente facettes face/dos au photon
			photon->weight *= -__fdividef(abs(nx*photon->vx + ny*photon->vy + nz*photon->vz),photon->vz) ;
			
		}
		// Projection de la surface apparente de la facette sur le plan horizontal
		photon->weight = __fdividef( photon->weight, cBeta );
		
		temp = -(nx*photon->vx + ny*photon->vy + nz*photon->vz);
		
		if (abs(temp) > 1.01F){
			printf("ERREUR dans la fonction surface: variable temp supérieure à 1.01\n");
			printf(" temp = %f\n", temp);
			printf("nx=%f, ny=%f, nz=%f\tvx=%f, vy=%f, vz=%f\n", nx,ny,nz,photon->vx,photon->vy,photon->vz);
		}
		
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

		if(sTh<=nind){	// Calcul des coefficients de reflexion ( parallele et perpendiculaire )
// 			cot = __cosf( asinf(__fdividef(sTh,nind)) );
			temp = __fdividef(sTh,nind);
			cot = sqrtf( 1.0F - temp*temp );
			ncot = nind*cot;
			ncTh = nind*cTh;
			rpar = __fdividef(cot - ncTh,cot + ncTh);
			rpar2 = rpar*rpar;
			rper = __fdividef(cTh - ncot,cTh + ncot);
			rper2 = rper*rper;
			ReflTot = 0;
			
			// Rapport de l'intensite speculaire reflechie
			rat=__fdividef(photon->stokes2*rper2 + photon->stokes1*rpar2,photon->stokes1+photon->stokes2);
		}
		
		if( (ReflTot==1) || (SURd==1) || ( (SURd==3)&&(RAND<rat) ) ){
			//Nouveau parametre pour le photon apres reflexion speculaire
			photon->stokes2 *= rper2;
			photon->stokes1 *= rpar2;
			photon->stokes4 *= -rpar*rper;
			photon->stokes3 *= -rpar*rper;
			
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
			

			
			if (SURd==1){ /*On pondere le poids du photon par le coefficient de reflexion dans le cas 
d'une reflexion speculaire sur le dioptre (mirroir parfait)*/
				photon->weight *= rat;
			}
// 			cond = (SURd==1);
// 			photon->weight = photon->weight*rat*cond + photon->weight*(!cond);
			
			
		}
		
		else{	// Transmission par le dioptre
			
			tpar = __fdividef( 2*cTh,ncTh+ cot);
			tper = __fdividef( 2*cTh,cTh+ ncot);
			
			photon->stokes2 *= tper*tper;
			photon->stokes1 *= tpar*tpar;
			photon->stokes3 *= -tpar*tper;
			photon->stokes4 *= -tpar*tper;
			
			alpha = __fdividef(cTh,nind) - cot;
			photon->vx = __fdividef(photon->vx,nind) + alpha*nx;
			photon->vy = __fdividef(photon->vy,nind) + alpha*ny;
			photon->vz = __fdividef(photon->vz,nind) + alpha*nz;
			photon->ux = __fdividef( nx+cot*photon->vx,sTh )*nind;
			photon->uy = __fdividef( ny+cot*photon->vy,sTh )*nind;
			photon->uz = __fdividef( nz+cot*photon->vz,sTh )*nind;
			
			// Le photon est renvoyé dans l'atmosphère
// 			photon->loc = ;
			
			/* On pondere le poids du photon par le coefficient de transmission dans le cas d'une reflexion
			speculaire sur le dioptre plan (ocean diffusant) */
			if( SURd == 2)
				photon->weight *= (1-rat);
			
		}
		
		if( SIMd == -1) // Dioptre seul
			photon->loc=SPACE;
		
	}
	
	else if( DIOPTREd == 3){// Reflexion surface lambertienne
		
		float thetab;	// angle de diffusion (entre le vecteur avt et après reflexion)
		float uxn,vxn,uyn,vyn,uzn,vzn;	// Vecteur du photon après reflexion
		float cTh2 = RAND;
		cTh = sqrtf( cTh2 );
		sTh = sqrtf( 1.0F - cTh2 );

		phi = RAND*DEUXPI;	//angle azimutal
		cPhi = __cosf(phi);
		sPhi = __sinf(phi);
		
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
		thetab = acosf( fmin( fmax(-1,temp),1 ) );
		if( thetab==0){
			photon->loc=SPACE;
			printf("theta nul\n");
			return;
		}
		
		// (Produit scalaire V.Unew)/sin(theta)
		temp = __fdividef( photon->vx*uxn + photon->vy*uyn + photon->vz*uzn, __sinf(thetab) );
		psi = acosf( fmin( fmax(-1,temp),1 ) );	// angle entre le plan (u,v)old et (u,v)new
		
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
		
		if( SIMd == -1) // Dioptre seul
			photon->loc=SPACE;
		else
			photon->loc=ATMOS;
		
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
__device__ void exit(Photon* photon, Variables* var, Tableaux tab, unsigned long long* nbPhotonsThr
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
	int loc = photon->loc;
	// Remise à zéro de la localisation du photon
	photon->loc = NONE;
	
	#ifdef SORTIEINT
	/* On compte le nombre de boucles qu'a effectué un photon pour vérifier qu'il n'y a pas de bug de
	stationnarité.*/
	if( (iloop-photon->numBoucle)<NBLOOPd ){
	atomicAdd(tab.nbBoucle + (iloop-photon->numBoucle),1);
	}
	else{
		printf("Problème dans le comptage des boucles, débordement tableau\n");
	}
	#endif
	

	// Si le photon est absorbé on ne fait rien d'autre
	if(loc == ABSORBED)
	{
		// nbPhotonsThr est le nombre de photons traités par le thread, on l'incrémente
		(*nbPhotonsThr)++;
		return;
	}

// si son poids est anormalement élevé on le compte comme une erreur. Test effectué uniquement en présence de dioptre
	if( (photon->weight > WEIGHTMAX) && (SIMd!=-2)){
		// printf("Erreur poids trop eleve\n");
		atomicAdd(&(var->erreurpoids), 1);
		return;
	}
	// Sinon on traite le photon et on l'ajoute dans le tableau tabPhotons de ce thread
// 	else
// 	{
		#ifdef SORTIEINT
		//Sauvegarde du poids pour debug
		tab.poids[*nbPhotonsThr] = photon->weight;
		#endif

	// Incrémentation du nombre de photons traités par le thread
	(*nbPhotonsThr)++;
		// Création d'un float theta qui sert à modifier les nombres de Stokes
		float theta = acosf(fmin(1.F, fmax(-1.F, STHSd * photon->vx + CTHSd * photon->vz)) );
		// Si theta = 0 on l'ignore (cas où le photon repart dans la direction solaire)
		if(theta == 0.F)
		{
// 			printf("Erreur theta nul\n");
			atomicAdd(&(var->erreurtheta), 1);
			// Incrémentation du nombre de photons traités par le thread
			return;
		}

// 		else{
			// Création d'un angle psi qui sert à modifier les nombres de Stokes
			float psi;
			int ith=0, iphi=0;
			// Initialisation de psi
			calculPsi(photon, &psi, theta);
			// Modification des nombres de Stokes
			float cPsi = __cosf(psi);
			float sPsi = __sinf(psi);
			
			modifStokes(photon, psi, cPsi, sPsi, 0);
			
			// On modifie ensuite le poids du photon
			photon->weight = __fdividef(photon->weight, photon->stokes1 + photon->stokes2);

				// Calcul de la case dans laquelle le photon sort
				calculCase(&ith, &iphi, photon, var);
				// Rangement du photon dans sa case, et incrémentation de variables
				if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)))
				{
					// Rangement dans le tableau des paramètres pondérés du photon

					atomicAdd(tab.tabPhotons+(0 * NBTHETAd * NBPHId + ith * NBPHId + iphi), photon->weight * photon->stokes1);

					atomicAdd(tab.tabPhotons+(1 * NBTHETAd * NBPHId + ith * NBPHId + iphi), photon->weight * photon->stokes2);

					atomicAdd(tab.tabPhotons+(2 * NBTHETAd * NBPHId + ith * NBPHId + iphi), photon->weight * photon->stokes3);
							
					atomicAdd(tab.tabPhotons+(3 * NBTHETAd * NBPHId + ith * NBPHId + iphi), photon->weight * photon->stokes4);
		
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
// 			}
// 	}
	
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
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
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
		float psi2 = 2.F * psi;
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
	if(vxy < VALMIN)
	{
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurvxy), 1);
		#endif
// 		if(photon->vy < 0.F) *iphi = NBPHId - 1;
		if(photon->vy < 0.F) *iphi = __float2int_rd( __fdividef(PI*NBPHId, DEUXPI));
		else *iphi = 0;
	}

	else	// Sinon on calcule iphi
	{
		// On place d'abord le photon dans un demi-cercle
		float cPhiP = __fdividef(photon->vx, vxy); //cosPhiPhoton
		// Cas limite où phi est très proche de 0, la formule générale ne marche pas
		if(cPhiP >= 1.F) *iphi = 0;
		// Cas limite où phi est très proche de PI, la formule générale ne marche pas
		else if(cPhiP <= -1.F) *iphi = (NBPHId / 2) - 1;
		// Cas général
		else *iphi = __float2int_rd(__fdividef(acosf(cPhiP) * NBPHId, DEUXPI));
		// Puis on place le photon dans l'autre demi-cercle selon vy
		if(photon->vy < 0.F) *iphi = NBPHId - 1 - *iphi;
		#ifdef PROGRESSION
		// Lorsque vy=0 on décide par défaut que le photon reste du côté vy>0
		if(photon->vy == 0.F) atomicAdd(&(var->erreurvy), 1);
		#endif
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

	float TAUMAX = TAUATM / CTHSbis; //tau initial du photon
	cudaMemcpyToSymbol(TAUMAXd, &TAUMAX, sizeof(float));	

}



// Détermination de la nouvelle direction du photon lorsqu'il est diffusé par aérosol
__device__ void scatterAer(Photon* photon, Tableaux tab
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

	float zang=0;
	float theta=0;
	float psi=0;
	float cPsi, sPsi, cTh, sTh;
	int iang;

	/** Initialisation aléatoire des paramètres **/
	zang = RAND*(NFAERd-1);
	iang= __float2int_rd(zang);
	
// 	if (iang==NFAERd){
// // 		printf("Correction de iang\n");
// 		theta = tab.faer[(NFAERd-1)*5+4];
// 	}
// 	printf("zang=%f, iang=%d, zang2=%f\n",zang,iang, zang2);
	
// 	if( iang==NFAERd-1){
// 		theta = tab.faer[(NFAERd-1)*5+4];	// L'accès à faer[x][y] se fait par
// 																						// faer[y*5+x]
// 	}
// 	else{
		zang = zang - iang;
	theta = tab.faer[iang*5+4]+ zang*( tab.faer[(iang+1)*5+4]-tab.faer[iang*5+4] );	// L'accès à faer[x][y] se fait par
//faer[y*5+x]
// 	}

	psi = RAND*DEUXPI;

	/** Rotation des paramètres de Stokes **/
	cPsi = __cosf(psi);
	sPsi = __sinf(psi);
	cTh = __cosf(theta);
	sTh = __sinf(theta);

	modifStokes(photon, psi, cPsi, sPsi, 1);

	/** Création de 2 vecteurs provisoires w et v **/
	float wx, wy, wz, vx, vy, vz;
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
	
	/** Changement du poids et des nombres de stokes du photon **/
	float stokes1 = photon->stokes1;
	float stokes2 = photon->stokes2;
	float faer1 = tab.faer[iang*5+0];
	float faer2 = tab.faer[iang*5+1];

	// Calcul du poids après diffusion
	photon->weight *= __fdividef( 2.0F*(stokes1*faer1+stokes2*faer2) , stokes1+stokes2);

	// Calcul des parametres de Stokes du photon apres diffusion
	photon->stokes1 = 2.0F*faer1*stokes1;
	photon->stokes2 = 2.0F*faer2*stokes2;
	photon->stokes3 *= tab.faer[iang*5+2];
	photon->stokes4 = 0.F;

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

__device__ void scatterTot(Photon* photon, Tableaux tab
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
// 	float propMol = 0;	// Proportion de molécules par rapport aux aérosols dans l'atmosphère
	// Recherche de la couche où se situe le photon
	float tauBis = TAUATMd-photon->tau;
	int icouche=1;

	while( (tab.tauCouche[icouche] < (tauBis)) && icouche<NATM )
		icouche++;

// 	propMol = tab.pMol[icouche];

	if( RAND <= tab.pMol[icouche]){
		// Le photon rencontre une molécule, et donc est diffusé par celle-ci
		scatterMol(photon, etatThr
		#if defined(RANDMWC) || defined(RANDMT)
		, configThr
		#endif
		#ifdef TRAJET
		, idx, evnt
		#endif
				);
	}

	else{	// Le photon rencontre un aérosol
		scatterAer(photon, tab, etatThr
					#if defined(RANDMWC) || defined(RANDMT)
					, configThr
					#endif
					#ifdef TRAJET
					, idx, evnt
					#endif
							);
		photon->weight *= W0AERd;
	}
}

__device__ void calculCommunScatter( Photon* photon, float* cTh, Tableaux tab, int icouche
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

	float zang=0, theta=0;
	int iang;
	float stokes1, stokes2;
	float cTh2;
	
	stokes1 = photon->stokes1;
	stokes2 = photon->stokes2;
	
	if( RAND<=tab.pMol[icouche] ){	// Theta calculé pour la diffusion moléculaire
		*cTh =  2.F * RAND - 1.F; //cosThetaPhoton
		cTh2 = (*cTh)*(*cTh);
		// Calcul du poids après diffusion
		photon->weight *= __fdividef(1.5F * ((1.F+GAMAd)*stokes2+((1.F-GAMAd)*cTh2+2.F*GAMAd)*stokes1), (1.F+2.F*GAMAd) *
		(stokes1+stokes2));
		// Calcul des parametres de Stokes du photon apres diffusion
		photon->stokes2 += GAMAd * stokes1;
		photon->stokes1 = ( (1.F - GAMAd) * cTh2 + GAMAd) * stokes1 + GAMAd * photon->stokes2;
// 		photon->stokes1 = cTh2*stokes1 + GAMAd*( (1-cTh2)*stokes1+stokes2 );
		photon->stokes3 *= (1.F - GAMAd) * (*cTh);
		photon->stokes4 = 0.F /*(1.F - 3.F * GAMAd) * (*cTh) * photon->stokes4*/;
	}
	
	else{	// Aérosols
		zang = RAND*(NFAERd-1);
		iang= __float2int_rd(zang);
		
		zang = zang - iang;
		theta = tab.faer[iang*5+4]+ zang*( tab.faer[(iang+1)*5+4]-tab.faer[iang*5+4] );	// L'accès à faer[x][y] se fait par

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
		photon->stokes4 = 0.F;
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

	float tauBis = TAUATMd-photon->tau;
	int icouche=1;
	float cTh=0, sTh, psi, cPsi, sPsi;
	float wx, wy, wz, vx, vy, vz;
	
	while( (tab.tauCouche[icouche] < (tauBis)) && icouche<NATM )
		icouche++;
	
	psi = RAND * DEUXPI; //psiPhoton
	cPsi = __cosf(psi); //cosPsiPhoton
	sPsi = __sinf(psi); //sinPsiPhoton
	
	// Modification des nombres de Stokes
	modifStokes(photon, psi, cPsi, sPsi, 1);
	
	calculCommunScatter( photon, &cTh, tab, icouche
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

