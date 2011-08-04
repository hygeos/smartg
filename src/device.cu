
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
		, unsigned long long* x, unsigned int* a, float* tableau1
		, curandState_t *globalRand, float* tableau2
		, EtatMT* etat, ConfigMT* config, float* tableau3
		#endif
		#ifdef TRAJET
		, Evnt* evnt //récupération d'informations
		#endif
			       )
{
	// idx est l'indice du thread considéré
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
	
	#ifdef TABRAND
	// DEBUG Recuperation des nombres aleatoires generes par les differentes fonctions random
	//RandomMWC
	if(idx < 5)
		if(tableau1[50] == 0.f)
		{
			int k = 0;
			if(tableau1[0] != 0.f) k = 50;
			for(int j = 0; j < 10; j++) tableau1[k+idx*10+j] = randomMWCfloat_co(x+idx,a+idx);
		}
	//RandomCuda
	if(idx < 5)
		if(tableau2[50] == 0.f)
		{
			int k = 0;
			if(tableau2[0] != 0.f) k = 50;
			for(int j = 0; j < 10; j++) tableau2[k+idx*10+j] = curand_uniform(globalRand+idx);
		}
	//RandomMT
	if(idx < 5)
		if(tableau3[50] == 0.f)
		{
			int k = 0;
			if(tableau3[0] != 0.f) k = 50;
			for(int j = 0; j < 10; j++) tableau3[k+idx*10+j] = randomMTfloat(etat+idx, config+idx);
		}
	#endif
	
	// Création de variable propres à chaque thread
	unsigned int iloop;
	unsigned int nbPhotonsThr = 0; // nombre de photons taités par le thread
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; // nombre de photons taités par le thread et resortis dans l'espace
	#endif
	
	Photon photon; // On associe une structure de photon au thread
	photon.loc = NONE; // Initialement le photon n'est nulle part, il doit être initialisé
	
	// Dans cette boucle on simule le parcours du photon, puis on le réinitialise, ... Le thread lance plusieurs photons
	for(iloop = 0; iloop < NBLOOPd; iloop++)
	{
		// Si le photon est à NONE on l'initialise et on le met à ATMOS
		if(photon.loc == NONE) init(&photon
				#ifdef TRAJET
				, idx, evnt
				#endif
					   );
		
		// Si le photon est à ATMOS on le avancer jusqu'à SURFACE, ou SPACE, ou ATMOS s'il subit une diffusion
		if(photon.loc == ATMOS) move(&photon, tab
				, idx
				#ifdef TRAJET
				, evnt
				#endif
					    );
		
		// Si le photon est encore à ATMOS il subit une diffusion et reste dans ATMOS
		if(photon.loc == ATMOS) scatter(&photon
				, tab
				, idx
				#ifdef TRAJET
				, evnt
				#endif
					       );
		
		// Si le photon est à SURFACE on le met à ABSORBED
		if(photon.loc == SURFACE) surfac(&photon
				, var, tab
				, idx
				#ifdef TRAJET
				, evnt
				#endif
						);
		
		// Si le photon est dans SPACE ou ABSORBED on récupère ses infos et on le remet à NONE
		if(photon.loc == ABSORBED || photon.loc == SPACE) exit(&photon
				, var, tab
				, &nbPhotonsThr
				#ifdef PROGRESSION
				, &nbPhotonsSorThr
				#endif
				#ifdef TRAJET
				, idx, evnt
				#endif
					       );
	}
	
	// Après la boucle on rassemble les nombres de photons traités par chaque thread
	atomicAddULL(&var->nbPhotons, nbPhotonsThr);
	
	#ifdef PROGRESSION
	// On rassemble les nombres de photons traités et sortis de chaque thread
	atomicAddULL(&var->nbPhotonsSor, nbPhotonsSorThr);
	// On incrémente avncement qui compte le nombre d'appels du Kernel
	atomicAddULL(&var->nbThreads, 1);
	#endif
}

// Fonction qui initialise les generateurs du random cuda
__global__ void initRandCUDA(curandState_t* globalRand, unsigned long long seed)
{
	// Pour chaque thread on initialise son generateur avec le meme seed mais un idx different
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
	curand_init(seed, idx, 0, globalRand+idx);
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
	photon->tau = TAURAYd;
	photon->weight = WEIGHTINIT;
	// Initialisation des paramètres de stokes du photon
	photon->stokes1 = 0.5F;
	photon->stokes2 = 0.5F;
	photon->stokes3 = 0.F;
	photon->stokes4 = 0.F;
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<20) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		if(i <20 )
		{
			// "1"représente l'événement "initialisation" du photon
			evnt[i].action = 1;
			// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
		}
	}
	#endif
}

// Fonction device qui traite les photons dans l'atmosphère en les faisant avancer
__device__ void move(Photon* photon, Tableaux tab
		, int idx
		#ifdef TRAJET
		, Evnt* evnt
		#endif
		    )
{
	// TO DO
	// tirage épaisseur optique avec diffusion forcée
	// p->tau += -LOG(1.F-MersenneTwisterGenerateFloat(pmtState, initialConfig)*(1.F-EXP(-tau_max))) * p->v.z;
	
	// Tirage de la nouvelle épaisseur optique du photon sans diffusion forcée
	photon->tau += -__logf(1.F - RAND) * photon->vz;
	
	// Si tau<0 le photon atteint la surface
	if(photon->tau < 0.F) photon->loc = SURFACE;
	// Si tau>TAURAY le photon atteint l'espace
	else if(photon->tau > TAURAYd) photon->loc = SPACE;
	// Sinon il a rencontré une molécule, on ne fait rien car il reste dans l'atmosphère, et va être traité par scatter
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<20) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		if(i <20 )
		{
			// "2"représente l'événement "move" du photon
			evnt[i].action = 2;
			// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
		}
	}
	#endif
}

// Fonction device qui traite les photons qui sont encore dans l'atmosphère après "move" : rencontre avec une molécule
__device__ void scatter(Photon* photon,
		Tableaux tab
		, int idx
		#ifdef TRAJET
		, Evnt* evnt
		#endif
		       )
{
	// On détermine la nouvelle direction du photon
	float cThP, cThP2, sThP, psiP, cPsiP, sPsiP;
	// On prend un cosinus aléatoirement entre -1 et 1, il s'agit du cosinus de l'angle que fait la nouvelle direction avec l'ancienne, on prend le cos aléatoirement pour l'équiprobabilité dans toutes les directons
	cThP = 2.F * RAND - 1.F; //cosThetaPhoton
	cThP2 = cThP * cThP; //cosThetaPhotonCarré
	sThP = sqrtf(1.F - cThP2);//sinThetaPhoton
	// On prend un angle phi aléatoirement entre 0 et 2.PI, il s'agit de l'angle entre le nouveau vecteur projeté dans le plan orthogonal à v et le vecteur u
	psiP = RAND * DEUXPI; //psiPhoton
	cPsiP = __cosf(psiP); //cosPsiPhoton
	sPsiP = __sinf(psiP); //sinPsiPhoton
	
	// Dans certains cas on change les nombres de stokes du photon
	if ((photon->stokes1 != photon->stokes2) || (photon->stokes3 != 0.F))
	{
		float cPsiP2 = cPsiP * cPsiP; //cosPsiPhotonCarré
		float sPsiP2 = sPsiP * sPsiP; //sinPsiPhotonCarré
		float psiP2 = 2.F * psiP; //2*PsiPhoton
		float s2PsiP   = __sinf(psiP2); //sin(2*PsiPhoton)
		
		float stokes1, stokes2, stokes3;
		stokes1 = photon->stokes1;
		stokes2 = photon->stokes2;
		stokes3 = photon->stokes3;
		float a = __fdividef(s2PsiP * stokes3, 2.F);
		
		photon->stokes1 = cPsiP2 * stokes1 + sPsiP2 * stokes2 + a;
		photon->stokes2 = sPsiP2 * stokes1 + cPsiP2 * stokes2 - a;
		photon->stokes3 = s2PsiP * (stokes2 - stokes1) + __cosf(psiP2) * stokes3;
	}

	// Création de 2 vecteurs provisoires w et v
	float wx, wy, wz, vx, vy, vz;
	// w est le rotationnel entre l'ancien vecteur u et l'ancien vecteur v du photon
	wx = photon->uy * photon->vz - photon->uz * photon->vy;
	wy = photon->uz * photon->vx - photon->ux * photon->vz;
	wz = photon->ux * photon->vy - photon->uy * photon->vx;
	// v est le nouveau vecteur v du photon
	vx = cThP * photon->vx + sThP * ( cPsiP * photon->ux + sPsiP * wx );
	vy = cThP * photon->vy + sThP * ( cPsiP * photon->uy + sPsiP * wy );
	vz = cThP * photon->vz + sThP * ( cPsiP * photon->uz + sPsiP * wz );
	// Changement du vecteur u (orthogonal au vecteur vitesse du photon)
	photon->ux = __fdividef(cThP * vx - photon->vx, sThP);
	photon->uy = __fdividef(cThP * vy - photon->vy, sThP);
	photon->uz = __fdividef(cThP * vz - photon->vz, sThP);
	// Changement du vecteur v (vitesse du photon)
	photon->vx = vx;
	photon->vy = vy;
	photon->vz = vz;
	
	// Changement du poids et des nombres de stokes du photon
	float gam = __fdividef(DEPO, 2.F-DEPO);
	float stokes1 = photon->stokes1;
	float stokes2 = photon->stokes2;
	// Calcul du poids après diffusion
	photon->weight *= __fdividef(1.5F * ((1.F+gam)*stokes2+((1.F-gam)*cThP2+2.F*gam)*stokes1), (1.F+2.F*gam) * (stokes1+stokes2));
	// Calcul des parametres de Stokes du photon apres diffusion
	photon->stokes2 = stokes2 + gam * stokes1;
	photon->stokes1 = ( (1.F - gam) * cThP2 + gam) * stokes1 + gam * photon->stokes2;
	photon->stokes3 = (1.F - gam) * cThP * photon->stokes3;
	photon->stokes4 = (1.F - 3.F * gam) * cThP * photon->stokes4;
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
		// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<20) i++;
		// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		if(i <20 )
		{
			// "3"représente l'événement "scatter" du photon
			evnt[i].action = 3;
			// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
		}
	}
	#endif
}

// Fonction device qui traite les photons atteignant le sol
__device__ void surfac(Photon* photon,
		Variables* var, Tableaux tab
		, int idx
		#ifdef TRAJET
		, Evnt* evnt //récupération d'informations
		#endif
		      )
{
	// Pour l'instant le sol absorbe tous les photons
	photon->loc = ABSORBED;
	
	// TO DO
	/*
	if(rand_MWC_co(&(tab.x),&(tab.a))<0.5F) photon->loc = ABSORBED;
	else
	{
		photon->loc = ATMOS;
		photon->tau = 0.F;
		photon->vz = -photon->vz;
	}
	*/
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
			// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<20) i++;
			// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		if(i <20 )
		{
				// "4"représente l'événement "surface" du photon
			evnt[i].action = 4;
				// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
		}
	}
	#endif
}

// Fonction device qui traite les photons absorbés ou atteignant l'espace
__device__ void exit(Photon* photon,
		Variables* var, Tableaux tab
		, unsigned int* nbPhotonsThr
		#ifdef PROGRESSION
		, unsigned int* nbPhotonsSorThr
		#endif
		#ifdef TRAJET
		, int idx, Evnt* evnt
		#endif
		    )
{
	int loc = photon->loc;
	// Remise à zéro de la localisation du photon
	photon->loc = NONE;
	// Si le photon est absorbé on ne fait rien d'autre
	if(loc == ABSORBED)
	{
		// nbPhotonsThr est le nombre de photons traités par le thread, on l'incrémente
		(*nbPhotonsThr)++;
	}
	// Le photon est sorti dans l'espace, si son poids est anormalement élevé on le compte comme une erreur
	else if(photon->weight > WEIGHTMAX)
	{
		atomicAdd(&var->erreurpoids, 1);
	}
	// Sinon on traite le photon et on l'ajoute dans le tableau tabPhotons de ce thread
	else
	{
		// Création d'un float theta qui sert à modifier les nombres de Stokes
		float theta = acosf(fmin(1.F, fmax(-1.F, STHSd * photon->vx + CTHSd * photon->vz)));
		// Si theta = 0 on l'ignore (cas où le photon repart dans la direction solaire)
		if(theta == 0.F)
		{
			atomicAdd(&var->erreurtheta, 1);
		}
		else
		{
			// Traitement du photon, poids et nombres de Stokes
			calculsPhoton(theta, photon);
			// Calcul de la case dans laquelle le photon sort
			int ith, iphi;
			calculCase(&ith, &iphi, photon, var);
			// Rangement du photon dans sa case, et incrémentation de variables
			if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)))
			{
				// Rangement dans le tableau du poids des photons
				atomicAddULL(tab.tabPhotons+(0 * NBTHETAd * NBPHId + ith * NBPHId + iphi), __float2uint_rn(photon->weight * photon->stokes1 * SCALEFACTOR + 0.5F));
				atomicAddULL(tab.tabPhotons+(1 * NBTHETAd * NBPHId + ith * NBPHId + iphi), __float2uint_rn(photon->weight * photon->stokes2 * SCALEFACTOR + 0.5F));

				// Incrémentation du nombre de photons traités par le thread
				(*nbPhotonsThr)++;
				#ifdef PROGRESSION
				// Incrémentation du nombre de photons sortis dans l'espace pour ce thread
				(*nbPhotonsSorThr)++;
				#endif
		
			}
			else
			{
				#ifdef PROGRESSION
				atomicAdd(&var->erreurcase, 1);
				#endif
			}
		}
	}
	
	#ifdef TRAJET
	// Récupération d'informations sur le premier photon traité
	if(idx == 0)
	{
		int i = 0;
			// On cherche la première action vide du tableau
		while(evnt[i].action != 0 && i<20) i++;
			// Et on remplit la première case vide des tableaux (tableaux de 20 cases)
		if(i <20 )
		{
				// "5"représente l'événement "exit" du photon
			evnt[i].action = 5;
				// On récupère le tau et le poids du photon
			evnt[i].tau = photon->tau;
			evnt[i].poids = photon->weight;
		}
	}
	#endif
}

// Fonction qui traite les photons sortants dans l'espace: changement de poids et de Stokes
__device__ void calculsPhoton(float theta, Photon* photon)
{
	// Création d'un angle psi qui sert à modifier les nombres de Stokes
	float psi;
	if (theta < 0.0025F)
	{
		psi = acosf(fmin(1.F - VALMIN, fmax(-(1.F - VALMIN), - CTHSd * photon->ux + STHSd * photon->uz)));
	}
	else
	{
		psi = acosf(fmin(1.F, fmax(-1.F, __fdividef(STHSd * photon->ux + CTHSd * photon->uz, __sinf(theta)))));
	}
	float sign = STHSd * (photon->uy * photon->vz - photon->uz * photon->vy) + CTHSd * (photon->ux * photon->vy - photon->uy * photon->vx);
	if (sign < 0.F) psi = -psi;
	// On modifie les nombres de Stokes grâce à psi
	if((photon->stokes1 != photon->stokes2) || (photon->stokes3 != 0.F))
	{
		float cp = __cosf(psi);
		float sp = __sinf(psi);
		float cp2 = cp * cp;
		float sp2 = sp * sp;
		float psi2 = 2.F * psi;
		float stokes1, stokes2, stokes3;
		stokes1 = photon->stokes1;
		stokes2 = photon->stokes2;
		stokes3 = photon->stokes3;
		float s2p = __sinf(psi2);
		float s2ps2u = __fdividef(s2p * stokes3, 2.F);
		photon->stokes1 = cp2 * stokes1 + sp2 * stokes2 + s2ps2u;
		photon->stokes2 = sp2 * stokes1 + cp2 * stokes2 - s2ps2u;
		photon->stokes3 = s2p * (stokes2 - stokes1) + __cosf(psi2) * stokes3;
	}
	// On modifie ensuite le poids du photon
	photon->weight = __fdividef(photon->weight, photon->stokes1 + photon->stokes2);
}

// Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
__device__ void calculCase(int* ith, int* iphi, Photon* photon, Variables* var)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->vx * photon->vx + photon->vy * photon->vy);
	// Calcul de la valeur de ithv
	*ith = __float2int_rd(__fdividef(acosf(photon->vz) * NBTHETAd, DEMIPI));
	// Si le photon ressort très près du zénith on ne peut plus calculer iphi,
	// on est à l'intersection de toutes les cases du haut
	if(vxy < VALMIN)
	{
		#ifdef PROGRESSION
		atomicAdd(&var->erreurvxy, 1);
		#endif
		if(photon->vy < 0.F) *iphi = NBPHId - 1;
		else *iphi = 0;
	}
	// Sinon on calcule iphi
	else
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
		if(photon->vy == 0.F) atomicAdd(&var->erreurvy, 1);
		#endif
	}
}

// Fonction random MWC qui renvoit un float de [0.1[ à partir d'un generateur (x+a)
__device__ float randomMWCfloat_co(unsigned long long* x,unsigned int* a)
{
	//Generate a random number [0,1)
	*x=(*x&0xffffffffull)*(*a)+(*x>>32);
	return __fdividef(__uint2float_rz((unsigned int)(*x)),(float)0x100000000);// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)
}//end __device__ rand_MWC_co

// Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
__device__ float randomMWCfloat_oc(unsigned long long* x,unsigned int* a)
{
	//Generate a random number (0,1]
	return 1.0f-randomMWCfloat_co(x,a);
}//end __device__ rand_MWC_oc

// Fonction random Mersenne Twister qui renvoit un float de ]0.1] à partir d'un generateur (etat+config)
__device__ float randomMTfloat(EtatMT* etat, ConfigMT* config)
{
	//Convert to (0, 1] float
	return ((float)randomMTuint(etat, config) + 1.0f) / 4294967296.0f;
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
	cudaMemcpyToSymbol(THETASOLd, &THETASOL, sizeof(float));
	cudaMemcpyToSymbol(LAMBDAd, &LAMBDA, sizeof(float));
	cudaMemcpyToSymbol(TAURAYd, &TAURAY, sizeof(float));
	cudaMemcpyToSymbol(TAUAERd, &TAUAER, sizeof(float));
	cudaMemcpyToSymbol(W0AERd, &W0AER, sizeof(float));
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
	cudaMemcpyToSymbol(NBSTOKESd, &NBSTOKES, sizeof(int));
	cudaMemcpyToSymbol(PROFILd, &PROFIL, sizeof(int));
	cudaMemcpyToSymbol(SIMd, &SIM, sizeof(int));
	cudaMemcpyToSymbol(SURd, &SUR, sizeof(int));
	cudaMemcpyToSymbol(DIOPTREd, &DIOPTRE, sizeof(int));
	cudaMemcpyToSymbol(DIFFFd, &DIFFF, sizeof(int));
	
	float THSbis = THETASOL/180*PI; //thetaSolaire
	cudaMemcpyToSymbol(THSd, &THSbis, sizeof(float));

	float CTHSbis = cosf(THSbis); //cosThetaSolaire
	cudaMemcpyToSymbol(CTHSd, &CTHSbis, sizeof(float));

	float STHSbis = sinf(THSbis); //sinThetaSolaire
	cudaMemcpyToSymbol(STHSd, &STHSbis, sizeof(float));

	float TAUMAXbis = TAURAY / CTHSbis; //tau initial du photon
	cudaMemcpyToSymbol(TAUMAXd, &TAUMAXbis, sizeof(float));
}
