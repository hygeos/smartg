
	  //////////////
	 // INCLUDES //
	//////////////

#include "communs.h"
#include "device.h"

	  //////////////////////
	 // FONCTIONS DEVICE //
	//////////////////////

// Fonction device qui lance tous les threads, leur associe des photons, et les fait tourner en boucle
__global__ void lancementKernel(Random* random //variables fonction aléatoire
		, Constantes* constantes //constantes
		, Progress* progress
		, unsigned long long* tabPhotons //récupération d'informations

		#ifdef TABNBPHOTONS
		, unsigned long long* tabNbPhotons //récupération d'informations
		#endif

		#ifdef TRAJET
		, Evnt* evnt //récupération d'informations
		#endif
			       )
{
	// idx est l'indice du thread considéré
	int idx = (blockIdx.x * YGRID + blockIdx.y) * XBLOCK * YBLOCK + (threadIdx.x * YBLOCK + threadIdx.y);
	
	// Création de variable propres à chaque thread
	unsigned int iloop;
	unsigned int nbPhotonsThr = 0; // nombre de photons taités par le thread
	
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; // nombre de photons taités par le thread et resortis dans l'espace
	#endif
	
	Photon photon; // On associe une structure de photon au thread
	photon.loc = NONE; // Initialement le photon n'est nulle part, il doit être initialisé
	
	// Dans cette boucle on simule le parcours du photon, puis on le réinitialise, ... Le thread lance plusieurs photons
	for(iloop = 0; iloop<NBLOOP; iloop++)
	{
		// Si le photon est à NONE on l'initialise et on le met à ATMOS
		if(photon.loc == NONE) init(&photon //photon
				, constantes //&(constantes.cThS), &(constantes.sThS) //constantes
				#ifdef TRAJET
				, idx, evnt//, tauevnt, poidsevnt //récupération d'informations
				#endif
					   );

		// Si le photon est à SPACE on récupère ses infos et on le remet à NONE
		if(photon.loc == SPACE) exit(&photon //photon
				, constantes //constantes
				, &nbPhotonsThr //récupération d'informations
						
				#ifdef PROGRESSION
				, &nbPhotonsSorThr //récupération d'informations
				#endif
				
				, progress //récupération d'informations
				, tabPhotons //récupération d'informations

				#ifdef TABNBPHOTONS
				, tabNbPhotons //récupération d'informations
				#endif

				#ifdef TRAJET
				, idx, evnt //récupération d'informations
				#endif
					    );
		
		// Si le photon est à ATMOS on le avancer jusqu'à SURFACE, ou SPACE, ou ATMOS s'il subit une diffusion
		if(photon.loc == ATMOS) move(&photon //photon
				, constantes //constantes
				, random+idx //variables fonction aléatoire
				#ifdef TRAJET
				, idx, evnt //récupération d'informations
				#endif
					    );
		
		// Si le photon est encore à ATMOS il subit une diffusion et reste dans ATMOS
		if(photon.loc == ATMOS) scatter(&photon //photon
				, random+idx //variables fonction aléatoire
				#ifdef TRAJET
				, idx, evnt //récupération d'informations
				#endif
					       );
		
		// Si le photon est à SURFACE on le met à ABSORBED
		if(photon.loc == SURFACE) surfac(&photon //photon
				, random+idx //variables fonction aléatoire
				#ifdef TRAJET
				, idx, evnt //récupération d'informations
				#endif
						);
		
		// Si le photon est à ABSORBED on récupère ses infos et on le remet à NONE
		if(photon.loc == ABSORBED) exit(&photon //photon
				, constantes//constantes
				, &nbPhotonsThr //récupération d'informations
						
				#ifdef PROGRESSION
				, &nbPhotonsSorThr //récupération d'informations
				#endif
				
				, progress//erreur //récupération d'informations
				, tabPhotons //récupération d'informations
						
				#ifdef TABNBPHOTONS
				, tabNbPhotons //récupération d'informations
				#endif

				#ifdef TRAJET
				, idx, evnt //récupération d'informations
				#endif
					       );
	}
	
	// Après la boucle on rassemble les nombres de photons traités par chaque thread
	atomicAddULL(&(progress->nbPhotons), nbPhotonsThr);
	
	#ifdef PROGRESSION
	// On rassemble les nombres de photons traités et sortis de chaque thread
	atomicAddULL(&(progress->nbPhotonsSor), nbPhotonsSorThr);
	// On incrémente avncement qui compte le nombre d'appels du Kernel
	atomicAddULL(&(progress->nbThreads), 1);
	#endif
}

// Fonction device qui initialise le photon associé à un thread
__device__ void init(Photon* photon, //photon
		Constantes* constantes //constantes	
		#ifdef TRAJET
		, int idx, Evnt* evnt //récupération d'informations
		#endif
		    )
{
	// Initialisation du vecteur vitesse
	photon->vx = -constantes->sThS;
	photon->vy = 0.F;
	photon->vz = -constantes->cThS;
	// Initialisation du vecteur orthogonal au vecteur vitesse
	photon->ux = -photon->vz;
	photon->uy = 0.F;
	photon->uz = photon->vx;
	// Le photon est initialement dans l'atmosphère, et tau peut être vu comme sa hauteur par rapport au sol
	photon->loc = ATMOS;
	photon->tau = TAU;
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

// Fonction device qui traite les photons atteignant le sol
__device__ void surfac(Photon* photon, //photon
		Random* random //variables fonction aléatoire
		#ifdef TRAJET
		, int idx, Evnt* evnt //récupération d'informations
		#endif
		      )
{
	// Pour l'instant le sol absorbe tous les photons
	photon->loc = ABSORBED;
	
	// TO DO
	/*
	if(rand_MWC_co(&(random->x),&(random->a))<0.5F)
	{
		photon->loc = ABSORBED;
	}
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
__device__ void exit(Photon* photon, //photon
		Constantes* constantes //constantes
		, unsigned int* nbPhotonsThr //récupération d'informations
				
		#ifdef PROGRESSION
		, unsigned int* nbPhotonsSorThr //récupération d'informations
		#endif
		, Progress* progress //récupération d'informations
		, unsigned long long* tabPhotons //récupération d'informations
		#ifdef TABNBPHOTONS
		, unsigned long long* tabNbPhotons //récupération d'informations
		#endif
		
		#ifdef TRAJET
		, int idx, Evnt* evnt //récupération d'informations
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
		atomicAdd(&(progress->erreurpoids), 1);
	// Sinon on traite le photon et on l'ajoute dans le tableau tabPhotons de ce thread
	else
	{
		// Création d'un float theta qui sert à modifier les nombres de Stokes
		float theta = acosf(fmin(1.F, fmax(-1.F, (constantes->sThS) * photon->vx + (constantes->cThS) * photon->vz)));
		// Si theta = 0 on l'ignore (cas où le photon repart dans la direction solaire)
		if(theta == 0.F)
			atomicAdd(&(progress->erreurtheta), 1);
		else
		{
			// Traitement du photon, poids et nombres de Stokes
			calculsPhoton(theta, constantes, photon);
			// Calcul de la case dans laquelle le photon sort
			int ith, iphi;
			calculCase(&ith, &iphi, photon, progress);
			// Rangement du photon dans sa case, et incrémentation de variables
			if(((ith >= 0) && (ith < NBTHETA)) && ((iphi >= 0) && (iphi < NBPHI)))
			{
				// Rangement dans le tableau du poids des photons
				atomicAddULL(&tabPhotons[0 * NBTHETA * NBPHI + ith * NBPHI + iphi], (unsigned int)(photon->weight * photon->stokes1 * SCALEFACTOR + 0.5F));
				atomicAddULL(&tabPhotons[1 * NBTHETA * NBPHI + ith * NBPHI + iphi], (unsigned int)(photon->weight * photon->stokes2 * SCALEFACTOR + 0.5F));
				
				#ifdef TABNBPHOTONS
				// Rangement dans le tableau du nombre de photons
				atomicAddULL(&tabNbPhotons[ith*NBPHI + iphi], 1);
				#endif
				// Incrémentation du nombre de photons traités par le thread
				(*nbPhotonsThr)++;
				#ifdef PROGRESSION
				// Incrémentation du nombre de photons sortis dans l'espace pour ce thread
				(*nbPhotonsSorThr)++;
				#endif
		
			}
			else atomicAdd(&(progress->erreurcase), 1);
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

// Fonction device qui traite les photons dans l'atmosphère en les faisant avancer
__device__ void move(Photon* photon, //photon
		Constantes* constantes //constantes
		, Random* random //variables fonction aléatoire
		#ifdef TRAJET
		, int idx, Evnt* evnt //récupération d'informations
		#endif
		    )
{
	// TO DO
	// tirage épaisseur optique avec diffusion forcée
	// p->tau += -LOG(1.F-MersenneTwisterGenerateFloat(pmtState, initialConfig)*(1.F-EXP(-tau_max))) * p->v.z;
	
	// Tirage de la nouvelle épaisseur optique du photon sans diffusion forcée
	photon->tau += -__logf(1.F - rand_MWC_co(&(random->x),&(random->a))) * photon->vz;
	
	// Si tau<0 le photon atteint la surface
	if(photon->tau < 0.F) photon->loc = SURFACE;
	// Si tau>tauMax le photon atteint l'espace
	else if(photon->tau > TAU) photon->loc = SPACE;
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
__device__ void scatter(Photon* photon, //photon
		Random* random //variables fonction aléatoire
		#ifdef TRAJET
		, int idx, Evnt* evnt //récupération d'informations
		#endif
		       )
{
	// On détermine la nouvelle direction du photon
	float cThP, cThP2, sThP, psiP, cPsiP, sPsiP;
	// On prend un cosinus aléatoirement entre -1 et 1, il s'agit du cosinus de l'angle que fait la nouvelle direction avec l'ancienne, on prend le cos aléatoirement pour l'équiprobabilité dans toutes les directons
	cThP = 2.F * rand_MWC_co(&(random->x),&(random->a)) - 1.F; //cosThetaPhoton
	cThP2 = cThP * cThP; //cosThetaPhotonCarré
	sThP = sqrtf(1.F - cThP2);//sinThetaPhoton
	// On prend un angle phi aléatoirement entre 0 et 2.PI, il s'agit de l'angle entre le nouveau vecteur projeté dans le plan orthogonal à v et le vecteur u
	psiP = rand_MWC_co(&(random->x),&(random->a)) * PImul2; //psiPhoton
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
		float a = s2PsiP / 2.F * stokes3;
		
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
	photon->ux = (cThP * vx - photon->vx) / sThP;
	photon->uy = (cThP * vy - photon->vy) / sThP;
	photon->uz = (cThP * vz - photon->vz) / sThP;
	// Changement du vecteur v (vitesse du photon)
	photon->vx = vx;
	photon->vy = vy;
	photon->vz = vz;
	
	// Changement du poids et des nombres de stokes du photon
	float gam = DEPO/(2.F-DEPO);
	float stokes1 = photon->stokes1;
	float stokes2 = photon->stokes2;
	// Calcul du poids après diffusion
	photon->weight *= 1.5F / (1.F+2.F*gam) / (stokes1+stokes2) * ((1.F+gam)*stokes2+((1.F-gam)*cThP2+2.F*gam)*stokes1);
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

// Fonction qui traite les photons sortants dans l'espace: changement de poids et de Stokes
__device__ void calculsPhoton(float theta, Constantes* constantes, Photon* photon)
{
	// Création d'un angle psi qui sert à modifier les nombres de Stokes
	float psi;
	if (theta < 0.0025F)
	{
		psi = acosf(fmin(1.F - VALMIN, fmax(-(1.F - VALMIN), -(constantes->cThS) * photon->ux + (constantes->sThS) * photon->uz)));
	}
	else
	{
		psi = acosf(fmin(1.F, fmax(-1.F, ((constantes->sThS) * photon->ux + (constantes->cThS) * photon->uz) / (__sinf(theta)))));
	}
	float sign = (constantes->sThS) * (photon->uy * photon->vz - photon->uz * photon->vy) + (constantes->cThS) * (photon->ux * photon->vy - photon->uy * photon->vx);
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
		float s2ps2u = s2p / 2.F * stokes3;
		photon->stokes1 = cp2 * stokes1 + sp2 * stokes2 + s2ps2u;
		photon->stokes2 = sp2 * stokes1 + cp2 * stokes2 - s2ps2u;
		photon->stokes3 = s2p * (stokes2 - stokes1) + __cosf(psi2) * stokes3;
	}
	// On modifie ensuite le poids du photon
	photon->weight = photon->weight / (photon->stokes1 + photon->stokes2);
}

// Fonction qui calcule la position (ith, iphi) du photon dans le tableau de sortie
__device__ void calculCase(int* ith, int* iphi, Photon* photon, Progress* progress)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->vx * photon->vx + photon->vy * photon->vy);
	// Calcul de la valeur de ithv
	*ith = (int)(acosf(photon->vz) * (NBTHETA) / (PIdiv2));
	// Si le photon ressort très près du zénith on ne peut plus calculer iphi,
	// on est à l'intersection de toutes les cases du haut
	if(vxy < VALMIN)
	{
		atomicAdd(&(progress->erreurvxy), 1);
		if(photon->vy < 0.F) *iphi = NBPHI - 1;
		else *iphi = 0;
	}
	// Sinon on calcule iphi
	else
	{
		// On place d'abord le photon dans un demi-cercle
		float cPhiP = photon->vx / vxy; //cosPhiPhoton
		// Cas limite où phi est très proche de 0, la formule générale ne marche pas
		if(cPhiP >= 1.F) *iphi = 0;
		// Cas limite où phi est très proche de PI, la formule générale ne marche pas
		else if(cPhiP <= -1.F) *iphi = (NBPHI / 2) - 1;
		// Cas général
		else *iphi = (int)(acosf(cPhiP) * (NBPHI) / (PImul2));
		// Puis on place le photon dans l'autre demi-cercle selon vy
		if(photon->vy < 0.F) *iphi = NBPHI - 1 - *iphi;
		// Lorsque vy=0 on décide par défaut que le photon reste du côté vy>0
		if(photon->vy == 0.F) atomicAdd(&(progress->erreurvy), 1);
	}
}

__device__ float rand_MWC_co(unsigned long long* x,unsigned int* a)
{
		//Generate a random number [0,1)
	*x=(*x&0xffffffffull)*(*a)+(*x>>32);
	return __fdividef(__uint2float_rz((unsigned int)(*x)),(float)0x100000000);// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)
}//end __device__ rand_MWC_co

__device__ float rand_MWC_oc(unsigned long long* x,unsigned int* a)
{
		//Generate a random number (0,1]
	return 1.0f-rand_MWC_co(x,a);
}//end __device__ rand_MWC_oc

// Device function to add an unsigned integer to an unsigned long long
__device__ void atomicAddULL(unsigned long long* address, unsigned int add)
{
	if(atomicAdd((unsigned int*)address,add)+add<add)
		atomicAdd(((unsigned int*)address)+1,1u);
}
