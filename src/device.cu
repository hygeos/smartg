
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

/*************************************************************/
/*************************************************************/
/*          MENTION LICENCE POUR RNGs                        */
/*************************************************************/
/*         Philox 4x32 7                                     */
/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/***************************************************************/
/*          FIN LICENCES RNGs                                  */
/***************************************************************/



/**********************************************************
*	> Includes
***********************************************************/

#include "communs.h"
#include "device.h"
#include <math.h>


/**********************************************************
*	> Kernel
***********************************************************/

/* lancementKernel
* Kernel de lancement et gestion de la simulation
* Les fonctions de plus bas niveau sont appelées en fonction de la localisation du photon
* Il peut être important de rappeler que le kernel lance tous les threads mais effectue des calculs similaires. La boucle de la
* fonction va donc être effectuée pour chaque thread du block de la grille
* A TESTER: Regarder pour effectuer une réduction de l'atomicAdd
*/



__device__ void launchKernel(Variables* var, Tableaux tab
		, Init* init
			       )
{
	// idx est l'indice du thread considéré
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
    int loc_prev;
    int count_level;
    int this_thread_active = 1;


	// Paramètres de la fonction random en mémoire locale
	#ifdef RANDMWC
	unsigned long long etatThr;
	unsigned int configThr;
	configThr = tab.config[idx];
	etatThr = tab.etat[idx];
	#endif
	#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
        curandSTATE etatThr;
	etatThr = tab.etat[idx];
	#endif
	#ifdef RANDMT
	ConfigMT configThr;
	EtatMT etatThr;
	configThr = tab.config[idx];
	etatThr = tab.etat[idx];
	#endif
        #ifdef RANDPHILOX4x32_7
        //la clef se defini par l'identifiant global (unique) du thread...
        //...et par la clef utilisateur ou clef par defaut
        //ce systeme garanti l'existence de 2^32 generateurs differents par run et...
        //...la possiblite de reemployer les memes sequences a partir de la meme clef utilisateur
        //(plus d'infos dans "communs.h")
        philox4x32_key_t configThr = {{idx, tab.config}};
        //le compteur se defini par trois mots choisis au hasard (il parait)...
        //...et un compteur definissant le nombre d'appel au generateur
        //ce systeme garanti l'existence de 2^32 nombres distincts pouvant etre genere par thread,...
        //...et ce sur l'ensemble du process (et non pas 2^32 par thread par appel au kernel)
        //(plus d'infos dans "communs.h")
        philox4x32_ctr_t etatThr = {{tab.etat[idx], 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
        #endif

	
	// Création de variable propres à chaque thread
	unsigned long long nbPhotonsThr = 0; 	// Nombre de photons traités par le thread
	
	#ifdef PROGRESSION
	unsigned int nbPhotonsSorThr = 0; 		// Nombre de photons traités par le thread et ressortis dans l'espace
	#endif
	
	Photon ph; 		// On associe une structure de photon au thread
	ph.loc = NONE;	// Initialement le photon n'est nulle part, il doit être initialisé
	
	
    atomicAdd(&(var->nThreadsActive), 1);

    //
    // main loop
    //
    while (var->nThreadsActive > 0) {

        if ((var->nbPhotons > NBLOOPd) && this_thread_active) {
            this_thread_active = 0;
            atomicAdd(&(var->nThreadsActive), -1);
        }

		// Si le photon est à NONE on l'initialise et on le met à la localisation correspondant à la simulaiton en cours
		if((ph.loc == NONE) && this_thread_active){
			
			initPhoton(&ph, tab
				, init
			    , &etatThr
			    #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
			    , &configThr
			    #endif
					);
			
		}
		

        //
		// Deplacement
        //
        // -> Si OCEAN ou ATMOS
        loc_prev = ph.loc;
		if( (ph.loc == ATMOS) || (ph.loc == OCEAN)){

            #ifdef SPHERIQUE
            if (ph.loc == ATMOS)
                move_sp(&ph, tab, init
                        , &etatThr
                        #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
                        , &configThr
                        #endif
                                );
            else 
            #endif
               move_pp(&ph,tab.z, tab.h, tab.pMol , tab.abs , tab.ho, &etatThr
                        #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
                        , &configThr
                        #endif
                                );
		}

        //
        // count after move:
        // count the photons in space and reaching surface from above or below
        //
        count_level = -1;
        if (ph.loc == SPACE) {
            count_level = UPTOA;

            // increment the photon counter
            // (for this thread)
            nbPhotonsThr++;

            #ifdef PROGRESSION
            nbPhotonsSorThr++;
            #endif

            // reset the photon location (always)
            ph.loc = NONE;
        } else if ((ph.loc == SURF0M) || (ph.loc == SURF0P)) {
            if ((loc_prev == ATMOS) || (loc_prev == SPACE)) count_level = DOWN0P;
            if (loc_prev == OCEAN) count_level = UP0M;
        }

        // count the photons
        countPhoton(&ph, tab, count_level
                #ifdef PROGRESSION
                , var
                #endif
                );


		syncthreads();

		
        //
		// Diffusion
        //
        // -> dans ATMOS ou OCEAN
		if( (ph.loc == ATMOS) || (ph.loc == OCEAN)){
	
			scatter(&ph, tab.faer, tab.ssa , tab.foce , tab.sso, tab.ip, tab.ipo, &etatThr
			#if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
			, &configThr
			#endif
				);

		}
		syncthreads();


        //
		// Reflection
        //
        // -> in SURFACE
        loc_prev = ph.loc;
        if ((ph.loc == SURF0M) || (ph.loc == SURF0P)){
           // Eventually evaluate Downward 0+ and Upward 0- radiance


           if( ENVd==0 ) { // si pas d effet d environnement
			if( DIOPTREd!=3 )
				surfaceAgitee(&ph, tab.alb, &etatThr
					#if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
					, &configThr
					#endif
						);

			else
				surfaceLambertienne(&ph, tab.alb, &etatThr
                                        #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
					, &configThr
					#endif
						);
           }

           else {
                float dis=0;
                dis = sqrtf((ph.x-X0d)*(ph.x-X0d) +(ph.y-Y0d)*(ph.y-Y0d));
                if( dis > ENV_SIZEd) {
				     surfaceLambertienne(&ph, tab.alb, &etatThr
                                        #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
					 , &configThr
					      #endif
						);
                }
                else {
				     surfaceAgitee(&ph, tab.alb, &etatThr
					        #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
					 , &configThr
					        #endif
						);
                }
           }
		}
		syncthreads();

        //
		// Reflection
        //
        // -> in SEAFLOOR
		if(ph.loc == SEAFLOOR){
		     surfaceLambertienne(&ph, tab.alb, &etatThr
                                    #if defined(RANDMWC) || defined(RANDMT) || defined(RANDPHILOX4x32_7)
			 , &configThr
			      #endif
			);
        }
		syncthreads();


        //
        // count after surface:
        // count the photons leaving the surface towards the ocean or atmosphere
        //
        count_level = -1;
        if ((loc_prev == SURF0M) || (loc_prev == SURF0P)) {
            if ((ph.loc == ATMOS) || (ph.loc == SPACE)) count_level = UP0P;
            if (ph.loc == OCEAN) count_level = DOWN0M;
        }
        countPhoton(&ph, tab, count_level
                #ifdef PROGRESSION
                , var
                #endif
                );



		if(ph.loc == ABSORBED){
			ph.loc = NONE;
			nbPhotonsThr++;
		}
		syncthreads();

		

        // from time to time, transfer the per-thread photon counter to the
        // global counter
        if (nbPhotonsThr % 100 == 0) {
            atomicAdd(&(var->nbPhotons), nbPhotonsThr);
            nbPhotonsThr = 0;
        }

	}


	// Après la boucle on rassemble les nombres de photons traités par chaque thread

	atomicAdd(&(var->nbPhotons), nbPhotonsThr);

	#ifdef PROGRESSION
	// On rassemble les nombres de photons traités et sortis de chaque thread
	atomicAdd(&(var->nbPhotonsSor), nbPhotonsSorThr);

	// On incrémente avncement qui compte le nombre d'appels du Kernel
	atomicAdd(&(var->nbThreads), 1);
	#endif

        #ifdef RANDPHILOX4x32_7
	// Sauvegarde de l'état du random pour que les nombres ne soient pas identiques à chaque appel du kernel
	tab.etat[idx] = etatThr[0];
        #else
	// Sauvegarde de l'état du random pour que les nombres ne soient pas identiques à chaque appel du kernel
	tab.etat[idx] = etatThr;
        #endif


}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph, Tableaux tab
		,  Init* init
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
        #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
		    )
{
	// Initialisation du vecteur vitesse
	ph->vx = - STHVd;
	ph->vy = 0.F;
	ph->vz = - CTHVd;

	
	// Initialisation du vecteur orthogonal au vecteur vitesse
	ph->ux = -ph->vz;
	ph->uy = 0.F;
	ph->uz = ph->vx;
	
    // Initialisation de la longueur d onde
     //mono chromatique
	ph->ilam = __float2uint_rz(RAND * NLAMd);
	ph->wavel = tab.lambda[ph->ilam];
    atomicAdd(tab.nbPhotonsInter+ph->ilam, 1);

    if ((SIMd == -2) || (SIMd == 1) || (SIMd == 2)) {

        //
        // Initialisation du photon au sommet de l'atmosphère
        //

        ph->x = init->x0;
        ph->y = init->y0;
        ph->z = init->z0;
        ph->couche = 0;   // top of atmosphere

        #ifdef SPHERIQUE
        ph->rayon = sqrtf(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z );
        #endif

        // !! DEV on ne calucle pas d ep optique ici
        ph->loc = ATMOS;
        ph->tau = tab.h[NATMd + ph->ilam*(NATMd+1)]; 

    } else if ((SIMd == -1) || (SIMd == 0) || (SIMd == 3)) {

        //
        // Initialisation du photon à la surface ou dans l'océan
        //
        ph->x = 0.;
        ph->y = 0.;
        #ifdef SPHERIQUE
        ph->z = RTER;
        #else
        ph->z = 0;
        #endif

        ph->tau = 0.f;

        if (SIMd == 3) {
            ph->loc = OCEAN;
        } else {
            ph->loc = SURF0P;
        }

    } else ph->loc = NONE;
	

	ph->weight = WEIGHTINIT;
	
	// Initialisation des paramètres de stokes du photon
	ph->stokes1 = 0.5F;
	ph->stokes2 = 0.5F;
	ph->stokes3 = 0.F;
	ph->stokes4 = 0.F;

}



#ifdef SPHERIQUE
__device__ void move_sp(Photon* ph, Tableaux tab, Init* init
        #ifdef RANDMWC
        , unsigned long long* etatThr, unsigned int* configThr
        #endif
        #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
        #endif
        #ifdef RANDMT
        , EtatMT* etatThr, ConfigMT* configThr
        #endif
        #ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
        #endif
            ) {

    float tauRdm;
    float hph = 0.;  // cumulative optical thickness
    float vzn, delta;
    float d_cur, r_cur2, h_cur;
    int sign_direction;
    int i_layer_fw, i_layer_bh; // index or layers forward and hehind the photon
    float costh, sinth;
    int ilam = ph->ilam*(NATMd+1);  // wavelength offset in optical thickness table

    #ifdef DEBUG
    int niter = 0;
    if ((ph->couche >= NATMd) || (ph->couche < 0)) {
        printf("Fatal error, wrong index (%d)\n", ph->couche);
    }
    #endif

    // Random Optical Thickness to go through
    tauRdm = -logf(1.F-RAND);

    vzn = __fdividef( ph->vx*ph->x + ph->vy*ph->y + ph->vz*ph->z , ph->rayon);

    // a priori value for sign_direction:
    // sign_direction may change sign from -1 to +1 if the photon does not
    // cross lower layer
    if (vzn <= 0) sign_direction = -1;
    else sign_direction = 1;

    while (1) {

        #ifdef DEBUG
        niter++;

        if (niter > 100) {
            printf("niter=%d break\n", niter);
            break;
        }
        #endif

        //
        // stopping criteria
        //
        if (ph->couche == NATMd) {
            ph->loc = SURF0P;
            ph->couche -= 1;  // next time photon enters move_sp, it's at layers NATM-1
            break;
        }
        if (ph->couche < 0) {
            ph->loc = SPACE;
            break;
        }

        //
        // determine the index of the next potential layer
        //
        if (sign_direction < 0) {
            // photon goes down
            // (towards higher indices)
            i_layer_fw = ph->couche + 1;
            i_layer_bh = ph->couche;
        } else {
            // photon goes up
            // (towards lower indices)
            i_layer_fw = ph->couche;
            i_layer_bh = ph->couche + 1;
        }

        //
        // initializations
        //
        costh = vzn;
        sinth = sqrtf(1.f-costh*costh);
        r_cur2 = ph->rayon*ph->rayon;

        //
        // calculate the distance d_cur from the current position to the fw layer
        //
        // ri : radius of next layer boundary ri=zi+RTER
        // r  : radius of current point along the path 
        // costh: cosine of the path direction with the vertical of the current point
        // In the triangle we have ri**2 = d_cur**2 + r**2 - 2 *d_cur *r * costh
        // or: d_cur**2 - 2 *r * costh * d_cur + r**2-ri**2 = 0 , to be solved for d_cur
        // determinant delta is : 4 r**2 costh**2 - 4(r**2-ri**2)
        // it simplifies to:
        delta = 4.f*((tab.z[i_layer_fw]+RTER)*(tab.z[i_layer_fw]+RTER) - r_cur2*sinth*sinth);

        if (delta < 0) {
            if (sign_direction > 0) {
                //#ifdef DEBUG
                //printf("Warning sign_direction (idx=%d, niter=%d, lay=%d, delta=%f, alt=%f zlay1=%f zlay2=%f vzn=%f)\n",
                //        idx, niter, ph->couche, delta, ph->rayon-RTER,
                //        tab.z[i_layer_fw],
                //        tab.z[i_layer_bh],
                //        vzn);
                //#endif

                // because of numerical uncertainties, a downward photon may
                // not strictly be between zi and zi+1
                // in rare case of grazing angle there is sometimes no intersection
                // with current layer because photon is actually slightly above it.
                // therefore we consider that delta=0 such that the photon is
                // tangent to the layer
                delta = 0.;
            } else {
                // no intersection, with lower layer, we should to towards higher layer
                sign_direction = 1;
                continue;
            }
        }

        /* Now, there are two real solutions for d_cur:
        *
        * if photon goes towards higher layers (sign_direction == 1) and costh>0
        * => we keep the smallest solution in abs. val   (both terms are of opposite signs)
        *
        * if photon goes towards lower layers (sign_direction == -1) and costh<0
        * => we keep the smallest solution in abs. val   (both terms are of opposite signs)
        *
        * if photon goes towards higher layers (sign_direction == 1) and costh<0
        * => we keep the greatest solution in abs. val   (both terms are of same signs)
        *
        * NOTE: the solution is always the smallest positive one
        */
        d_cur = 0.5f*(-2.*ph->rayon*costh + sign_direction*sqrtf(delta));


        //
        // calculate the optical thickness h_cur to the next layer
        // We compute the layer extinction coefficient of the layer DTau/Dz and multiply by the distance within the layer
        //
        h_cur = __fdividef(abs(tab.h[i_layer_bh+ilam] - tab.h[i_layer_fw+ilam])*d_cur,
                          abs(tab.z[i_layer_bh] - tab.z[i_layer_fw]));


        //
        // update photon position
        //
        if (hph + h_cur > tauRdm) {
            // photon stops within the layer
            d_cur *= (tauRdm-hph)/h_cur;
            ph->x = ph->x + ph->vx*d_cur;
            ph->y = ph->y + ph->vy*d_cur;
            ph->z = ph->z + ph->vz*d_cur;
            ph->rayon = sqrtf(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z);
            ph->weight *= 1.f - tab.abs[ph->couche+ilam];
            ph->prop_aer = 1.f - tab.pMol[ph->couche+ilam];

            break;
        } else {
            // photon advances to the next layer
            hph += h_cur;
            ph->x = ph->x + ph->vx*d_cur;
            ph->y = ph->y + ph->vy*d_cur;
            ph->z = ph->z + ph->vz*d_cur;
            ph->couche -= sign_direction;
            ph->rayon = sqrtf(ph->x*ph->x + ph->y*ph->y + ph->z*ph->z);
            vzn = __fdividef( ph->vx*ph->x + ph->vy*ph->y + ph->vz*ph->z , ph->rayon);
        }


    }
}
#endif // SPHERIQUE


__device__ void move_pp(Photon* ph,float*z, float* h, float* pMol , float *abs , float* ho
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
		    ) {


	float Dsca=0.f, dsca=0.f;

	ph->tau += -logf(1.f - RAND)*ph->vz;

	float tauBis;
    int icouche;

	if (ph->loc == OCEAN){  
        if (ph->tau > 0) {
           ph->tau = 0.F;
           ph->loc = SURF0M;
           if (SIMd == 3){
              ph->loc = SPACE;
           }
           return;
        }
        // Si tau<TAUOCEAN le photon atteint le fond 
        else if( ph->tau < ho[NOCEd + ph->ilam *(NOCEd+1)] ){
            ph->loc = SEAFLOOR;
            ph->tau = ho[NOCEd + ph->ilam *(NOCEd+1)];
            return;
        }

        // Calcul de la couche dans laquelle se trouve le photon
        tauBis =  ho[NOCEd + ph->ilam *(NOCEd+1)] - ph->tau;
        icouche = 1;

        while ((ho[icouche+ ph->ilam *(NOCEd+1)] > (tauBis)) && (icouche < NOCEd)) {
            icouche++;
        }
        ph->couche = icouche;

    }

    if (ph->loc == ATMOS) {

        // Si tau<0 le photon atteint la surface
        if(ph->tau < 0.F){
            ph->loc = SURF0P;
            ph->tau = 0.F;
        return;
        }
        // Si tau>TAUATM le photon atteint l'espace
        else if( ph->tau > h[NATMd + ph->ilam *(NATMd+1)] ){
            ph->loc = SPACE;
            return;
        }
        
        // Sinon il reste dans l'atmosphère, et va subit une nouvelle diffusion
        
        // Calcul de la couche dans laquelle se trouve le photon
        tauBis =  h[NATMd + ph->ilam *(NATMd+1)] - ph->tau;
        icouche = 1;
        
        while ((h[icouche+ ph->ilam *(NATMd+1)] < (tauBis)) && (icouche < NATMd)) {
            icouche++;
        }
        
        ph->couche = icouche;
        ph->prop_aer = 1.f - pMol[ph->couche+ph->ilam*(NATMd+1)];
        ph->weight = ph->weight * (1.f - abs[ph->couche+ph->ilam*(NATMd+1)]);

    }

        float phz,rdist;
        Dsca= fabs(h[icouche] - h[icouche-1]) ;
        dsca= fabs(tauBis - h[icouche-1]) ;

        //calcul de la nouvelle altitude du photon
        phz=z[icouche-1]+(dsca/Dsca)*(z[icouche]-z[icouche-1]);
        rdist=(phz-ph->z)/ph->vz;
        ph->z = phz;
        ph->x = ph->x + ph->vx*rdist;
        ph->y = ph->y + ph->vy*rdist;

}



/* scatter
* Diffusion du photon par une molécule ou un aérosol
* Modification des paramètres de stokes et des vecteurs U et V du photon (polarisation, vitesse)
*/
__device__ void scatter( Photon* ph, float* faer, float* ssa , float* foce , float* sso, int* ip, int* ipo
			#ifdef RANDMWC
			, unsigned long long* etatThr, unsigned int* configThr
			#endif
			#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                        , curandSTATE* etatThr
                        #endif
			#ifdef RANDMT
			, EtatMT* etatThr, ConfigMT* configThr
			#endif
                        #ifdef RANDPHILOX4x32_7
                        , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
                        #endif
			){

	float cTh=0.f, sTh, psi, cPsi, sPsi;
	float wx, wy, wz, vx, vy, vz;
	
	
	/* Les calculs qui différent pour les aérosols et les molécules sont regroupés dans cette partie.
	 * L'idée à termes est de réduire au maximum cette fonction, en calculant également la fonction de phase pour les
	 * molécules, à la manière des aérosols.
	*/

	float zang=0.f, theta=0.f;
	int iang, ilay, ipha;
	float stokes1, stokes2, stokes3, stokes4, norm;
	float cTh2;
	float prop_aer = ph->prop_aer;
	
	stokes1 = ph->stokes1;
	stokes2 = ph->stokes2;
	
	
	///////// Possible de mettre dans une fonction séparée, mais attention aux performances /////////
	///////// Faire également attention à bien passer le pointeur de cTh et le modifier dans la fonction /////////
	
	if(ph->loc!=OCEAN){
        ilay = ph->couche + ph->ilam*(NATMd+1); // atm layer index
        ipha  = ip[ilay]; // atm phase function index
		if( prop_aer<RAND ){

			// Get Teta (see Wang et al., 2012)
			float b = (RAND - 4.0 * ALPHAd - BETAd) / (2.0 * ALPHAd);
			float expo = 1./2.;
			float base = ACUBEd + b*b;
			float tmp  = pow(base, expo);
			expo = 1./3.;
			base = -b + tmp;
			float u = pow(base,expo);
			cTh     = u - Ad / u;  						       

			if (cTh < -1.0) cTh = -1.0;
			if (cTh >  1.0) cTh =  1.0;
			cTh2 = cTh * cTh;
			
			/////////////
			//  Get Phi

			// Biased sampling scheme for phi
			psi = RAND * DEUXPI;	//psiPhoton
			cPsi = __cosf(psi);	//cosPsiPhoton
			sPsi = __sinf(psi);     //sinPsiPhoton		

			// Calcul des parametres de Stokes du photon apres diffusion
			
			// Rotation des paramètres de stokes
			rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3,  psi,
				     &ph->stokes1, &ph->stokes2, &ph->stokes3 );

			// Calcul des parametres de Stokes du photon apres diffusion
			float cross_term;
			stokes1 = ph->stokes1;
			stokes2 = ph->stokes2;
			cross_term  = DELTA_PRIMd * (stokes1 + stokes2);
			ph->stokes1 = 3./2. * (  DELTAd  * stokes1 + cross_term );
			ph->stokes2 = 3./2. * (  DELTAd  * cTh2 * stokes2 + cross_term );			
			ph->stokes3 = 3./2. * (  DELTAd * cTh  * ph->stokes3 );
			ph->stokes4 = 3./2. * (  DELTAd * DELTA_SECOd * cTh * ph->stokes4 );
			// bias sampling scheme
			float phase_func;
			phase_func = 3./4. * DELTAd * (cTh2+1.0) + 3.0 * DELTA_PRIMd;
			ph->stokes1 /= phase_func;  
			ph->stokes2 /= phase_func;  
			ph->stokes3 /= phase_func;     		
			ph->stokes4 /= phase_func;     		


		}
		else{
			// Aérosols
			zang = RAND*(NFAERd-2);
			iang= __float2int_rd(zang);
			zang = zang - iang;
			/* L'accès à faer[x][y] se fait par faer[y*5+x] */
			theta = faer[ipha*NFAERd*5+iang*5+4]+ zang*( faer[ipha*NFAERd*5+(iang+1)*5+4]-faer[ipha*NFAERd*5+iang*5+4] );
			//theta = faer[iang*5+4]+ zang*( faer[(iang+1)*5+4]-faer[iang*5+4] );
			cTh = __cosf(theta);

			//////////////
			//  Get Phi

			// biased sampling scheme for phi
			psi = RAND * DEUXPI;	//psiPhoton
			cPsi = __cosf(psi);	//cosPsiPhoton
			sPsi = __sinf(psi);     //sinPsiPhoton		
			// Rotation des paramètres de stokes
			rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3,   psi,
				     &ph->stokes1, &ph->stokes2, &ph->stokes3);

            stokes3=ph->stokes3;
            stokes4=ph->stokes4;
			// Calcul des parametres de Stokes du photon apres diffusion
			ph->stokes1 *= faer[ipha*NFAERd*5+iang*5+0];
			ph->stokes2 *= faer[ipha*NFAERd*5+iang*5+1];
			ph->stokes3 = stokes3*faer[ipha*NFAERd*5+iang*5+2] - stokes4*faer[ipha*NFAERd*5+iang*5+3];
			ph->stokes4 = stokes4*faer[ipha*NFAERd*5+iang*5+2] + stokes3*faer[ipha*NFAERd*5+iang*5+3];

			float debias;
			debias = __fdividef( 2., faer[ipha*NFAERd*5+iang*5+0] + faer[ipha*NFAERd*5+iang*5+1] );
			ph->stokes1 *= debias;  
			ph->stokes2 *= debias;  
			ph->stokes3 *= debias;  
			ph->stokes4 *= debias;  

			ph->weight *= ssa[ilay];
			
		}
		
	}
	else{	/* Photon dans l'océan */
	    float prop_raman=1., new_wavel;
        ilay = ph->couche + ph->ilam*(NOCEd+1); // oce layer index
        ipha  = ipo[ilay]; // oce phase function index

        // we fix the proportion of Raman to 2% at 488 nm, !! DEV
        //prop_raman = 0.02 * pow ((1.e7/ph->wavel-3400.)/(1.e7/488.-3400.),5); // Raman scattering to pure water scattering ratio

	    if(prop_raman <RAND ){
            // diffusion Raman
            // Phase function similar to Rayleigh
		    // Get Teta (see Wang et al., 2012)
			float b = (RAND - 4.0 * ALPHAd - BETAd) / (2.0 * ALPHAd);
			float expo = 1./2.;
			float base = ACUBEd + b*b;
			float tmp  = pow(base, expo);
			expo = 1./3.;
			base = -b + tmp;
			float u = pow(base,expo);
			cTh     = u - Ad / u;  						       

			if (cTh < -1.0) cTh = -1.0;
			if (cTh >  1.0) cTh =  1.0;
			cTh2 = cTh * cTh;
			
			/////////////
			//  Get Phi
			// Biased sampling scheme for phi
			psi = RAND * DEUXPI;	//psiPhoton
			cPsi = __cosf(psi);	//cosPsiPhoton
			sPsi = __sinf(psi);     //sinPsiPhoton		

			// Calcul des parametres de Stokes du photon apres diffusion
			
			// Rotation des paramètres de stokes
			rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3, psi,
				     &ph->stokes1, &ph->stokes2, &ph->stokes3);

			// Calcul des parametres de Stokes du photon apres diffusion
			float cross_term;
			stokes1 = ph->stokes1;
			stokes2 = ph->stokes2;
			cross_term  = DELTA_PRIMd * (stokes1 + stokes2);
			ph->stokes1 = 3./2. * (  DELTAd  * stokes1 + cross_term );
			ph->stokes2 = 3./2. * (  DELTAd  * cTh2 * stokes2 + cross_term );			
			ph->stokes3 = 3./2. * (  DELTAd * cTh  * ph->stokes3 );
			ph->stokes4 = 3./2. * (  DELTAd * DELTA_SECOd * cTh * ph->stokes4 );
			// bias sampling scheme
			float phase_func;
			phase_func = 3./4. * DELTAd * (cTh2+1.0) + 3.0 * DELTA_PRIMd;
			ph->stokes1 /= phase_func;  
			ph->stokes2 /= phase_func;  
			ph->stokes3 /= phase_func;     		
			ph->stokes4 /= phase_func;     		

            // Changement de longueur d onde
            new_wavel  = 22.94 + 0.83 * (ph->wavel) + 0.0007 * (ph->wavel)*(ph->wavel);
            ph->weight /= new_wavel/ph->wavel;
            ph->wavel = new_wavel;
		  }

	  else{
          // diffusion elastique
		
		zang = RAND*(NFOCEd-2);
		iang = __float2int_rd(zang);
		zang = zang - iang;

		theta = foce[ipha*NFOCEd*5+iang*5+4]+ zang*( foce[ipha*NFOCEd*5+(iang+1)*5+4]-foce[ipha*NFOCEd*5+iang*5+4] );
		
		cTh = __cosf(theta);

        //////////////
        //  Get Phi

        // biased sampling scheme for phi
        psi = RAND * DEUXPI;	//psiPhoton
        cPsi = __cosf(psi);	//cosPsiPhoton
        sPsi = __sinf(psi);     //sinPsiPhoton
        // Rotation des paramètres de stokes
        rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3, psi,
                &ph->stokes1, &ph->stokes2, &ph->stokes3);


		stokes3 = ph->stokes3;
		stokes4 = ph->stokes4;
        // Calcul des parametres de Stokes du photon apres diffusion
        ph->stokes1 *= foce[ipha*NFOCEd*5+iang*5+0];
        ph->stokes2 *= foce[ipha*NFOCEd*5+iang*5+1];
        ph->stokes3 = stokes3*foce[ipha*NFOCEd*5+iang*5+2] - stokes4*foce[ipha*NFOCEd*5+iang*5+3];
        ph->stokes4 = stokes4*foce[ipha*NFOCEd*5+iang*5+2] + stokes3*foce[ipha*NFOCEd*5+iang*5+3];

        float debias;
        debias = __fdividef( 2., foce[ipha*NFOCEd*5+iang*5+0] + foce[ipha*NFOCEd*5+iang*5+1] );
        ph->stokes1 *= debias;
        ph->stokes2 *= debias;
        ph->stokes3 *= debias;
        ph->stokes4 *= debias;

		ph->weight *= sso[ilay];

	 } // elastic scattering

	/** Roulette russe **/
	if( ph->weight < WEIGHTRR ){
		if( RAND < __fdividef(ph->weight,WEIGHTRR) ){
			ph->weight = WEIGHTRR;
		}
		else{
				ph->loc = ABSORBED;
			}
		}
		
    } //photon in ocean
   ////////// Fin séparation ////////////
	
	sTh = sqrtf(1.F - cTh*cTh);	// sinThetaPhoton
	
	/** Création de 2 vecteurs provisoires w et v **/
	float vx_s, vy_s, vz_s, ux_s, uy_s, uz_s;	// Parametres du photon sauves pour optimisation
	vx_s = ph->vx;
	vy_s = ph->vy;
	vz_s = ph->vz;
	ux_s = ph->ux;
	uy_s = ph->uy;
	uz_s = ph->uz;
	// w est le rotationnel entre l'ancien vecteur u et l'ancien vecteur v du photon
	wx = uy_s * vz_s - uz_s * vy_s;
	wy = uz_s * vx_s -ux_s * vz_s;
	wz = ux_s * vy_s - uy_s * vx_s;
	// v est le nouveau vecteur v du photon
	vx = cTh * vx_s + sTh * ( cPsi * ux_s + sPsi * wx );
	vy = cTh * vy_s + sTh * ( cPsi * uy_s + sPsi * wy );
	vz = cTh * vz_s + sTh * ( cPsi * uz_s + sPsi * wz );
	// Changement du vecteur u (orthogonal au vecteur vitesse du photon)
    ph->ux = cTh * vx - vx_s;
    ph->uy = cTh * vy - vy_s;
    ph->uz = cTh * vz - vz_s;
	
	
	// Changement du vecteur v (vitesse du photon)
	ph->vx = vx;
	ph->vy = vy;
	ph->vz = vz;

    // renormalisation
    norm=sqrtf(ph->vx*ph->vx+ph->vy*ph->vy+ph->vz*ph->vz);
    ph->vx/=norm;
    ph->vy/=norm;
    ph->vz/=norm;
    norm=sqrtf(ph->ux*ph->ux+ph->uy*ph->uy+ph->uz*ph->uz);
    ph->ux/=norm;
    ph->uy/=norm;
    ph->uz/=norm;


}


/* surfaceAgitee
* Reflexion sur une surface agitée ou plane en fonction de la valeur de DIOPTRE
*/
__device__ void surfaceAgitee(Photon* ph, float* alb
		#ifdef RANDMWC
		, unsigned long long* etatThr, unsigned int* configThr
		#endif
		#if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                , curandSTATE* etatThr
                #endif
		#ifdef RANDMT
		, EtatMT* etatThr, ConfigMT* configThr
		#endif
		#ifdef RANDPHILOX4x32_7
                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
		#endif
			){
	
	if( SIMd == -2){ // Atmosphère , la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
	
	// Réflexion sur le dioptre agité
	float theta;	// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float cTh, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	
	float sig = 0.F;
	float beta = 0.F;	// Angle par rapport à la verticale du vecteur normal à une facette de vagues 
	float sBeta;
	float cBeta;
	
	float alpha ;	//Angle azimutal du vecteur normal a une facette de vagues
	
	float nind;
	float temp;
	
	float nx, ny, nz;	// Coordonnées du vecteur normal à une facette de vague
	float s1, s2, s3 ;
    float stokes3, stokes4;
	
	float rpar, rper, rparper, rparper_cross;	// Coefficient de reflexion parallèle et perpendiculaire
	float rpar2;		// Coefficient de reflexion parallèle au carré
	float rper2;		// Coefficient de reflexion perpendiculaire au carré
	float rat;			// Rapport des coefficients de reflexion perpendiculaire et parallèle
	float ReflTot;		// Flag pour la réflexion totale sur le dioptre
	float cot;			// Cosinus de l'angle de réfraction du photon
	float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
	float tpar, tper;	//
    float geo_trans_factor;
	
	
	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	
	/** Calcul du theta impact et phi impact **/
	//NOTE: Dans le code Fortran, ce calcul est effectué dans atmos
	float icp, isp, ict, ist;	// Sinus et cosinus de l'angle d'impact
	float vxn, vyn, vzn, uxn, uyn, uzn;
	
	
	/** Calcul de l'angle entre l'axe z et la normale au point d'impact **/
	/*NOTE: le float pour les calculs suivant fait une erreur de 2.3% 
	 * par exemple (theta_float=0.001196 / theta_double=0.0011691
	 * Mais ils sont bien plus performant et cette erreur ne pose pas de problème jusqu'à présent.
	 * De plus, l'angle d'impact n'est pas calculé mais directement les cosinus et sinus de cet angle.
	*/
	if( ph->z > 0. ){
		ict = __fdividef(ph->z,RTER);
		
		if(ict>1.f){
				ict = 1.f;
		}

		ist = sqrtf( 1.f - ict*ict );
   
		if(ph->x >= 0.f) ist = -ist;
		
        icp = __fdividef(ph->x,sqrtf(ph->x*ph->x + ph->y*ph->y));
        isp = sqrtf( 1.f - icp*icp );
        if (isnan(isp)) { // avoid numerical instabilities where x,y are close to zero (or icp>1)
            icp = 1.F;
            isp = 0.F;
        }
        
        if( ph->y < 0.f ) isp = -isp;
	}
	else{
		// Photon considéré comme perdu
        #ifdef DEBUG
        printf("Error in surfaceAgitee ph->z = %f\n", ph->z);
        #endif
		ph->loc = ABSORBED;	// Correspondant au weight=0 en Fortran
		return;
	}
	
	
	/** Il faut exprimer Vx,y,z et Ux,y,z dans le repère de la normale au point d'impact **/
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
    // DR Estimation of the probability P of interaction of the photon with zentih angle theta with a facet of slope beta and azimut alpha	
    // DR P_alpha_beta : Probability of occurence of a given azimuth and slope
    // DR P_alpha_beta = P_Cox_Munk(beta) * P(alpha | beta), conditional probability, for normal incidence, independent variables and P(alpha|beta)=P(alpha)=1/2pi
    // DR following Plass75:
    // DR Pfacet : Probability of occurence of a facet
    // DR Pfacet = projected area of the facet divided by unit area of the possible interaction surface * P_alpha_beta
    // DR Pfacet = P_alpha_beta / cos(beta)
    // DR for non normal incident angle, the probability of interaction between the photon and the facet is proportional to the surface of the facet seen by the photon so
    // DR that is cosine of incident angle of photon on the facet theta_inc=f(alpha,beta,theta)
    // DR P # Pfacet * cos(theta_inc) for cos(theta_inc) >0
    // DR P = 0 for cos(theta_inc)<=0
    // DR for having a true probability, one has to normalize this to 1. The A normalization factor depends on theta and is the sum on all alpha and beta with the condition
    // DR cos(theta_inc)>0 (visible facet)
    // DR A = Sum_0_2pi Sumr_0_pi/2 P_alpha_beta /cos(beta) cos(theta_inc) dalpha dbeta
    // DR Finally P = 1/A * P_alpha_beta  /cos(beta) cos(theta_inc)
	if( DIOPTREd !=0 ){
        theta = DEMIPI;
        // DR Computation of P_alpha_beta = P_Cox_Munk(beta) * P(alpha | beta)
        // DR we draw beta first according to Cox_Munk isotropic and then draw alpha, conditional probability
        // DR rejection method: to exclude unphysical azimuth (leading to incident angle theta >=PI/2)
        // DR we continue until acceptable value for alpha
		sig = sqrtf(0.003F + 0.00512f *WINDSPEEDd);
		beta = atanf( sig*sqrtf(-__logf(RAND)) );
        while(theta>=DEMIPI){
           alpha = DEUXPI * RAND;
	       sBeta = __sinf( beta );
	       cBeta = __cosf( beta );
	       nx = sBeta*__cosf( alpha );
	       ny = sBeta*__sinf( alpha );
	
           // compute relative index of refraction
           // DR a: air, b: water , Mobley 2015 nind = nba = nb/na
	       if (ph->loc == SURF0M) {
		       nind = __fdividef(1.f,NH2Od);
		       nz = -cBeta;
	       }
	       else{
		       nind = NH2Od;
		       nz = cBeta;
	       }
	       temp = -(nx*ph->vx + ny*ph->vy + nz*ph->vz);
	       theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), temp ) ));
        }
	}
    else{
        beta = 0;
        alpha = DEUXPI * RAND;
	    sBeta = __sinf( beta );
	    cBeta = __cosf( beta );
	    nx = sBeta*__cosf( alpha );
	    ny = sBeta*__sinf( alpha );
	
        if (ph->loc == SURF0M) {
		    nind = __fdividef(1.f,NH2Od);
		    nz = -cBeta;
	    }
	    else{
		    nind = NH2Od;
		    nz = cBeta;
	    }
	    temp = -(nx*ph->vx + ny*ph->vy + nz*ph->vz);
	    theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), temp ) ));
    }

	cTh = __cosf(theta);
	sTh = __sinf(theta);

    // Anorm factor modelled with a simple linear fit that represents the departure from vz,
    // (Anorm-vz)
    // ^                                               +
    // |                                              + 
    // |                                             + 
    // |                                            + 
    // |                                           + 
    // ++++++++++++++++++++++++++++++++++++++++++++--------> (theta)
    // 0                                          |        90
    //                                        Theta_thres=f(Windspeed)
    // The slope of the model is constant=0.004 and threshold depends on windspeed. Below threshold on theta, all slopes
    // are possible and thus A=1/vz
    float Anorm;
    float slopeA=0.00377;
    float theta_thres;
    theta_thres = 83.46 - WINDSPEEDd; // between 1 and 15 m/s
    float avz = abs(ph->vz);
    float aavz = acosf(avz)*360./DEUXPI;
    if(aavz > theta_thres){
       Anorm = avz + slopeA * (aavz - theta_thres);
    }
    else{
       Anorm = avz;
    }

    // DR probability of slope interaction with photon corection factor, biased sampling correction of pure Cox_Munk probability function
    ph->weight *= __fdividef(abs(cTh), cBeta * Anorm);

	// Rotation of Stokes parameters
	s1 = ph->stokes1;
	s2 = ph->stokes2;
	s3 = ph->stokes3;

	if( (s1!=s2) || (s3!=0.F) ){

		temp = __fdividef(nx*ph->ux + ny*ph->uy + nz*ph->uz,sTh);
		psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

		if( (nx*(ph->uy*ph->vz-ph->uz*ph->vy) + ny*(ph->uz*ph->vx-ph->ux*ph->vz) + nz*(ph->ux*ph->vy-ph->uy*ph->vx) ) <0 ){
			psi = -psi;
		}

        rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3, psi,
                &ph->stokes1, &ph->stokes2, &ph->stokes3);
	}

	if( sTh<=nind){
		temp = __fdividef(sTh,nind);
		cot = sqrtf( 1.0F - temp*temp );
		ncTh = nind*cTh;
		ncot = nind*cot;
		rpar = __fdividef(ncTh - cot,ncTh  + cot); // DR Mobley 2015 sign convention
		rper = __fdividef(cTh - ncot,cTh + ncot);
		rpar2 = rpar*rpar;
		rper2 = rper*rper;
        rparper = rpar * rper;
        rparper_cross = 0.;
        // DR rat is the energetic reflection factor used to normalize the R and T matrix (see Xun 2014)
		rat =  __fdividef(ph->stokes1*rper2 + ph->stokes2*rpar2,ph->stokes1+ph->stokes2);
		//rat = 0.5 * (rper2 + rpar2); // DR see Xun 2014, eq 15 strange ....
		ReflTot = 0;
	}
	else{
		cot = 0.f;
		rpar = 1.f;
		rper = 1.f;
        rat = 1.f;
        // DR rat is normalizing the relection matrix
		rpar2 = rpar*rpar;
		rper2 = rper*rper;
        rparper = __fdividef(2.*sTh*sTh*sTh*sTh, 1.-(1.+nind * nind)*cTh*cTh) - 1.; // DR !! Mobley 2015
        rparper_cross = -__fdividef(2.*cTh*sTh*sTh*sqrtf(sTh*sTh-nind*nind), 1.-(1.+nind * nind)*cTh*cTh); // DR !! Mobley 2015
		ReflTot = 1;
	}

    stokes3 = ph->stokes3;	
    stokes4 = ph->stokes4;	
	if( (ReflTot==1) || (SURd==1) || ( (SURd==3)&&(RAND<rat) ) ){

		
		ph->stokes1 *= rper2;
		ph->stokes2 *= rpar2;
		ph->stokes3 = rparper*stokes3 + rparper_cross*stokes4; // DR Mobley 2015 sign convention
		ph->stokes4 = rparper*stokes4 - rparper_cross*stokes3; // DR Mobley 2015 sign convention
		
		ph->vx += 2.F*cTh*nx;
		ph->vy += 2.F*cTh*ny;
		ph->vz += 2.F*cTh*nz;
		ph->ux = __fdividef( nx-cTh*ph->vx,sTh );
		ph->uy = __fdividef( ny-cTh*ph->vy,sTh );
		ph->uz = __fdividef( nz-cTh*ph->vz,sTh );
		

        // DR Normalization of the reflexion matrix
        // DR the reflection coefficient is taken into account:
        // DR once in the random selection (Rand < rat)
        // DR once in the reflection matrix multiplication
        // DR so twice and thus we normalize by rat (Xun 2014).
        // DR not to be applied for forced reflection (SUR=1 or total reflection) where there is no random selection
		if (SURd==3 && ReflTot==0) {
			ph->weight /= rat;
			}



        //
        // photon next location
        //
        if (ph->loc == SURF0P) {
            if (ph->vz > 0) {  // avoid multiple reflexion above the surface
                // SURF0P becomes ATM or SPACE
                if( SIMd==-1 || SIMd==0 ){
                    ph->loc = SPACE;
                } else{
                    ph->loc = ATMOS;
                }
            } // else, no change of location
        } else {
            if (ph->vz < 0) {  // avoid multiple reflexion under the surface
                // SURF0M becomes OCEAN or ABSORBED
                if( SIMd==1 ){
                    ph->loc = ABSORBED;
                } else{
                    ph->loc = OCEAN;
                }
            } // else, no change of location
        }


	} // Reflection

	else{	// Transmission

		
        geo_trans_factor = nind* cot/cTh; // DR Mobley 2015 OK , see Xun 2014
		tpar = __fdividef( 2*cTh,ncTh+ cot);
		tper = __fdividef( 2*cTh,cTh+ ncot);
		
		ph->stokes2 *= tpar*tpar*geo_trans_factor;
		ph->stokes1 *= tper*tper*geo_trans_factor;
		ph->stokes3 *= tpar*tper*geo_trans_factor; //DR positive factor Mobley 2015
		ph->stokes4 *= tpar*tper*geo_trans_factor; //DR positive factor Mobley 2015
		
		alpha  = __fdividef(cTh,nind) - cot;
		ph->vx = __fdividef(ph->vx,nind) + alpha*nx;
		ph->vy = __fdividef(ph->vy,nind) + alpha*ny;
		ph->vz = __fdividef(ph->vz,nind) + alpha*nz;
		ph->ux = __fdividef( nx+cot*ph->vx,sTh )*nind;
		ph->uy = __fdividef( ny+cot*ph->vy,sTh )*nind;
		ph->uz = __fdividef( nz+cot*ph->vz,sTh )*nind;

		
        // DR Normalization of the transmission matrix
        // the transmission coefficient is taken into account:
        // once in the random selection (Rand > rat)
        // once in the transmission matrix multiplication
        // so we normalize by (1-rat) (Xun 2014).
        // Not to be applied for forced transmission (SUR=2)
        if ( SURd == 3) 
            ph->weight /= (1-rat);

        //
        // photon next location
        //
        if (ph->loc == SURF0M) {
            if (ph->vz > 0) {
                // SURF0P becomes ATM or SPACE
                if( SIMd==-1 || SIMd==0 ){
                    ph->loc = SPACE;
                } else{
                    ph->loc = ATMOS;
                }
            } else {
                // multiple transmissions (vz<0 after water->air transmission)
                ph->loc = SURF0P;
            }
        } else {
            if (ph->vz < 0) {  // avoid multiple reflexion under the surface
                // SURF0M becomes OCEAN or ABSORBED
                if( SIMd==-1 || SIMd==1 ){
                    ph->loc = ABSORBED;
                } else{
                    ph->loc = OCEAN;
                }
            } else {
                // multiple transmissions (vz<0 after water->air transmission)
                // (for symmetry, but should not happen)
                ph->loc = SURF0P;
            }
        }

	} // Transmission

	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	/** Retour dans le repère d'origine **/
	
	// Re-projection vers le repères de direction de photon. L'angle à prendre pour la projection est -angleImpact
	isp = -isp;
	ist = -ist;
	
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
}



/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* ph, float* alb
						#ifdef RANDMWC
						, unsigned long long* etatThr, unsigned int* configThr
						#endif
                                                #if defined(RANDCUDA) || defined (RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
                                                , curandSTATE* etatThr
						#endif
						#ifdef RANDMT
						, EtatMT* etatThr, ConfigMT* configThr
						#endif
                                                #ifdef RANDPHILOX4x32_7
                                                , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr
                                                #endif
						){
	
	if( SIMd == -2){ 	// Atmosphère ou océan seuls, la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
	
	float uxn,vxn,uyn,vyn,uzn,vzn;	// Vecteur du photon après reflexion
	float cTh2 = RAND;
	float cTh = sqrtf( cTh2 );
	float sTh = sqrtf( 1.0F - cTh2 );
	
	float phi = RAND*DEUXPI;	//angle azimutal
	float cPhi = __cosf(phi);
	float sPhi = __sinf(phi);
	
    #ifdef SPHERIQUE
	float icp, isp, ict, ist;	// Sinus et cosinus de l'angle d'impact
    #endif
	

	/** Séparation du code pour atmosphère sphérique ou parallèle **/
	#ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	/** Calcul du theta impact et phi impact **/
	
    if (ph->loc != SEAFLOOR){

	
	/** Calcul de l'angle entre l'axe z et la normale au point d'impact **/
	/*NOTE: le float pour les calculs suivant fait une erreur de 2.3% 
	* par exemple (theta_float=0.001196 / theta_double=0.0011691
	* Mais ils sont bien plus performant et cette erreur ne pose pas de problème jusqu'à présent.
	* De plus, l'angle d'impact n'est pas calculé mais directement les cosinus et sinus de cet angle.
	*/
	if( ph->z > 0. ){
		ict = __fdividef(ph->z,RTER);
		
		if(ict>1.f){
			ict = 1.f;
		}
		
		ist = sqrtf( 1.f - ict*ict );
		
		if(ph->x >= 0.f) ist = -ist;
		
		if( sqrtf(ph->x*ph->x + ph->y*ph->y)<1.e-6 ){
			/*NOTE En fortran ce test est à 1.e-8, relativement au double utilisés, peut peut être être supprimer ici*/
			icp = 1.f;
		}
		else{
			icp = __fdividef(ph->x,sqrtf(ph->x*ph->x + ph->y*ph->y));
			isp = sqrtf( 1.f - icp*icp );
			
			if( ph->y < 0.f ) isp = -isp;
		}
	}
	else{
		// Photon considéré comme perdu
		ph->loc = ABSORBED;	// Correspondant au weight=0 en Fortran
		return;
	}
	
	
	/** Il faut exprimer Vx,y,z et Ux,y,z dans le repère de la normale au point d'impact **/
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

    } // photon not seafloor
	
	#endif
	
	
	/** calcul u,v new **/
	vxn = cPhi*sTh;
	vyn = sPhi*sTh;
	vzn = cTh;
	
	uxn = cPhi*cTh;
	uyn = sPhi*cTh;
	uzn = -sTh;
	

	// Depolarisation du Photon
	float norm;
	norm = ph->stokes1 + ph->stokes2;
	ph->stokes1 = 0.5 * norm;
	ph->stokes2 = 0.5 * norm;
    ph->stokes3 = 0.0;
    ph->stokes4 = 0.0;

	
	ph->vx = vxn;
	ph->vy = vyn;
	ph->vz = vzn;
	ph->ux = uxn;
	ph->uy = uyn;
	ph->uz = uzn;
	

    if (DIOPTREd!=4 && ((ph->loc == SURF0M) || (ph->loc == SURF0P))){
	  // Si le dioptre est seul, le photon est mis dans l'espace
	  bool test_s = ( SIMd == -1);
	  ph->loc = SPACE*test_s + ATMOS*(!test_s);
    }
	
    if (ph->loc != SEAFLOOR){

	  ph->weight *= alb[0+ph->ilam*2];

	  #ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	  /** Retour dans le repère d'origine **/
	  // Re-projection vers le repères de direction de photon. L'angle à prendre pour la projection est -angleImpact
	  isp = -isp;
	  ist = -ist;
	
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
    } // not seafloor 

    else {
	  ph->weight *= alb[1+ph->ilam*2];
      ph->loc = OCEAN;
    }
    
}



__device__ void countPhoton(Photon* ph,
        Tableaux tab,
        int count_level
		#ifdef PROGRESSION
		, Variables* var   // TODO: remove nbPhotonsSorThr
		#endif
		    ) {

    if (count_level < 0) {
        // don't count anything
        return;
    }

    // don't count the photons directly transmitted
    if ((ph->weight == WEIGHTINIT) && (ph->stokes1 == ph->stokes2) && (ph->stokes3 == 0.f) && (ph->stokes4 == 0.f)) {
        return;
    }

    float *tabCount; // pointer to the "counting" array:
                     // may be TOA, or BOA down, and so on

    float theta = acosf(fmin(1.F, fmax(-1.F, 0.f * ph->vx + 1.f * ph->vz)));
    #ifdef SPHERIQUE
    if(ph->vz<=0.f) {
         // do not count the downward photons leaving atmosphere
         return;
    }
    #endif

    // Si theta = 0 on l'ignore
    // (cas où le photon repart dans la direction de visée)
	if(theta == 0.F)
	{
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurtheta), 1);
		#endif
		return;
	}

	float psi;
	int ith=0, iphi=0, il=0;
	// Initialisation de psi
	calculPsi(ph, &psi, theta);
	
	// Rotation of stokes parameters
    float s1, s2, s3, s4;
    rotateStokes(ph->stokes1, ph->stokes2, ph->stokes3,  psi,
            &s1, &s2, &s3);
    s4 = ph->stokes4;
	// Calcul de la case dans laquelle le photon sort
	calculCase(&ith, &iphi, &il, ph 
			   #ifdef PROGRESSION
			   , var
			   #endif
			   );
	
  	if( ph->vy<0.f )
    		s3 = -s3;
	
    // Change sign convention for compatibility with OS
    s3 = -s3;

	float tmp = s1;
	s1 = s2;
	s2 = tmp;
	

	float weight = ph->weight;

    #ifdef DEBUG
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
    if (isnan(weight)) printf("(idx=%d) Error, weight is NaN\n", idx);
    if (isnan(s1)) printf("(idx=%d) Error, s1 is NaN\n", idx);
    if (isnan(s2)) printf("(idx=%d) Error, s2 is NaN\n", idx);
    if (isnan(s3)) printf("(idx=%d) Error, s3 is NaN\n", idx);
    #endif

	// Rangement du photon dans sa case, et incrémentation de variables
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (il >= 0) && (il < NLAMd) && (!isnan(weight)))
	{
        // select the appropriate level (count_level)
        tabCount = tab.tabPhotons + count_level*5*NBTHETAd*NBPHId*NLAMd;

        // count in that level
        atomicAdd(tabCount+(0 * NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi), weight * s1);
        atomicAdd(tabCount+(1 * NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi), weight * s2);
        atomicAdd(tabCount+(2 * NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi), weight * s3);
        atomicAdd(tabCount+(3 * NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi), weight * s4);
        atomicAdd(tabCount+(4 * NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi), 1.);
	}
	else
	{
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurcase), 1);
		#endif
	}

}



//
// Rotation of the stokes parameters by an angle psi between the incidence and
// the emergence planes
// input: 3 stokes parameters s1, s2, s3, (s4 does not need to be rotated)
//        rotation angle psi in radians
// output: 3 rotated stokes parameters s1r, s2r, s3r,
//
__device__ void rotateStokes(float s1, float s2, float s3, float psi,
        float *s1r, float *s2r, float *s3r)
{
    float cPsi = __cosf(psi);
    float sPsi = __sinf(psi);
    float cPsi2 = cPsi * cPsi;
    float sPsi2 = sPsi * sPsi;
    float twopsi = 2.F*psi;
    float a, s2Psi;
    s2Psi = __sinf(twopsi);
    a = 0.5f*s2Psi*s3;
    *s1r = cPsi2 * s1 + sPsi2 * s2 - a;
    *s2r = sPsi2 * s1 + cPsi2 * s2 + a;
    *s3r = s2Psi * (s1 - s2) + __cosf(twopsi) * s3;
}



/* calculPsi
* Calcul du psi pour la direction de sortie du photon
*/
__device__ void calculPsi(Photon* photon, float* psi, float theta)
{
	float sign;
// 	if (theta >= 0.05F)
// 	{
		*psi = acosf(fmin(1.F, fmax(-1.F, __fdividef(0.f * photon->ux + 1.f * photon->uz, __sinf(theta)))));
// 	}
// 	else
// 	{
// 		*psi = acosf(fmin(1.F - VALMIN, fmax(-(1.F - VALMIN), - 1.f * photon->ux + 0.f * photon->uz)));
// 	}
	
	sign = 0.f * (photon->uy * photon->vz - photon->uz * photon->vy) + 1.f * (photon->ux * photon->vy - photon->uy * photon->vx);
	if (sign < 0.F) *psi = -(*psi);
}


/* calculCase
* Fonction qui calcule la position (ith, iphi) et l'indice spectral (il) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void calculCase(int* ith, int* iphi, int* il, Photon* photon
			#ifdef PROGRESSION
			, Variables* var
			#endif 
			)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->vx * photon->vx + photon->vy * photon->vy);

	// Calcul de la valeur de ithv
	// _rn correspond à round to the nearest integer
	*ith = __float2int_rd(__fdividef(acosf(fabsf(photon->vz)) * NBTHETAd, DEMIPI));
	//*ith = __float2int_rn(__fdividef(acosf(fabsf(photon->vz)) * NBTHETAd, DEMIPI));

	// Calcul de la valeur de il
    // DEV!!
    *il = photon->ilam;

	/* Si le photon ressort très près du zénith on ne peut plus calculer iphi,
	 on est à l'intersection de toutes les cases du haut */
	
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
		
		// Puis on place le photon dans l'autre demi-cercle selon vy, utile uniquement lorsque l'on travail sur tous l'espace
// 		if(photon->vy < 0.F) *iphi = NBPHId - 1 - *iphi;
		#ifdef PROGRESSION
		// Lorsque vy=0 on décide par défaut que le photon reste du côté vy>0
		if(photon->vy == 0.F) atomicAdd(&(var->erreurvy), 1);
		#endif
	}
	
	else{
		// Photon très près du zenith
		#ifdef PROGRESSION
		atomicAdd(&(var->erreurvxy), 1);
		#endif
// 		/*if(photon->vy < 0.F) *iphi = NBPHId - 1;
// 		else*/ *iphi = 0;
		if(photon->vy >= 0.F)  *iphi = 0;
		else *iphi = NBPHId - 1;
	}
	
}



/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/

#ifdef RANDCUDA
/* initRandCUDA
* Fonction qui initialise les generateurs du random cuda
*/
__global__ void initRandCUDA(curandState_t* etat, unsigned long long seed)
{
	// Pour chaque thread on initialise son generateur avec le meme seed mais un idx different
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
	curand_init(seed, idx, 0, etat+idx);
}
#endif
#if defined(RANDCURANDSOBOL32) || defined (RANDCURANDSCRAMBLEDSOBOL32)
/* initRandCUDANDQRNGs
* Fonction qui initialise le generateur (scrambled) sobol 32 de curand
*/
__global__ void initRandCUDANDQRNGs
(
    curandSTATE* etat,
    curandDirectionVectors32_t *rngDirections
)
{
    // Pour chaque thread on initialise son generateur avec le meme seed mais un idx different
    unsigned int gID = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x));
    curand_init(
        //seule 20000 dimensions sont disponibles... le % permet de ne pas planter ici en segfault, mais...
        //...attention a la pertinence des resultats ici, si on depasse les 20000 threads !
        rngDirections[gID % 20000],
        #ifdef RANDCURANDSCRAMBLEDSOBOL32
        3, //aucune indication sur la pertinence de cette valeur...
        #endif
        /*0*/gID,
        etat+gID
               );
}
#endif


#ifdef RANDMT
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
#endif


#ifdef RANDMWC
/* randomMWCfloat
* Fonction random MWC qui renvoit un float de ]0.1] à partir d'un generateur (x+a)
*/
__device__ float randomMWCfloat(unsigned long long* x,unsigned int* a)
{
	//Generate a random number (0,1]
	*x=(*x&0xffffffffull)*(*a)+(*x>>32);
	return __fdividef(__uint2float_rz((unsigned int)(*x)) + 1.0f,(float)0x100000000);
}

#endif

#ifdef RANDPHILOX4x32_7
/* initPhilox4x32_7Compteur
* Fonction qui initialise la partie variable du compteur des philox
*/
__global__ void initPhilox4x32_7Compteur(unsigned int* tab, unsigned int compteurInit)
{
    unsigned int gID = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + blockIdx.y * gridDim.x));

    tab[gID] = compteurInit;
}

/* randomPhilox4x32_7float
* Fonction random Philox-4x32-7 qui renvoit un float dans ]0;1]
*/
__device__ float randomPhilox4x32_7float(philox4x32_ctr_t* ctr, philox4x32_key_t* key)
{
    //Recuperation d'un unsigned int pour retourner un float dans ]0;1]
    return __fdividef(__uint2float_rz(randomPhilox4x32_7uint(ctr, key)) + 1.0f, 4294967296.0f);
}

/* randomPhilox4x32_7uint
* Fonction random Philox-4x32-7 qui renvoit un uint à partir d'un generateur (etat+config)
* TODO A noter que 4 valeurs sont en fait generees, un seul uint peut etre renvoye, donc 3 sont perdus
* En pratique les valeurs generees sont des int32. Il y a donc une conversion vers uint32 de realisee
*/
__device__ unsigned int randomPhilox4x32_7uint(philox4x32_ctr_t* ctr, philox4x32_key_t* key)
{
    //variable de retour
    philox4x32_ctr_t res;
    //generation de 4 int32
    res = philox4x32_R(7, *ctr, *key);
    //increment du premier mot de 32bits du compteurs
    (*ctr).v[0]++;
    //conversion d'un des mots generes sous forme d'unsigned int
    return (unsigned int) res[0];
}
#endif

extern "C" {
    __global__ void lancementKernelPy(Variables* var, Tableaux *tab
	, Init* init
    ) {
        launchKernel(var, *tab,init);
    }
}
