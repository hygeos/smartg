
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

#include <helper_math.h>
#include <stdio.h>
/**********************************************************
*	> Kernel
***********************************************************/


extern "C" {
	__global__ void launchKernel(
							 struct Spectrum *spectrum, float *X0,
							 struct Phase *faer, struct Phase *foce,
							 unsigned long long *errorcount, int *nThreadsActive, void *tabPhotons,
							 unsigned long long *Counter,
							 unsigned long long *NPhotonsIn,
							 unsigned long long *NPhotonsOut,
							 float *tabthv, float *tabphi,
							 struct Profile *prof_atm,
							 struct Profile *prof_oc,
							 long long *wl_proba_icdf,
							 unsigned int *philox_data
							 ) {

    // current thread index
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
	int loc_prev;
	int count_level;
	int this_thread_active = 1;
	unsigned long long iloop = 0;

	// philox_data:
	// index 0: seed (config)
	// index 1 to last: status

	// Paramètres de la fonction random en mémoire locale
	//la clef se defini par l'identifiant global (unique) du thread...
	//...et par la clef utilisateur ou clef par defaut
	//ce systeme garanti l'existence de 2^32 generateurs differents par run et...
	//...la possiblite de reemployer les memes sequences a partir de la meme clef utilisateur
	//(plus d'infos dans "communs.h")
	philox4x32_key_t configThr = {{idx, philox_data[0]}};
	//le compteur se defini par trois mots choisis au hasard (il parait)...
	//...et un compteur definissant le nombre d'appel au generateur
	//ce systeme garanti l'existence de 2^32 nombres distincts pouvant etre genere par thread,...
	//...et ce sur l'ensemble du process (et non pas 2^32 par thread par appel au kernel)
	//(plus d'infos dans "communs.h")
	philox4x32_ctr_t etatThr = {{philox_data[idx+1], 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

	
	// Création de variable propres à chaque thread
	unsigned long long nbPhotonsThr = 0; 	// Nombre de photons traités par le thread
	

	Photon ph, ph_le; 		// On associe une structure de photon au thread
	//Photon ph_le2

	ph.loc = NONE;	// Initialement le photon n'est nulle part, il doit être initialisé

	atomicAdd(nThreadsActive, 1);

    //
    // main loop
    //
	while (*nThreadsActive > 0) {
		iloop += 1;

        if (((Counter[0] > NBLOOPd)
			 && this_thread_active
			 && (ph.loc == NONE))
			|| (iloop > MAX_LOOP)  // avoid infinite loop
                                       // when photons don't end
			) {
            this_thread_active = 0;
            atomicAdd(nThreadsActive, -1);
        }

        // Si le photon est à NONE on l'initialise et on le met à la localisation correspondant à la simulaiton en cours
        if((ph.loc == NONE) && this_thread_active){

            initPhoton(&ph, prof_atm, spectrum, X0, NPhotonsIn, wl_proba_icdf,
                       &etatThr , &configThr);
            iloop = 1;
            #ifdef DEBUG_PHOTON
            display("INIT", &ph);
            #endif

        }


			//
			// Deplacement
			//
			// -> Si OCEAN ou ATMOS
			loc_prev = ph.loc;
			if( (ph.loc == ATMOS) || (ph.loc == OCEAN)){

            #ifdef SPHERIQUE
            if (ph.loc == ATMOS)
                move_sp(&ph, prof_atm, 0, 0 , &etatThr , &configThr);
            else 
            #endif
                move_pp(&ph, prof_atm, prof_oc, &etatThr , &configThr);
            #ifdef DEBUG_PHOTON
            display("MOVE", &ph);
            #endif
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

            // reset the photon location (always)
            ph.loc = NONE;
            #ifdef DEBUG_PHOTON
            display("SPACE", &ph);
            #endif

        } else if ((ph.loc == SURF0P) && (loc_prev != SURF0P)) {
            count_level = DOWN0P;
        } else if ((ph.loc == SURF0M) && (loc_prev != SURF0M)) {
            count_level = UP0M;
        }

			// count the photons
        
			/* Cone Sampling */
			if (LEd ==0) countPhoton(&ph, prof_atm, tabthv, tabphi, count_level,
                errorcount, tabPhotons, NPhotonsOut);


			syncthreads();

		
			//
			// Scatter
			//
			// -> dans ATMOS ou OCEAN
			if( (ph.loc == ATMOS) || (ph.loc == OCEAN)) {

            /* Scattering Local Estimate */
            if (LEd == 1) {
			int NK, up_level, down_level, count_level_le;
			int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
			int iph0 = idx%NBPHId;
			if (ph.loc == ATMOS) {
			NK=2;
			up_level = UPTOA;
			down_level = DOWN0P;
		}
			if (ph.loc == OCEAN) {
			NK=1;
			up_level = UP0M;
		}
			for(int k=0; k<NK; k++){
			if (k==0) count_level_le = up_level;
			else count_level_le = down_level;

                    for (int iph=0; iph<NBPHId; iph++){
                        for (int ith=0; ith<NBTHETAd; ith++){
                            copyPhoton(&ph, &ph_le);
                            ph_le.iph = (iph + iph0)%NBPHId;
                            ph_le.ith = (ith + ith0)%NBTHETAd;
                            scatter(&ph_le, prof_atm, prof_oc, faer, foce,
                                    1, tabthv, tabphi,
                                    count_level_le, &etatThr , &configThr);

                            #ifdef DEBUG_PHOTON
                            if (k==0) display("SCATTER LE UP", &ph_le);
                            else display("SCATTER LE DOWN", &ph_le);
                            #endif

                            #ifdef SPHERIQUE
                            if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, count_level_le , &etatThr , &configThr);
                            #ifdef DEBUG_PHOTON
                            display("MOVE LE", &ph_le);
                            #endif
                            #endif

                            countPhoton(&ph_le, prof_atm, tabthv, tabphi, count_level_le,
                                    errorcount, tabPhotons, NPhotonsOut);
                        }
                    }
                }
            }

            /* TEST DOUBLE LOCAL ESTIMATE IN OCEAN */
            // Scattering Double Local Estimate in Ocean in case of dioptre 
            /*if (LEd == 1 && ph.loc==OCEAN && SIMd != -2) {
                int NK, up_level, down_level, count_level_le;
                int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                int iph0 = idx%NBPHId;
                copyPhoton(&ph, &ph_le);
                scatter(&ph_le, prof_atm, prof_oc, faer, foce,
                            1, tabthv, tabphi,
                            UP0M2, &etatThr , &configThr);
                ph_le.weight *= expf(-fabs(ph_le.tau/ph_le.vz));
                ph_le.loc=SURF0M;
                ph_le.tau=0.F;
                NK=2;
                up_level = UP0P;
                down_level = DOWN0M;
                for(int k=0; k<NK; k++){
                    if (k==0) count_level_le = up_level;
                    else count_level_le = down_level;

                    for (int iph=0; iph<NBPHId; iph++){
                        for (int ith=0; ith<NBTHETAd; ith++){
                            copyPhoton(&ph_le, &ph_le2);
                            ph_le2.iph = (iph + iph0)%NBPHId;
                            ph_le2.ith = (ith + ith0)%NBTHETAd;

                            surfaceAgitee(&ph_le2, 1, tabthv, tabphi,
                                count_level_le, &etatThr , &configThr);

                            #ifdef DEBUG_PHOTON
                            if (k==0) display("SURFACE LE2 UP", &ph_le2);
                            else display("SURFACE LE2 DOWN", &ph_le2);
                            #endif
                            countPhoton(&ph_le2, prof_atm, tabthv, tabphi, count_level_le, errorcount, tabPhotons, NPhotonsOut);

                            if (k==0) { 
                             countPhoton(&ph_le2, prof_atm, tabthv, tabphi, UPTOA , errorcount, tabPhotons, NPhotonsOut);
                            }

                        }
                    }
                }
            }*/  //Double LE

            /* Scattering Propagation */
            scatter(&ph, prof_atm, prof_oc, faer, foce,
                    0, tabthv, tabphi, 0,
                    &etatThr , &configThr);
            #ifdef DEBUG_PHOTON
            display("SCATTER", &ph);
            #endif

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
			if( DIOPTREd!=3 ) {
                /* Surface Local Estimate */
                if (LEd == 1 && SIMd != -2) {
                /* TEST Double LE */
                //if ((LEd == 1) && (SIMd != -2 && ph.loc == SURF0P)) {
                  int NK=2, count_level_le;
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;
                  for(int k=0; k<NK; k++){
                    if (k==0) count_level_le = UP0P;
                    else count_level_le = DOWN0M;

                for (int ith=0; ith<NBTHETAd; ith++){
                      for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.iph = (iph + iph0)%NBPHId;
                        ph_le.ith = (ith + ith0)%NBTHETAd;

                        surfaceAgitee(&ph_le, 1, tabthv, tabphi,
                                count_level_le, &etatThr , &configThr);

                        #ifdef DEBUG_PHOTON
                        if (k==0) display("SURFACE LE UP", &ph_le);
                        else display("SURFACE LE DOWN", &ph_le);
                        #endif

                        countPhoton(&ph_le, prof_atm, tabthv, tabphi, count_level_le, errorcount, tabPhotons, NPhotonsOut);
                        if (k==0) { 
                            #ifdef SPHERIQUE
                            move_sp(&ph_le, prof_atm, 1, UPTOA , &etatThr , &configThr);
                            #endif
                            countPhoton(&ph_le, prof_atm, tabthv, tabphi, UPTOA , errorcount, tabPhotons, NPhotonsOut);
                        }
                      }
                    }
                  }
                }
				surfaceAgitee(&ph, 0, tabthv, tabphi,
                        count_level, &etatThr , &configThr);
            }

			else { 
                if (LEd == 1 && SIMd != -2) {
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;
                  for (int ith=0; ith<NBTHETAd; ith++){
                    for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.iph = (iph + iph0)%NBPHId;
                        ph_le.ith = (ith + ith0)%NBTHETAd;
				        surfaceLambertienne(&ph_le, 1, tabthv, tabphi, spectrum, &etatThr , &configThr);
                        countPhoton(&ph_le, prof_atm, tabthv, tabphi, UP0P,  errorcount, tabPhotons, NPhotonsOut);
                        countPhoton(&ph_le, prof_atm, tabthv, tabphi, UPTOA, errorcount, tabPhotons, NPhotonsOut);
                    }
                  }
                }
				surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &etatThr , &configThr);
            } // DIOPTRE=!3
           } // ENV=0

           else {
                float dis=0;
                dis = sqrtf((ph.pos.x-X0d)*(ph.pos.x-X0d) +(ph.pos.y-Y0d)*(ph.pos.y-Y0d));
                if( dis > ENV_SIZEd) {
                    surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &etatThr , &configThr);
                }
                else {
                    surfaceAgitee(&ph, 0, tabthv, tabphi, count_level, &etatThr , &configThr);
                }
           } // ENV=1
            #ifdef DEBUG_PHOTON
             display("SURFACE", &ph);
            #endif
		}
		syncthreads();

        //
		// Reflection
        //
        // -> in SEAFLOOR
        if(ph.loc == SEAFLOOR){
           if (LEd == 1 && SIMd != -2) {
              int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
              int iph0 = idx%NBPHId;
              for (int ith=0; ith<NBTHETAd; ith++){
                for (int iph=0; iph<NBPHId; iph++){
                    copyPhoton(&ph, &ph_le);
                    ph_le.iph = (iph + iph0)%NBPHId;
                    ph_le.ith = (ith + ith0)%NBTHETAd;
				    surfaceLambertienne(&ph_le, 1, tabthv, tabphi, spectrum, &etatThr , &configThr);
                    countPhoton(&ph_le, prof_atm, tabthv, tabphi, UP0M,  errorcount, tabPhotons, NPhotonsOut);
                }
              }
            }
			surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &etatThr , &configThr);
            #ifdef DEBUG_PHOTON
            display("SEAFLOOR", &ph);
            #endif
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
        
        /* Cone Sampling */
        if (LEd == 0) countPhoton(&ph, prof_atm, tabthv, tabphi, count_level, errorcount, tabPhotons, NPhotonsOut);



		if(ph.loc == ABSORBED){
			ph.loc = NONE;
			nbPhotonsThr++;
		}
		syncthreads();

		

        // from time to time, transfer the per-thread photon counter to the
        // global counter
        if (nbPhotonsThr % 100 == 0) {
            atomicAdd(Counter, nbPhotonsThr);
            nbPhotonsThr = 0;
        }

	}


	// Après la boucle on rassemble les nombres de photons traités par chaque thread

	atomicAdd(Counter, nbPhotonsThr);

    if (ph.loc != NONE) {
        atomicAdd(errorcount+ERROR_MAX_LOOP, 1);
    }

	// Sauvegarde de l'état du random pour que les nombres ne soient pas identiques à chaque appel du kernel
    philox_data[idx+1] = etatThr[0];

}
}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph, struct Profile *prof_atm,
                           struct Spectrum *spectrum, float *X0, unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf,
                           philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr)
{
	// Initialisation du vecteur vitesse
	ph->v.x = - STHVd;
	ph->v.y = 0.F;
	ph->v.z = - CTHVd;

	
	// Initialisation du vecteur orthogonal au vecteur vitesse
	ph->u.x = -ph->v.z;
	ph->u.y = 0.F;
	ph->u.z = ph->v.x;
	
    // Initialisation de la longueur d onde
     //mono chromatique
    if (NWLPROBA == 0) {
        ph->ilam = __float2uint_rz(RAND * NLAMd);
    } else {
        ph->ilam = wl_proba_icdf[__float2uint_rz(RAND * NWLPROBA)];
    }

	ph->wavel = spectrum[ph->ilam].lambda;
    atomicAdd(NPhotonsIn+ph->ilam, 1);

    if ((SIMd == -2) || (SIMd == 1) || (SIMd == 2)) {

        //
        // Initialisation du photon au sommet de l'atmosphère
        //

        ph->pos.x = X0[0];
        ph->pos.y = X0[1];
        ph->pos.z = X0[2];
        ph->couche = 0;   // top of atmosphere

        #ifdef SPHERIQUE
		ph->rayon = length(ph->pos);
        #endif

        // !! DEV on ne calucle pas d ep optique ici
        ph->loc = ATMOS;
        ph->tau = prof_atm[NATMd + ph->ilam*(NATMd+1)].tau;

    } else if ((SIMd == -1) || (SIMd == 0) || (SIMd == 3)) {

        //
        // Initialisation du photon à la surface ou dans l'océan
        //
        ph->pos = make_float3(0.,0.,0.);
        #ifdef SPHERIQUE
        ph->pos.z = RTER;
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
	ph->stokes.x = 0.5F;
	ph->stokes.y = 0.5F;
	ph->stokes.z = 0.F;
	ph->stokes.w = 0.F;

}



#ifdef SPHERIQUE
__device__ void move_sp(Photon* ph, struct Profile *prof_atm, int le, int count_level , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr) {

    float tauRdm;
    float hph = 0.;  // cumulative optical thickness
    float vzn, delta1, h_cur;
    float d_tot = 0.;
    float d;
    float rat;
    int sign_direction;
    int i_layer_fw, i_layer_bh; // index or layers forward and behind the photon
    float costh, sinth2;
    int ilam = ph->ilam*(NATMd+1);  // wavelength offset in optical thickness table

    if (ph->couche == 0) ph->couche = 1;

    #ifdef DEBUG
    int niter = 0;
    // ph->couche is indexed
    // from 1 (TOA layer between interfaces 0 and 1)
    // to NATM (bottom layer between interfaces NATM-1 to NATM)
    if ((ph->couche > NATMd) || (ph->couche <= 0)) {
        printf("Fatal error, wrong index (%d)\n", ph->couche);
    }
    #endif

    // Random Optical Thickness to go through
    if (!le) tauRdm = -logf(1.F-RAND);
    // in LE photon is forced to exit upward or downward
    else tauRdm = 1e6;
    

    vzn = __fdividef( dot(ph->v, ph->pos), ph->rayon);
    #ifndef ALT_MOVE
    costh = vzn;
    sinth2 = 1.f-costh*costh;
    #endif

    // a priori value for sign_direction:
    // sign_direction may change sign from -1 to +1 if the photon does not
    // cross lower layer
    if (vzn <= 0) sign_direction = -1;
    else sign_direction = 1;

    while (1) {

        #ifdef DEBUG
        niter++;

        if (niter > 2*NATMd+1) {
            printf("niter=%d break\n", niter);
            break;
        }
        #endif

        //
        // stopping criteria
        //
        if (ph->couche == NATMd+1) {
            ph->loc = SURF0P;
            ph->tau = 0.;
            ph->couche -= 1;  // next time photon enters move_sp, it's at layers NATM
            #ifdef DEBUG
            if (dot(ph->v, ph->pos) > 0) {
                printf("Warning, vzn > 0 at SURF0P in move_sp (vzn=%f)\n", vzn);
            }
            #endif
            break;
        }
        if (ph->couche <= 0) {
            ph->loc = SPACE;
            break;
        }

        //
        // determine the index of the next potential layer
        //
        if (sign_direction < 0) {
            // photon goes down
            // (towards higher indices)
            i_layer_fw = ph->couche;
            i_layer_bh = ph->couche - 1;
        } else {
            // photon goes up
            // (towards lower indices)
            i_layer_fw = ph->couche - 1;
            i_layer_bh = ph->couche;
        }

        #ifdef ALT_MOVE
        // initializations
        costh = vzn;
        sinth2 = 1.f-costh*costh;
        #endif

        //
        // calculate the distance d to the fw layer
        #ifndef ALT_MOVE
        // from the initial position
        #else
        // from the current position
        #endif
        //
        // ri : radius of next layer boundary ri=zi+RTER
        // r  : radius of current point along the path 
        // costh: angle between the position vector and the direction vector
        // In the triangle we have ri² = d² + r² + 2*d*r*costh
        // or: d**2 + 2*r*costh*d + r**2-ri**2 = 0 , to be solved for d
        // delta = 4.r².costh² - 4(r²-ri²) = 4*r²*((ri/r)²-sinth²) = 4*r²*delta1
        // with delta1 = (ri/r)²-sinth²
        rat = (prof_atm[i_layer_fw].z+RTER)/ph->rayon;
        delta1 = rat*rat - sinth2;   // same sign as delta

        if (delta1 < 0) {
            if (sign_direction > 0) {
                #ifdef DEBUG
                printf("Warning sign_direction (niter=%d, lay=%d, delta1=%f, alt=%f zlay1=%f zlay2=%f vzn=%f)\n",
                        niter, ph->couche, delta1, ph->rayon-RTER,
                        prof_atm[i_layer_fw].z,
                        prof_atm[i_layer_bh].z,
                        vzn);
                #endif

                // because of numerical uncertainties, a downward photon may
                // not strictly be between zi and zi+1
                // in rare case of grazing angle there is sometimes no intersection
                // with current layer because photon is actually slightly above it.
                // therefore we consider that delta=0 such that the photon is
                // tangent to the layer
                delta1 = 0.;
            } else {
                // no intersection, with lower layer, we should go towards higher layer
                sign_direction = 1;
                continue;
            }
        }

        /* Now, there are two real solutions for d
        *  The solution is the smallest positive one
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
        */
        /* d = 0.5f*(-2.*ph->rayon*costh + sign_direction*2*ph->rayon*sqrtf(delta1)); simplified to: */
        d = ph->rayon*(-costh + sign_direction*sqrtf(delta1));
        #ifdef DEBUG
        if (d < 0) {
            #ifndef ALT_MOVE
            printf("Warning in move_sp (d=%f < 0 ; vzn=%f, sqrt(delta1)=%f)\n",
                d, vzn, sqrtf(delta1));
            #else
            printf("(alt_move) Warning in move_sp (d=%f < 0 ; vzn=%f, sqrt(delta1)=%f)\n",
                d, vzn, sqrtf(delta1));
            #endif
        } else if (d_tot > d) {
            printf("Error in move_sp (d_tot=%f > d=%f)\n", d_tot, d);
        }
        #endif


        //
        // calculate the optical thickness h_cur to the next layer
        // We compute the layer extinction coefficient of the layer DTau/Dz and multiply by the distance within the layer
        //
        #ifndef ALT_MOVE
        h_cur = __fdividef(abs(prof_atm[i_layer_bh+ilam].tau - prof_atm[i_layer_fw+ilam].tau)*(d - d_tot),
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        #else
        h_cur = __fdividef(abs(prof_atm[i_layer_bh+ilam].tau - prof_atm[i_layer_fw+ilam].tau)*d,
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        #endif


        //
        // update photon position
        //
        if (hph + h_cur > tauRdm) {
            // photon stops within the layer
            #ifndef ALT_MOVE
            d_tot += (d - d_tot)*(tauRdm - hph)/h_cur;
            #else
            d *= (tauRdm-hph)/h_cur;
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->rayon = length(ph->pos);
            ph->weight *= 1.f - prof_atm[ph->couche+ilam].abs;
            ph->prop_aer = 1.f - prof_atm[ph->couche+ilam].pmol;

            #ifdef DEBUG
            vzn = __fdividef( dot(ph->v, ph->pos) , ph->rayon);
            #endif
            #endif

            break;
        } else {
            // photon advances to the next layer
            hph += h_cur;
            ph->couche -= sign_direction;
            #ifndef ALT_MOVE
            d_tot = d;
            #else
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->rayon = length(ph->pos);
            vzn = __fdividef( dot(ph->v, ph->pos) , ph->rayon);
            #endif
        }

    }
    if (le) {
        if (( (count_level==UPTOA)  && (ph->loc==SPACE ) ) || ( (count_level==DOWN0P) && (ph->loc==SURF0P) )) ph->weight *= __expf(-(hph + h_cur));
        else ph->weight = 0.;
    }
    #ifndef ALT_MOVE
    //
    // update the position of the photon
    //
    ph->pos = operator+(ph->pos, ph->v*d_tot);
    ph->rayon = length(ph->pos);
    ph->weight *= 1.f - prof_atm[ph->couche+ilam].abs;
    ph->prop_aer = 1.f - prof_atm[ph->couche+ilam].pmol;
    #endif
}
#endif // SPHERIQUE


__device__ void move_pp(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
        philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr) {

	float Dsca=0.f, dsca=0.f;

	ph->tau += -logf(1.f - RAND)*ph->v.z;

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
        else if( ph->tau < prof_oc[NOCEd + ph->ilam *(NOCEd+1)].tau ){
            ph->loc = SEAFLOOR;
            ph->tau = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].tau;
            return;
        }

        // Calcul de la couche dans laquelle se trouve le photon
        tauBis = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].tau - ph->tau;
        icouche = 1;

        while ((prof_oc[icouche+ ph->ilam *(NOCEd+1)].tau > (tauBis)) && (icouche < NOCEd)) {
            icouche++;
        }
        ph->couche = icouche;

    }

    if (ph->loc == ATMOS) {

        float phz,rdist;

        // Si tau<0 le photon atteint la surface
        if(ph->tau < 0.F){
            ph->loc = SURF0P;
            ph->tau = 0.F;

            // move the photon forward down to the surface
            // the linear distance is ph->z/ph->vz
            operator+=(ph->pos, ph->v * fabs(ph->pos.z/ph->v.z));
            ph->pos.z = 0.;
        return;
        }
        // Si tau>TAUATM le photon atteint l'espace
        else if( ph->tau > prof_atm[NATMd + ph->ilam *(NATMd+1)].tau ){
            ph->loc = SPACE;
            return;
        }
        
        // Sinon il reste dans l'atmosphère, et va subit une nouvelle diffusion
        
        // Calcul de la couche dans laquelle se trouve le photon
        tauBis =  prof_atm[NATMd + ph->ilam *(NATMd+1)].tau - ph->tau;
        icouche = 1;
        
        while ((prof_atm[icouche+ ph->ilam *(NATMd+1)].tau < (tauBis)) && (icouche < NATMd)) {
            icouche++;
        }
        
        ph->couche = icouche;
        ph->prop_aer = 1.f - prof_atm[ph->couche+ph->ilam*(NATMd+1)].pmol;
        ph->weight = ph->weight * (1.f - prof_atm[ph->couche+ph->ilam*(NATMd+1)].abs);

        Dsca= fabs(prof_atm[icouche+ph->ilam*(NATMd+1)].tau - prof_atm[icouche-1+ph->ilam*(NATMd+1)].tau);
        dsca= fabs(tauBis - prof_atm[icouche-1+ph->ilam*(NATMd+1)].tau) ;

        // calculate new photon position
        phz = __fdividef(dsca,Dsca) * (prof_atm[icouche].z - prof_atm[icouche-1].z) + prof_atm[icouche-1].z; 
        rdist=  fabs(__fdividef(phz-ph->pos.z, ph->v.z));
        operator+= (ph->pos, ph->v*rdist);
        ph->pos.z = phz;

    }

}


__device__ void scatter(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
        struct Phase *faer, struct Phase *foce,
        int le,
        float* tabthv, float* tabphi, int count_level,
        philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr){

	float cTh=0.f ;
	float zang=0.f, theta=0.f;
	int iang, ilay, ipha;
	float4 stokes;
	float cTh2, psi, sign;
	float prop_aer = ph->prop_aer;
	
    if (le){
        /* in case of LE the photon units vectors, scattering angle and Psi rotation angle are determined by output zenith and azimuth angles*/
        float thv, phi;
        float3 v;

        if (count_level==UP0M2) { /* In case of Double Local Estimate, the first direction is chosen randomly */
			phi = RAND * DEUXPI;
            thv = RAND * DEMIPI;
            sign = 1.0F;
        }
        else {
            if (count_level==DOWN0P) sign = -1.0F;
            else sign = 1.0F;
            phi = tabphi[ph->iph];
            thv = tabthv[ph->ith];
        }
        v.x = __cosf(phi) * __sinf(thv);
        v.y = __sinf(phi) * __sinf(thv);
        v.z = sign * __cosf(thv);
        theta = ComputeTheta(ph->v, v);
        cTh = __cosf(theta);
		if (cTh < -1.0) cTh = -1.0;
		if (cTh >  1.0) cTh =  1.0;
        cTh2 = cTh * cTh;
        ComputePsiLE(ph->u, ph->v, v, &psi, &ph->u); 
        ph->v = v;
    }

    /* Scattering in atmosphere */
	if(ph->loc!=OCEAN){
		/************************************/
		/* Rayleigh and Aerosols scattering */
		/************************************/
        ilay = ph->couche + ph->ilam*(NATMd+1); // atm layer index
		/* atm phase function index */
		if( prop_aer < RAND ){ipha  = 0;} // Rayleigh index
		else {ipha  = prof_atm[ilay].iphase + 1;} // Aerosols index

		float P11, P12, P22, P33, P43, P44;
		if(!le) {
			/* in the case of propagation (not LE) the photons scattering angle and Psi rotation angle are determined randomly*/
			/////////////
			// Get Theta from Cumulative Distribution Function
			zang = RAND*(NF-1);
			iang= __float2int_rd(zang);
			zang = zang - iang;

			theta = (1.-zang)*faer[ipha*NF+iang].p_ang + zang*faer[ipha*NF+iang+1].p_ang;
			cTh = __cosf(theta);

			/////////////
			//  Get Phi
			//  Biased sampling scheme for psi 1)
			psi = RAND * DEUXPI;	

			/////////////
			// Get Scattering matrix from CDF
			P11 = faer[ipha*NF+iang].p_P11;
			P12 = faer[ipha*NF+iang].p_P12;
			P22 = faer[ipha*NF+iang].p_P22;
			P33 = faer[ipha*NF+iang].p_P33;
			P43 = faer[ipha*NF+iang].p_P43;
			P44 = faer[ipha*NF+iang].p_P44;

			// int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
			// if (P12 != 0){
			// if (idx == 0)
			// 	printf("P11 = %.3f, P12 = %.3f, P22 = %.3f, P33 = %.3f, P43 = %.3f, P44 = %.3f\n", P11, P12, P22, P33, P43, P44);
			// }
		}
		else {
			/////////////
			// Get Index of scattering angle and Scattering matrix directly 
			zang = theta * NF/PI ;
			iang = __float2int_rd(zang);
			zang = zang - iang;

			if (abs(cTh) < 1) {
				P11 = (1-zang)*faer[ipha*NF+iang].a_P11 + zang*faer[ipha*NF+iang+1].a_P11;
				P12 = (1-zang)*faer[ipha*NF+iang].a_P12 + zang*faer[ipha*NF+iang+1].a_P12;
				P22 = (1-zang)*faer[ipha*NF+iang].a_P22 + zang*faer[ipha*NF+iang+1].a_P22;
				P33 = (1-zang)*faer[ipha*NF+iang].a_P33 + zang*faer[ipha*NF+iang+1].a_P33;
				P43 = (1-zang)*faer[ipha*NF+iang].a_P43 + zang*faer[ipha*NF+iang+1].a_P43;
				P44 = (1-zang)*faer[ipha*NF+iang].a_P44 + zang*faer[ipha*NF+iang+1].a_P44;
			}
			else if (cTh >=1) {
				P11 = faer[ipha*NF].a_P11;
				P12 = faer[ipha*NF].a_P12;
				P22 = faer[ipha*NF].a_P22;
				P33 = faer[ipha*NF].a_P33;
				P43 = faer[ipha*NF].a_P43;
				P44 = faer[ipha*NF].a_P44;
			}
			else {
				P11 = faer[ipha*NF+(NF-1)].a_P11;
				P12 = faer[ipha*NF+(NF-1)].a_P12;
				P22 = faer[ipha*NF+(NF-1)].a_P22;
				P33 = faer[ipha*NF+(NF-1)].a_P33;
				P43 = faer[ipha*NF+(NF-1)].a_P43;
				P44 = faer[ipha*NF+(NF-1)].a_P44;
			}
		}

		// Stokes vector rotation
		rotateStokes(ph->stokes, psi, &ph->stokes);

		// Scattering matrix multiplication
		float4x4 P_scatter = make_float4x4(
			P11, P12, 0., 0.,
			P12, P22, 0., 0.,
			0., 0., P33, -P43,
			0., 0., P43, P44
			);

		ph->stokes = mul(P_scatter, ph->stokes);

		// stokes=ph->stokes;

		// ph->stokes.x = stokes.x * P11 + stokes.y * P12;
		// ph->stokes.y = stokes.y * P22 + stokes.x * P12;
		// ph->stokes.z = stokes.z * P33 - stokes.w * P43;
		// ph->stokes.w = stokes.w * P33 + stokes.z * P44;

		if (!le){
			// Bias sampling scheme 2): Debiasing
			float debias;
			debias = __fdividef( 2., P11 + P22 + 2*P12 );
			operator*=(ph->stokes, debias); 
		}

		else ph->weight /= 4.F; //Phase function normalization

		if( prop_aer >= RAND ){
			// Photon weight reduction due to the aerosol single scattering albedo of the current layer
			ph->weight *= prof_atm[ilay].ssa;
		}
	}
	else{	/* Photon dans l'océan */
	    float prop_raman=1., new_wavel;
        ilay = ph->couche + ph->ilam*(NOCEd+1); // oce layer index
        ipha  = prof_oc[ilay].iphase + 1; // oce phase function index

        // we fix the proportion of Raman to 2% at 488 nm, !! DEV
        //prop_raman = 0.02 * pow ((1.e7/ph->wavel-3400.)/(1.e7/488.-3400.),5); // Raman scattering to pure water scattering ratio
	    if(prop_raman <RAND ){
            /***********************/
            /* Raman scattering    */
            /* Phase function      */
            /* similar to Rayleigh */
            /***********************/
            if(!le) {
                /* in the case of propagation (not LE) the photons scattering angle and Psi rotation angle are determined randomly*/
			    /////////////
			    // Get Theta (see Wang et al., 2012)
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
			    //  Biased sampling scheme for psi 1)
			    psi = RAND * DEUXPI;
            }

			// Stokes vector rotation
			rotateStokes(ph->stokes, psi, &ph->stokes );

			// Scattering matrix multiplication
			float cross_term;
			stokes.x = ph->stokes.x;
			stokes.y = ph->stokes.y;
			cross_term  = DELTA_PRIMd * (ph->stokes.x + ph->stokes.y);
			ph->stokes.x = 3./2. * (  DELTAd  * stokes.x + cross_term );
			ph->stokes.y = 3./2. * (  DELTAd  * cTh2 * stokes.y + cross_term );			
			ph->stokes.z = 3./2. * (  DELTAd  * cTh  * ph->stokes.z );
			ph->stokes.w = 3./2. * (  DELTAd  * DELTA_SECOd * cTh * ph->stokes.w );

            if (!le){
			    // Bias sampling scheme 2): Debiasing
			    float phase_func;
			    phase_func = 3./4. * DELTAd * (cTh2+1.0) + 3.0 * DELTA_PRIMd;
				operator/=(ph->stokes, phase_func);    		
            }

            else ph->weight /= 4.F; //Phase function normalization

            /* Wavelength change */
            new_wavel  = 22.94 + 0.83 * (ph->wavel) + 0.0007 * (ph->wavel)*(ph->wavel);
            ph->weight /= new_wavel/ph->wavel;
            ph->wavel = new_wavel;
		  }

	  else{
            /***********************/
            /* Elastic scattering */
            /***********************/
            float P11,P22,P33,P43;
            if(!le) {
                /* in the case of propagation (not LE) the photons scattering angle and Psi rotation angle are determined randomly*/
			    /////////////
                // Get Theta from Cumulative Distribution Function
			    zang = RAND*(NF-2);
			    iang= __float2int_rd(zang);
			    zang = zang - iang;
			    theta = (1.-zang)*foce[ipha*NF+iang].p_ang + zang*foce[ipha*NF+iang+1].p_ang;
			    cTh = __cosf(theta);

			    /////////////
			    //  Get Phi
			    //  Biased sampling scheme for psi 1)
			    psi = RAND * DEUXPI;	

                /////////////
                // Get Scattering matrix from CDF
                P11 = foce[ipha*NF+iang].p_P11;
                P22 = foce[ipha*NF+iang].p_P22;
                P33 = foce[ipha*NF+iang].p_P33;
                P43 = foce[ipha*NF+iang].p_P43;
            }

            else {
                /////////////
                // Get Index of scattering angle and Scattering matrix directly 
                zang = theta * (NF-1)/PI ;
                iang = __float2int_rd(zang);
			    zang = zang - iang;
                if (abs(cTh) < 1) {
                    P11 = (1.-zang)*foce[ipha*NF+iang].a_P11 + zang*foce[ipha*NF+iang+1].a_P11;
                    P22 = (1.-zang)*foce[ipha*NF+iang].a_P22 + zang*foce[ipha*NF+iang+1].a_P22;
                    P33 = (1.-zang)*foce[ipha*NF+iang].a_P33 + zang*foce[ipha*NF+iang+1].a_P33;
                    P43 = (1.-zang)*foce[ipha*NF+iang].a_P43 + zang*foce[ipha*NF+iang+1].a_P43;
                }
                else if (cTh >=1) {
                    P11 = foce[ipha*NF].a_P11;
                    P22 = foce[ipha*NF].a_P22;
                    P33 = foce[ipha*NF].a_P33;
                    P43 = foce[ipha*NF].a_P43;
                }
                else {
                    P11 = foce[ipha*NF+NF-1].a_P11;
                    P22 = foce[ipha*NF+NF-1].a_P22;
                    P33 = foce[ipha*NF+NF-1].a_P33;
                    P43 = foce[ipha*NF+NF-1].a_P43;
                }
            }

			// Stokes vector rotation
			rotateStokes(ph->stokes, psi, &ph->stokes);

			// Scattering matrix multiplication
            stokes.z=ph->stokes.z;
            stokes.w=ph->stokes.w;
			ph->stokes.x *= P11;
			ph->stokes.y *= P22;
			ph->stokes.z = stokes.z * P33 - stokes.w * P43;
			ph->stokes.w = stokes.w * P33 + stokes.z * P43;

            if (!le){
			    // Bias sampling scheme 2): Debiasing
			    float debias;
			    debias = __fdividef( 2., P11 + P22 );
				operator*=(ph->stokes, debias);
            }

            else ph->weight /= 4.F; //Phase function normalization

            // Photon weight reduction due to the aerosol single scattering albedo of the current layer
			ph->weight *= prof_oc[ilay].ssa;
			
		} /* elastic scattering*/

	/** Russian roulette for propagating photons **/
     if (!le) {
	  if( ph->weight < WEIGHTRR ){
		if( RAND < __fdividef(ph->weight,WEIGHTRR) ){
			ph->weight = WEIGHTRR;
		}
		else{
				ph->loc = ABSORBED;
			}
		}
     }
		
    } //photon in ocean

   ////////// Fin séparation ////////////
   
    if (!le){
        modifyUV( ph->v, ph->u, cTh, psi, &ph->v, &ph->u) ;
    }

}


__device__ void surfaceAgitee(Photon* ph, int le, float* tabthv, float* tabphi, int count_level , philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr) {
	
	if( SIMd == -2){ // Atmosphère , la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
	
	// Réflexion sur le dioptre agité
	float theta;	// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float cTh, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	
	float sig, sig2  ;
	float beta = 0.F;	// Angle par rapport à la verticale du vecteur normal à une facette de vagues 
	float sBeta;
	float cBeta;
	
	float alpha ;	//Angle azimutal du vecteur normal a une facette de vagues
	
	float nind; // relative index of refrection 
	float temp;
	
    // coordinates of the normal to the wave facet in the original axis
	float3 no;

    // coordinates of the normal to the wave facet in the local axis (Nx, Ny, Nz)
	float3 n_l;

	float s1, s2, s3 ;
    float4 stokes;
	
	float rpar, rper, rparper, rparper_cross;	// Coefficient de reflexion parallèle et perpendiculaire
	float rpar2;		// Coefficient de reflexion parallèle au carré
	float rper2;		// Coefficient de reflexion perpendiculaire au carré
	float rat;			// Rapport des coefficients de reflexion perpendiculaire et parallèle
	int ReflTot;		// Flag pour la réflexion totale sur le dioptre
	float cot;			// Cosinus de l'angle de réfraction du photon
	float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
	float tpar, tper, tparper, tpar2, tper2;	//
    float geo_trans_factor;
    int iter=0;
    float vzn;  // projection of V on the local vertical
    float thv, phi;
	float3 v;

    // Determination of the relative refractive index
    // a: air, b: water , Mobley 2015 nind = nba = nb/na
    // and sign for further computation
    float sign;
    if (ph->loc == SURF0M)  {
        nind = __fdividef(1.f,NH2Od);
        sign = -1;
    }
    else  {
        nind = NH2Od;
        sign = 1;
    }
    
	
    #ifdef SPHERIQUE
    // define 3 vectors Nx, Ny and Nz in cartesian coordinates which define a
    // local orthonormal basis at the impact point.
    // Nz is the local vertical direction, the direction of the 2 others does not matter
    // because the azimuth is chosen randomly
	float3 Nx, Ny, Nz;

	Nz = ph->pos/RTER; // Nz is the vertical at the impact point

    // Ny is chosen arbitrarily by cross product of Nz with axis X = (1,0,0)
	Ny = cross(Nz, make_float3(1.0,0.0,0.0));

    // Nx is the cross product of Ny and Nz
	Nx = cross(Ny, Nz);
 
	// Normalizatioin
	Ny = normalize(Ny);
	Nz = normalize(Nz);

    #ifdef DEBUG
    // we check that there is no upward photon reaching surface0+
    if ((ph->loc == SURF0P) && (dot(ph->v, ph->pos) > 0)) {
        // upward photon when reaching the surface at (0+)
        printf("Warning, vzn>0 (vzn=%f) with SURF0+ in surfaceAgitee\n", dot(ph->v, ph->pos));
    }
    #endif
    #endif

	
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


    sig = sqrtf(0.003F + 0.00512f *WINDSPEEDd);
    sig2= sig * sig;

    if (!le) {
	 if( DIOPTREd !=0 ){
        // Rough surface

        theta = DEMIPI;
        // DR Computation of P_alpha_beta = P_Cox_Munk(beta) * P(alpha | beta)
        // DR we draw beta first according to Cox_Munk isotropic and then draw alpha, conditional probability
        // DR rejection method: to exclude unphysical azimuth (leading to incident angle theta >=PI/2)
        // DR we continue until acceptable value for alpha

        while (theta >= DEMIPI) {
           iter++;
           if (iter >= 100) {
                // safety check
                #ifdef DEBUG
                printf("Warning, photon rejected in RoughSurface while loop\n");
                printf("%i  V=(%f,%f,%f) beta,alpha=(%f,%f) \n",
                        iter,
                        ph->v.x,
                        ph->v.y,
                        ph->v.z,
                        beta/PI*180,
                        alpha/PI*180
                      );
                #endif
                ph->loc = NONE;
                break;
           }
           beta = atanf( sig*sqrtf(-__logf(RAND)) );
           alpha = DEUXPI * RAND;
           sBeta = __sinf( beta );
           cBeta = __cosf( beta );

           // the facet has coordinates
           // (sin(beta)*cos(alpha), sin(beta)*sin(alpha), cos(beta)) in axis (Nx, Ny, Nz)

           // Normal of the facet in the local frame
           n_l.x = sign * sBeta * __cosf( alpha );
           n_l.y = sign * sBeta * __sinf( alpha );
           n_l.z = sign * cBeta;

           cTh = -(dot(n_l, ph->v));
           theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));
        }
     } else {
        // Flat surface
        beta  = 0.F;
        cBeta = 1.F;
        n_l.x   = 0.F;
        n_l.y   = 0.F;
        n_l.z   = sign;

        cTh = -(dot(n_l, ph->v));
        theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));
     }
    } /* not le*/

    if (le) {
     float sign_le = 1.F;
     if (count_level==DOWN0M) sign_le = -1.F;

     phi = tabphi[ph->iph];
     thv = tabthv[ph->ith];
     v.x  = cosf(phi) * sinf(thv);
     v.y  = sinf(phi) * sinf(thv);
     v.z  = sign_le * cosf(thv);  
     
     // Normal to the facet in the global frame
     // 1) Determination of the half direction vector
     if ((ph->loc==SURF0P) && (count_level==DOWN0M) ||
         (ph->loc==SURF0M) && (count_level==UP0P))   { // Refraction geometry
        // vector equation for determining the half direction h = - (ni*i + no*o)
        // or h = - (i + nind*o)
        // h is pointing toward the incoming ray
		 no = sign*(operator-(ph->v, v*nind));
     }
     if ((ph->loc==SURF0P) && (count_level==UP0P) ||
         (ph->loc==SURF0M) && (count_level==DOWN0M)) { // Reflection geometry
        // vector equation for determining the half direction h = sign(i dot o) (i + o)
		 no = operator-(v, ph->v);
     }

     // 2) Normalization of the half direction vector
     no=normalize(no);

     // Incidence angle in the local frame
     cTh = fabs(-dot(no, ph->v));
     theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));

     #ifdef SPHERIQUE
     // facet slope
     cBeta = 1./RTER * fabs(dot(no, ph->pos));
     beta  = fabs(acosf(cBeta));
     #else
     cBeta = fabs(no.z);
     beta  = acosf(no.z);
     #endif
	 if( (DIOPTREd == 0) && (fabs(beta) >= 1e-6)) {  //  for a flat ocean beta shall be stricly zero 
        ph->weight = 0.;
        return;
     }
    
    } /*le*/


    // express the coordinates of the normal to the wave facet in the original
    // axis instead of local axis (Nx, Ny, Nz)
    if (!le) {
    #ifdef SPHERIQUE
	no = operator+(operator+(n_l.x*Nx, n_l.y*Ny), n_l.z*Nz);
    #else
    no = n_l;
    #endif
    }

	sTh = __sinf(theta);

    #ifdef SPHERIQUE
    // avz is the projection of V on the local vertical
	float avz = fabs(dot(ph->pos, ph->v))/RTER;
    #else
    float avz = fabs(ph->v.z);
    #endif

	// Rotation of Stokes parameters
	s1 = ph->stokes.x;
	s2 = ph->stokes.y;
	s3 = ph->stokes.z;

	if( (s1!=s2) || (s3!=0.F) ){

		temp = __fdividef(dot(no, ph->u), sTh);
		psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

		if( dot(no, cross(ph->u, ph->v)) <0 ){
			psi = -psi;
		}

        rotateStokes(ph->stokes, psi, &ph->stokes);
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
        rparper_cross = 0.F;
		tpar = __fdividef( 2.F*cTh,ncTh+ cot);
		tper = __fdividef( 2.F*cTh,cTh+ ncot);
        tpar2= tpar * tpar;
        tper2= tper * tper;
        tparper = tpar * tper;
        // DR rat is the energetic reflection factor used to normalize the R and T matrix (see Xun 2014)
		rat =  __fdividef(ph->stokes.x*rper2 + ph->stokes.y*rpar2,ph->stokes.x+ph->stokes.y);
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
        tpar = 0.;
        tper = 0.;
        tpar2 =0.;
        tper2 =0.;
        tparper =0.;
		ReflTot = 1;
	}

    // Weighting
    float Anorm;
    // Ross et al 2005
    // Normalization of the slope probability density function taking into acount shadows
    // used to LE weight, and also to debias random draw of slope in non LE
    float nu = __fdividef(1.F, tanf(acosf(avz))*(sqrtf(2.) * sig));
    Anorm = 1.F + __fdividef(__expf(-nu*nu) - nu * sqrtf(PI) * erfcf(nu),2.F * nu * sqrtf(PI));
    Anorm *= avz;

    ph->weight *= __fdividef(fabs(cTh), cBeta *  Anorm ); // Common to all photons, cBeta for surface area unit correction
    
    if (le && (DIOPTREd!=0)) {
     if ((ph->loc==SURF0P) && (count_level==UP0P) ||
         (ph->loc==SURF0M) && (count_level==DOWN0M)) { // Reflection geometry
            ph->weight  *=
                 __fdividef( __expf(-(1.F-cBeta*cBeta)/(cBeta*cBeta*sig2)) ,  cBeta*cBeta*cBeta * sig2)
                *__fdividef(1.F, 4.F * fabs(cTh) );
     }
     if ((ph->loc==SURF0P) && (count_level==DOWN0M) ||
         (ph->loc==SURF0M) && (count_level==UP0P))   { // Refraction geometry
            if (sTh <= nind) ph->weight  *=
                 __fdividef( __expf(-(1.F-cBeta*cBeta)/(cBeta*cBeta*sig2)) ,  cBeta*cBeta*cBeta * sig2)
                *__fdividef(nind*nind * cot, (ncot - cTh)*(ncot - cTh)); // See Zhai et al., 2010
            else ph->weight = 0.F;
     }
     if (ph->weight <= 1e-15) {
         ph->weight = 0.;
         //return;
     }
    }

    stokes.z = ph->stokes.z;	
    stokes.w = ph->stokes.w;	
    int condR=1;
    if (!le) condR = (SURd==3)&&(RAND<rat);

	if (  (!le && (condR || (SURd==1) || ReflTot) )
       || ( le && (ph->loc==SURF0M) && (count_level == DOWN0M) )
       || ( le && (ph->loc==SURF0P) && (count_level == UP0P)   )
       ){	// Reflection

		ph->stokes.x *= rper2;
		ph->stokes.y *= rpar2;
		ph->stokes.z = rparper*stokes.z + rparper_cross*stokes.w; // DR Mobley 2015 sign convention
		ph->stokes.w = rparper*stokes.w - rparper_cross*stokes.z; // DR Mobley 2015 sign convention
		
        if (le) { ph->v = v; }
        else { operator+=(ph->v, (2.F*cTh)*no); }

		ph->u = operator/(operator-(no, cTh*ph->v), sTh);	

        // DR Normalization of the reflexion matrix
        // DR the reflection coefficient is taken into account:
        // DR once in the random selection (Rand < rat)
        // DR once in the reflection matrix multiplication
        // DR so twice and thus we normalize by rat (Xun 2014).
        // DR not to be applied for forced reflection (SUR=1 or total reflection) where there is no random selection
		if (SURd==3 && !ReflTot && !le) {
		//if (SURd==3 && ReflTot==0) {
			ph->weight /= rat;
			}

        #ifdef SPHERIQUE
        vzn = dot(ph->v, ph->pos); // produit scalaire
        #else
        vzn = ph->v.z;
        #endif

        //
        // photon next location
        //
        if (!le) {
         if (ph->loc == SURF0P) {
            if (vzn > 0) {  // avoid multiple reflexion above the surface
                // SURF0P becomes ATM or SPACE
                if( SIMd==-1 || SIMd==0 ){
                    ph->loc = SPACE;
                } else{
                    ph->loc = ATMOS;
                }
            } // else, no change of location
         } else {
            if (vzn < 0) {  // avoid multiple reflexion under the surface
                // SURF0M becomes OCEAN or ABSORBED
                if( SIMd==1 ){
                    ph->loc = ABSORBED;
                } else{
                    ph->loc = OCEAN;
                }
            } // else, no change of location
         }
        }


	} // Reflection

	else if (  (!le && !condR) 
            //|| ( le && (ph->loc==SURF0M) && (count_level == UP0P  ) )
            //|| ( le && (ph->loc==SURF0P) && (count_level == DOWN0M) )
            || ( le && (ph->loc==SURF0M) && (count_level == UP0P  ) && !ReflTot )
            || ( le && (ph->loc==SURF0P) && (count_level == DOWN0M) && !ReflTot )
            ){	// Transmission

        geo_trans_factor = nind* cot/cTh; // DR Mobley 2015 OK , see Xun 2014, Zhai et al 2010
		
		ph->stokes.x *= tper2*geo_trans_factor;
		ph->stokes.y *= tpar2*geo_trans_factor;
		ph->stokes.z *= tparper*geo_trans_factor;
		ph->stokes.w *= tparper*geo_trans_factor;
		
		alpha  = __fdividef(cTh, nind) - cot;

        if (le) { ph->v = v; }
        else { ph->v = operator+(operator/(ph->v, nind), alpha*no); }
		ph->u = operator/(operator+(no, cot*ph->v), sTh )*nind;

        #ifdef SPHERIQUE
        vzn = dot(ph->v, ph->pos); // produit scalaire
        #else
        vzn = ph->v.z;
        #endif


        // DR Normalization of the transmission matrix
        // the transmission coefficient is taken into account:
        // once in the random selection (Rand > rat)
        // once in the transmission matrix multiplication
        // so we normalize by (1-rat) (Xun 2014).
        // Not to be applied for forced transmission (SUR=2)
        if ( (SURd == 3 ) && !le) 
            ph->weight /= (1-rat);

        //
        // photon next location
        //
        if (!le) {
         if (ph->loc == SURF0M) {
            if (vzn > 0) {
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
            if (vzn < 0) {  // avoid multiple reflexion under the surface
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
        }

	} // Transmission
}


/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* ph, int le, float* tabthv, float* tabphi, struct Spectrum *spectrum, philox4x32_ctr_t* etatThr, philox4x32_key_t* configThr) {
	
	if( SIMd == -2){ 	// Atmosphère ou océan seuls, la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
	
	float3 u_n, v_n;	// Vecteur du photon après reflexion
    float phi;
    float cTh, sTh, cPhi, sPhi;

    if (le) {
        cTh  = cosf(tabthv[ph->ith]);  
        phi  = tabphi[ph->iph];
        ph->weight *= cTh;
    }
    else {
        float ddis=0.0F;
        if ((LEd==0) || (LEd==1 && RAND>ddis)) {
            // Standard sampling
	        cTh = sqrtf( RAND );
	        phi = RAND*DEUXPI;
        }
        else {
            // DDIS sampling , Buras and Mayer
            float Om = 0.001;
	        cTh = sqrtf(1.F-RAND*Om);
            phi = RAND*DEUXPI;
            ph->weight *= DEUXPI*(1. -sqrtf(1.F-Om));
        }
    }

	sTh = sqrtf( 1.0F - cTh*cTh );
	cPhi = __cosf(phi);
	sPhi = __sinf(phi);
	
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
	if( ph->pos.z > 0. ){
		ict = __fdividef(ph->pos.z,RTER);
		
		if(ict>1.f){
			ict = 1.f;
		}
		
		ist = sqrtf( 1.f - ict*ict );
		
		if(ph->pos.x >= 0.f) ist = -ist;
		
		if( sqrtf(ph->pos.x*ph->pos.x + ph->pos.y*ph->pos.y)<1.e-6 ){
			/*NOTE En fortran ce test est à 1.e-8, relativement au double utilisés, peut peut être être supprimer ici*/
			icp = 1.f;
		}
		else{
			icp = __fdividef(ph->pos.x,sqrtf(ph->pos.x*ph->pos.x + ph->pos.y*ph->pos.y));
			isp = sqrtf( 1.f - icp*icp );
			
			if( ph->pos.y < 0.f ) isp = -isp;
		}
	}
	else{
		// Photon considéré comme perdu
		ph->loc = ABSORBED;	// Correspondant au weight=0 en Fortran
		return;
	}
	
	/** Il faut exprimer Vx,y,z et Ux,y,z dans le repère de la normale au point d'impact **/
	v_n.x= ict*icp*ph->v.x - ict*isp*ph->v.y + ist*ph->v.z;
	v_n.y= isp*ph->v.x + icp*ph->v.y;
	v_n.z= -icp*ist*ph->v.x + ist*isp*ph->v.y + ict*ph->v.z;
	
	u_n.x= ict*icp*ph->u.x - ict*isp*ph->u.y + ist*ph->u.z;
	u_n.y= isp*ph->u.x + icp*ph->u.y;
	u_n.z= -icp*ist*ph->u.x + ist*isp*ph->u.y + ict*ph->u.z;
	
	ph->v = v_n;
	ph->u = u_n;

    } // photon not seafloor
	#endif //SPHERICAL

	/** calcul u,v new **/
	v_n.x = cPhi*sTh;
	v_n.y = sPhi*sTh;
	v_n.z = cTh;
	
	u_n.x = cPhi*cTh;
	u_n.y = sPhi*cTh;
	u_n.z = -sTh;

	// Depolarisation du Photon
	float norm;
	norm = ph->stokes.x + ph->stokes.y;
	ph->stokes.x = 0.5 * norm;
	ph->stokes.y = 0.5 * norm;
    ph->stokes.z = 0.0;
    ph->stokes.w = 0.0;

	ph->v = v_n;
	ph->u = u_n;
	
    if (DIOPTREd!=4 && ((ph->loc == SURF0M) || (ph->loc == SURF0P))){
	  // Si le dioptre est seul, le photon est mis dans l'espace
	  bool test_s = ( SIMd == -1);
	  ph->loc = SPACE*test_s + ATMOS*(!test_s);
    }
	
    if (ph->loc != SEAFLOOR){

	  ph->weight *= spectrum[ph->ilam].alb_surface;

	  #ifdef SPHERIQUE	/* Code spécifique à une atmosphère sphérique */
	  /** Retour dans le repère d'origine **/
	  // Re-projection vers le repères de direction de photon. L'angle à prendre pour la projection est -angleImpact
	  isp = -isp;
	  ist = -ist;
	
	  v_n.x= ict*icp*ph->v.x - ict*isp*ph->v.y + ist*ph->v.z;
	  v_n.y= isp*ph->v.x + icp*ph->v.y;
	  v_n.z= -icp*ist*ph->v.x + ist*isp*ph->v.y + ict*ph->v.z;
	
	  u_n.x= ict*icp*ph->u.x - ict*isp*ph->u.y + ist*ph->u.z;
	  u_n.y= isp*ph->u.x + icp*ph->u.y;
	  u_n.z= -icp*ist*ph->u.x + ist*isp*ph->u.y + ict*ph->u.z;
	
	  ph->v = v_n;
	  ph->u = u_n;
	  #endif
    } // not seafloor 

    else {
	  ph->weight *= spectrum[ph->ilam].alb_seafloor;
      ph->loc = OCEAN;
    }
    
}



__device__ void countPhoton(Photon* ph,
        struct Profile *prof_atm,
        float *tabthv, float *tabphi,
        int count_level,
		unsigned long long *errorcount,
        void *tabPhotons, unsigned long long *NPhotonsOut
        ) {

    if (count_level < 0) {
        // don't count anything
        return;
    }

    // don't count the photons directly transmitted
    if ((ph->weight == WEIGHTINIT) && (ph->stokes.x == ph->stokes.y) && (ph->stokes.z == 0.f) && (ph->stokes.w == 0.f)) {
        return;
    }

    #ifdef DOUBLE 
    double *tabCount;                   // pointer to the "counting" array:
    double dweight;
	double4 ds;                         // replace ds1, ds2, ds3, ds4
    #else                               // may be TOA, or BOA down, and so on
    float *tabCount; 
    #endif

    float theta = acosf(fmin(1.F, fmax(-1.F, ph->v.z)));
    #ifdef SPHERIQUE
    if(ph->v.z<=0.f) {
         // do not count the downward photons leaving atmosphere
         // DR April 2016, test flux for spherical shell
        //return;
    }
    #endif

	if(theta == 0.F)
	{
		atomicAdd(errorcount+ERROR_THETA, 1);
	}


	float psi=0.;
	int ith=0, iphi=0, il=0;
    float4 st; // replace s1, s2, s3, s4
    int II, JJ;

    if (theta != 0.F) {
        ComputePsi(ph, &psi, theta);
    }
    else {
        // Compute Psi in the special case of nadir
        float ux_phi;
        float uy_phi;
        float cos_psi;
        float sin_psi;
        float eps=1e-4;

            ux_phi = cosf(tabphi[ph->iph]);
            uy_phi = sinf(tabphi[ph->iph]);
            cos_psi = (ux_phi*ph->u.x + uy_phi*ph->u.y);
            if( cos_psi > 1.0) cos_psi = 1.0;
            if( cos_psi < -1.0) cos_psi = -1.0;
            sin_psi = sqrtf(1.0 - (cos_psi*cos_psi) );
            if(( abs( (ph->u.x*cos_psi - ph->u.y*sin_psi) - ux_phi ) < eps ) and ( abs( (ph->u.x*sin_psi + ph->u.y*cos_psi) - uy_phi ) < eps )) {
                psi = -acosf(cos_psi);
            }
            else{
                psi = acosf(cos_psi);
            } 
    }

    rotateStokes(ph->stokes, psi, &st);
    st.w = ph->stokes.w;
	// Calcul de la case dans laquelle le photon sort
	if (LEd == 0) ComputeBox(&ith, &iphi, &il, ph, errorcount);
    else {
        ith = ph->ith;
        iphi= ph->iph;
        il = ph->ilam;
        #ifndef SPHERIQUE
        float tau_le;
        if (count_level==UPTOA) tau_le = prof_atm[NATMd + ph->ilam *(NATMd+1)].tau;
        if ((count_level==DOWN0P) || (count_level==UP0M) || (count_level==UP0P) ) tau_le = 0.F;
        ph->weight *= __expf(__fdividef(-fabs(tau_le - ph->tau), abs(ph->v.z))); // LE attenuation to count_level
        #endif
    }
	
  	/*if( ph->vy<0.f )
    		s3 = -s3;*/  // DR 
	
    // Change sign convention for compatibility with OS
    st.z = -st.z;

	float tmp = st.x;
	st.x = st.y;
	st.y = tmp;
	

	float weight = ph->weight;
	if (FLUXd==1 && LEd==0) weight /= fabs(ph->v.z);

    #ifdef DEBUG
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
    if (isnan(weight)) printf("(idx=%d) Error, weight is NaN, %d\n", idx,ph->loc);
    if (isnan(st.x)) printf("(idx=%d) Error, s1 is NaN\n", idx);
    if (isnan(st.y)) printf("(idx=%d) Error, s2 is NaN\n", idx);
    if (isnan(st.z)) printf("(idx=%d) Error, s3 is NaN\n", idx);
    #endif

	// Rangement du photon dans sa case, et incrémentation de variables
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (il >= 0) && (il < NLAMd) && (!isnan(weight)))
	{
        II = NBTHETAd*NBPHId*NLAMd;
        JJ = il*NBTHETAd*NBPHId + ith*NBPHId + iphi;

        #ifdef DOUBLE 
            dweight = (double)weight;
            ds = make_double4(st.x, st.y, st.z, st.w);

            // select the appropriate level (count_level)
            tabCount = (double*)tabPhotons + count_level*NPSTKd*NBTHETAd*NBPHId*NLAMd;

            DatomicAdd(tabCount+(0*II+JJ), dweight*(ds.x+ds.y));
            DatomicAdd(tabCount+(1*II+JJ), dweight*(ds.x-ds.y));
            DatomicAdd(tabCount+(2*II+JJ), dweight*ds.z);
            DatomicAdd(tabCount+(3*II+JJ), dweight*ds.w);
        #else
            // select the appropriate level (count_level)
            tabCount = (float*)tabPhotons + count_level*NPSTKd*NBTHETAd*NBPHId*NLAMd;

            atomicAdd(tabCount+(0*II+JJ), weight * (st.x+st.y));
            atomicAdd(tabCount+(1*II+JJ), weight * (st.x-st.y));
            atomicAdd(tabCount+(2*II+JJ), weight * st.z);
            atomicAdd(tabCount+(3*II+JJ), weight * st.w);
        #endif
        atomicAdd(NPhotonsOut + ((count_level*NLAMd + il)*NBTHETAd + ith)*NBPHId + iphi, 1);
	}
	else
	{
		atomicAdd(errorcount+ERROR_CASE, 1);
	}

}



//
// Rotation of the stokes parameters by an angle psi between the incidence and
// the emergence planes
// input: float4 stokes parameters
//        rotation angle psi in radians
// output: float 4 rotated stokes parameters
//
__device__ void rotateStokes(float4 s, float psi, float4 *sr)
{
    float cPsi = __cosf(psi); float sPsi = __sinf(psi); float cPsi2 = cPsi * cPsi; float sPsi2 = sPsi * sPsi;
	float twopsi = 2.F*psi;  float s2Psi = __sinf(twopsi); float a = 0.5f*s2Psi;

	float3x3 L = make_float3x3(
		cPsi2, sPsi2, -a,                
		sPsi2, cPsi2, a,                 
		s2Psi, -s2Psi, __cosf(twopsi)   
		);

    // Since s(4) do not change by the rotation, multiply the 3x3 matrix L(psi) by the 3 first terms of s
	(*sr) = mul(L,s); // see the function "mul" in helper_math.h for more infos
}


/* ComputePsi
*/
__device__ void ComputePsi(Photon* ph, float* psi, float theta)
{
    // see Rammella et al. Three Monte Carlo programs of polarized light transport into scattering media: part I Optics Express, 2005, 13, 4420
    double wz;
    wz = (double)ph->v.x * (double)ph->u.y - (double)ph->v.y * (double)ph->u.x;
    *psi = atan2(wz, -1.*(double)ph->u.z); 
}


/* ComputeBox
* Fonction qui calcule la position (ith, iphi) et l'indice spectral (il) du photon dans le tableau de sortie
* La position correspond à une boite contenu dans l'espace de sortie
*/
__device__ void ComputeBox(int* ith, int* iphi, int* il,
                           Photon* photon, unsigned long long *errorcount)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->v.x * photon->v.x + photon->v.y * photon->v.y);

	// Calcul de la valeur de ithv
	// _rn correspond à round to the nearest integer
	*ith = __float2int_rd(__fdividef(acosf(fabsf(photon->v.z)) * NBTHETAd, DEMIPI));
	//*ith = __float2int_rn(__fdividef(acosf(fabsf(photon->vz)) * NBTHETAd, DEMIPI));

	// Calcul de la valeur de il
    // DEV!!
    *il = photon->ilam;

	/* Si le photon ressort très près du zénith on ne peut plus calculer iphi,
	 on est à l'intersection de toutes les cases du haut */
	
	if(vxy >= VALMIN)
	{	//on calcule iphi
	
		// On place d'abord le photon dans un demi-cercle
		float cPhiP = __fdividef(photon->v.x, vxy); //cosPhiPhoton
		// Cas limite où phi est très proche de 0, la formule générale ne marche pas
		//if(cPhiP >= 1.F) *iphi = 0;
		// Cas limite où phi est très proche de PI, la formule générale ne marche pas
		//else if(cPhiP <= -1.F) *iphi = (NBPHId) - 1;
		// Size of the angular boxes
        float dphi = __fdividef(2.F*PI,NBPHId);

        // Boxes centred on 0., dphi, 2dphi, ..., 180-dphi, 180., 180.+dphi,...., 360-dphi .
        // Boxes indices 0, 1, 2, ..., NBPHI/2-1, NBPHI/2, NBPHI/2 +1,..., NBPHI-2, NBPHI -1
        // So 2 boxes on 0 and 180 + NBPHI/2-1 boxes with vy>0 and NBPHI/2 -1 boxes with vy<0
        // Total NBPHI boxes from 0 to NBPHI -1; NBPHI has to be even
        // if the azimuth is within the zeroth boxe centered on 0. of width dphi/2 (half width dphi/4)
        if(cPhiP >= cosf(dphi/2.)) *iphi = 0;
        // if the azimuth is in the middle box centered on 180.
        else if(cPhiP <= -cosf(dphi/2.)) *iphi = NBPHId/2;
		else {
            /* otherwise it lies in a dphi box whose index (starting from 1) is given by the ratio of
             Phi -dphi/4. to the possible phi range that is PI-dphi/2. multiplied by the number of boxes NBPHId/2-1*/
            *iphi = __float2int_rd(__fdividef((acosf(cPhiP)-dphi/2.) * (NBPHId/2-1.0F), PI-dphi)) + 1;
		
		    // Puis on place le photon dans l'autre demi-cercle selon vy, utile uniquement lorsque l'on travail sur tous l'espace
   		    if(photon->v.y < 0.F) *iphi = NBPHId - *iphi;
            }
		// Lorsque vy=0 on décide par défaut que le photon reste du côté vy>0
		if(photon->v.y == 0.F) atomicAdd(errorcount+ERROR_VXY, 1);
	}
	
	else{
		// Photon très près du zenith
		atomicAdd(errorcount+ERROR_VXY, 1);
// 		/*if(photon->vy < 0.F) *iphi = NBPHId - 1;
// 		else*/ *iphi = 0;
		if(photon->v.y >= 0.F)  *iphi = 0;
		else *iphi = NBPHId - 1;
	}
	
}

#ifdef DEBUG_PHOTON
__device__ void display(const char* desc, Photon* ph) {
    //
    // display the status of the photon (only for thread 0)
    //
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);

    if (idx==0) {
        printf("%16s X=(%6.3f,%6.3f,%6.3f) V=(%6.3f,%6.3f,%6.3f) U=(%6.3f,%6.3f,%6.3f) S=(%6.3f,%6.3f,%6.3f,%6.3f) tau=%6.3f weight=%11.3e loc=",
               desc,
               ph->pos.x, ph->pos.y, ph->pos.z,
               ph->v.x,ph->v.y,ph->v.z,
               ph->u.x,ph->u.y,ph->u.z,
               ph->stokes.x, ph->stokes.y,
               ph->stokes.z, ph->stokes.w,
               ph->tau, ph->weight
               );
        switch(ph->loc) {
            case 0: printf("SPACE"); break;
            case 1: printf("ATMOS"); break;
            case 2: printf("SURF0P"); break;
            case 3: printf("SURF0M"); break;
            case 4: printf("ABSORBED"); break;
            case 5: printf("NONE"); break;
            case 6: printf("OCEAN"); break;
            case 7: printf("SEAFLOOR"); break;
            default:
                    printf("UNDEFINED");
        }
        printf("\n");
    }
}
#endif

__device__ void modifyUV( float3 v0, float3 u0, float cTh, float psi, float3 *v1, float3 *u1){ 
    float sTh, cPsi, sPsi;
	float3 v, u, w;

    sPsi = __sinf(psi);
    cPsi = __cosf(psi);
    sTh = sqrtf(1.F - cTh*cTh);
	w = cross(u0, v0); // w : cross product entre l'ancien vec u et l'ancien vec v du photon
	v = operator+(cTh * v0, sTh * (operator+(cPsi * u0, sPsi * w))); // v est le nouveau vecteur v du photon
	// Changement du vecteur u (orthogonal au vecteur vitesse du photon)
    if (cTh <= -1.F) {
		u = -u0;}
    else if (cTh >= 1.F){
        u  = u0;}
    else {u = operator-(cTh * v, v0);}

	*v1 = normalize(v); // v1 = v normalized
	*u1 = normalize(u); // u1 = u normalized
}

__device__ void ComputePsiLE( float3 u0, float3 v0, float3 v1, float* psi, float3* u1){
	float prod_scal, den, y1, cpsi, spsi;	
	float EPS6 = 1e-9;	
	float3 w0, w1;

	// compute former w
	w0 = cross(u0, v0); // w : cross product entre l'ancien vec u et l'ancien vec v du photon
	w1 = cross(v1, v0);	// compute the normal to the new scattering plan i.e. new w vector

	den = length(w1); // Euclidean length also called L2-norm
	if (den < EPS6) {
		prod_scal =  dot(v0, v1);
		if (prod_scal < 0.0)
			w1 = w0;       // diffusion vers l'avant
		else{ w1 = -w0; }   // diffusion vers l'arriere
	}

	else{ operator/=(w1, den); } // Hors Sujet : need to see if "__fdividef" is better
	
	cpsi = dot(w0,w1); 	//  Compute the scalar product between w0 and w1

	if (cpsi >  1.0) 
		cpsi =  1.0;
	if (cpsi < -1.0) 
		cpsi = -1.0;
	spsi = sqrtf(1.0 - cpsi * cpsi);
	if (spsi >  1.0) 
		spsi =  1.0;

	// Change of reference frame, look for the expression of  {vx1, vy1, vz1}
	// in the base linked to the photon before the scattering event = old
	// scattering plan. 
	// Let say that x1, y1, z1 are the new coordinate of cos_dir_sensor
	y1 = dot(w0,v1);
	// --- Sign of spsi
	if (y1 < 0.0) 
		spsi = -spsi;

	*psi = acosf(cpsi);
	if (spsi<0)
		*psi = 2*PI - *psi;

	// get the new u vector
	*u1 = cross(v1, w1);	
}

__device__ float ComputeTheta(float3 v0, float3 v1){
	// compute the diffusion angle theta between
	// to direction cosine {vx0, vy0, vz0} and {vx1, vy1, vz1} 

	float cs;
	float theta;
	
	//--- Find cos(theta) and sin(theta)
	cs =  dot(v1,v0)  ;//  produit scalaire
	
	// test cs to avois acos(cs)=NaN
	if(cs>+1) cs = 1.00;
	if(cs<-1) cs = -1.00;
		
	//--- compute theta
	
	theta = acosf(cs);

	return(theta);		
}

__device__ void copyPhoton(Photon* ph, Photon* ph_le) {
    ph_le->v = ph->v; //float3
    ph_le->u = ph->u; // float3
    ph_le->stokes = ph->stokes; //float4
    ph_le->loc = ph->loc;
    ph_le->tau = ph->tau;
    ph_le->couche = ph->couche;
    ph_le->weight = ph->weight;
    ph_le->wavel = ph->wavel;
    ph_le->ilam = ph->ilam;
    ph_le->prop_aer = ph->prop_aer;
    ph_le->pos = ph->pos; // float3
    #ifdef SPHERIQUE
    ph_le->rayon = ph->rayon;
    ph_le->taumax = ph->taumax;
    #endif
}

/**********************************************************
*	> Fonctions liées au générateur aléatoire
***********************************************************/


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

__device__ double DatomicAdd(double* address, double val)
{
        unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
           assumed = old;
           old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                __longlong_as_double(assumed)));

                // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
}
