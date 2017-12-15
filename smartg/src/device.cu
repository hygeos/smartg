
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
#include "geometry.h"
#include "transform.h"
#include "shapes.h"
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
							 void *rng_state
							 ) {

    // current thread index
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
	int loc_prev;
	int count_level;
	int this_thread_active = 1;
	unsigned long long iloop = 0;

    struct RNG_State rngstate;
    #ifdef PHILOX
	// philox_data:
	// index 0: seed (config)
	// index 1 to last: status

	// Paramètres de la fonction random en mémoire locale
	//la clef se defini par l'identifiant global (unique) du thread...
	//...et par la clef utilisateur ou clef par defaut
	//ce systeme garanti l'existence de 2^32 generateurs differents par run et...
	//...la possiblite de reemployer les memes sequences a partir de la meme clef utilisateur
	//(plus d'infos dans "communs.h")
	philox4x32_key_t configThr = {{idx, ((unsigned int *)rng_state)[0]}};
	//le compteur se defini par trois mots choisis au hasard (il parait)...
	//...et un compteur definissant le nombre d'appel au generateur
	//ce systeme garanti l'existence de 2^32 nombres distincts pouvant etre genere par thread,...
	//...et ce sur l'ensemble du process (et non pas 2^32 par thread par appel au kernel)
	//(plus d'infos dans "communs.h")
	philox4x32_ctr_t etatThr = {{((unsigned int *)rng_state)[idx+1], 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

    rngstate.configThr = configThr;
    rngstate.etatThr = etatThr;

    #endif
    #ifdef CURAND_PHILOX
    // copy RNG state in local memory
    rngstate.state = ((curandStatePhilox4_32_10_t *)rng_state)[idx];
    #endif

	
	// Création de variable propres à chaque thread
	unsigned long long nbPhotonsThr = 0; 	// Nombre de photons traités par le thread
	

	Photon ph, ph_le; 		// On associe une structure de photon au thread
	//Photon ph_le2

	bigCount = 1;   // Initialisation de la variable globale bigCount (voir geometry.h)

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

            initPhoton2(&ph, prof_atm, prof_oc, spectrum, X0, NPhotonsIn, wl_proba_icdf, tabthv, tabphi,
                       &rngstate);
            //initPhoton(&ph, prof_atm, prof_oc, spectrum, X0, NPhotonsIn, wl_proba_icdf,
             //          &rngstate);
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
           move_sp(&ph, prof_atm, 0, 0 , &rngstate);
        else 
        #endif
        move_pp(&ph, prof_atm, prof_oc, &rngstate);
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
        } else if (ph.loc == SEAFLOOR) {
            count_level = DOWNB;
        }

		// count the photons
        
		/* Cone Sampling */
		if (LEd ==0) countPhoton(&ph, prof_atm, prof_oc, tabthv, tabphi, count_level,
            errorcount, tabPhotons, NPhotonsOut);

		__syncthreads();
		
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
			        NK=2;
                    up_level = UP0M;
			        down_level = DOWNB;
		        }

                // Loop on levels for counting (for upward and backward)
			    for(int k=0; k<NK; k++){
			        if (k==0) count_level_le = up_level;
			        else count_level_le = down_level;

                    // Double Loop on directions
                    for (int iph=0; iph<NBPHId; iph++){
                        for (int ith=0; ith<NBTHETAd; ith++){
                            // Copy of the propagation photon to to the virtual, local estiumate photon
                            copyPhoton(&ph, &ph_le);
                            // Computation of the index of the direction
                            ph_le.iph = (iph + iph0)%NBPHId;
                            ph_le.ith = (ith + ith0)%NBTHETAd;
                            // Scatter the virtual photon, using le=1, and count_level for the scattering angle computation
                            scatter(&ph_le, prof_atm, prof_oc, faer, foce,
                                    1, tabthv, tabphi,
                                    count_level_le, &rngstate);

                            #ifdef DEBUG_PHOTON
                            if (k==0) display("SCATTER LE UP", &ph_le);
                            else display("SCATTER LE DOWN", &ph_le);
                            #endif

                            #ifdef SPHERIQUE
                            // !! in case of spherical geometry, the attenuation is computation using the move_sp function
                            // in plane parallel, this is done in the next countPhoton function
                            if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, count_level_le , &rngstate);
                            #ifdef DEBUG_PHOTON
                            display("MOVE LE", &ph_le);
                            #endif
                            #endif

                            // Finally count the virtual photon
                            countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, count_level_le,
                                    errorcount, tabPhotons, NPhotonsOut);

                        } //directions
                    } // directions
                } // levels
            } // LE

            /* TEST DOUBLE LOCAL ESTIMATE IN OCEAN */
            // Scattering Double Local Estimate in Ocean in case of dioptre 
            /*if (LEd == 1 && ph.loc==OCEAN && SIMd != -2) {
                int NK, up_level, down_level, count_level_le;
                int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                int iph0 = idx%NBPHId;
                copyPhoton(&ph, &ph_le);
                scatter(&ph_le, prof_atm, prof_oc, faer, foce,
                            1, tabthv, tabphi,
                            UP0M2, &rngstate);
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
                                          count_level_le, &rngstate);

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

            /* Scattering Propagation , using le=0 and propagation photon */
            scatter(&ph, prof_atm, prof_oc, faer, foce,
                    0, tabthv, tabphi, 0,
                    &rngstate);
            #ifdef DEBUG_PHOTON
            display("SCATTER", &ph);
            #endif

		} // photon in ATMOS or OCEAN
		__syncthreads();


        //
		// Reflection
        //
        // -> in SURFACE
        loc_prev = ph.loc;
        if ((ph.loc == SURF0M) || (ph.loc == SURF0P)){
           // Eventually evaluate Downward 0+ and Upward 0- radiance

           // if not environment effects 
           if( ENVd==0 ) { 

           // if not a Lambertian surface
			if( DIOPTREd!=3 ) {
                /* Surface Local Estimate (not evaluated if atmosphere only simulation)*/
                if (LEd == 1 && SIMd != -2) {
                ///* TEST Double LE */
                //if ((LEd == 1) && (SIMd != -2 && ph.loc == SURF0P)) {
                  int NK, count_level_le;
                  if (NOCEd==0) NK=1;
                  else NK=2;
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

                        // Reflect or Tramsit the virtual photon, using le=1, and count_level for the scattering angle computation
                        if (BRDFd != 0)
                            surfaceBRDF(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);
                        else 
                            surfaceAgitee(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);

                        #ifdef DEBUG_PHOTON
                        if (k==0) display("SURFACE LE UP", &ph_le);
                        else display("SURFACE LE DOWN", &ph_le);
                        #endif

                        // Count the photon up to the counting levels (at the surface UP0P or DOW0M)
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, count_level_le, errorcount, tabPhotons, NPhotonsOut);

                        // Only for upward photons count also them up to TOA
                        if (k==0) { 
                            #ifdef SPHERIQUE
                            // for spherical case attenuation if performed usin move_sp
                            if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                            #endif
                            // Final counting at the TOA
                            countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UPTOA , errorcount, tabPhotons, NPhotonsOut);
                        }
                        // Only for downward photons count also them up to Bottom 
                        if (k==1) { 
                            // Final counting at the B 
                            countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, DOWNB , errorcount, tabPhotons, NPhotonsOut);
                        }
                      }//direction
                    }//direction
                  }// counting levels
                } //LE

                // Propagation of photon using le=0
                if (BRDFd != 0)
				    surfaceBRDF(&ph, 0, tabthv, tabphi,
                              count_level, &rngstate);
                else
				    surfaceAgitee(&ph, 0, tabthv, tabphi,
                              count_level, &rngstate);
            } // Not lambertian


            // Lambertian case
			else { 
                if (LEd == 1 && SIMd != -2) {
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;
                  for (int ith=0; ith<NBTHETAd; ith++){
                    for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.iph = (iph + iph0)%NBPHId;
                        ph_le.ith = (ith + ith0)%NBTHETAd;
				        surfaceLambertienne(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);
                        // Only two levels for counting by definition
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UP0P,  errorcount, tabPhotons, NPhotonsOut);
                        #ifdef SPHERIQUE
                        // for spherical case attenuation if performed usin move_sp
                        if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                        #endif
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, NPhotonsOut);
                    }//direction
                  }//direction
                } //LE

                //Propagation of Lamberatian reflection with le=0
				surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &rngstate);
            } // Lambertian
           } // ENV=0

           // Environment effects, no LE computed yet
           else {
                float dis=0;
                dis = sqrtf((ph.pos.x-X0d)*(ph.pos.x-X0d) +(ph.pos.y-Y0d)*(ph.pos.y-Y0d));
                if( dis > ENV_SIZEd) {
                 if (LEd == 1 && SIMd != -2) {
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;
                  for (int ith=0; ith<NBTHETAd; ith++){
                    for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.iph = (iph + iph0)%NBPHId;
                        ph_le.ith = (ith + ith0)%NBTHETAd;
				        surfaceLambertienne(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);
                        // Only two levels for counting by definition
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UP0P,  errorcount, tabPhotons, NPhotonsOut);
                        #ifdef SPHERIQUE
                        // for spherical case attenuation if performed usin move_sp
                        if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                        #endif
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, NPhotonsOut);
                    }//direction
                  }//direction
                 } //LE
                 //Propagation of Lamberatian reflection with le=0
                    surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &rngstate);
                }// dis
                else {
                 if (LEd == 1 && SIMd != -2) {
                 ///* TEST Double LE */
                 //if ((LEd == 1) && (SIMd != -2 && ph.loc == SURF0P)) {
                  int NK, count_level_le;
                  if (NOCEd==0) NK=1;
                  else NK=2;
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

                        // Reflect or Tramsit the virtual photon, using le=1, and count_level for the scattering angle computation
                        if (BRDFd != 0)
                            surfaceBRDF(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);
                        else
                            surfaceAgitee(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);

                        #ifdef DEBUG_PHOTON
                        if (k==0) display("SURFACE LE UP", &ph_le);
                        else display("SURFACE LE DOWN", &ph_le);
                        #endif

                        // Count the photon up to the counting levels (at the surface UP0P or DOW0M)
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, count_level_le, errorcount, tabPhotons, NPhotonsOut);

                        // Only for upward photons count also them up to TOA

                        #ifdef DEBUG_PHOTON
                        if (k==0) display("SURFACE LE UP", &ph_le);
                        else display("SURFACE LE DOWN", &ph_le);
                        #endif

                        // Count the photon up to the counting levels (at the surface UP0P or DOW0M)
                        countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, count_level_le, errorcount, tabPhotons, NPhotonsOut);

                        // Only for upward photons count also them up to TOA
                        if (k==0) { 
                            #ifdef SPHERIQUE
                            // for spherical case attenuation if performed usin move_sp
                            if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                            #endif
                            // Final counting at the TOA
                            countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UPTOA , errorcount, tabPhotons, NPhotonsOut);
                        }
                        // Only for downward photons count also them up to Bottom 
                        if (k==1) { 
                            // Final counting at the B 
                            countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, DOWNB , errorcount, tabPhotons, NPhotonsOut);
                        }
                      }//direction
                    }//direction
                  }// counting levels
                 } //LE
                // Propagation of photon using le=0
                    if (BRDFd != 0)
                        surfaceBRDF(&ph, 0, tabthv, tabphi, count_level, &rngstate);
                    else
                        surfaceAgitee(&ph, 0, tabthv, tabphi, count_level, &rngstate);
                } //dis
           } // ENV=1

           #ifdef DEBUG_PHOTON
           display("SURFACE", &ph);
           #endif
		}

		__syncthreads();

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
				    surfaceLambertienne(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);
                    //  contribution to UP0M level
                    countPhoton(&ph_le, prof_atm, prof_oc, tabthv, tabphi, UP0M,   errorcount, tabPhotons, NPhotonsOut);
                }
              }
            } //LE

			surfaceLambertienne(&ph, 0, tabthv, tabphi, spectrum, &rngstate);
            #ifdef DEBUG_PHOTON
            display("SEAFLOOR", &ph);
            #endif
         }
        __syncthreads();


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
        if (LEd == 0) countPhoton(&ph, prof_atm, prof_oc, tabthv, tabphi, count_level, errorcount, tabPhotons, NPhotonsOut);



		if(ph.loc == ABSORBED){
			ph.loc = NONE;
			nbPhotonsThr++;
		}
		if(ph.loc == REMOVED){
			ph.loc = NONE;
		}
		__syncthreads();

		

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

    #ifdef PHILOX
	// Sauvegarde de l'état du random pour que les nombres
    // ne soient pas identiques à chaque appel du kernel
    ((unsigned int *)rng_state)[idx+1] = rngstate.etatThr[0];
    #endif
    #ifdef CURAND_PHILOX
    ((curandStatePhilox4_32_10_t *)rng_state)[idx] = rngstate.state;
    #endif

}
}


/**********************************************************
*	> Modélisation phénomènes physiques
***********************************************************/
/* initPhoton2
   New init with sensor class
*/

__device__ void initPhoton2(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                           struct Spectrum *spectrum, float *X0, unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf, float* tabthv, float* tabphi,
                           struct RNG_State *rngstate) {
    float dz, dz_i, delta_i, epsilon;
    float cTh, sTh, phi;
    int ilayer;

    ph->nint = 0;
	ph->weight = WEIGHTINIT;

	// Stokes parameters initialization according to natural sunlight
	ph->stokes.x = 0.5F;
	ph->stokes.y = 0.5F;
	ph->stokes.z = 0.F;
	ph->stokes.w = 0.F;
	
    // Wavelength initialization
    if (NWLPROBA == 0) { 
        ph->ilam = __float2uint_rz(RAND * NLAMd);
    } else {
        ph->ilam = wl_proba_icdf[__float2uint_rz(RAND * NWLPROBA)];
    }
	ph->wavel = spectrum[ph->ilam].lambda;


    // Position and optical thicknesses initializations
    ph->pos = make_float3(POSXd,POSYd,POSZd);
    ph->loc = LOCd;
    #ifdef SPHERIQUE
	ph->radius = length(ph->pos);
    #endif

    if(ph->loc == SURF0P){
        ph->layer   = NATMd;
        ph->tau     = 0.F;
        ph->tau_abs = 0.F;
        epsilon     = 0.F;
        ph->pos.z   = 0.F;
        #ifdef SPHERIQUE
        ph->pos.z = RTER;
        #endif
        #ifdef ALIS
        for (int k=0; k<NLOWd; k++) {
            ph->tau_sca[k] = 0.F;
        }
        #endif
    }

    if(ph->loc == SURF0M){
        ph->layer   = NOCEd;
        ph->tau     = 0.F;
        ph->tau_abs = 0.F;
        epsilon     = 0.F;
        ph->pos.z   = 0.F;
        #ifdef SPHERIQUE
        ph->pos.z = RTER;
        #endif
        #ifdef ALIS
        for (int k=0; k<NLOWd; k++) {
            ph->tau_sca[k] = 0.F; ;
        }
        #endif
    }

    if(ph->loc == SEAFLOOR){
        ph->layer   = 0;
        ph->tau     = get_OD(BEERd, prof_oc[NOCEd +ph->ilam*(NOCEd+1)]);
        ph->tau_abs = prof_oc[NOCEd +ph->ilam*(NOCEd+1)].OD_abs;
        epsilon     = 0.F;
        ph->pos.z   = prof_oc[NOCEd].z;
        #ifdef ALIS
        int DL=(NLAMd-1)/(NLOWd-1);
        for (int k=0; k<NLOWd; k++) {
            ph->tau_sca[k] = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]) ;
        }
        #endif
    }

    if(ph->loc == OCEAN){
        ilayer = 1;
        while (( prof_oc[ilayer].z > POSZd) && (ilayer < NOCEd)) {
            ilayer++;
        }
        ph->layer = ilayer;
        dz_i    = fabs(prof_oc[ilayer].z - prof_oc[ilayer-1].z);
        dz      = fabs(POSZd - prof_oc[ilayer-1].z) ;
        epsilon = fabs(__fdividef(dz,dz_i));

        delta_i = fabs(get_OD(BEERd, prof_oc[ilayer+ph->ilam*(NOCEd+1)]) - get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]));
        ph->tau = epsilon * delta_i + (get_OD(BEERd, prof_oc[NOCEd+ph->ilam*(NOCEd+1)])-
                                       get_OD(BEERd, prof_oc[ilayer+ph->ilam*(NOCEd+1)])); 

        delta_i = fabs(prof_oc[ilayer+ph->ilam*(NOCEd+1)].OD_abs - prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs);
        ph->tau_abs = epsilon * delta_i + (prof_oc[NOCEd+ph->ilam*(NOCEd+1)].OD_abs -
                                           prof_oc[ilayer+ph->ilam*(NOCEd+1)].OD_abs); 
        #ifdef ALIS
        int DL=(NLAMd-1)/(NLOWd-1);
        for (int k=0; k<NLOWd; k++) {
            delta_i = fabs(get_OD(BEERd, prof_oc[ilayer+k*DL*(NOCEd+1)]) - get_OD(BEERd, prof_oc[ilayer-1+k*DL*(NOCEd+1)]));
            ph->tau_sca[k] = epsilon * delta_i + (get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)])-
                                                  get_OD(1,prof_oc[ilayer + k*DL*(NOCEd+1)]));
        }
        #endif
    }

    if(ph->loc == ATMOS){
        ilayer = 1;
        float POSZd_alt; 
        #ifdef SPHERIQUE
        POSZd_alt = POSZd - RTER;
        #else
        POSZd_alt = POSZd;
        #endif
        while (( prof_atm[ilayer].z > POSZd_alt) && (ilayer < NATMd)) {
            ilayer++;
        }
        ph->layer = ilayer;
        dz_i    = fabs(prof_atm[ilayer-1].z - prof_atm[ilayer].z);
        dz      = fabs(POSZd_alt - prof_atm[ilayer].z) ;
        epsilon = fabs(__fdividef(dz,dz_i));

        delta_i = fabs(get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)]) - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
        ph->tau = epsilon * delta_i + (get_OD(BEERd, prof_atm[NATMd+ph->ilam*(NATMd+1)])-
                                       get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)])); 
        delta_i = fabs(prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs - prof_atm[ilayer-1+ph->ilam*(NATMd+1)].OD_abs);
        ph->tau_abs = epsilon * delta_i + (prof_atm[NATMd+ph->ilam*(NATMd+1)].OD_abs -
                                           prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs); 
        #ifdef ALIS
        int DL=(NLAMd-1)/(NLOWd-1);
        for (int k=0; k<NLOWd; k++) {
            delta_i = fabs(get_OD(BEERd, prof_atm[ilayer+k*DL*(NATMd+1)]) - get_OD(BEERd, prof_atm[ilayer-1+k*DL*(NATMd+1)]));
            ph->tau_sca[k] = epsilon * delta_i + (get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)])-
                                                  get_OD(1,prof_atm[ilayer + k*DL*(NATMd+1)]));
        }
        #endif
    }

    // Direction initialization
    if (TYPEd != 0) {
        // Standard sampling of zenith angle for lambertian emittor (for planar flux)
	    cTh = sqrtf(1.F-RAND*sinf(FOVd*DEUXPI/360.));
        // for spherical flux, adjust weight as a function of cTh
        float weight_irr = fabs(cTh);
        if (TYPEd == 2 && weight_irr > 0.001f) ph->weight /= weight_irr;
        
	    phi = RAND*DEUXPI;
        sTh = sqrtf(1.F - cTh*cTh);
	    ph->v.x   = cosf(phi)*sTh;
	    ph->v.y   = sinf(phi)*sTh;
	    ph->v.z   = cTh;
	    // Initialization of the orthogonal vector to the propagation
        ph->u.x   = cosf(phi)*cTh;
        ph->u.y   = sinf(phi)*cTh;
	    ph->u.z   = -sTh;
    }
    else {
        // One fixed direction (for radiance)
        ph->v.x = 0.F;
        ph->v.y = 0.F;
        ph->v.z = 1.F;
	    // Initialization of the orthogonal vector to the propagation
        ph->u.x = 1.F;
        ph->u.y = 0.F;
	    ph->u.z = 0.F;
    }


    // Rotations of v and u in the detector direction THDEG,PHDEG
    float cPh;
    if (MId != 0) { // Multiple Init Direction
        if (MId <=0) { 
            // Random selection of Zenith init angle
            ph->ith = __float2uint_rz(RAND * NBTHETAd);
            // Random selection of Azimuth init angle
            if (MId == -1) ph->iph = __float2uint_rz(RAND * NBPHId);
            else {
                ph->iph = ph->ith; // Zip option iph follows ith;
            }
        }
        else {
            // Random selection of Zenith and (zipped) Azimuth init angles according to MId and NLAMd
            int NL = NLAMd/MId;
            int NT = NBTHETAd/MId;
            int offset = ph->ilam/NL;
            ph->ith = __float2uint_rz(RAND * NT) + offset*NT;
            ph->iph = ph->ith;
        }
        cTh = cosf(DEUXPI/2. - tabthv[ph->ith]);
        cPh = cosf(DEUXPI/2. - tabphi[ph->iph]);
    }
    else {
        ph->ith = 0;
        ph->iph = 0;
        cTh = cosf(THDEGd*DEUXPI/360.);
        cPh = cosf(PHDEGd*DEUXPI/360.);
    }
    sTh       = sqrtf(1.F - cTh*cTh);
    float sPh = sqrtf(1.F - cPh*cPh);
	float3x3 LTh = make_float3x3(
		cTh,  0.F,  sTh,                
		0.F,  1.F,  0.F,                 
	   -sTh,  0.F,  cTh                 
        );
	float3x3 LPh = make_float3x3(
		cPh, -sPh,  0.F,                
		sPh,  cPh,  0.F,                 
		0.F,  0.F,  1.F                 
        );
	ph->v = mul(LTh,ph->v);
	ph->v = mul(LPh,ph->v);
	ph->u = mul(LTh,ph->u);
	ph->u = mul(LPh,ph->u);


    // init specific ALIS quantities
    #ifdef ALIS
    ph->nevt = 0;
    ph->layer_prev[ph->nevt]   = ph->layer;
    ph->vz_prev[ph->nevt]      = ph->v.z;
    ph->epsilon_prev[ph->nevt] = epsilon;
    for (int k=0; k<NLOWd; k++) {
        ph->weight_sca[k] = 1.0F;
    }
    #endif

    // Init photon counters
    #ifdef ALIS
    for (int k=0; k<NLAMd; k++) atomicAdd(NPhotonsIn + k, 1);
    #else
    if (MId != 0) {
        atomicAdd(NPhotonsIn + (ph->ilam*NBTHETAd + ph->ith)*NBPHId + ph->iph, 1);
    }
    else atomicAdd(NPhotonsIn + ph->ilam, 1);
    #endif

    #ifdef BACK
    //ph->Mf= make_diag_float4x4 (1.F);
    ph->M = make_diag_float4x4 (1.F);
    #endif

    }

/* initPhoton
* Initialise le photon dans son état initial avant l'entrée dans l'atmosphère
*/
__device__ void initPhoton(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                           struct Spectrum *spectrum, float *X0, unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf,
                           struct RNG_State *rngstate) {
    ph->nint = 0;

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

    #ifdef ALIS
    //ph->ilam = 0;
    //ph->ilam = (NLAMd-1)/2;
    for (int k=0; k<NLAMd; k++) atomicAdd(NPhotonsIn +k, 1);
    #else
    atomicAdd(NPhotonsIn+ph->ilam, 1);
    #endif

	ph->wavel = spectrum[ph->ilam].lambda;


    if ((SIMd == -2) || (SIMd == 1) || (SIMd == 2)) {

        //
        // Initialisation du photon au sommet de l'atmosphère
        //

        ph->pos.x = X0[0];
        ph->pos.y = X0[1];
        ph->pos.z = X0[2];
        ph->layer = 0;   // top of atmosphere

        #ifdef SPHERIQUE
		ph->radius = length(ph->pos);
        #endif

        ph->loc = ATMOS;
        ph->tau = get_OD(BEERd, prof_atm[NATMd + ph->ilam*(NATMd+1)]) ;
        ph->tau_abs = prof_atm[NATMd + ph->ilam*(NATMd+1)].OD_abs;

    } else if ((SIMd == -1) || (SIMd == 0) || (SIMd == 3)) {

        //
        // Initialisation du photon à la surface ou dans l'océan
        //
        ph->pos = make_float3(0.,0.,0.);
        #ifdef SPHERIQUE
        ph->pos.z = RTER;
        #endif

        ph->tau = 0.f;
        ph->tau_abs = 0.f;

        if (SIMd == 3) {
            ph->loc = OCEAN;
            ph->layer = NOCEd;
        } else {
            ph->loc = SURF0P;
            ph->layer = NATMd;
        }

    } else ph->loc = NONE;
	
    #ifdef ALIS
    int DL=(NLAMd-1)/(NLOWd-1);
    ph->nevt = 0;
    ph->layer_prev[ph->nevt] = ph->layer;
    ph->vz_prev[ph->nevt] = ph->v.z;
    ph->epsilon_prev[ph->nevt] = 0.f;
    for (int k=0; k<NLOWd; k++) {
        ph->weight_sca[k] = 1.0f;
        ph->tau_sca[k] = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]) ;
    }
    #endif

	ph->weight = WEIGHTINIT;
	
	// Initialisation des paramètres de stokes du photon
	ph->stokes.x = 0.5F;
	ph->stokes.y = 0.5F;
	ph->stokes.z = 0.F;
	ph->stokes.w = 0.F;

    #ifdef BACK
    //ph->Mf= make_diag_float4x4 (1.F);
    ph->M = make_diag_float4x4 (1.F);

    /*float theta = acosf(fmin(1.F, fmax(-1.F, ph->v.z)));
    float psi;
    if (theta != 0.F) {
        ComputePsi(ph, &psi, theta);
    }
    else {
        // Compute Psi in the special case of zenith
        float ux_phi;
        float uy_phi;
        float cos_psi;
        float sin_psi;
        float eps=1e-4;

        ux_phi  = 1.F;
        uy_phi  = 0.F;
        cos_psi = (ux_phi*ph->u.x + uy_phi*ph->u.y);
        if( cos_psi >  1.0) cos_psi =  1.0;
        if( cos_psi < -1.0) cos_psi = -1.0;
        sin_psi = sqrtf(1.0 - (cos_psi*cos_psi));
        if( (abs((ph->u.x*cos_psi-ph->u.y*sin_psi)-ux_phi) < eps) && (abs((ph->u.x*sin_psi+ph->u.y*cos_psi)-uy_phi) < eps) ) {
                psi = -acosf(cos_psi);
        }
        else{
                psi = acosf(cos_psi);
        } 
      }
    float4x4 L;
    rotationM(psi,&L);
    ph->M = L;*/
    #endif
}



#ifdef SPHERIQUE
__device__ void move_sp(Photon* ph, struct Profile *prof_atm, int le, int count_level,
                        struct RNG_State *rngstate) {

    float tauRdm;
    float hph = 0.;  // cumulative optical thickness
    float vzn, delta1, h_cur, h_cur_abs, epsilon;
    float d_tot = 0.;
    float d;
    float rat;
    int sign_direction;
    int i_layer_fw, i_layer_bh; // index or layers forward and behind the photon
    float costh, sinth2;
    int ilam = ph->ilam*(NATMd+1);  // wavelength offset in optical thickness table

    if (ph->layer == 0) ph->layer = 1;

    #ifdef DEBUG
    int niter = 0;
    // ph->layer is indexed
    // from 1 (TOA layer between interfaces 0 and 1)
    // to NATM (bottom layer between interfaces NATM-1 to NATM)
    if ((ph->layer > NATMd) || (ph->layer <= 0)) {
        printf("Fatal error, wrong index (%d)\n", ph->layer);
    }
    #endif

    // Random Optical Thickness to go through
    if (!le) tauRdm = -logf(1.F-RAND);
    // if called with le mode, it serves to compute the transmission
    // from photon last intercation position to TOA, thus 
    // photon is forced to exit upward or downward and tauRdm is chosen to be an upper limit
    else tauRdm = 1e6;
    

    vzn = __fdividef( dot(ph->v, ph->pos), ph->radius);
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
        if (ph->layer == NATMd+1) {
            ph->loc = SURF0P;
            ph->tau = 0.;
            ph->layer -= 1;  // next time photon enters move_sp, it's at layers NATM
            #ifdef DEBUG
            if (dot(ph->v, ph->pos) > 0) {
                printf("Warning, vzn > 0 at SURF0P in move_sp (vzn=%f)\n", vzn);
            }
            #endif
            break;
        }
        if (ph->layer <= 0) {
            ph->loc = SPACE;
            break;
        }

        //
        // determine the index of the next potential layer
        //
        if (sign_direction < 0) {
            // photon goes down
            // (towards higher indices)
            i_layer_fw = ph->layer;
            i_layer_bh = ph->layer - 1;
        } else {
            // photon goes up
            // (towards lower indices)
            i_layer_fw = ph->layer - 1;
            i_layer_bh = ph->layer;
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
        rat = (prof_atm[i_layer_fw].z+RTER)/ph->radius;
        delta1 = rat*rat - sinth2;   // same sign as delta

        if (delta1 < 0) {
            if (sign_direction > 0) {
                #ifdef DEBUG
                printf("Warning sign_direction (niter=%d, lay=%d, delta1=%f, alt=%f zlay1=%f zlay2=%f vzn=%f)\n",
                        niter, ph->layer, delta1, ph->radius-RTER,
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
        /* d = 0.5f*(-2.*ph->radius*costh + sign_direction*2*ph->radius*sqrtf(delta1)); simplified to: */
        d = ph->radius*(-costh + sign_direction*sqrtf(delta1));
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
        // calculate the optical thicknesses h_cur and h_cur_abs to the next layer
        // We compute the layer extinction coefficient of the layer DTau/Dz and multiply by the distance within the layer
        //
        #ifndef ALT_MOVE
        h_cur = __fdividef(abs(get_OD(BEERd,prof_atm[i_layer_bh+ilam]) - get_OD(BEERd,prof_atm[i_layer_fw+ilam]))*(d - d_tot),
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        h_cur_abs = __fdividef(abs(prof_atm[i_layer_bh+ilam].OD_abs - prof_atm[i_layer_fw+ilam].OD_abs)*(d - d_tot),
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        #else
        h_cur = __fdividef(abs(get_OD(BEERd,prof_atm[i_layer_bh+ilam]) - get_OD(BEERd,prof_atm[i_layer_fw+ilam]))*d,
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        h_cur_abs = __fdividef(abs(prof_atm[i_layer_bh+ilam].OD_abs - prof_atm[i_layer_fw+ilam].OD_abs)*d,
                          abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z));
        #endif


        //
        // update photon position
        //
        if (hph + h_cur > tauRdm) {
            // photon stops within the layer
            epsilon = (tauRdm - hph)/h_cur;
            #ifndef ALT_MOVE
            d_tot += (d - d_tot)*epsilon;
            #else
            d *= epsilon;
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->radius = length(ph->pos);

            #ifdef DEBUG
            vzn = __fdividef( dot(ph->v, ph->pos) , ph->radius);
            #endif
            #endif
            if (BEERd == 1) ph->weight *= __expf(-( epsilon * h_cur_abs));

            break;
        } else {
            // photon advances to the next layer
            hph += h_cur;
            ph->layer -= sign_direction;
            #ifndef ALT_MOVE
            d_tot = d;
            #else
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->radius = length(ph->pos);
            vzn = __fdividef( dot(ph->v, ph->pos) , ph->radius);
            #endif
            if (BEERd == 1) ph->weight *= __expf(-( h_cur_abs));
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
    ph->radius = length(ph->pos);
    #endif

    ph->prop_aer = 1.f - prof_atm[ph->layer+ilam].pmol;
    if (BEERd == 0) ph->weight *= prof_atm[ph->layer+ilam].ssa;
}
#endif // SPHERIQUE


__device__ void move_pp(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                        struct RNG_State *rngstate) {

	float delta_i=0.f, delta=0.f, epsilon;
    float phz, rdist, tauBis;
    int ilayer;
    #ifdef ALIS
    float dsca_dl, dsca_dl0=-ph->tau ;
    int DL=(NLAMd-1)/(NLOWd-1);
    #else
    float ab;
    #endif


	ph->tau += -logf(1.f - RAND)*ph->v.z;

	if (ph->loc == OCEAN){  
        // If tau>0 photon is reaching the surface 
        if (ph->tau > 0) {

            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                ab =  0.F;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += 0.F;
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = 0.F;
                dsca_dl -= ph->tau_sca[k]; 
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl)-fabs(dsca_dl0),  fabs(ph->v.z)));
                ph->tau_sca[k] = 0.F;
            }
            #endif

            ph->tau = 0.F;
            ph->tau_abs = 0.F;
            ph->loc = SURF0M;
            if (SIMd == 3){
              ph->loc = SPACE;
            }
            ph->layer = NOCEd;

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = 1.f;
            #endif
           return;
        }
        // If tau<TAUOCEAN photon is reaching the sea bottom
        else if( ph->tau < get_OD(BEERd, prof_oc[NOCEd + ph->ilam *(NOCEd+1)]) ){

            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                ab = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].OD_abs;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += get_OD(1,prof_oc[NOCEd + ph->ilam*(NOCEd+1)]) ; 
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]);
                dsca_dl -= ph->tau_sca[k]; 
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) - fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]);
            }
            #endif

            ph->loc = SEAFLOOR;
            ph->tau = get_OD(BEERd, prof_oc[NOCEd + ph->ilam *(NOCEd+1)]);
            ph->tau_abs = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].OD_abs;
            ph->layer = 0;

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = 0.f;
            #endif
            return;
        }

        // Calcul de la layer dans laquelle se trouve le photon
        tauBis = get_OD(BEERd, prof_oc[NOCEd + ph->ilam *(NOCEd+1)]) - ph->tau;
        ilayer = 1;

        while (( get_OD(BEERd, prof_oc[ilayer+ ph->ilam *(NOCEd+1)]) > (tauBis)) && (ilayer < NOCEd)) {
            ilayer++;
        }
        ph->layer = ilayer;
        ph->prop_aer = 1.f - prof_oc[ph->layer+ph->ilam*(NOCEd+1)].pmol;

        delta_i= fabs(get_OD(BEERd, prof_oc[ilayer+ph->ilam*(NOCEd+1)]) - get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]));
        delta= fabs(tauBis - get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)])) ;
        epsilon = __fdividef(delta,delta_i);
        
        #ifdef ALIS
        ph->nevt++;
        ph->layer_prev[ph->nevt] = ph->layer;
        ph->vz_prev[ph->nevt] = ph->v.z;
        ph->epsilon_prev[ph->nevt] = epsilon;
        #endif
            
        #ifndef ALIS
        if (BEERd == 0) ph->weight *= prof_oc[ph->layer+ph->ilam*(NOCEd+1)].ssa;
        else { // We compute the cumulated absorption OT at the new postion of the photon
            // photon new position in the layer
            ab = prof_oc[NOCEd+ph->ilam*(NOCEd+1)].OD_abs - 
                (epsilon * (prof_oc[ilayer+ph->ilam*(NOCEd+1)].OD_abs - prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs) +
                prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs);
            // absorption between start and stop
            ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            ph->tau_abs = ab;
        }
        #else
        // cumulated scattering OD at reference wavelength
        dsca_dl0 += get_OD(1,prof_oc[NOCEd + ph->ilam*(NOCEd+1)]) - 
            (epsilon * (get_OD(1,prof_oc[ilayer+ph->ilam*(NOCEd+1)]) - get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)])) +
            get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]));
        for (int k=0; k<NLOWd; k++) {
           // cumulated scattering relative OD wrt reference wavelength
            float tautmp = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]) - 
                (epsilon * (get_OD(1,prof_oc[ilayer+k*DL*(NOCEd+1)]) - get_OD(1,prof_oc[ilayer-1+k*DL*(NOCEd+1)])) +
                get_OD(1,prof_oc[ilayer-1+k*DL*(NOCEd+1)])) ;
            dsca_dl  = tautmp - ph->tau_sca[k]; 
            ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) -fabs(dsca_dl0), fabs(ph->v.z)));
            ph->tau_sca[k] = tautmp;
        }
        #endif
        
    } // Ocean

    if (ph->loc == ATMOS) {
        // If tau<0 photon is reaching the surface 
        if(ph->tau < 0.F){

            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                ab =  0.F;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += 0.F;
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = 0.F;
                dsca_dl -= ph->tau_sca[k]; 
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl)-fabs(dsca_dl0),  fabs(ph->v.z)));
                ph->tau_sca[k] = 0.F;
            }
            #endif

            ph->loc = SURF0P;
            ph->tau = 0.F;
            ph->tau_abs = 0.F;
            // move the photon forward down to the surface
            // the linear distance is ph->z/ph->vz
            operator+=(ph->pos, ph->v * fabs(ph->pos.z/ph->v.z));
            ph->pos.z = 0.;
            ph->layer = NATMd;

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = 1.f;
            #endif
        return;
        }
        // If tau>TAUATM photon is reaching space
        else if( ph->tau > get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)]) ){

            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                ab = prof_atm[NATMd + ph->ilam *(NATMd+1)].OD_abs;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += get_OD(1,prof_atm[NATMd + ph->ilam*(NATMd+1)]) ; 
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]);
                dsca_dl -= ph->tau_sca[k]; 
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) - fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]);
            }
            #endif

            ph->loc = SPACE;
            ph->layer = 0;

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = 0.f;
            #endif

            return;
        }
        
        // Sinon il reste dans l'atmosphère, et va subit une nouvelle diffusion
        
        // Calcul de la layer dans laquelle se trouve le photon
        tauBis =  get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)]) - ph->tau;
        ilayer = 1;
        
        while (( get_OD(BEERd, prof_atm[ilayer+ ph->ilam *(NATMd+1)]) < (tauBis)) && (ilayer < NATMd)) {
            ilayer++;
        }
        
        ph->layer = ilayer;
        ph->prop_aer = 1.f - prof_atm[ph->layer+ph->ilam*(NATMd+1)].pmol;

        delta_i= fabs(get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)]) - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
        delta= fabs(tauBis - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)])) ;
        epsilon = __fdividef(delta,delta_i);

        #ifdef ALIS
        ph->nevt++;
        ph->layer_prev[ph->nevt] = ph->layer;
        ph->vz_prev[ph->nevt] = ph->v.z;
        ph->epsilon_prev[ph->nevt] = epsilon;
        #endif

        #ifndef ALIS
        if (BEERd == 0) ph->weight *= prof_atm[ph->layer+ph->ilam*(NATMd+1)].ssa;
        else { // We compute the cumulated absorption OT at the new postion of the photon
            // photon new position in the layer
            ab = prof_atm[NATMd+ph->ilam*(NATMd+1)].OD_abs - 
                (epsilon * (prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs - prof_atm[ilayer-1+ph->ilam*(NATMd+1)].OD_abs) +
                prof_atm[ilayer-1+ph->ilam*(NATMd+1)].OD_abs);
            // absorption between start and stop
            ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            ph->tau_abs = ab;
        }
        #else

        // cumulated scattering OD at reference wavelength
        dsca_dl0 += get_OD(1,prof_atm[NATMd + ph->ilam*(NATMd+1)]) - 
            (epsilon * (get_OD(1,prof_atm[ilayer+ph->ilam*(NATMd+1)]) - get_OD(1,prof_atm[ilayer-1+ph->ilam*(NATMd+1)])) +
            get_OD(1,prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
        for (int k=0; k<NLOWd; k++) {
           // cumulated scattering relative OD wrt reference wavelength
            float tautmp = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]) - 
                (epsilon * (get_OD(1,prof_atm[ilayer+k*DL*(NATMd+1)]) - get_OD(1,prof_atm[ilayer-1+k*DL*(NATMd+1)])) +
                get_OD(1,prof_atm[ilayer-1+k*DL*(NATMd+1)])) ;
            dsca_dl  = tautmp - ph->tau_sca[k]; 
            ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) -fabs(dsca_dl0), fabs(ph->v.z)));
            ph->tau_sca[k] = tautmp;
        }
        #endif

        // calculate new photon position
        phz = epsilon * (prof_atm[ilayer].z - prof_atm[ilayer-1].z) + prof_atm[ilayer-1].z; 
        rdist=  fabs(__fdividef(phz-ph->pos.z, ph->v.z));
        operator+= (ph->pos, ph->v*rdist);
        ph->pos.z = phz;

    } //ATMOS

}


__device__ void scatter(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
        struct Phase *faer, struct Phase *foce,
        int le,
        float* tabthv, float* tabphi, int count_level,
        struct RNG_State *rngstate) {

	float cTh=0.f ;
	float zang=0.f, theta=0.f;
	int iang, ilay, ipha;
	float psi, sign;
	float prop_aer = ph->prop_aer, RANDTWO;
	RANDTWO = RAND;
	struct Phase *func;
	float prop_raman=1., new_wavel;
	float P11, P12, P22, P33, P43, P44;

    ph->nint += 1;
    if (le){
        /* in case of LE the photon units vectors, scattering angle and Psi rotation angle are determined by output zenith and azimuth angles*/
        float thv, phi;
        float3 v;
        float EPS = 1e-12;

        if (count_level==DOWN0P || count_level==DOWNB) sign = -1.0F;
        else sign = 1.0F;
        phi = tabphi[ph->iph];
        thv = tabthv[ph->ith];
        if (thv < EPS) thv = EPS;
        v.x = cosf(phi) * sinf(thv);
        v.y = sinf(phi) * sinf(thv);
        v.z = sign * cosf(thv);
        theta = ComputeTheta(ph->v, v);
        cTh = __cosf(theta);
		if (cTh < -1.0) cTh = -1.0;
		if (cTh >  1.0) cTh =  1.0;
        ComputePsiLE(ph->u, ph->v, v, &psi, &ph->u); 
        ph->v = v;
    }


    /* Scattering in atmosphere */
	if(ph->loc!=OCEAN){
		/************************************/
		/* Rayleigh and Aerosols scattering */
		/************************************/
		func = faer; // atm phases
        ilay = ph->layer + ph->ilam*(NATMd+1); // atm layer index

		/* atm phase function index */
		if( prop_aer < RAND ){ipha  = 0;} // Rayleigh index
		else {ipha  = prof_atm[ilay].iphase + 1;} // Aerosols index
	}
	/* Scattering in ocean */
	else{
        /***********************************/
		/* Raman and Elastic scattering    */
		/***********************************/
		func = foce; // oce phases
        ilay = ph->layer + ph->ilam*(NOCEd+1); // oce layer index

		/* ocean phase function index */
		if( prop_raman < RANDTWO ){ipha  = 0;} // raman index
		else if ( prop_aer < RAND ){ipha  = 0;} // Rayleigh index
		else {ipha  = prof_oc[ilay].iphase + 1;} // Aerosols index
	}

	if(!le) {
		/* in the case of propagation (not LE) the photons scattering angle and Psi
		   rotation angle are determined randomly */
		/////////////
		// Get Theta from Cumulative Distribution Function
		zang = RAND*(NF-1);
		iang= __float2int_rd(zang);
		zang = zang - iang;

		theta = (1.-zang)*func[ipha*NF+iang].p_ang + zang*func[ipha*NF+iang+1].p_ang;
		cTh = __cosf(theta);

		/////////////
		// Get Scattering matrix from CDF
		/*P11 = func[ipha*NF+iang].p_P11;
		P12 = func[ipha*NF+iang].p_P12;
		P22 = func[ipha*NF+iang].p_P22;
		P33 = func[ipha*NF+iang].p_P33;
		P43 = func[ipha*NF+iang].p_P43;
		P44 = func[ipha*NF+iang].p_P44;*/
		P11 = (1-zang)*func[ipha*NF+iang].p_P11 + zang*func[ipha*NF+iang+1].p_P11;
		P12 = (1-zang)*func[ipha*NF+iang].p_P12 + zang*func[ipha*NF+iang+1].p_P12;
		P22 = (1-zang)*func[ipha*NF+iang].p_P22 + zang*func[ipha*NF+iang+1].p_P22;
		P33 = (1-zang)*func[ipha*NF+iang].p_P33 + zang*func[ipha*NF+iang+1].p_P33;
		P43 = (1-zang)*func[ipha*NF+iang].p_P43 + zang*func[ipha*NF+iang+1].p_P43;
		P44 = (1-zang)*func[ipha*NF+iang].p_P44 + zang*func[ipha*NF+iang+1].p_P44;

        #ifndef BIAS
		/////////////
		//  Get Psi
		//  Rejection method for sampling psi  : !!!! NEW !!!!
        float fpsi_cond=0.F; 
        float fpsi=0.F; 
        float gamma=0.F; 
        float Q = ph->stokes.y - ph->stokes.x;
        float U = ph->stokes.z;
        float DoLP = __fdividef(sqrtf(Q*Q+U*U), ph->stokes.x + ph->stokes.y);
        float K = __fdividef(P22-P11,P11+P22+2*P12);
        if (abs(Q) > 0.F) gamma   = 0.5F * atan2((double)U,(double)Q);
        float fpsi_cond_max = (1.F + DoLP * fabs(K)  )/DEUXPI;
        int niter=0;
        while (fpsi >= fpsi_cond)
            {
            niter++;
		    psi = RAND * DEUXPI;	
            fpsi= RAND * fpsi_cond_max;
            fpsi_cond = (1.F + DoLP * K * cosf(2*(psi-gamma)))/DEUXPI;
            if (niter >= 100) {
                // safety check
                #ifdef DEBUG
                printf("Warning, photon rejected in scatter while loop\n");
                printf("%i  S=(%f,%f), DoLP, gamma=(%f,%f) psi,theta=(%f,%f) \n",
                        niter,
                        Q,
                        U,
                        DoLP,
                        gamma,
                        psi/PI*180,
                        theta/PI*180
                      );
                #endif
                ph->loc = NONE;
                break;
              }
		    }
		int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
		if (idx == -1){
			printf("P11 = %.3f, P12 = %.3f, P22 = %.3f, P33 = %.3f, P43 = %.3f, P44 = %.3f\n", P11, P12, P22, P33, P43, P44);
            printf("%i  S=(%f,%f), DoLP, gamma=(%f,%f) psi,fpsi,fpsi_cond,theta=(%f,%f,%f,%f) \n",
                        niter,
                        Q,
                        U,
                        DoLP,
                        gamma,
                        psi/PI*180,
                        fpsi,
                        fpsi_cond,
                        theta/PI*180
                      );
		 }

        #else
		/////////////
		//  Get Phi
		//  Biased sampling scheme for psi 1)
		psi = RAND * DEUXPI;	
        #endif
	}
	else {
		/////////////
		// Get Index of scattering angle and Scattering matrix directly 
		zang = theta * (NF-1)/PI ;
		iang = __float2int_rd(zang);
		zang = zang - iang;

		P11 = (1-zang)*func[ipha*NF+iang].a_P11 + zang*func[ipha*NF+iang+1].a_P11;
		P12 = (1-zang)*func[ipha*NF+iang].a_P12 + zang*func[ipha*NF+iang+1].a_P12;
		P22 = (1-zang)*func[ipha*NF+iang].a_P22 + zang*func[ipha*NF+iang+1].a_P22;
		P33 = (1-zang)*func[ipha*NF+iang].a_P33 + zang*func[ipha*NF+iang+1].a_P33;
		P43 = (1-zang)*func[ipha*NF+iang].a_P43 + zang*func[ipha*NF+iang+1].a_P43;
		P44 = (1-zang)*func[ipha*NF+iang].a_P44 + zang*func[ipha*NF+iang+1].a_P44;
	}

	// Stokes vector rotation
	rotateStokes(ph->stokes, psi, &ph->stokes);

	// Scattering matrix multiplication
	float4x4 P_scatter = make_float4x4(
		P11, P12, 0. , 0.  ,
		P12, P22, 0. , 0.  ,
		0. , 0. , P33, -P43,
		0. , 0. , P43, P44
		);

	ph->stokes = mul(P_scatter, ph->stokes);

    #ifdef BACK
    float4x4 L;
    //float4x4 Lf;
    rotationM(DEUXPI-psi,&L);
    ph->M   = mul(ph->M,mul(L,P_scatter));
    //rotationM(psi,&Lf);
    //ph->Mf  = mul(mul(P_scatter,Lf),ph->Mf);
    #endif

	if (!le){
		// Bias sampling scheme 2): Debiasing
		float debias = 1.F;
        #ifdef BIAS
		debias = __fdividef( 2., P11 + P22 + 2*P12 ); // Debias is equal to the inverse of the phase function
		operator*=(ph->stokes, debias); 
        #else
        debias = __fdividef( 1., ph->stokes.x + ph->stokes.y);
		operator*=(ph->stokes, debias); 
        #endif
        #ifdef BACK
        ph->M  = mul(ph->M ,   make_diag_float4x4(debias));
        //ph->Mf = mul(ph->Mf ,  make_diag_float4x4(debias));
        #endif
	}

	else {
        //ph->weight /= 4.F; // Phase function normalization
		operator*=(ph->stokes, 0.25F); 
        #ifdef BACK
        ph->M  = mul(ph->M ,   make_diag_float4x4(0.25F));
        //ph->Mf = mul(ph->Mf ,  make_diag_float4x4(0.25F));
        #endif
    }


	if(ph->loc!=OCEAN){
		/************************/
		/* Photon in atmosphere */
		/************************/
        #ifdef ALIS
        int DL=(NLAMd-1)/(NLOWd-1);
        float P11_aer_ref, P11_ray, P22_aer_ref, P22_ray, P_ref;
        float pmol= prof_atm[ph->layer+ ph->ilam*(NATMd+1)].pmol;
        
        if (pmol <1.) {
		    zang = theta * (NF-1)/PI ;
		    iang = __float2int_rd(zang);
		    zang = zang - iang;
		    int ipharef  = prof_atm[ph->layer+ph->ilam*(NATMd+1)].iphase + 1; 
            // Phase functions of aerosols and Rayliegh, and mixture of both at reference wavelength
		    P11_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P11 + zang*func[ipharef*NF+iang+1].a_P11;
		    P11_ray     = (1-zang)*func[0      *NF+iang].a_P11 + zang*func[0      *NF+iang+1].a_P11;
		    P22_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P22 + zang*func[ipharef*NF+iang+1].a_P22;
		    P22_ray     = (1-zang)*func[0      *NF+iang].a_P22 + zang*func[0      *NF+iang+1].a_P22;
            P_ref     = (P11_ray+P22_ray) * pmol + (P11_aer_ref+P22_aer_ref) * (1.-pmol);
        }

        for (int k=0; k<NLOWd; k++) {
            ph->weight_sca[k] *= __fdividef(get_OD(1,prof_atm[ph->layer+ k*DL*(NATMd+1)]), 
                                            get_OD(1,prof_atm[ph->layer + ph->ilam*(NATMd+1)]));

            if (pmol <1.) {
		        int iphak  = prof_atm[ph->layer+k*DL*(NATMd+1)].iphase + 1; 
                float pmol_k = prof_atm[ph->layer+ k*DL*(NATMd+1)].pmol;
                // Phase functions of aerosols  at other wavelengths, Rayleigh is supposed to be constant with wavelength
		        float P11_aer = (1-zang)*func[iphak*NF+iang].a_P11 + zang*func[iphak*NF+iang+1].a_P11;
		        float P22_aer = (1-zang)*func[iphak*NF+iang].a_P22 + zang*func[iphak*NF+iang+1].a_P22;
                // Phase functions of the mixture of aerosols and Rayliegh at other wavelengths
                float P_k   = (P11_ray+P22_ray) * pmol_k + (P11_aer+P22_aer) * (1.-pmol_k);
                ph->weight_sca[k] *= __fdividef(P_k, P_ref);
                //int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
                //if (idx==0) printf("%d %d %f %d %f %f %f %f\n",ipha,ph->layer,pmol,k,pmol_k,
                //        ph->weight_sca[k],P_k,P_ref);
            }
        }
        #endif
	}
	else{
		/*******************/
		/* Photon in ocean */
		/*******************/
		if( prop_raman < RANDTWO ){	// Raman
            /* Wavelength change */
            new_wavel  = 22.94 + 0.83 * (ph->wavel) + 0.0007 * (ph->wavel)*(ph->wavel);
            ph->weight /= new_wavel/ph->wavel;
            ph->wavel = new_wavel;
		}		
	    else{ // Elastic
            #ifdef ALIS
            int DL=(NLAMd-1)/(NLOWd-1);
            float P11_aer_ref, P11_ray, P22_aer_ref, P22_ray, P_ref;
            float pmol= prof_oc[ph->layer+ ph->ilam*(NOCEd+1)].pmol;
            if (pmol <1.) {
		        zang = theta * (NF-1)/PI ;
		        iang = __float2int_rd(zang);
		        zang = zang - iang;
		        int ipharef  = prof_oc[ph->layer+ph->ilam*(NOCEd+1)].iphase + 1; 
                // Phase functions of aerosols and Rayliegh, and mixture of both at reference wavelength
		        P11_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P11 + zang*func[ipharef*NF+iang+1].a_P11;
		        P11_ray     = (1-zang)*func[0      *NF+iang].a_P11 + zang*func[0      *NF+iang+1].a_P11;
		        P22_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P22 + zang*func[ipharef*NF+iang+1].a_P22;
		        P22_ray     = (1-zang)*func[0      *NF+iang].a_P22 + zang*func[0      *NF+iang+1].a_P22;
                P_ref     = (P11_ray+P22_ray) * pmol + (P11_aer_ref+P22_aer_ref) * (1.-pmol);
            }
            for (int k=0; k<NLOWd; k++) {
                 ph->weight_sca[k] *= __fdividef(get_OD(1,prof_oc[ph->layer+ k*DL*(NOCEd+1)]), 
                     get_OD(1,prof_oc[ph->layer + ph->ilam*(NOCEd+1)]));
                if (pmol <1.) {
		            int iphak  = prof_oc[ph->layer+k*DL*(NOCEd+1)].iphase + 1; 
                    float pmol_k = prof_oc[ph->layer+ k*DL*(NOCEd+1)].pmol;
                    // Phase functions of aerosols  at other wavelengths, Rayleigh is supposed to be constant with wavelength
		            float P11_aer = (1-zang)*func[iphak*NF+iang].a_P11 + zang*func[iphak*NF+iang+1].a_P11;
		            float P22_aer = (1-zang)*func[iphak*NF+iang].a_P22 + zang*func[iphak*NF+iang+1].a_P22;
                    // Phase functions of the mixture of aerosols and Rayliegh at other wavelengths
                    float P_k   = (P11_ray+P22_ray) * pmol_k + (P11_aer+P22_aer) * (1.-pmol_k);
                    ph->weight_sca[k] *= __fdividef(P_k, P_ref);
                }
            }
            #endif
		}

	} //ocean
		
	if (!le){
		/** Russian roulette for propagating photons **/
		if( ph->weight < WEIGHTRR ){
			if( RAND < __fdividef(ph->weight,WEIGHTRR) ){ph->weight = WEIGHTRR;}
			else{ph->loc = ABSORBED;}
		}
		modifyUV( ph->v, ph->u, cTh, psi, &ph->v, &ph->u) ;
	}
    else {
        ph->weight /= fabs(ph->v.z);
    }
	
}


__device__ void surfaceAgitee(Photon* ph, int le,
                              float* tabthv, float* tabphi, int count_level,
                              struct RNG_State *rngstate) {
	
	if( SIMd == -2){ // Atmosphère , la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
    ph->nint += 1;
	
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
    // coordinates of the half direction vector of the wave facet in the original axis (see Walter 2007)
    float3 half;
    // coordinates of the normal to the wave facet in the local axis (Nx, Ny, Nz)
	float3 n_l;

	float rpar, rper, rparper, rparper_cross;	// Coefficient de reflexion parallèle et perpendiculaire
	float rpar2;		// Coefficient de reflexion parallèle au carré
	float rper2;		// Coefficient de reflexion perpendiculaire au carré
	float rat;			// Rapport des coefficients de reflexion perpendiculaire et parallèle
	int ReflTot;		// Flag pour la réflexion totale sur le dioptre
	float cot;			// Cosinus de l'angle de réfraction du photon
	float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
	float tpar, tper, tparper, tpar2, tper2;	//
    int iter=0;
    float vzn;  // projection of V on the local vertical
    float thv, phi;
	float3 v, v_l;

    // Reflection  and Transmission Matrices
    float4x4 R, T;

    // Determination of the relative refractive index
    // a: air, b: water , Mobley 2015 nind = nba = nb/na
    // in general nind = n_t/n_i or no/ni (transmitted over incident or output versus input)
    // and sign for further computation, sign positive for upward facet normal for reflection
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

	Nz = ph->pos; // Nz is the vertical at the impact point

    // Ny is chosen arbitrarily by cross product of Nz with axis X = (1,0,0)
	Ny = cross(Nz, make_float3(1.0,0.0,0.0));

    // Nx is the cross product of Ny and Nz
	Nx = cross(Ny, Nz);
 
	// Normalization
	Nx = normalize(Nx);
	Ny = normalize(Ny);
	Nz = normalize(Nz);

    #ifdef DEBUG
    // we check that there is no upward photon reaching surface0+
    if ((ph->loc == SURF0P) && (dot(ph->v, ph->pos) > 0)) {
        // upward photon when reaching the surface at (0+)
        printf("Warning, vzn>0 (vzn=%f) with SURF0+ in surfaceAgitee, %f %f %f %f %f %f\n", dot(ph->v, ph->pos),Nz.x,Nz.y,Nz.z, ph->pos.x,ph->pos.y,ph->pos.z);
    }
    #endif

    /* Compute the photon v vector in the local frame */
    v_l.x = dot(ph->v,Nx);
    v_l.y = dot(ph->v,Ny);
    v_l.z = dot(ph->v,Nz);

    #else
    v_l = ph->v;
    #endif

    if (ph->loc==SURF0M) v_l = ph->v;

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


    /* SAMPLING */

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

           // Normal of the facet in the local frame
           n_l.x = sign * sBeta * __cosf( alpha );
           n_l.y = sign * sBeta * __sinf( alpha );
           n_l.z = sign * cBeta;

           // Compute incidence angle //
           cTh = -(dot(n_l,v_l));
           theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));
        }
     } else {
        // Flat surface
        beta  = 0.F;
        cBeta = 1.F;
        n_l.x   = 0.F;
        n_l.y   = 0.F;
        n_l.z   = sign;

        cTh = -(dot(n_l, v_l));
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
     // We refer to Walter 2007
     // i : input unit vector, directed outward facet, so i=-ph->v
     // o : output unit vector, so o=v

     // 1) Determination of the half direction vector
     if ((ph->loc==SURF0P) && (count_level==DOWN0M) ||
         (ph->loc==SURF0M) && (count_level==UP0P))   { // Refraction geometry
        // vector equation for determining the half direction half = - (no*o + ni*i)
        // or half = - (nind*o + i)
        // The convention in Walter is h pointing towards the medieum with lowest index of refraction
        /*****/
        // So
		 half = operator-(v*nind, ph->v) *(-1.F*sign);
         // test : exclude facets whose normal are not on the same side as incoming photons
         if ((half.z * sign) < 0) {
             ph->weight = 0.F;
             ph->loc=REMOVED;
             return;
         }
     }
     if ((ph->loc==SURF0P) && (count_level==UP0P) ||
         (ph->loc==SURF0M) && (count_level==DOWN0M)) { // Reflection geometry
        // vector equation for determining the half direction h = (o + i)
		 half = operator-(v, ph->v);
     }


     // 2) Normalization of the half direction vector: facet normal unit vector
     no=normalize(half);
     //no=normalize(no);

     // Incidence angle
     cTh = fabs(-dot(no, ph->v));
     theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));

     #ifdef SPHERIQUE
     // facet slope
     cBeta = fabs(dot(no, Nz));
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

	sTh = __sinf(theta);

    // express the coordinates of the normal to the wave facet in the original
    // axis instead of local axis (Nx, Ny, Nz)
    if (!le) {
    #ifdef SPHERIQUE
	no = operator+(operator+(n_l.x*Nx, n_l.y*Ny), n_l.z*Nz);
    #else
    no = n_l;
    #endif
    }


    #ifdef SPHERIQUE
    // avz is the projection of V on the local vertical
	float avz = fabs(dot(Nz, ph->v));
    #else
    float avz = fabs(ph->v.z);
    #endif

	// Rotation of Stokes parameters

	//temp = dot(cross(ph->v,ph->u),normalize(cross(ph->v,no)));
    // Simplification :
	temp = __fdividef(dot(no, ph->u), sTh);
	psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

	if( dot(no, cross(ph->u, ph->v)) <0 ){
		psi = -psi;
	}

    rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L = make_diag_float4x4 (1.F);
    rotationM(DEUXPI-psi,&L);
    #endif

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
		rat =  __fdividef(rper2 + rpar2, 2.F);
        //rat =  __fdividef(ph->stokes.x*rper2 + ph->stokes.y*rpar2,ph->stokes.x+ph->stokes.y);
		ReflTot = 0;
        //int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
        //if (idx==0 && (ph->loc==SURF0M) && (count_level==UP0P) && (le)) printf("DEB %.3f %.3f %.3f %.4f\n", v.z, no.z, ph->v.z,psi);
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
    float p,qv,LambdaS,LambdaR,jac;

    // Ross et al 2005, Ross and Dion, 2007, Zeisse 1995
    // Slope sampling bias correction using the normalized interaction PDF q
    // weight has to be multiplied by q/p, where p is the slope PDF
    // Coefficient Lambda for normalization of q taking into acount slope shadowing and hiding
    // Including wave shadows is performed at the end after the outgoing direction is calculated

    // 1. Source direction
    LambdaS  =  Lambda(avz,sig);

    // Bias sampling correction
    if (!le) ph->weight *= __fdividef(fabs(cTh), cBeta * (1.F + LambdaS) * avz );
    
    if (le && (DIOPTREd!=0)) {
     // The weight depends on the normalized VISIBLE interaction PDF qv (Ross 2005) 
     // Compute p 
     p =  __fdividef( __expf(-(1.F-cBeta*cBeta)/(cBeta*cBeta*sig2)) ,  cBeta*cBeta*cBeta * sig2); 

     if ((ph->loc==SURF0P) && (count_level==UP0P) ||
         (ph->loc==SURF0M) && (count_level==DOWN0M)) { // Reflection geometry
            qv  = __fdividef(p * fabs(cTh), cBeta * fabs(v.z));
            // Multiplication by the reflection Jacobian
            jac = __fdividef(1.F, 4.F * fabs(cTh) );
     }
     if ((ph->loc==SURF0P) && (count_level==DOWN0M) ||
         (ph->loc==SURF0M) && (count_level==UP0P))   { // Refraction geometry
            if (sTh <= nind) {
                qv  =  __fdividef(p * fabs(cTh), cBeta * fabs(v.z));
                // Multiplication by the refraction Jacobian
                jac = __fdividef(nind*nind * cot, (ncot - cTh)*(ncot - cTh)); // See Zhai et al., 2010
            }
            else qv = 0.F;
            #ifdef BACK
            /* for reciprocity of transmission function see Walter 2007 */
            #endif
     }

     // 2. Reflected/Refracted direction, Normalization of qv
     LambdaR  =  Lambda(fabs(v.z),sig);

     if (WAVE_SHADOWd)
         qv /= (1.F + LambdaS + LambdaR);
     else
         qv /= (1.F + LambdaS);

     // apply the BRDF to the weight
     ph->weight *= __fdividef(qv * jac , avz);


    } /*le */

    int condR=1;
    if (!le) condR = (SURd==3)&&(RAND<rat);

	if (  (!le && (condR || (SURd==1) || ReflTot) )
       || ( le && (ph->loc==SURF0M) && (count_level == DOWN0M) )
       || ( le && (ph->loc==SURF0P) && (count_level == UP0P)   )
       ){	// Reflection

	    R= make_float4x4(
		    rper2, 0., 0., 0.,
		    0., rpar2, 0., 0.,
		    0., 0.,  rparper, rparper_cross,
		    0., 0., -rparper_cross, rparper
		    );

        ph->stokes = mul(R,ph->stokes);

        #ifdef BACK
        ph->M   = mul(ph->M,mul(L,R));
        #endif
		
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
        vzn = dot(ph->v, Nz);
        #else
        vzn = ph->v.z;
        #endif

        //
        // photon next location
        //
         if (ph->loc == SURF0P) {
            if (vzn > 0) {  // avoid multiple reflexion above the surface
                // SURF0P becomes ATM or SPACE
                if( SIMd==-1 || SIMd==0 ){
                    ph->loc = SPACE;
                } else{
                    ph->loc = ATMOS;
                }
            } // else, no change of location
        else if (SINGLEd) ph->loc = REMOVED;
     } else {
        if (vzn < 0) {  // avoid multiple reflexion under the surface
            // SURF0M becomes OCEAN or ABSORBED
            if( SIMd==1 ){
                ph->loc = ABSORBED;
            } else{
                ph->loc = OCEAN;
            }
        } // else, no change of location
        else if (SINGLEd) ph->loc = REMOVED;
     }


} // Reflection


else if (  (!le && !condR) 
        //|| ( le && (ph->loc==SURF0M) && (count_level == UP0P  ) )
        //|| ( le && (ph->loc==SURF0P) && (count_level == DOWN0M) )
        || ( le && (ph->loc==SURF0M) && (count_level == UP0P  ) && !ReflTot )
        || ( le && (ph->loc==SURF0P) && (count_level == DOWN0M) && !ReflTot )
        ){	// Transmission

    float geo_trans_factor = nind* cot/cTh; // DR Mobley 2015 OK , see Xun 2014, Zhai et al 2010
    T= make_float4x4(
        tper2, 0., 0., 0.,
        0., tpar2, 0., 0.,
        0., 0., tparper, 0.,
        0., 0., 0., tparper
        );
    
    ph->stokes = mul(T,ph->stokes);
    ph->weight *= geo_trans_factor;

    #ifdef BACK
    ph->M   = mul(ph->M,mul(L,T));
    //ph->weight /= nind*nind;
    #endif
    
    alpha  = __fdividef(cTh, nind) - cot;

    if (le) { ph->v = v; }
    else { ph->v = operator+(operator/(ph->v, nind), alpha*no); }
    ph->u = operator/(operator+(no, cot*ph->v), sTh )*nind;

    #ifdef SPHERIQUE
    vzn = dot(ph->v, Nz);
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
            if (SINGLEd) ph->loc = REMOVED;
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
            ph->loc = SURF0M;
            if (SINGLEd) ph->loc = REMOVED;
        }
     }

	} // Transmission

    LambdaR  =  Lambda(fabs(ph->v.z),sig);
    if (WAVE_SHADOWd && !le) 
        ph->weight *= __fdividef(1.F + LambdaS, 1.F + LambdaR + LambdaS);

    #ifdef BACK
    if (!le){
        if (!WAVE_SHADOWd) ph->weight *= __fdividef(avz * (1.F + LambdaS), fabs(ph->v.z) * (1.F + LambdaR));
        // else   ph->weight *= __fdividef(avz, fabs(ph->v.z));
    }
    #endif

    if (!le) {
		/* Russian roulette for propagating photons **/
		if( ph->weight < WEIGHTRR ){
			if( RAND < __fdividef(ph->weight,WEIGHTRR) ){ph->weight = WEIGHTRR;}
			else{ph->loc = ABSORBED;}
		}
    }

}

/* Surface BRDF */
__device__ void surfaceBRDF(Photon* ph, int le,
                              float* tabthv, float* tabphi, int count_level,
                              struct RNG_State *rngstate) {
	
	if( SIMd == -2){ // Atmosphère , la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
    ph->nint += 1;
	
	// Réflexion sur le dioptre agité
	float theta;	// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float cTh, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	
	float sig, sig2  ;
	float cBeta;
	
	float nind; // relative index of refrection 
	float temp;
	
    // coordinates of the normal to the wave facet in the original axis
	float3 no;

	float rpar, rper, rparper, rparper_cross;	// Coefficient de reflexion parallèle et perpendiculaire
	float rpar2;		// Coefficient de reflexion parallèle au carré
	float rper2;		// Coefficient de reflexion perpendiculaire au carré
	float cot;			// Cosinus de l'angle de réfraction du photon
	float ncot, ncTh;	// ncot = nind*cot, ncoi = nind*cTh
    float thv, phi;
	float3 v;

    // Reflection  and Transmission Matrices
    float4x4 R;

    // Determination of the relative refractive index
    // a: air, b: water , Mobley 2015 nind = nba = nb/na
    // and sign for further computation
    nind = NH2Od;
     
    #ifdef SPHERIQUE
    // define 3 vectors Nx, Ny and Nz in cartesian coordinates which define a
    // local orthonormal basis at the impact point.
    // Nz is the local vertical direction, the direction of the 2 others does not matter
    // because the azimuth is chosen randomly
	float3 Nx, Ny, Nz;

	Nz = ph->pos; // Nz is the vertical at the impact point

    // Ny is chosen arbitrarily by cross product of Nz with axis X = (1,0,0)
	Ny = cross(Nz, make_float3(1.0,0.0,0.0));

    // Nx is the cross product of Ny and Nz
	Nx = cross(Ny, Nz);
 
	// Normalizatioin
	Nx = normalize(Nx);
	Ny = normalize(Ny);
	Nz = normalize(Nz);

    #ifdef DEBUG
    // we check that there is no upward photon reaching surface0+
    if ((ph->loc == SURF0P) && (dot(ph->v, ph->pos) > 0)) {
        // upward photon when reaching the surface at (0+)
        printf("Warning, vzn>0 (vzn=%f) with SURF0+ in surfaceBRDF\n", dot(ph->v, ph->pos));
    }
    #endif
    #endif

    sig = sqrtf(0.003F + 0.00512f *WINDSPEEDd);
    sig2= sig * sig;

    // Rough surface
    if (le) {
     phi = tabphi[ph->iph];
     thv = tabthv[ph->ith];
    }
    else {
	 phi = RAND*DEUXPI;
	 thv = acosf(sqrtf( RAND ));
    }

    v.x  = cosf(phi) * sinf(thv);
    v.y  = sinf(phi) * sinf(thv);
    v.z  = cosf(thv);  
     
    // vector equation for determining the half direction h = sign(i dot o) (i + o)
	no = operator-(v, ph->v);

    // 2) Normalization of the half direction vector
    no=normalize(no);

    // Incidence angle in the local frame
    cTh = fabs(-dot(no, ph->v));
    theta = acosf( fmin(1.00F-VALMIN, fmax( -(1.F-VALMIN), cTh ) ));

    #ifdef SPHERIQUE
    // facet slope
    cBeta = fabs(dot(no, Nz));
    #else
    cBeta = fabs(no.z);
    #endif

	sTh = __sinf(theta);

    #ifdef SPHERIQUE
    // avz is the projection of V on the local vertical
	float avz = fabs(dot(Nz, ph->v));
    #else
    float avz = fabs(ph->v.z);
    #endif

	// Rotation of Stokes parameters
    temp = __fdividef(dot(no, ph->u), sTh);
	psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

	if( dot(no, cross(ph->u, ph->v)) <0 ){
		psi = -psi;
	}

    rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L = make_diag_float4x4 (1.F);
    rotationM(DEUXPI-psi,&L);
    #endif

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

    // Weighting
    float LambdaS;
    LambdaS  =  Lambda(avz,sig);

    ph->weight *= __fdividef(1.F, cBeta * avz * fabs(v.z) ); // Common to all photons, cBeta for surface area unit correction
    
    ph->weight *=  __fdividef( __expf(-(1.F-cBeta*cBeta)/(cBeta*cBeta*sig2)) ,  cBeta*cBeta*cBeta * sig2)
                 * __fdividef(1.F, 4.F  );
    if (ph->weight <= 1e-15) {
         ph->weight = 0.;
         //return;
    }

	R= make_float4x4(
	   rper2, 0., 0., 0.,
	   0., rpar2, 0., 0.,
	   0., 0., rparper, rparper_cross,
	   0., 0., -rparper_cross, rparper
	   );

    ph->stokes = mul(R,ph->stokes);

    #ifdef BACK
    ph->M   = mul(ph->M,mul(L,R));
    #endif
		
    ph->v = v;
	ph->u = operator/(operator-(no, cTh*ph->v), sTh);	

        // photon next location
    if( SIMd==-1 || SIMd==0 ){
        ph->loc = SPACE;
    } else {
          ph->loc = ATMOS;
    }

    if (WAVE_SHADOWd) {
        // Add Wave shadowing
        // compute wave shadow outgoing photon
        float LambdaR;
        LambdaR  =  Lambda(fabs(ph->v.z),sig);
        //compute wave shadow function incoming and outgoing photon
        ph->weight *= __fdividef(1.F, 1.F + LambdaR + LambdaS);
    }

    if (!le) {
		/* Russian roulette for propagating photons **/
		if( ph->weight < WEIGHTRR ){
			if( RAND < __fdividef(ph->weight,WEIGHTRR) ){ph->weight = WEIGHTRR;}
			else{ph->loc = ABSORBED;}
		}
    }

}

/* surfaceLambertienne
* Reflexion sur une surface lambertienne
*/
__device__ void surfaceLambertienne(Photon* ph, int le,
                                    float* tabthv, float* tabphi, struct Spectrum *spectrum,
                                    struct RNG_State *rngstate) {
	
	if( SIMd == -2){ 	// Atmosphère ou océan seuls, la surface absorbe tous les photons
		ph->loc = ABSORBED;
		return;
	}
	
    ph->nint += 1;
	float3 u_n, v_n;	// Vecteur du photon après reflexion
    float phi;
    float cTh, sTh, cPhi, sPhi;

    if (le) {
        cTh  = cosf(tabthv[ph->ith]);  
        phi  = tabphi[ph->iph];
        //ph->weight *= cTh;
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
		ph->loc = ABSORBED;	
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
    float4x4 L = make_float4x4(
                    0.5F, 0.5F, 0.F, 0.F,
                    0.5F, 0.5F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F 
            );
    ph->stokes = mul(L,ph->stokes);

    #ifdef BACK
    ph->M = mul(L,ph->M);
    #endif
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
        struct Profile *prof_atm, struct Profile *prof_oc,
        float *tabthv, float *tabphi,
        int count_level,
		unsigned long long *errorcount,
        void *tabPhotons, unsigned long long *NPhotonsOut
        ) {

    if (count_level < 0 || ph->loc==REMOVED || ph->loc==ABSORBED) {
        // don't count anything
        return;
    }

    // don't count the photons directly transmitted
    //if ((ph->weight == WEIGHTINIT) && (ph->stokes.x == ph->stokes.y) && (ph->stokes.z == 0.f) && (ph->stokes.w == 0.f)) {
    if (ph->nint == 0) {
        return;
    }

    #ifdef DOUBLE 
    double *tabCount;                   // pointer to the "counting" array:
    double dweight;
	double4 ds;                         // replace ds1, ds2, ds3, ds4
    #ifdef ALIS
    double dwsca, dwabs;
    #endif
    #else                               // may be TOA, or BOA down, and so on
    float *tabCount; 
    #endif

    float theta = acosf(fmin(1.F, fmax(-1.F, ph->v.z)));
    #ifdef SPHERIQUE
    if(ph->v.z<=0.f) {
         // do not count the downward photons leaving atmosphere
         // DR April 2016, test flux for spherical shell
         // !! TO TEST in glitter + spherical !!
         //return;
         // !! TO TEST in glitter + spherical !!
    }
    #endif

	float psi=0.;
	int ith=0, iphi=0, il=0;
    float4 st; // replace s1, s2, s3, s4
    int II, JJ;


    if ((theta != 0.F) && (theta!= acosf(-1.F))) {
       ComputePsi(ph, &psi, theta);
    }
    else {
       if (LEd == 0) {
          atomicAdd(errorcount+ERROR_THETA, 1);
          return;
       }
       else {
          // Compute Psi in the special case of zenith
          ComputePsiZenith(ph,&psi,tabphi[ph->iph]);
       }
    }

    rotateStokes(ph->stokes, psi, &st);
    st.w = ph->stokes.w;

    #ifdef BACK
    float4x4 L;
    float4 stback = make_float4(0.5F, 0.5F, 0., 0.);
    //float4 stforw = make_float4(0.5F, 0.5F, 0., 0.);
    //rotationM(psi,&L);
	//ph->Mf = mul(L,ph->Mf);
    rotationM(DEUXPI-psi,&L);
	ph->M = mul(ph->M,L);
    stback = mul(ph->M, stback);
    //stforw = mul(ph->Mf,stforw);
    st = stback;
    #endif

	float weight = ph->weight;
    #ifdef ALIS
        float weight_sca[MAX_NLOW];
        for (int k=0; k<NLOWd; k++) {
            weight_sca[k] = ph->weight_sca[k];
        }
    #endif

	// Compute Box for outgoing photons in case of cone sampling
	if (LEd == 0) ComputeBox(&ith, &iphi, &il, ph, errorcount);

    // For virtual (LE) photons the direction is stored within photon structure
    // Moreover we compute also final attenuation for LE 
    else {
        ith = ph->ith;
        iphi= ph->iph;
        il  = ph->ilam;

        if (!(   (SIMd==-1) 
              || (NATMd==0 && (count_level==UPTOA || count_level==UP0P)) 
              || (NOCEd==0 && count_level==UP0P)
             )
           ){
        // Computation of final attenutation only in PP
        #ifndef SPHERIQUE
        int layer_le;
        float tau_le;
        Profile *prof;
        int layer_end;

        // define correct start and end layers and profiles for LE
        if (count_level==UPTOA) {
            layer_le = 0; 
            layer_end= NATMd;
            prof = prof_atm;
        }
        if ((count_level==DOWN0P) || (count_level==DOWN0M) || (count_level==UP0P) || (count_level==UP0M) ) {
            if ((ph->loc == ATMOS) || (ph->loc == SURF0M) || (ph->loc == SURF0P) ) {
                layer_le = NATMd;
                layer_end= NATMd;
                prof = prof_atm;
            }
            if ((ph->loc == OCEAN) || (ph->loc == SEAFLOOR)) {
                layer_le = NOCEd;
                layer_end= NOCEd;
                prof = prof_oc;
            }
        }
        if (count_level==DOWNB) {
            layer_le = 0;
            layer_end= NOCEd;
            prof = prof_oc;
        }

        #ifndef ALIS
        // Attenuation of the current photon
        // First get the extinction optical depth at the counting level
        tau_le = prof[(layer_end-layer_le) + ph->ilam *(layer_end+1)].OD;
        // if BEER=0, photon variable tau corresponds to extinction
        if (BEERd == 0) weight *= expf(-fabs(__fdividef(tau_le - ph->tau, ph->v.z))); // LE attenuation to count_level
        // if BEER=1, photon variable tau corresponds to scattering only, need to add photon absorption variable
        else weight *= expf(-fabs(__fdividef(tau_le - (ph->tau+ph->tau_abs), ph->v.z))); // LE attenuation to count_level

        // Specific computation for ALIS
        #else
        float dsca_dl, dsca_dl0;
        int DL=(NLAMd-1)/(NLOWd-1);

        // Complete photon history toward space for further final absorption computation
        ph->layer = 0;
        ph->nevt++;
        ph->layer_prev[ph->nevt] = ph->layer;
        ph->vz_prev[ph->nevt] = ph->v.z;
        ph->epsilon_prev[ph->nevt] = 0.f;
        
        // Attenuation by scattering only of the main 'central' or 'reference' photon
        // First get the scattering optical depth at the counting level
        tau_le = prof[(layer_end-layer_le) + ph->ilam *(layer_end+1)].OD_sca;
        // LE attenuation to count_level without absorption, central wavelength
        dsca_dl0 = tau_le - ph->tau; 
        weight *= expf(-fabs(__fdividef(dsca_dl0, ph->v.z)));

        // Differential LE scattering attenuation to count_level for others 'scattering' wavelengths
        for (int k=0; k<NLOWd; k++) {
           dsca_dl = prof[(layer_end-layer_le) + k*DL*(layer_end+1)].OD_sca - ph->tau_sca[k]; 
           weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) -fabs(dsca_dl0), fabs(ph->v.z)));
        }
        #endif // NOT ALIS
        #endif // NOT SPHERIQUE
     } // SIMd  

    }   //LE
	
  	/*if( ph->vy<0.f )
    		s3 = -s3;*/  // DR 
	
    // Change sign convention for compatibility with OS
    st.z = -st.z;

	float tmp = st.x;
	st.x = st.y;
	st.y = tmp;
	


    float weight_irr = fabs(ph->v.z);
	//if (FLUXd==1 && LEd==0 & weight_irr > 0.01f) weight /= weight_irr;
    // In Forward mode, and in case of spherical flux, update the weight
	if (BACKd ==0 && FLUXd==2 && LEd==0 & weight_irr > 0.001f) weight /= weight_irr;

    #ifdef DEBUG
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
    if (isnan(weight)) printf("(idx=%d) Error, weight is NaN, %d\n", idx,ph->loc);
    if (isnan(st.x)) printf("(idx=%d) Error, s1 is NaN\n", idx);
    if (isnan(st.y)) printf("(idx=%d) Error, s2 is NaN\n", idx);
    if (isnan(st.z)) printf("(idx=%d) Error, s3 is NaN\n", idx);
    #endif

	// Rangement du photon dans sa case, et incrémentation de variables
    II = NBTHETAd*NBPHId*NLAMd;

    // Regular counting procedure
    #ifndef ALIS
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (il >= 0) && (il < NLAMd) && (!isnan(weight)))
	{
      JJ = il*NBTHETAd*NBPHId + ith*NBPHId + iphi;

      #ifdef DOUBLE 
      // select the appropriate level (count_level)
      tabCount = (double*)tabPhotons + count_level*NPSTKd*NBTHETAd*NBPHId*NLAMd;
      dweight = (double)weight;
      ds = make_double4(st.x, st.y, st.z, st.w);
      DatomicAdd(tabCount+(0*II+JJ), dweight*(ds.x+ds.y));
      DatomicAdd(tabCount+(1*II+JJ), dweight*(ds.x-ds.y));
      DatomicAdd(tabCount+(2*II+JJ), dweight*ds.z);
      DatomicAdd(tabCount+(3*II+JJ), dweight*ds.w);
      // If GTX 1000 or more recent use native double atomic add
      /*atomicAdd(tabCount+(0*II+JJ), dweight*(ds.x+ds.y));
      atomicAdd(tabCount+(1*II+JJ), dweight*(ds.x-ds.y));
      atomicAdd(tabCount+(2*II+JJ), dweight*ds.z);
      atomicAdd(tabCount+(3*II+JJ), dweight*ds.w);*/

      #else
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

    #else //ALIS
    int DL=(NLAMd-1)/(NLOWd-1);
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (!isnan(weight)))
    {
      // For all wavelengths
      for (il=0; il<NLAMd; il++) {
          float wabs = 1.0f;
          JJ = il*NBTHETAd*NBPHId + ith*NBPHId + iphi;

          // Linear interpolation upon wavelength of the scattering correction
          int ik=il/DL;
          float wsca;
          if (il != NLAMd-1) wsca = __fdividef((il-ik*DL)*1.0f,DL*1.0f) * (weight_sca[ik+1] - weight_sca[ik]) +
                          weight_sca[ik]; 
          else wsca = weight_sca[NLOWd-1];
          
          //  OR Polynomial fit for scattering correction, !!DEV
          /* 
          float wsca = 0.;
          for (int k=0; k<NLOWd; k++){
            float acc = 1.f;
            for (int j=0; j< NLOWd; j++) {
                if (j!=k) acc *= __fdividef((float)il-(float)j*DL,(float)k*DL-(float)j*DL); 
            }
            wsca += ph->weight_sca[k] * acc;
           }
          */
        
          // Computation of the absorption along photon s history
          for (int n=0; n<ph->nevt; n++){
              //Computing absorption optical depths form start to stop for all segments
              float tau_abs1, tau_abs2;
              if (ph->layer_prev[n+1] == 0) tau_abs2 = 0.;
              else tau_abs2 = (prof_atm[ph->layer_prev[n+1]   + il *(NATMd+1)].OD_abs -
                             prof_atm[ph->layer_prev[n+1]-1 + il *(NATMd+1)].OD_abs) *
                             ph->epsilon_prev[n+1] + prof_atm[ph->layer_prev[n+1]-1 + il *(NATMd+1)].OD_abs;
              if (ph->layer_prev[n] == 0) tau_abs1 = 0.;
              else tau_abs1 = (prof_atm[ph->layer_prev[n]   + il *(NATMd+1)].OD_abs -
                             prof_atm[ph->layer_prev[n]-1 + il *(NATMd+1)].OD_abs) *
                             ph->epsilon_prev[n] + prof_atm[ph->layer_prev[n]-1 + il *(NATMd+1)].OD_abs;
              wabs *= exp(-fabs(__fdividef(tau_abs2 - tau_abs1 , ph->vz_prev[n+1])));
          }

          #ifdef DOUBLE 
          tabCount = (double*)tabPhotons + count_level*NPSTKd*NBTHETAd*NBPHId*NLAMd;
          dweight = (double)weight;
          ds = make_double4(st.x, st.y, st.z, st.w);
          dwsca=(double)wsca;
          dwabs=(double)wabs;
          DatomicAdd(tabCount+(0*II+JJ), dweight * dwsca * dwabs * (ds.x+ds.y));
          DatomicAdd(tabCount+(1*II+JJ), dweight * dwsca * dwabs * (ds.x-ds.y));
          DatomicAdd(tabCount+(2*II+JJ), dweight * dwsca * dwabs * ds.z);
          DatomicAdd(tabCount+(3*II+JJ), dweight * dwsca * dwabs * ds.w);

          #else
          tabCount = (float*)tabPhotons + count_level*NPSTKd*NBTHETAd*NBPHId*NLAMd;
          atomicAdd(tabCount+(0*II+JJ), weight * wsca * wabs * (st.x+st.y));
          atomicAdd(tabCount+(1*II+JJ), weight * wsca * wabs * (st.x-st.y));
          atomicAdd(tabCount+(2*II+JJ), weight * wsca * wabs * st.z);
          atomicAdd(tabCount+(3*II+JJ), weight * wsca * wabs * st.w);
          #endif    

          atomicAdd(NPhotonsOut + ((count_level*NLAMd + il)*NBTHETAd + ith)*NBPHId + iphi, 1);
      }

    }
	else
	{
		atomicAdd(errorcount+ERROR_CASE, 1);
	}
    #endif //ALIS

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

//
// Rotation Matrix L from an angle psi between the incidence and
// the emergence planes
__device__ void rotationM(float psi, float4x4 *L)
{
    float cPsi = __cosf(psi); float sPsi = __sinf(psi); float cPsi2 = cPsi * cPsi; float sPsi2 = sPsi * sPsi;
	float twopsi = 2.F*psi;  float s2Psi = __sinf(twopsi); float a = 0.5f*s2Psi;

	*L = make_float4x4(
		cPsi2, sPsi2, -a, 0.f,               
		sPsi2, cPsi2, a, 0.f,                 
		s2Psi, -s2Psi, __cosf(twopsi), 0.f,
        0.f, 0.f, 0.f, 1.f
		);
}


/* ComputePsi
*/
__device__ void ComputePsi(Photon* ph, float* psi, float theta)
{
    // see Rammella et al. Three Monte Carlo programs of polarized light transport into scattering media: part I Optics Express, 2005, 13, 4420
    double wz;
    wz = (double)ph->v.x * (double)ph->u.y - (double)ph->v.y * (double)ph->u.x;
    *psi = atan2(wz, -1.e+00*(double)ph->u.z); 
}

/* ComputePsiZenith
*/
__device__ void ComputePsiZenith(Photon* ph, float* psi, float phi)
{
        // Compute Psi in the special case of zenith
        float ux_phi;
        float uy_phi;
        float cos_psi;
        float sin_psi;
        float eps=1e-4;

        ux_phi  = cosf(phi);
        uy_phi  = sinf(phi);
        cos_psi = (ux_phi*ph->u.x + uy_phi*ph->u.y);
        if( cos_psi >  1.0) cos_psi =  1.0;
        if( cos_psi < -1.0) cos_psi = -1.0;
        sin_psi = sqrtf(1.0 - (cos_psi*cos_psi));
        if( (abs((ph->u.x*cos_psi-ph->u.y*sin_psi)-ux_phi) < eps) && (abs((ph->u.x*sin_psi+ph->u.y*cos_psi)-uy_phi) < eps) ) {
                *psi = -acosf(cos_psi);
        }
        else{
                *psi = acosf(cos_psi);
        } 
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
        printf("%16s X=(%6.3f,%6.3f,%6.3f) V=(%6.3f,%6.3f,%6.3f) U=(%6.3f,%6.3f,%6.3f) S=(%6.3f,%6.3f,%6.3f,%6.3f) tau=%6.3f tau_abs=%6.3f weight=%11.3e loc=",
               desc,
               ph->pos.x, ph->pos.y, ph->pos.z,
               ph->v.x,ph->v.y,ph->v.z,
               ph->u.x,ph->u.y,ph->u.z,
               ph->stokes.x, ph->stokes.y,
               ph->stokes.z, ph->stokes.w,
               ph->tau,ph->tau_abs, ph->weight
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
            case 8: printf("REMOVED"); break;
            default:
                    printf("UNDEFINED");
        }
        #ifdef ALIS
        printf(" nevt=%2d",ph->nevt);
        printf(" wsca=");
        for (int k=0; k<NLOWd; k++) printf("%7.5f ",ph->weight_sca[k]);
        printf(" dtausca=");
        for (int k=0; k<NLOWd; k++) printf("%7.5f ",ph->tau_sca[k]);
        printf(" layers=");
        for (int k=0; k<ph->nevt+1; k++) printf("%3d ",ph->layer_prev[k]);
        printf(" vz=");
        for (int k=0; k<ph->nevt+1; k++) printf("%7.5f ",ph->vz_prev[k]);
        printf(" delta=");
        for (int k=0; k<ph->nevt+1; k++) printf("%7.5f ",ph->epsilon_prev[k]);
        #endif
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
	float EPS6 = 1e-4;	
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

	else{ operator/=(w1, den); }
	
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
    ph_le->tau_abs = ph->tau_abs;
    ph_le->layer = ph->layer;
    ph_le->weight = ph->weight;
    ph_le->wavel = ph->wavel;
    ph_le->ilam = ph->ilam;
    ph_le->prop_aer = ph->prop_aer;
    ph_le->pos = ph->pos; // float3
    ph_le->nint = ph->nint;
    #ifdef SPHERIQUE
    ph_le->radius = ph->radius;
    ph_le->taumax = ph->taumax;
    #endif
    #ifdef ALIS
    int k, kmax=ph->nevt+1;
    ph_le->nevt = ph->nevt;
    for (k=0; k<kmax; k++) ph_le->layer_prev[k] = ph->layer_prev[k];
    for (k=0; k<kmax; k++) ph_le->vz_prev[k] = ph->vz_prev[k];
    for (k=0; k<kmax; k++) ph_le->epsilon_prev[k] = ph->epsilon_prev[k];
    for (k=0; k<NLOWd; k++) ph_le->weight_sca[k] = ph->weight_sca[k];
    for (k=0; k<NLOWd; k++) ph_le->tau_sca[k] = ph->tau_sca[k];
    #endif
    #ifdef BACK
    int kk;
    for (kk=0; kk<16; kk++) ph_le->M[kk] = ph->M[kk];
    //for (kk=0; kk<16; kk++) ph_le->Mf[kk] = ph->Mf[kk];
    #endif

}

__device__ float get_OD(int BEERd, struct Profile prof) {  
    if (BEERd == 1) return prof.OD_sca;
    else            return prof.OD;
}


__device__ float Lambda(float avz, float sig) {
    float l;
    if (avz == 1.F) l = 1.F;
    else {
        float nu = __fdividef(1.F, tanf(acosf(avz))*(sqrtf(2.) * sig));
        l = __fdividef(__expf(-nu*nu) - nu * sqrtf(PI) * erfcf(nu),2.F * nu * sqrtf(PI));
    }
    return l;
}

#ifdef PHILOX

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

#endif
