/*
 * Copyright 2011-2020 HYGEOS
 *
 * This file is subject to the Smart-G licence.
 * Please see LICENCE.TXT for further details.
 */

/**********************************************************
*	> Includes
***********************************************************/

#include "communs.h"
#include "device.h"
#ifdef OBJ3D
#include "geometry.h"
#include "shapes.h"
#endif
#include "transform.h"

#include <math.h>
#include <math_constants.h>
#include <helper_math.h>
#include <stdio.h>
#include <cuda_fp16.h>
/*****/
#define _FLIP(x) (x%2==0 ? x+1 : x-1)


/****************************************************************************************************/
/****************************************************************************************************/
/****************************    MAIN KERNEL     ****************************************************/
/****************************************************************************************************/
/****************************************************************************************************/

extern "C" {
	__global__ void launchKernel(
							 struct Spectrum *spectrum, float *X0,
							 struct Phase *faer, struct Phase *foce,
							 unsigned long long *errorcount, int *nThreadsActive, void *tabPhotons, void *tabDist, void *tabHist, 
							 unsigned long long *Counter,
							 unsigned long long *NPhotonsIn,
							 unsigned long long *NPhotonsOut,
							 float *tabthv, float *tabphi, struct Sensor *tab_sensor,
							 struct Profile *prof_atm,
							 struct Profile *prof_oc,
                             struct Cell *cell_atm,
                             struct Cell *cell_oc,
							 long long *wl_proba_icdf,
							 long long *cell_proba_icdf,
							 void *rng_state
							 , void *tabObjInfo,
							 struct IObjets *myObjets,
							 struct GObj *myGObj,
							 struct IObjets *myRObj,
							 unsigned long long *nbPhCat,
							 void *wPhCat, void *wPhCat2,
							 void *wPhLoss,
							 void *wPhLoss2
                             /*#ifdef TIME
                             , unsigned long long *time_spent
                             #endif*/
							 ) {

    // current thread index
	unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;
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

	unsigned long long nbPhotonsThr = 0;  // Number of photons processed by the thread
	
	Photon ph, ph_le; 	// Photons structure for prapagation and Local Estimate (virtual photon)	
    float refrac_angle=0.F;

    bool mask_le = false;
	#ifdef OBJ3D
	IGeo geoStruc, geoStruc_le;
    float3 phit_le=make_float3(0.f, 0.f, 0.f);
	bigCount = 1;   // Initialisation de la variable globale bigCount (voir geometry.h)
    #endif


	while (this_thread_active > 0 and nThreadsActive[0] > 0) {
		iloop += 1;
		
		#ifdef OBJ3D
		/* ************************************************************************************************** */
		/* si on simule des objs on utilise cette astuce pour lancer exactement le nombre souhaité de photons */
		/* Si le nombre de ph lancés NBLOOPd > 256000 et que le compteur devient > (NBLOOPd-256000) alors     */
		/* on commence à diminuer le nombre de threads actif... ici à 999+1 = 1000 threads actif              */
        /* ************************************************************************************************** */
		if ((NBLOOPd > 50000) && idx > 999 && this_thread_active && Counter[0] >= (NBLOOPd-50000) && *nThreadsActive > 1000 && ph.loc == NONE)
		{
			this_thread_active = 0;
            atomicAdd(nThreadsActive, -1);
		}
		else if ((NBLOOPd > 5000) && idx > 99 && this_thread_active && Counter[0] >= (NBLOOPd-5000) && *nThreadsActive > 100 && ph.loc == NONE)
		{
			this_thread_active = 0;
            atomicAdd(nThreadsActive, -1);
		}
		else if((NBLOOPd > 500) && idx > 9 && this_thread_active &&Counter[0] >= (NBLOOPd-500) && *nThreadsActive > 10 && ph.loc == NONE)
		{
			this_thread_active = 0;
            atomicAdd(nThreadsActive, -1);
		}
		else if((NBLOOPd > 50) && idx > 0 && this_thread_active &&Counter[0] >= (NBLOOPd-50) && *nThreadsActive > 1 && ph.loc == NONE)
		{
			this_thread_active = 0;
            atomicAdd(nThreadsActive, -1);
		}
		#endif


        /* Termination rule */
        /* Active threads are deactivated if general loop counter has reached maximum or individual loop counter iloop imaximum exceeded*/
        /* Photons should be new ones (loc==None)*/
		if ((this_thread_active == 1 and Counter[0] >= NBLOOPd and ph.loc == NONE) or (iloop > MAX_LOOP))
		{
			this_thread_active = 0;
            atomicSub(nThreadsActive, 1);
		}

        
        /*--------------------------------------------------------------------------------------------------------  */
        /*                      INIT                                                                                */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns new photons (LOC == NONE)                                                              */
        /*--------------------------------------------------------------------------------------------------------  */
        if((ph.loc == NONE) && this_thread_active){

            initPhoton(&ph, prof_atm, prof_oc, tab_sensor, spectrum, X0, NPhotonsIn, wl_proba_icdf, cell_proba_icdf,
                       tabthv, tabphi, &rngstate
					   #ifdef OBJ3D
					   , myObjets
					   #endif
				);
			
            iloop = 1; // individual photon loop counter initialized
            #ifdef VERBOSE_PHOTON
			if (idx==0) {printf("\n");}
            display("INIT", &ph);
            #endif

        }

        /*--------------------------------------------------------------------------------------------------------  */
        /*                      MOVE                                                                                */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns photons in OCEAN or ATMOS                                                              */
        /*--------------------------------------------------------------------------------------------------------  */
        loc_prev = ph.loc;

        //--------------------------
        // Move
        //--------------------------
		if(ph.loc == ATMOS) {
           #ifdef VERBOSE_PHOTON
		   display("MOVE_START", &ph);
           #endif

           #ifdef SPHERIQUE
           /* Forced First Scattering option only for spherical mode*/
           /* in case of first scattering    */
           /* a virtual photon is propagated until TOA*/
           /* the maximum optical thickness seen by the real photon is recorded */
           if ((ph.nint==0) && FFSd) {
               copyPhoton(&ph, &ph_le);
               move_sp(&ph_le, prof_atm, 1, UPTOA , &rngstate);
               ph.taumax = ph_le.taumax;
               //if (idx==0)printf("%d %f\n",ph.is,ph.taumax);
           }
           /* the photon moves in spherical shell */
           /* it eventually uses the taumax as computed previously for Forced First Scattering*/
           move_sp(&ph, prof_atm, 0, 0 , &rngstate);

           #else // Plane Parallel
            #ifdef ALT_PP // Alternative PP move mode (very similar to Spherical move mode)
            move_pp2(&ph, prof_atm, prof_oc, 
                   #ifdef OPT3D
                   cell_atm, cell_oc,
                   #endif
                   0, 0 , &rngstate);
            #else // Fast PP move mode
            move_pp(&ph, prof_atm, prof_oc, &rngstate
				#ifdef OBJ3D
				   , &geoStruc, myObjets, myGObj, tabObjInfo
				#endif
			    );
            #endif // ALT or FAST

            #ifdef VERBOSE_PHOTON
		    display("MOVE_STOP", &ph);
            #endif

           #endif // SP or PP
        }

		if(ph.loc == OCEAN) {
           #ifdef VERBOSE_PHOTON
		   display("MOVE_START", &ph);
           #endif

           #ifdef ALT_PP // Alternative PP move mode
           move_pp2(&ph, prof_atm, prof_oc, 
                   #ifdef OPT3D
                   cell_atm, cell_oc,
                   #endif
                   0, 0 , &rngstate);
           #else // Fast PP mode
           move_pp(&ph, prof_atm, prof_oc, &rngstate
				#ifdef OBJ3D
				   , &geoStruc, myObjets, myGObj, tabObjInfo
				#endif
                );

           #ifdef VERBOSE_PHOTON
		   display("MOVE_STOP", &ph);
           #endif

           #endif // ALT or FAST
            }


        //--------------------------
        // count after move:
        // count the photons finishing in space (or source for thermal backward) 
        // and/or reaching surface from above or below
        //--------------------------
        count_level = NOCOUNT; // Initialize the counting level

        //
        // 1- Final counting level determination
        //
        #if defined(THERMAL) && defined(BACK)
        /* in the specific Thermal case in backward mode*/
        /* we count the photons in the artificial 'TOA' box*/
        /* if they have reached the SOURCE */
		if ((ph.loc == SOURCE) || (ph.loc == SPACE)) {
            if (ph.loc == SOURCE) count_level = UPTOA;

        #else // The general case
        /* photons having reached SPACE are counted in the TOA level*/
		if (ph.loc == SPACE) {
            count_level = UPTOA;
        #endif // Thermal backward or general
            
            /* Terminating photon life */
            // increment the photon counter (for this thread)
            nbPhotonsThr++;
            // reset the photon location
            ph.loc = NONE;

            #ifdef VERBOSE_PHOTON
            display("SPACE/SOURCE", &ph);
            #endif

        //
        // 2- Intermediate counting level determination
        //
        } else if ((ph.loc == SURF0P) && (loc_prev != SURF0P)) {
            /* Passing through the surface interface, downward */
            count_level = DOWN0P;
        } else if ((ph.loc == SURF0M) && (loc_prev != SURF0M)) {
            /* Passing through the surface interface, upward */
            count_level = UP0M; 
        } else if (ph.loc == SEAFLOOR) {
            /* reaching seafloor */
            count_level = DOWNB;
        }

        //
		// 3- Count the photons
        //
		/* Cone Sampling */
		if (LEd ==0) countPhoton(&ph, spectrum, prof_atm, prof_oc, tabthv, tabphi, count_level,
            errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut);

		#if defined(BACK) && defined(OBJ3D)
		if (count_level == UPTOA and LMODEd == 4 and LEd == 0) // the photon reach TOA
		{ countPhotonObj3D(&ph, 0, tabObjInfo, &geoStruc, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2);}
        #endif


		__syncthreads();
        

        /*--------------------------------------------------------------------------------------------------------  */
        /*                      SCATTER                                                                             */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns photons in OCEAN or ATMOS                                                              */
        /*--------------------------------------------------------------------------------------------------------  */
		if( ((ph.loc == ATMOS) || (ph.loc == OCEAN)) ) {
            //
		    // 1- Choose the scatterer
            //
            choose_scatterer(&ph, prof_atm, prof_oc,
                        #ifdef OPT3D
                        cell_atm, cell_oc,
                        #endif
                        spectrum, &rngstate); 

            #ifdef VERBOSE_PHOTON
            display("CHOOSE SCAT", &ph);
            #endif

            //
		    // 2- Scattering Local Estimate (LE)
            //
            if (LEd == 1) {
			    int NK, up_level, down_level, count_level_le;
			    int ith0 = idx%NBTHETAd; //index shifts in LE zenith and azimuth angles
                int iph0 = idx%NBPHId;
                /* Two levels for counting if photon is in ATMOS, 1) up TOA and 2) down 0+*/
			    if (ph.loc == ATMOS) {
			        NK=2;
			        up_level = UPTOA;
			        down_level = DOWN0P;
		        }
                /* Two levels for counting if photon is in OCEAN, 1) up 0- and 2) down Bottom*/
			    if (ph.loc == OCEAN) {
			        NK=2;
                    up_level = UP0M;
			        down_level = DOWNB;
		        }
                float phi, thv, cr;
                float3 v,u,uu;
                float3 no = normalize(ph.pos);
                float3x3 R;

                /* Loop on levels for counting (upward and downward) */
			    for(int k=0; k<NK; k++){
			        if (k==0) count_level_le = up_level;
			        else count_level_le = down_level;

                    // Double Loop on directions (zenith ang azimuths)
                    for (int ith=0; ith<NBTHETAd; ith++){
                        for (int iph=0; iph<NBPHId; iph++){


                            // Copy of the propagation photon to to the virtual, local estimate photon
                            copyPhoton(&ph, &ph_le);
                            // Computation of the indices of the direction
                            ph_le.ith = (ith + ith0)%NBTHETAd;
                            if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                            else ph_le.iph =  ph_le.ith;
                            // azimuth and zenith LE
                            phi = tabphi[ph_le.iph];
                            thv = tabthv[ph_le.ith];

                            //
                            // 2-1 Estimation of the refraction angle
                            //
                            // in case of atmospheric refraction determine the outgoing direction
                            if (REFRACd && ph_le.loc==ATMOS) {
                                DirectionToUV(thv, phi, &v, &u);
                                v = normalize(v);
                                u = normalize(u);
                                /* the virtual photon is prepared for propagation in LE direction*/
                                ph_le.v = v;
                                ph_le.u = u;
                                uu = normalize(cross(v, no));
                                int iter=0;
                                refrac_angle=0.F;
                                float ra=0.F;
                                // propagation //
                                //while((iter < 1)) {
                                while((iter < 0)) {
                                   /* propagation of the LE photon until TOA to estimate refraction angle */
                                   #ifdef SPHERIQUE
                                   move_sp(&ph_le, prof_atm, 1, UPTOA , &rngstate);
                                   if (ph_le.loc != SPACE) break;
                                   #endif
                                   /* the photon direction is extracted when when it comes out at the TOA */
                                   ph_le.v = normalize(ph_le.v);
                                   /* deviation angle from scattering point to the TOA*/
                                   cr = fabs(dot(v, ph_le.v));
                                   if (cr >= 1.F) cr=1.F;
                                   /* refraction angle */
                                   ra += acosf(cr);
                                   /* New virtual photon direction to compensate for refraction*/
                                   copyPhoton(&ph, &ph_le);
                                   ph_le.v   = v;
                                   ph_le.u   = u;
                                   /* We rotate the direction of the new LE photon around the unit vector uu, perpendicular
                                   to the current direction and the local vertical  by an angle equal to the refraction angle */
                                   R  = rotation3D(ra, uu);
                                   ph_le.v = mul(R, ph_le.v);
                                   iter++;
                                }
                                refrac_angle = ra;

                                /*Re-create the virtual photon for LE*/
                                copyPhoton(&ph, &ph_le);
                                ph_le.ith = (ith + ith0)%NBTHETAd;
                                if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                                else ph_le.iph =  ph_le.ith;
                            } // Atmospheric refraction

                            else refrac_angle = 0.F; // if no refraction option is chosen

                            //
                            // 2-2 Scatter the virtual photon, using le=1, and count_level for the scattering angle computation
                            //
                            scatter(&ph_le, prof_atm, prof_oc, 
                                    #ifdef OPT3D
                                    cell_atm, cell_oc,
                                    #endif
                                    faer, foce,
                                    1, refrac_angle, tabthv, tabphi,
                                    count_level_le, &rngstate);

                            #ifdef VERBOSE_PHOTON
                            if (k==0) display("SCATTER LE UP", &ph_le);
                            else display("SCATTER LE DOWN", &ph_le);
                            #endif

                            #ifdef SPHERIQUE
                            /* in spherical mode (ATMOS only), move the virtual photon until the counting level to calculate 
                            the extinction along the final path */
                            if (ph_le.loc==ATMOS) move_sp(&ph_le, prof_atm, 1, count_level_le , &rngstate);

                            #ifdef VERBOSE_PHOTON
                            display("MOVE LE", &ph_le);
                            #endif

                            #else
                             #ifdef ALT_PP
                            /* in alternative PP mode (for ATMOS or OCEAN), move the virtual photon until the counting level to calculate 
                            the extinction along the final path */
                             if ((ph_le.loc==ATMOS) || (ph_le.loc==OCEAN)) 
                                 move_pp2(&ph_le, prof_atm, prof_oc, 
                                         #ifdef OPT3D
                                         cell_atm, cell_oc,
                                         #endif
                                         1, count_level_le , &rngstate);
                             #endif // ALT PP
                            #endif // Spherical

                            // Finally count the virtual photon
                            /* in FAST PP mode the final extinction until the counting level is done in the countPhoton function */
							#if defined(OBJ3D)
                            mask_le = false;
                            copyIGeo(&geoStruc, &geoStruc_le);
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            if (!mask_le and count_level_le == UPTOA and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
							#endif
                            if (!mask_le) {countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, count_level_le, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }

                        } //directions
                    } // directions
                } // levels
            } // LE

            
            //
		    // 3- Scattering of the propagation photon (&ph and le=0 in scatter call)
            //
            scatter(&ph, prof_atm, prof_oc, 
                    #ifdef OPT3D
                    cell_atm, cell_oc,
                    #endif
                    faer, foce,
                    0, 0.F, tabthv, tabphi, 0,
                    &rngstate);

            #ifdef VERBOSE_PHOTON
            display("SCATTER", &ph);
            #endif

        } // photon in ATMOS or OCEAN
        


		__syncthreads();


        /*--------------------------------------------------------------------------------------------------------  */
        /*                      SURFACES  (1/2)                                                                     */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns photons in SURF0M , SURF0P                                                             */
        /*--------------------------------------------------------------------------------------------------------  */
        loc_prev = ph.loc;
        if ((ph.loc == SURF0M) || (ph.loc == SURF0P)){
           // Eventually evaluate Downward 0+ and Upward 0- radiance

           /* Define distance form the photon surface impact coordinates and the center of the ENV zone X0d,Y0d */
           float dis = sqrtf((ph.pos.x-X0d)*(ph.pos.x-X0d) +(ph.pos.y-Y0d)*(ph.pos.y-Y0d));

           ////////////////////////////
           // if no environment effects 
           // OR
           // photon is reflected by the target:
           // dis <= ENV_SIZEd if ENVd=1 (Environment outside disk) or dis >= ENV_SIZEd if ENVd=-1 (Environment inside disk)
           // or if ENV==2 it is a exponentially decreasing albedo between target and environment
           // or if ENV==3 and checkerboard return white square of albedo map
           // or abs(X) <= ENV_SIZEd if ENVd=4 (Environment outside disk) 
           // or abs(X) >= ENV_SIZEd if ENVd=-4 (Environment inside disk)
           ////////////////////////////
           if( (ENVd==0) || (ENVd==2) || 
               ((ENVd==3) && checkerboard(ph.pos, X0d, Y0d)) ||
               ((ENVd==1) && (dis<=ENV_SIZEd)) || ((ENVd==-1) && (dis>=ENV_SIZEd)) || 
               ((ENVd==4) && (abs(ph.pos.x)<=ENV_SIZEd)) || ((ENVd==-4) && (abs(ph.pos.x)>=ENV_SIZEd)) || ph.loc==SURF0M ) { 

            ////////////////////////////
            // if Air-Sea Interface 
            ////////////////////////////
			if( DIOPTREd<3 ) {
                //
		        // 1- Surface Local Estimate (not evaluated if atmosphere only simulation)*/
                //
                if (LEd == 1 && SIMd != ATM_ONLY) {
                  int NK, count_level_le;
                  if (NOCEd==0) NK=1; // if there is no ocean, just one level for contribution of surface to LE : up 0+ 
                  else NK=2; // otherwise 2 : up 0+ and down 0-
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;

                  /* Loop on levels for counting (upward and downward) */
                  for(int k=0; k<NK; k++){
                    if (k==0) count_level_le = UP0P;
                    else count_level_le = DOWN0M;

                    /* Double loop on zenith and azimuth LE */
                    for (int ith=0; ith<NBTHETAd; ith++){
                      for (int iph=0; iph<NBPHId; iph++){
                        // copy propagation to virtual photon
                        copyPhoton(&ph, &ph_le);
                        ph_le.ith = (ith + ith0)%NBTHETAd;
                        if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                        else ph_le.iph =  ph_le.ith;

                        // Reflect or Tramsit the virtual photon, using le=1 and count_level
                        if (BRDFd != 0)
                            surfaceBRDF(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);
                        else 
                            surfaceWaterRough(&ph_le, 1, tabthv, tabphi,
                                      count_level_le, &rngstate);

                        #ifdef VERBOSE_PHOTON
                        if (k==0) display("SURFACE LE UP", &ph_le);
                        else display("SURFACE LE DOWN", &ph_le);
                        #endif

                        // Count the photon up to the counting levels (at the surface UP0P or DOW0M)
                        #ifdef OBJ3D
                        mask_le = false;
                        copyIGeo(&geoStruc, &geoStruc_le);
                        mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                        #endif
                        if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, count_level_le, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }

                        // Only for upward photons in Atmopshere, count also them up to TOA
                        if (k==0) { 
                            // Final extinction computation n the atmosphere for SP and ALT_PP move mode
                            #ifdef SPHERIQUE
                            if (ph_le.loc==ATMOS)
                            {
                                move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                                #ifdef OBJ3D
                                mask_le = false;
                                mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                                #endif
                            }
                            #else // if not spheric
                            #ifdef ALT_PP
                            if (ph_le.loc==ATMOS)
                            {
                                move_pp2(&ph_le, prof_atm, prof_oc,
                                         #ifdef OPT3D
                                         cell_atm, cell_oc,
                                         #endif
                                        1, UPTOA, &rngstate);
                                #ifdef OBJ3D
                                mask_le = false;
                                mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                                #endif        
                            }
                            #endif // END ALT_PP
                            #endif // END not spheric
                            // Final extinction computation in FAST PP move mode and counting at the TOA for all move modes
                            if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UPTOA , errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                            #ifdef OBJ3D
                            if (!mask_le and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
                            #endif
                        }
                        // Only for downward photons in Ocean, count also them up to Bottom 
                        if (k==1) { 
                            // Final extinction computation in the ocean for ALT_PP move mode
                            #ifdef ALT_PP
                            if (ph_le.loc==OCEAN) 
                            {
                                move_pp2(&ph_le, prof_atm, prof_oc, 
                                         #ifdef OPT3D
                                         cell_atm, cell_oc,
                                         #endif
                                        1, DOWNB, &rngstate);
                                #ifdef OBJ3D
                                mask_le = false;
                                mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                                #endif
                            }
                            #endif // END ALT_PP
                            // Final extinction computation in FAST PP move mode and counting at the Bottom for all move modes
                            if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, DOWNB , errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                        }
                      }//direction
                    }//direction
                  }// counting levels
                } //LE

                //
		        // 3- REflection/Transmission of the propagation photon (&ph and le=0 in scatter call)
                //
                if (BRDFd != 0)
				    surfaceBRDF(&ph, 0, tabthv, tabphi,
                              count_level, &rngstate);
                else
				    surfaceWaterRough(&ph, 0, tabthv, tabphi,
                              count_level, &rngstate);
            } // Air-Sea interface 


            ////////////////////////////
            // BRDF interface
            ////////////////////////////
			else { 
                #ifdef SIF
                /* if Sun Induced Fluorescence is activated*/
			    /* Choose the emitter : Fluorescence or Solar reflection*/
                choose_emitter(&ph, prof_atm, prof_oc,  spectrum, 
                             &rngstate); 

                #ifdef VERBOSE_PHOTON
                display("CHOOSE EMITTER", &ph);
                #endif

                #endif

                ph.env = 0;
                //
		        // 1- Surface Local Estimate (not evaluated if atmosphere only simulation)*/
                //
                if (LEd == 1 && SIMd != ATM_ONLY) {
                  int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                  int iph0 = idx%NBPHId;

                  /* Double loop on zenith and azimuth LE */
                  for (int ith=0; ith<NBTHETAd; ith++){
                    for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.ith = (ith + ith0)%NBTHETAd;
                        if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                        else ph_le.iph =  ph_le.ith;

                        /* LE for BRDF type*/
                        surfaceLambert(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);

                        #ifdef VERBOSE_PHOTON
                        display("SURFACE LE UP", &ph_le);
                        #endif

                        // Only two levels for counting by definition (up 0+ and up TOA)
                        // 1) up 0+ for all move modes
                        #ifdef OBJ3D
                        mask_le = false;
                        copyIGeo(&geoStruc, &geoStruc_le);
                        mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                        #endif
                        if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UP0P,  errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }

                        // 2) up TOA for all move modes, need final extinction computation
                        // Final extinction computation in the atmosphere for SP and ALT_PP move mode
                        #ifdef SPHERIQUE
                        if (ph_le.loc==ATMOS)
                        {
                            move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                            #ifdef OBJ3D
                            mask_le = false;
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            #endif
                        }
                        #else // if not spheric
                        #ifdef ALT_PP
                        if (ph_le.loc==ATMOS) 
                        {
                            move_pp2(&ph_le, prof_atm, prof_oc, 
                                      #ifdef OPT3D
                                      cell_atm, cell_oc,
                                      #endif
                                      1, UPTOA , &rngstate);
                            #ifdef OBJ3D
                            mask_le = false;
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            #endif
                        }
                        #endif // END ALT_PP
                        #endif // END not spheric
                        // Final extinction computation in FAST PP move mode and counting at the TOA for all move modes
                        if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                        #ifdef OBJ3D
                        if (!mask_le and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
                        #endif

                    }//direction
                  }//direction
                } //LE

                //
		        // 2- Surface for propagation photon for the BRDF interface
                //
                surfaceLambert(&ph, 0, tabthv, tabphi, spectrum, &rngstate);

            } // BRDF interface (DIOPTRE=!3)
           } // No environment effects or interaction with the target


           ////////////////////////////
           // Environment effects
           // photon is reflected by the environment:
           // dis > ENV_SIZEd if ENVd=1 (Environment outside disk) or dis < ENV_SIZEd if ENVd=-1 (Environment inside disk)
           ////////////////////////////
           else if( ((ENVd==1) && (dis>ENV_SIZEd)) || 
                    ((ENVd==3) && !checkerboard(ph.pos, X0d, Y0d)) ||
                    ((ENVd==-1) && (dis<ENV_SIZEd)) || 
                    ((ENVd==4) && (abs(ph.pos.x)>ENV_SIZEd)) || 
                    ((ENVd==-4) && (abs(ph.pos.x)<ENV_SIZEd)) ) { 
                ph.env = 1;
                //
		        // 1- Surface Local Estimate (not evaluated if atmosphere only simulation)*/
                //
                if (LEd == 1 && SIMd != ATM_ONLY) {
                 int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
                 int iph0 = idx%NBPHId;

                 /* Double loop on zenith and azimuths LE*/
                 for (int ith=0; ith<NBTHETAd; ith++){
                    for (int iph=0; iph<NBPHId; iph++){
                        copyPhoton(&ph, &ph_le);
                        ph_le.ith = (ith + ith0)%NBTHETAd;
                        if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                        else ph_le.iph =  ph_le.ith;

                        /* LE for BRDF type*/
                        surfaceLambert(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);

                        // Only two levels for counting by definition
                        #ifdef OBJ3D
                        mask_le = false;
                        copyIGeo(&geoStruc, &geoStruc_le);
                        mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                        #endif
                        if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UP0P,  errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                        #ifdef SPHERIQUE
                        if (ph_le.loc==ATMOS)
                        {
                            move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                            #ifdef OBJ3D
                            mask_le = false;
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            #endif
                        }
                        #else // if not spheric
                        #ifdef ALT_PP
                        if (ph_le.loc==ATMOS)
                        {
                            move_pp2(&ph_le, prof_atm, prof_oc, 
                                    #ifdef OPT3D
                                    cell_atm, cell_oc,
                                    #endif
                                    1, UPTOA , &rngstate);
                            #ifdef OBJ3D
                            mask_le = false;
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            #endif
                        }
                        #endif // END ALT_PP
                        #endif // END not spheric
                        if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                        #ifdef OBJ3D
                        if (!mask_le and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
                        #endif
                    }//direction
                 }//direction
                } //LE

                //
		        // 2- Surface for Propagation photon*/
                //
                surfaceLambert(&ph, 0, tabthv, tabphi, spectrum, &rngstate);
           } // photon interaction with the environment 

           else {
                //
		        // When rare numerical problem at the surface/env boundary*/
                //
               ph.loc = REMOVED;
           }

           #ifdef VERBOSE_PHOTON
           display("SURFACE", &ph);
           #endif

        } // Photon at the surface (SURF0P or SURF0M)


		__syncthreads();


        /*--------------------------------------------------------------------------------------------------------  */
        /*                      SURFACES  (2/2)                                                                     */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns photons in SEAFLOOR                                                                    */
        /*--------------------------------------------------------------------------------------------------------  */
        if(ph.loc == SEAFLOOR){
           //
		   // 1- Seafloor Local Estimate*/
           //
           ph.env = 0;
           if (LEd == 1 && SIMd != ATM_ONLY) {
              int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
              int iph0 = idx%NBPHId;
              /* Double loop on zenith and azimuth LE*/
              for (int ith=0; ith<NBTHETAd; ith++){
                for (int iph=0; iph<NBPHId; iph++){
                    copyPhoton(&ph, &ph_le);
                    ph_le.ith = (ith + ith0)%NBTHETAd;
                    if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                    else ph_le.iph =  ph_le.ith;

                    /* LE on SEAFLOOR*/
                    surfaceLambert(&ph_le, 1, tabthv, tabphi, spectrum, &rngstate);

                    //  contribution to UP0M level
                    #ifdef ALT_PP                          
                    if (ph_le.loc==OCEAN)
                    {
                        move_pp2(&ph_le, prof_atm, prof_oc, 
                                #ifdef OPT3D
                                cell_atm, cell_oc,
                                #endif
                                1, UP0M, &rngstate);
                        #ifdef OBJ3D
                        mask_le = false;
                        copyIGeo(&geoStruc, &geoStruc_le);
                        mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj)
                        #endif
                    } 
                    #endif
                    if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UP0M,   errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }

                } // directions
              } // directions
            } //LE

           //
		   // 2- Seafloor propagation*/
           //
           surfaceLambert(&ph, 0, tabthv, tabphi, spectrum, &rngstate);

           #ifdef VERBOSE_PHOTON
           display("SEAFLOOR", &ph);
           #endif

         } // Seafloor



        __syncthreads();



		#ifdef OBJ3D
        /*--------------------------------------------------------------------------------------------------------  */
        /*                      SURFACE OBJECTS                                                                     */
        /*--------------------------------------------------------------------------------------------------------  */
        /*       It concerns photons in OBJSURF                                                                     */
        /*--------------------------------------------------------------------------------------------------------  */
        if(ph.loc == OBJSURF)
		{

			if (geoStruc.type == RECEIVER and LMODEd != 4 and LEd == 0) // this is a receiver
			{ countPhotonObj3D(&ph, 0, tabObjInfo, &geoStruc, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2);}

			ph.weight_loss[0] = ph.weight;

			if (geoStruc.material == 1) // Lambertian Mirror
			{
				if (LEd == 1)
				{				
					int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
					int iph0 = idx%NBPHId;
					for (int ith=0; ith<NBTHETAd; ith++){
						for (int iph=0; iph<NBPHId; iph++){
							copyPhoton(&ph, &ph_le);
                            ph_le.ith = (ith + ith0)%NBTHETAd;
                            if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                            else ph_le.iph =  ph_le.ith;
							surfaceLambert3D(&ph_le, 1, tabthv, tabphi, spectrum,
												  &rngstate, &geoStruc);			
							// Only two levels for counting by definition
                            mask_le = false;
                            copyIGeo(&geoStruc, &geoStruc_le);
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UP0P, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                            #ifdef SPHERIQUE
							// for spherical case attenuation if performed usin move_sp
							if (ph_le.loc==ATMOS)
                            {
                                mask_le = false;
                                move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                                mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            }
						    #endif
							if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                            if (!mask_le and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
						}//direction
					}//direction
                } //LE
                
                //
		        // obsjsurf propagation*/
                //
				surfaceLambert3D(&ph, 0, tabthv, tabphi, spectrum,
                                      &rngstate, &geoStruc);
            } // END Lambertian Mirror

			else if (geoStruc.material == 2) // Matte
			{
				ph.loc = ABSORBED;
				ph.weight = 0.F;
				if (geoStruc.type == HELIOSTAT) ph.H+=1;
				else if (geoStruc.type == RECEIVER) ph.E+=1;
				ph.weight_loss[0] = 0.F;
            } // End Matte

			else if (geoStruc.material == 3) // Mirror
			{	
				if (LEd == 1)
				{				
					int ith0 = idx%NBTHETAd; //index shifts in LE geometry loop
					int iph0 = idx%NBPHId;
					for (int ith=0; ith<NBTHETAd; ith++){
						for (int iph=0; iph<NBPHId; iph++){
							copyPhoton(&ph, &ph_le);
							ph_le.ith = (ith + ith0)%NBTHETAd;
                            if (!ZIPd) ph_le.iph = (iph + iph0)%NBPHId;
                            else ph_le.iph =  ph_le.ith;
							Obj3DRoughSurf(&ph_le, 1, tabthv, tabphi, &geoStruc, &rngstate);
							// Only two levels for counting by definition
                            mask_le = false;
                            copyIGeo(&geoStruc, &geoStruc_le);
                            mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UP0P, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                            #ifdef SPHERIQUE
							// for spherical case attenuation if performed usin move_sp
                            // once spherical case with OBJ implemented, must check if le still works
							if (ph_le.loc==ATMOS)
                            {
                                mask_le = false;
                                move_sp(&ph_le, prof_atm, 1, UPTOA, &rngstate);
                                mask_le = geoTest(ph_le.pos, ph_le.v, &phit_le, &geoStruc_le, myObjets, myGObj);
                            }
						    #endif
							if (!mask_le) { countPhoton(&ph_le, spectrum, prof_atm, prof_oc, tabthv, tabphi, UPTOA, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut); }
                            if (!mask_le and LMODEd == 4) { countPhotonObj3D(&ph_le, 1, tabObjInfo, &geoStruc_le, nbPhCat, wPhCat, wPhCat2, prof_atm, wPhLoss, wPhLoss2); }
                            
						}//direction
					}//direction
                } //LE

                //
		        // obsjsurf propagation*/
                //
				Obj3DRoughSurf(&ph, 0, tabthv, tabphi, &geoStruc, &rngstate);
            } // End Mirror

			else {ph.loc = REMOVED;} // unknow material

			#ifndef BACK
			if (geoStruc.type == HELIOSTAT and ph.direct == 0 and ph.loc != REMOVED and ph.loc != ABSORBED)
			{
				ph.weight_loss[1] = 0.F;
				ph.weight_loss[2] = 0.F;
				ph.weight_loss[3] = 0.F;
				// Look if the ray is blocked by an obj located between the ref point and the rec
				if (geoTestMir(ph.pos, ph.v, myObjets, myGObj))
					ph.weight_loss[1] = ph.weight; // w_BM
				else
				{   // If the ray is not blocked look if the ray miss or not the rec
					if ( geoTestRec(ph.pos, ph.v, myRObj) )
						ph.weight_loss[2] = ph.weight; // w_SP
					else
						ph.weight_loss[3] = ph.weight; // w_SM
				}
				countLoss(&ph, &geoStruc, wPhLoss, wPhLoss2);
			}
			#endif

			#ifdef VERBOSE_PHOTON
			display("OBJSURF", &ph);
            #endif
		}
        __syncthreads();
		#endif
        /*--------------------------------------------------------------------------------------------------------  */
        
        

        //--------------------------
        // count after surface:
        // count the photons leaving the surface towards the ocean or atmosphere
        //--------------------------
        count_level = -1;
        if ((loc_prev == SURF0M) || (loc_prev == SURF0P)) {
            if ((ph.loc == ATMOS) || (ph.loc == SPACE)
				#ifdef OBJ3D
				|| (ph.loc == OBJSURF)
				#endif
				)
				count_level = UP0P;
            if (ph.loc == OCEAN) count_level = DOWN0M;
        }
		
        /* Cone Sampling */
        if (LEd == 0) countPhoton(&ph, spectrum, prof_atm, prof_oc, tabthv, tabphi, count_level, errorcount, tabPhotons, tabDist, tabHist, NPhotonsOut);


        //--------------------------
        // Final counts
        //--------------------------

        /* absorbed photons during their life are terimnated and initialized again, they are counted*/
		if(ph.loc == ABSORBED){
			ph.loc = NONE;
			nbPhotonsThr++;
		}
        /* removed photons during their life are terimnated and initialized again, they are NOT counted*/
		if(ph.loc == REMOVED){
			ph.loc = NONE;
        }
        

		__syncthreads();


		if (this_thread_active == 1) 
		{
		    atomicAdd(Counter, nbPhotonsThr);
			nbPhotonsThr = 0;
        }		

	} // Main While loop on active threads

	
    //--------------------------
    // Error counts
    //--------------------------

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

} /* launchKernel*/
} /* extern C*/




/****************************************************************************************************/
/****************************************************************************************************/
/****************************    PHYSICAL PROCESSES    **********************************************/
/****************************************************************************************************/
/****************************************************************************************************/



/*--------------------------------------------------------------------------------------------------*/
/*                      INIT PHOTONS                                                                */
/*--------------------------------------------------------------------------------------------------*/

__device__ void initPhoton(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc, 
                           struct Sensor *tab_sensor, struct Spectrum *spectrum, 
                           float *X0, unsigned long long *NPhotonsIn,
                           long long *wl_proba_icdf, long long *cell_proba_icdf, 
                           float* tabthv, float* tabphi,
                           struct RNG_State *rngstate
						   #ifdef OBJ3D
						   , struct IObjets *myObjets
						   #endif
	) {
    float cTh, sTh, phi;
    int ilayer;
    #if !defined(ALT_PP) && !defined(SPHERIQUE)
    float dz, dz_i, delta_i, epsilon;
    #endif
	
	//int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
	
    #ifdef OBJ3D
	ph->direct = 0;
	ph->H = 0;
	ph->E = 0;
	ph->S = 0;
	ph->weight_loss[0] = 0.F;
	ph->weight_loss[1] = 0.F;
	ph->weight_loss[2] = 0.F;
	ph->weight_loss[3] = 0.F;
	ph->v_i = make_float3(-DIRSXd, -DIRSYd, -DIRSZd);
    #endif
    
    /* Interaction counters*/
    ph->nint = 0; // total
    ph->nref = 0; // reflection on surface (main)
    ph->nenv = 0; // reflection on surface (environment)
    ph->nsfl = 0; // reflection on seafloor 
	ph->nrrs = 0; // number of RRS events
    ph->nvrs = 0; // number of VRS events

    ph->env  = 0;
	ph->weight = WEIGHTINIT;
    ph->taumax = 0.F;

	// Stokes parameters initialization according to natural sunlight
	ph->stokes.x = 0.5F;
	ph->stokes.y = 0.5F;
	ph->stokes.z = 0.F;
    ph->stokes.w = 0.F;

    #ifdef BACK
    // Initialize also photon cumulative Mueller matrix
    ph->M = make_diag_float4x4 (1.F);
    #endif

	ph->scatterer = UNDEF;
    ph->ith = 0;
    ph->iph = 0;
	
    // Sensor index initialization
    ph->is = __float2uint_rz(RAND * NSENSORd);

    /* ----------------------------------------------------------------------------------------- */
    // Wavelength index initialization
    /* ----------------------------------------------------------------------------------------- */

    /* 1/ Uniform distribution of the wavelengths */
    if (NWLPROBA == 0) { 
        #ifdef ALIS
        // The starting wavelength index is chosen randomly among the available ones
        // if NJACd = 0: no jacobian -> one unperturbed profile
        // otherwise the 'real' wavelengths number is limited to NLAMd/NJACd
        // other indices correspond to the repetition of the initial wavelengths
        // several times (NJACd) for perturbed atmospheric or oceanic profiles
        ph->ilam = __float2uint_rz(RAND * NLAMd/(NJACd+1));
        #else

        // Case of all sensors seeing the same wavelengths
        if (tab_sensor[ph->is].ILAM_0 == -1) ph->ilam = __float2uint_rz(RAND * NLAMd);
        // Case of each sensor is associated a range of wavelengths
        else ph->ilam =  __float2uint_rz(RAND * 
                         (tab_sensor[ph->is].ILAM_1 - tab_sensor[ph->is].ILAM_0))
                         + tab_sensor[ph->is].ILAM_0;
        #endif

    } 
    
    /* 2/ Pre-defined distribution of the wavelengths */
    else {
        ph->ilam = wl_proba_icdf[__float2uint_rz(RAND * NWLPROBA)];
    }

    /* photon wavelength */
	ph->wavel = spectrum[ph->ilam].lambda;
    /* ----------------------------------------------------------------------------------------- */


    /* ----------------------------------------------------------------------------------------- */
    // Position initialization
    /* ----------------------------------------------------------------------------------------- */

    // Read and define posiion using sensor attributes
    ph->pos = make_float3(tab_sensor[ph->is].POSX,
                          tab_sensor[ph->is].POSY,
                          tab_sensor[ph->is].POSZ);
    ph->loc = tab_sensor[ph->is].LOC;
	
	#ifdef OBJ3D
	Transform TRotZ; int mPP = 1; //char mPP[]="Point";
	TRotZ = TRotZ.RotateZ(tab_sensor[ph->is].PHDEG-180.);
	ph->pos = TRotZ(ph->pos, mPP);
	#endif
	
    #ifdef SPHERIQUE
	ph->radius = length(ph->pos);
    #endif


    ///////////////////////////////////////
    // Ocean
    /////////////////////////////////////
    /* Z is negative in the ocean */
    /* OD ARE NEGATIVE also */
    /*
    ^ Z        layer index i
    |                      0           SURF 0+
    |  0=z[SURF]======================|==============|===========|===|==========
    |                      1          |              |           |   |
    |      ---------------------------|--------------|-----------|---|----------
    |                      2          |              |           |   |
    |      ---------------------------|--------------|-----------|---|----------
    |                      .          |              |           |   |
    |                      .          |              |           |   |
    |                      .          |              |           |   |
    |      ---------------------------|--------------|-----------|---|----------
    |                      i-1        |             \|/          |  \|/
    |    z[i-1]-----------------------|--------------v-OD[i-1]--\|/--v--tau[i-1]
    |    z.................i..........|...................tau(z).v..............
    |    z[i]-------------------------|-----------------------------------------
    |                      .          |                                      
    |                      .          |                                     
    |                      .          |                                     
    |      ---------------------------|-----------------------------------------
    |                      NOCE-1     |                                     
    |      ---------------------------|-----------------------------------------
    |                      NOCE      \|/                                     
    |    z[NOCE]______________________v__OD[NOCE]_______________________________
           ////////////////////// SEAFLOOR /////////////////////////////////////
    */
    if(ph->loc == OCEAN){
        /* Determine layer index from vertical position */
        ilayer = 1;
        while (( prof_oc[ilayer].z > tab_sensor[ph->is].POSZ) && (ilayer < NOCEd)) {
            ilayer++;
        }
        ph->layer = ilayer;

        // Fast Move Mode additional initialization
        // partial geometrical thickness in the layer (from top) : 
        //      epsilon = (z[i-1]-z)/(z[i-1]-z[i]) ; 0<epsilon<1
        // optical thickness of the layer delta[i] = OD[i-1] - OD[i], delta[i]>0
        // partial optical thickness in the layer (from top) : epsilon * delta[i]
        // tau(z) = tau[i-1] - epsilon * delta[i]
        // tau[i-1] = OD[i-1]
        #if !defined(ALT_PP) && !defined(SPHERIQUE)
        dz_i    = fabs(prof_oc[ilayer-1].z - prof_oc[ilayer].z);
        dz      = fabs(prof_oc[ilayer-1].z - tab_sensor[ph->is].POSZ) ; // Distance from the layer top
        epsilon = fabs(__fdividef(dz,dz_i));
        delta_i = fabs(get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]) 
                     - get_OD(BEERd, prof_oc[ilayer  +ph->ilam*(NOCEd+1)]));
        ph->tau = get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]) - epsilon * delta_i; 
        delta_i = fabs(prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs
                     - prof_oc[ilayer  +ph->ilam*(NOCEd+1)].OD_abs);
        ph->tau_abs =  prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs - epsilon * delta_i; 
        #endif

        // FAST Move Mode ALIS specific additional initialization
        #if defined(ALIS) && !defined(ALT_PP) && !defined(SPHERIQUE)
        int DL=(NLAMd-1)/(NLOWd-1);
        for (int k=0; k<NLOWd; k++) {
            delta_i = fabs(get_OD(BEERd, prof_oc[ilayer-1+k*DL*(NOCEd+1)])
                         - get_OD(BEERd, prof_oc[ilayer  +k*DL*(NOCEd+1)]));
            ph->tau_sca[k] = get_OD(1,prof_oc[ilayer-1 + k*DL*(NOCEd+1)]) - epsilon * delta_i;
        }
        #endif
    }

    ///////////////////////////////////////
    // Seafloor
    /////////////////////////////////////
    if(ph->loc == SEAFLOOR){
        ph->layer   = NOCEd;
        ph->pos.z   = prof_oc[NOCEd].z;
        
        // Fast Move Mode additional initialization
        #if !defined(ALT_PP) && !defined(SPHERIQUE)
        ph->tau     = get_OD(BEERd, prof_oc[NOCEd +ph->ilam*(NOCEd+1)]);
        ph->tau_abs = prof_oc[NOCEd +ph->ilam*(NOCEd+1)].OD_abs;
        epsilon = 0.F;
        #endif

        // FAST Move Mode ALIS specific additional initialization
        #if defined(ALIS) && !defined(ALT_PP) && !defined(SPHERIQUE)
        int DL=(NLAMd-1)/(NLOWd-1);
        for (int k=0; k<NLOWd; k++) {
            ph->tau_sca[k] = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]) ;
        }
        #endif
    }

    /////////////////////////////////////
    // Atmosphere or object near surface
    /////////////////////////////////////
    /*
    Z          layer index i
    ^                      0           TOA
    |  HTOA===========================|==============|==========================
    |                      1          |              |
    |      ---------------------------|--------------|--------------------------
    |                      2          |              |
    |      ---------------------------|--------------|--------------------------
    |                      .          |              |
    |                      .          |              |
    |                      .          |              |
    |      ---------------------------|--------------|--------------------------
    |                      i-1        |              |  
    |    z[i-1]-----------------------|--------------|--------------------------
    |    z.................i..........|.............\|/...............tau(z)....
    |    z[i]-------------------------|--------------v---OD[i]------/|\-----tau[i]
    |                      .          |                              |    /|\
    |                      .          |                              |     |
    |                      .          |                              |     |
    |      ---------------------------|------------------------------|-----|----
    |                      NATM-1     |                              |     |
    |      ---------------------------|------------------------------|-----|----
    |                      NATM      \|/                             |     | 
    |    z[NATM]______________________v__OD[NATM]____________________|_____|____
           /////////////////////////////////////////////////////////////////////
    */
    if((ph->loc == ATMOS)
	   #ifdef OBJ3D
	   || (ph->loc == OBJSURF)
	   #endif
		){

        // Altitude determination
        float POSZd_alt;
        #ifdef SPHERIQUE
        float3 rad = make_float3(tab_sensor[ph->is].POSX,
                           tab_sensor[ph->is].POSY,
                           tab_sensor[ph->is].POSZ);
        POSZd_alt = length(rad) - RTER;
        #else
        POSZd_alt = tab_sensor[ph->is].POSZ;
        #endif

        /* Determine layer index */
        /* if 3D atmospheric properties mode (OPT3D) is not chosen*/
        #ifndef OPT3D
        ilayer=1;   // initialization in the UPPER layer just below TOA
                    // the layer LOWER bound altitude is prof_atm[ilayer].z 
                    // So while this altitude is above sensor altitude we continue to go down
        while (( prof_atm[ilayer].z > POSZd_alt) && (ilayer < NATMd)) {
            ilayer++;
        }
        ph->layer = ilayer;

        // Fast Move Mode additional initialization
        #if !defined(ALT_PP) && !defined(SPHERIQUE)
        // partial geometrical thickness in the layer (from bottom) : 
        //      epsilon = (z-z[i])/(z[i-1]-z[i]) ; 0<epsilon<1
        // optical thickness of the layer delta[i] = OD[i] - OD[i-1]
        // partial optical thickness in the layer (from bottom) : epsilon * delta[i]
        // tau(z) = tau[i] + epsilon * delta[i]
        // tau[i] = OD[NATM] - OD[i]
        dz_i    = fabs(prof_atm[ilayer-1].z - prof_atm[ilayer].z);
        dz      = fabs(POSZd_alt - prof_atm[ilayer].z) ; 
        epsilon = fabs(__fdividef(dz,dz_i));
        delta_i = fabs(get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)]) 
                     - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
        ph->tau =   get_OD(BEERd, prof_atm[NATMd+ph->ilam*(NATMd+1)]) //
                  - get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)])
                  + epsilon * delta_i; 

        // Same scheme for absorption optical thickness
        delta_i = fabs(prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs 
                     - prof_atm[ilayer-1+ph->ilam*(NATMd+1)].OD_abs);
        ph->tau_abs =   prof_atm[NATMd+ph->ilam*(NATMd+1)].OD_abs
                      - prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs
                      + epsilon * delta_i; 
        #endif

        // FAST Move Mode ALIS specific additional initialization
        #if defined(ALIS) && !defined(ALT_PP) && !defined(SPHERIQUE)
        // Index interval of the wavelengths where to compute scattering corrections
        int DL=(NLAMd-1)/(NLOWd-1);
        // Loop on correction wavelength indices
        for (int k=0; k<NLOWd; k++) {
            delta_i = fabs(get_OD(BEERd, prof_atm[ilayer+k*DL*(NATMd+1)]) 
                         - get_OD(BEERd, prof_atm[ilayer-1+k*DL*(NATMd+1)]));
            ph->tau_sca[k] = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)])
                           - get_OD(1,prof_atm[ilayer + k*DL*(NATMd+1)])
                           + epsilon * delta_i;
        }
        #endif

        /* OPT3D cell initialization */
        /* in the case of OPT3D, the user specifies the layer (cell) initial index in the inputs */
        #else
        ph->layer = tab_sensor[ph->is].IBOX;
        #endif
    }

    /////////////////////////////////////
    // SURFACE (0+ or 0-)
    /////////////////////////////////////
    if((ph->loc == SURF0P) || (ph->loc == SURF0M)){
        if (ph->loc == SURF0P) ph->layer   = NATMd;
        else                   ph->layer   = 0;

        // Fast Move Mode additional initialization
        ph->tau     = 0.F;
        ph->tau_abs = 0.F;

        // FAST Move Mode ALIS specific additional initialization
        #if defined(ALIS) && !defined(ALT_PP) && !defined(SPHERIQUE)
        for (int k=0; k<NLOWd; k++) ph->tau_sca[k] = 0.F;
        #endif
    }


    /* ----------------------------------------------------------------------------------------- */
    //   Direction initialization
    /* ----------------------------------------------------------------------------------------- */

    ///////////////
    // Direction Sampling
    ///////////////
    if (tab_sensor[ph->is].TYPE != 0) {
        // Standard sampling of zenith angle for lambertian sensor/emittor (as for planar flux for example)
        // the zenith angle is limited by the conic Field Of View (FOV) of the sensor
        float sThmin = sinf(radiansd(tab_sensor[ph->is].FOV));
	    cTh = sqrtf(1.F-RAND*sThmin*sThmin);
        // for spherical flux, adjust weight as a function of cTh
        float weight_irr = fabs(cTh);
        if (tab_sensor[ph->is].TYPE == 2 && weight_irr > 0.001f) ph->weight /= weight_irr;
        
        // Azimuth is sampled uniformly
	    phi = RAND*DEUXPI;
        sTh = sqrtf(1.F - cTh*cTh);

        // Basic v photon vector with the  global Z direction as reference
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
        // Basic v photon vector with the  global Z direction as reference
        ph->v.x = 0.F;
        ph->v.y = 0.F;
        ph->v.z = 1.F;
	    // Initialization of the orthogonal vector to the propagation
        ph->u.x = 1.F;
        ph->u.y = 0.F;
	    ph->u.z = 0.F;
    }

    ///////////////
    // Direction Rotations of v and u in the detector direction THDEG,PHDEG
    ///////////////
	float cPh, sPh;
    #ifdef OBJ3D
	float THRAD, PHRAD;
	#endif

    /* THERMAL FORWARD specific case */
    /* !!!!! DEV               !!!!!!*/
    #if defined(THERMAL) && !defined(BACK) 
    ph->scatterer = THERMAL_EM;
    float dz;
    // 1D plane parallel forward thermal initialization
    if (NCELLPROBA !=0) {
        ph->layer = cell_proba_icdf[__float2uint_rz(RAND * NCELLPROBA) + ph->ilam*NCELLPROBA];
        dz  = fabs(prof_atm[ph->layer-1].z - prof_atm[ph->layer].z);
    } 
    else {
        ph->layer = __float2uint_rz(RAND * NATMd);
        dz  = fabs(prof_atm[ph->layer-1].z - prof_atm[ph->layer].z);
        float kabs= fabs(prof_atm[ph->layer-1+ph->ilam*(NATMd+1)].OD_abs 
                       - prof_atm[ph->layer  +ph->ilam*(NATMd+1)].OD_abs); 
        kabs /= dz;
        ph->weight= (2*DEUXPI) * kabs *  BPlanck(ph->wavel*1e-9, prof_atm[ph->layer].T) ;
    }
    #ifdef SPHERIQUE
    ph->pos   = make_float3(0., 0., RAND *dz + prof_atm[ph->layer].z + RTER);
    ph->radius = length(ph->pos);
    #else
    ph->pos   = make_float3(0., 0., RAND *dz + prof_atm[ph->layer].z);
    #endif
    ph->loc   = ATMOS;

    /* GENERAL case */
    #else 
	#ifdef OBJ3D
	THRAD = tab_sensor[ph->is].THDEG*DEUXPI/360.;
	PHRAD = tab_sensor[ph->is].PHDEG*DEUXPI/360.;
	cTh = cosf(THRAD);
	cPh = cosf(PHRAD);
	sTh = sinf(THRAD);
	sPh = sinf(PHRAD);
    #else 
	cTh = cosf(tab_sensor[ph->is].THDEG*DEUXPI/360.);
	cPh = cosf(tab_sensor[ph->is].PHDEG*DEUXPI/360.);
	sTh = sqrtf(1.F - cTh*cTh);
	sPh = sinf(tab_sensor[ph->is].PHDEG*DEUXPI/360.);
    #endif

    /* Compute the rotation matrices */
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

    /* Rotation of v and u vectors */
	ph->v = mul(LTh,ph->v);
	ph->v = mul(LPh,ph->v);
	ph->u = mul(LTh,ph->u);
	ph->u = mul(LPh,ph->u);

    #endif // THERMAL/NON THERMAL FORWARD


    /* ----------------------------------------------------------------------------------------- */
    //   ALIS specific initialization
    /* ----------------------------------------------------------------------------------------- */

    /* Fast Move Mode */
    #ifdef ALIS
    /* we record an event list where layers indices in the ocean
    are counted negative for differentiating them from atmospheric layers*/
    #if !defined(ALT_PP) && !defined(SPHERIQUE)
    ph->nevt = 0;
    if (ph->loc == ATMOS) ph->layer_prev[ph->nevt]   = ph->layer;
    if (ph->loc == OCEAN || ph->loc == SURF0M) ph->layer_prev[ph->nevt]   = -ph->layer;
    // the record consists also in the direction and epsilon in each layer
    ph->vz_prev[ph->nevt]      = ph->v.z;
    ph->epsilon_prev[ph->nevt] = epsilon;

    /* Standard Move Mode */
    #else
    /* We initialize the cumulative distance counters in each layer in atmosphere and ocean*/
    for (int k=0; k<(NATM_ABSd+1); k++) ph->cdist_atm[k]= 0.F;
    for (int k=0; k<(NOCE_ABSd+1); k++) ph->cdist_oc[k] = 0.F;
    #endif

    /* Initialize scattering corrections */
    for (int k=0; k<NLOWd; k++) ph->weight_sca[k] = 1.0F;

    /* Initialize SIF */
    ph->nsif = 0;
    #ifdef SIF
	ph->emitter = UNDEF;
    #endif
    #endif

    /* ----------------------------------------------------------------------------------------- */
    //   General photons counters initialization
    /* ----------------------------------------------------------------------------------------- */
    #ifdef ALIS
    // a photon launched in ALIS represent all the wavelengths, so increment all wavelength boxes
    for (int k=0; k<NLAMd; k++) atomicAdd(NPhotonsIn + NLAMd*ph->is + k, 1);
    #else
    // in general increment the randomly chosen particular wavelength box
    atomicAdd(NPhotonsIn + NLAMd*ph->is + ph->ilam, 1);
    #endif



    /* ----------------------------------------------------------------------------------------- */
    //   OBJ3D Specific initialization
    /* ----------------------------------------------------------------------------------------- */
    #ifdef OBJ3D
    #if !defined(BACK)
	if (LMODEd == 1) // Marche que pour le mode forward restreint
	{		
		/* ***************************************************************************************** */
		/* Créer la surface en TOA qui visera un reflecteur avec prise en compte des transformations */
		/* ***************************************************************************************** */
        #ifdef DOUBLE
		// Valeurs de l'angle zenital Theta et l'angle azimutal Phi (ici Phi pour l'instant imposé à 0)
		double sunTheta = 180-tab_sensor[ph->is].THDEG, sunPhi=tab_sensor[ph->is].PHDEG-180;

        // One fixed direction (for radiance) inverse of the initiale pos of the obj
		double3 vdouble = make_double3(0., 0., -1.);
		
	    // Initialization of the orthogonal vector to the propagation
		double3 udouble = make_double3(-1., 0., 0.);

		if (SUN_DISCd != 0)
		{
			double PHconed, THconed;
			PHconed = RAND*360;
			// Sampling between 0 and SUN_DISCd degrees uniformly following mu² -> acos(sqrt())
			THconed = (    acos(   sqrt(1-(  RAND*(1-cos(radiansd(SUN_DISCd)))  ))   )*180    )/PI;

			// Creation of transforms to consider alpha and beta for the computation of photon dirs
			Transformd TPHconed, TTHconed;
			TPHconed = TPHconed.RotateZ(PHconed); TTHconed = TTHconed.RotateY(THconed);
		
			// Apply transforms to vector u and v in function to alpha and beta
			vdouble = TPHconed(   Vectord(  TTHconed( Vectord(vdouble) )  )   );
			udouble = TPHconed(   Vectord(  TTHconed( Vectord(udouble) )  )   );
		}
		
		// Creation des tranformations (pour le calcul de la direction du photon)	
		Transformd TThetad, TPhid;
		TThetad = TThetad.RotateY(sunTheta);
		TPhid = TPhid.RotateZ(sunPhi);		

		// Application des transformation sur les vecteurs u et v en fonction de Theta et Phi
		vdouble = TPhid(   Vectord(  TThetad( Vectord(vdouble) )  )   );
		udouble = TPhid(   Vectord(  TThetad( Vectord(udouble) )  )   );

		ph->v = make_float3(float(vdouble.x), float(vdouble.y), float(vdouble.z));
		ph->u = make_float3(float(udouble.x), float(udouble.y), float(udouble.z));
		#else // IF NOT DOUBLE
		// Valeurs de l'angle zenital Theta et l'angle azimutal Phi (ici Phi pour l'instant imposé à 0)
		float sunTheta = 180-tab_sensor[ph->is].THDEG, sunPhi=tab_sensor[ph->is].PHDEG-180;

        // One fixed direction (for radiance)
		float3 vfloat = make_float3(0., 0., -1.);
		
	    // Initialization of the orthogonal vector to the propagation
		float3 ufloat = make_float3(-1., 0., 0.);

		if (SUN_DISCd != 0)
		{
			float PHcone, THcone;
			PHcone = RAND*360;
			THcone = (    acosf(   sqrtf(1-(  RAND*(1-cosf(radians(SUN_DISCd)))  ))   )*180    )/PI;

			// Creation of transforms to consider alpha and beta for the computation of photon dirs
			Transform TPHcone, TTHcone;
			TPHcone = TPHcone.RotateZ(PHcone); TTHcone = TTHcone.RotateY(THcone);
		
			// Apply transforms to vector u and v in function to alpha and beta
			vfloat = TPHcone(   Vectorf(  TTHcone( Vectorf(vfloat) )  )   );
			ufloat = TPHcone(   Vectorf(  TTHcone( Vectorf(ufloat) )  )   );
		}
		
		//Creation des tranformations (pour le calcul de la direction du photon)
		Transform TTheta, TPhi;
		TTheta = TTheta.RotateY(sunTheta);
		TPhi = TPhi.RotateZ(sunPhi);		

		// Application des transformation sur les vecteurs u et v en fonction de Theta et Phi
		vfloat = TPhi(   Vectorf(  TTheta( Vectorf(vfloat) )  )   );
		ufloat = TPhi(   Vectorf(  TTheta( Vectorf(ufloat) )  )   );

		ph->v = vfloat;
		ph->u = ufloat;
        #endif // END IF DOUBLE OR FLOAT
	
		// Récupération de l'objet réflecteur
		IObjets objP;
		objP.type = 0;

		// Permet de choisir aléatoirement un miroir
		float randMirPrev = -1;
		float randMir;		
		for (int i=0; i<nObj; i++)
		{
			if (myObjets[i].type == 1) // if equal to reflector
			{
				randMir = RAND;
				if (randMir > randMirPrev)
				{
					randMirPrev = randMir;
					objP = myObjets[i];
				}
			}
		}
    
		if (objP.type == 1) // S'il y a un réflecteur 
		{
			#ifdef DOUBLE
			// // Création des transformations depuis les valeurs python du reflecteur		
			Transformd Tid;
			double posxd, posyd;

			// add rotation transformation
			Tid = DaddRotAndParseOrder(Tid, objP);

			if (objP.mvTz != 0) { // si diff de 0 alors il y a une translation en z
				double timeOned;
				timeOned = (tab_sensor[ph->is].POSZ-objP.mvTz)/vdouble.z;
				posxd = timeOned*vdouble.x;
				posyd = timeOned*vdouble.y;
			} // Les Translations en x et y sont prises en compte à la fin

			// Si l'objet plan est un rectangle avec p0 le point min et p3 le point max, nous pouvons faire ce qui suit
			double xMinPd = objP.p0x, yMinPd = objP.p0y, xMaxPd = objP.p3x, yMaxPd = objP.p3y;
			
			// Tirer aléatoirement une position sur la surface du miroir dans sa position initiale
			double3 posTransd = make_double3(   (  ( (xMaxPd-xMinPd)*double(RAND) ) + xMinPd  ), (  ( (yMaxPd-yMinPd)*double(RAND) ) + yMinPd  ), 0.  );
			
			// Application des transfos de rot du miroir à cette entité	
			posTransd = Tid(Pointd(posTransd));
			
			// Projection des positions x et y suivant la direction solaire sur la surface de base de l'entité
			double timeTwod;
			timeTwod = posTransd.z/vdouble.z;
			posTransd.x -= timeTwod*vdouble.x;
			posTransd.y -= timeTwod*vdouble.y;

			// On veut lancer les photons depuis TOA + prise en compte des transfos de translation en x et y			
			posTransd.x +=  posxd + double(objP.mvTx);
			posTransd.y +=  posyd + double(objP.mvTy);			
			posTransd.z = tab_sensor[ph->is].POSZ;
			
			// mise à jour de la position finale du photon
			ph->pos=make_float3(float(posTransd.x), float(posTransd.y), float(posTransd.z));		
			#else // IF NOT DOUBLE
			// // Création des transformations depuis les valeurs python du reflecteur
			Transform Ti;
			// add rotation transformation
			Ti = addRotAndParseOrder(Ti, objP);

			if (objP.mvTz != 0) { // si diff de 0 alors il y a une translation en z
				float timeOne;
				timeOne = (tab_sensor[ph->is].POSZ-objP.mvTz)/ph->v.z;
				ph->pos.x = timeOne*ph->v.x;
				ph->pos.y = timeOne*ph->v.y;
			} // Les Translations en x et y sont prises en compte à la fin			

			// Si l'objet plan est un rectangle avec p0 le point min et p3 le point max, nous pouvons faire ce qui suit
			float xMinP = objP.p0x, yMinP = objP.p0y, xMaxP = objP.p3x, yMaxP = objP.p3y;
			
			// Tirer aléatoirement une position sur la surface du miroir dans sa position initiale
			float3 posTrans = make_float3(   (  ( (xMaxP-xMinP)*RAND ) + xMinP  ), (  ( (yMaxP-yMinP)*RAND ) + yMinP  ), 0.  );
			
			// Application des transfos de rot du miroir à cette entité
			posTrans = Ti(Pointf(posTrans));
			
			// Projection des positions x et y suivant la direction solaire sur la surface de base de l'entité
			float timeTwo;
			timeTwo = posTrans.z/ph->v.z;
			posTrans.x -= timeTwo*ph->v.x;
			posTrans.y -= timeTwo*ph->v.y;

			// On veut lancer les photons depuis TOA + prise en compte des transfos de translation en x et y
			posTrans.x +=  ph->pos.x + objP.mvTx;
			posTrans.y +=  ph->pos.y + objP.mvTy;			
			posTrans.z = tab_sensor[ph->is].POSZ;
			
			// mise à jour de la position finale du photon
			ph->pos=posTrans;
			#endif // END IF DOUBLE OR FLOAT
		}
		/* ***************************************************************************************** */		
	} // LMODE == 1
	
	if (LMODEd == 2) // Full Forward mode
	{
		float3 cusForwPos = make_float3( ((CFXd * RAND) - 0.5*CFXd), ((CFYd * RAND) - 0.5*CFYd), 0.);
		ph->pos.x = PXd + cusForwPos.x + CFTXd;
		ph->pos.y = PYd + cusForwPos.y + CFTYd;
		ph->pos.z = PZd;

		if(PZd<120.) ph->loc = ATMOS;

		if (ALDEGd > 1e-6)
		{
			// Initialization of the cosine vector (v), the orthogonal vector (u), ...
			float3 vfloat = make_float3(0., 0., -1.); float3 ufloat = make_float3(-1., 0., 0.);
			float PHcone, THcone, sunTheta = 180-tab_sensor[ph->is].THDEG, sunPhi=tab_sensor[ph->is].PHDEG-180;

			if (TYPEd == 1)
			{
				// Lambertian cone sampling, see Dutré 2003
				float sinTHCone, sinTH;
				PHcone = 360*RAND;
				sinTHCone = __sinf(radians(ALDEGd));
				sinTH = sqrtf(RAND)*sinTHCone;
				THcone = asinf(sinTH)*180./CUDART_PI_F;
			}
			if (TYPEd == 2)
			{
				// Isotropic cone sampling, see Dutré 2003
				float cosTHCone;
				PHcone = 360*RAND;
				cosTHCone = __cosf(radians(ALDEGd));
				THcone = acosf(RAND*(cosTHCone-1)+1)*180./CUDART_PI_F;
				//ph->weight *= __cosf(radians(THcone));
			}
			if (TYPEd == 3) //in development
			{
				//disk cone sampling
				//Mixte between disk sampling (Dunn & Shultis 2011) and cone sampling 
				float rd, Rd;
				PHcone = 360*RAND;
				Rd = __tanf(radians(ALDEGd));
				rd = Rd*sqrtf(RAND);
				THcone = atanf(rd)*180/CUDART_PI_F;
				// // Use of the koepke limb model
				// float Gamm;
				// Gamm = GammaL(550., rd/Rd);
				// ph->weight*=Gamm;
			}
			// Transformations to consider the solar cone sampling	
			Transform TPHcone, TTHcone;
			TPHcone = TPHcone.RotateZ(PHcone); TTHcone = TTHcone.RotateY(THcone);		
			vfloat = TPHcone(   Vectorf(  TTHcone( Vectorf(vfloat) )  )   );
			ufloat = TPHcone(   Vectorf(  TTHcone( Vectorf(ufloat) )  )   );
            // Transformations to consider the sun direction
			Transform TTheta, TPhi;
			TTheta = TTheta.RotateY(sunTheta); TPhi = TPhi.RotateZ(sunPhi);
			vfloat = TPhi(   Vectorf(  TTheta( Vectorf(vfloat) )  )   );
			ufloat = TPhi(   Vectorf(  TTheta( Vectorf(ufloat) )  )   );
			
			// Update the values of u and v
			ph->v = normalize(make_float3(vfloat.x, vfloat.y, vfloat.z));
			ph->u = normalize(make_float3(ufloat.x, ufloat.y, ufloat.z));

			if (TYPEd == 2)
			{
				ph->weight *= dot(ph->v, make_float3(0., 0., -1.));
			}
		} // END ALDEGd > 0

	} //END LMODEd == 2
    #else // else if Backward modes
	if (LMODEd == 3 or LMODEd == 4) // common part between mode B and BR (cusBackward)
	{		
		// Initialization of the cosine vector (v), the orthogonal vector (u), ...
		double3 vdouble2 = make_double3(0., 0., 1.); double3 udouble2 = make_double3(1., 0., 0.);
		double PHconed, THconed;

		if (TYPEd == 1)
		{
			// Lambertian cone sampling, see Dutré 2003
			double sinTHCone, sinTH;
			PHconed = 360*RAND;
			sinTHCone = sin(radiansd(ALDEGd));
			sinTH = sqrt(RAND)*sinTHCone;
			THconed = asin(sinTH)*180./CUDART_PI;
		}
		else if (TYPEd == 2)
		{
			// Isotropic cone sampling, see Dutré 2003
			double cosTHconed;
			PHconed = 360*RAND;
			cosTHconed = cos(radiansd(ALDEGd));
			THconed = acos(RAND*(cosTHconed-1)+1)*180./CUDART_PI;
			ph->weight *= cos(radiansd(THconed));
		}
		
		// Creation of transforms
		Transformd TPHconed, TTHconed;
		TPHconed = TPHconed.RotateZ(PHconed); TTHconed = TTHconed.RotateY(THconed);
		
		// Apply transforms to vector u and v
		vdouble2 = TPHconed(   Vectord(  TTHconed( Vectord(vdouble2) )  )   );
		udouble2 = TPHconed(   Vectord(  TTHconed( Vectord(udouble2) )  )   );

		// Creation of transforms to consider theta and phi for the computation of photon dirs
		Transformd TTheta, TPhi;
		TTheta = TTheta.RotateY(tab_sensor[ph->is].THDEG);
		TPhi = TPhi.RotateZ(tab_sensor[ph->is].PHDEG);		

		// Apply transforms to vector u and v in function to theta and phi
		vdouble2 = TPhi(   Vectord(  TTheta( Vectord(vdouble2) )  )   );
		udouble2 = TPhi(   Vectord(  TTheta( Vectord(udouble2) )  )   );

		// update of u and v
		ph->v = make_float3(float(vdouble2.x), float(vdouble2.y), float(vdouble2.z));
		ph->u = make_float3(float(udouble2.x), float(udouble2.y), float(udouble2.z));

	} //END LMODEd == 3 or LMODEd == 4
	if (LMODEd == 4) // LMODE = "BR" Backward with receiver
	{
		// Collect the receiver object
		IObjets objP;
		objP = myObjets[nObj];
		
		Transform TR; // declare the transformation of the receiver

		// If x, y or z values are different to 0 then there is a translation
		if (objP.mvTx != 0 or objP.mvTy != 0 or objP.mvTz != 0) {
			Transform TmT;
			TmT = TR.Translate(make_float3(objP.mvTx, objP.mvTy, objP.mvTz));
			TR = TmT; }

		// Add rotation tranformations
		TR = addRotAndParseOrder(TR, objP); //see the function

		float sizeX = nbCx*TCd; float sizeY = nbCy*TCd;
		Pointf p_t(sizeX*0.5 - (RAND*sizeX), sizeY*0.5 - (RAND*sizeY), 0.);
		ph->posIni = make_float3(p_t.x, p_t.y, p_t.z);
		
		// Apply transfo and update the value of the photon position
		ph->pos = TR(p_t);
	} //END LMODEd == 4
	#endif // END !defined(BACK)
    #endif //END OBJ3D
    }


/*--------------------------------------------------------------------------------------------------*/
/*                      MOVE PHOTONS                                                                */
/*--------------------------------------------------------------------------------------------------*/
/*                      1) Spherical mode                                                           */
/*                      (ATMOS)
/*--------------------------------------------------------------------------------------------------*/

#ifdef SPHERIQUE
__device__ void move_sp(Photon* ph, struct Profile *prof_atm, int le, int count_level,
                        struct RNG_State *rngstate) {
	
    float tauRdm;
    float hph = 0.;  // cumulative optical thickness
    float vzn, delta1, h_cur, tau_cur, epsilon, AMF;
    #ifndef ALIS
    float h_cur_abs, tau_cur_abs;
    #endif
    float d;
    float rat;
    int sign_direction;
    int i_layer_fw, i_layer_bh; // index or layers forward and behind the photon
    float costh, sinth2;
    int ilam = ph->ilam*(NATMd+1);  // wavelength offset in optical thickness table
    float3 no, v0, u0;
	//int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);

    if (ph->layer == 0) ph->layer = 1;

    // Random Optical Thickness to go through
    // Two cases : (i) Forced First Scattering (taumax!=0 and no interaction yet)
    // the random optical thickness draw is biased (limited to tau_max)
    // (ii): general case, the classical exponential probability law is used 
    if (!le) {
        if ((ph->taumax != 0.F) && (ph->nint==0)) {
            tauRdm = -logf(1.F-RAND*(1.F-exp(-ph->taumax)));
            ph->weight *= (1.F-exp(-ph->taumax));
        }
        else tauRdm = -logf(1.F-RAND);
    }
    // if called with LE mode, it serves to compute the transmission
    // from photon last interaction position to TOA, thus 
    // photon is forced to exit upward or downward and tauRdm is chosen to be an upper limit
    else tauRdm = 1e6;

    ph->radius = length(ph->pos);
    vzn = __fdividef( dot(ph->v, ph->pos), ph->radius);

    // a priori value for sign_direction:
    // sign_direction may change sign from -1 to +1 if the photon does not
    // cross lower layer
    if (vzn <= 0) sign_direction = -1;
    else sign_direction = 1;

    if (REFRACd) {
        // we store initial photon direction
        v0=ph->v;
        u0=ph->u;
    }
    while (1) {

        //
        // stopping criteria
        //
        if (ph->layer == NATMd+1) {
            ph->loc = SURF0P;
            ph->tau = 0.;
            ph->layer -= 1;  // next time photon enters move_sp, it's at layers NATM
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

        // initializations
        costh = vzn;
        sinth2 = 1.f-costh*costh;
        //
        // calculate the distance d to the fw layer
        // from the current position
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
        AMF = __fdividef(d, abs(prof_atm[i_layer_bh].z - prof_atm[i_layer_fw].z)); // Air Mass Factor

        //
        // calculate the optical thicknesses h_cur and h_cur_abs to the next layer
        // We compute the layer extinction coefficient of the layer DTau/Dz and multiply by the distance within the layer
        //
        tau_cur = abs(get_OD(BEERd,prof_atm[i_layer_bh+ilam]) - get_OD(BEERd,prof_atm[i_layer_fw+ilam]));
        h_cur   = tau_cur * AMF;
        #ifndef ALIS
        tau_cur_abs = abs(prof_atm[i_layer_bh+ilam].OD_abs - prof_atm[i_layer_fw+ilam].OD_abs) ;
        h_cur_abs = tau_cur_abs * AMF;
        #endif

        //
        // update photon position (two cases)
        //

        // 1. photon stops within the layer
        if (hph + h_cur > tauRdm) {
            // fraction of maximum optical depth traveled in the final layer
            epsilon = (tauRdm - hph)/h_cur;
            // length traveled in the final layer
            d *= epsilon;
            // AMF the final layer
            AMF*= epsilon;

            // photon is located at his final position
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->radius = length(ph->pos);
            #ifndef ALIS
            // in the general case absorption is computed until final position if BEER
            if (BEERd == 1) ph->weight *= __expf(-( epsilon * h_cur_abs));
            #else
            // for ALIS case record cumulative distances and scatering corrections
            float tau;
            ph->cdist_atm[ph->layer] += d;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                tau = abs(get_OD(1,prof_atm[i_layer_bh + k*DL*(NATMd+1)]) 
                        - get_OD(1,prof_atm[i_layer_fw + k*DL*(NATMd+1)]));
			    ph->weight_sca[k] *= exp(-(tau-tau_cur)*AMF);
            }
            #endif
            break;

        } 
        
        // 2. photon advances to the next layer
        else {
            // cumulative OD is updated by the total OD of the current layer
            hph += h_cur;
            // photon is located at his new position
            ph->pos = operator+(ph->pos, ph->v*d);
            ph->radius = length(ph->pos);
            // update local vertical
            no = operator/(ph->pos, ph->radius);
            vzn = dot(ph->v, no);

            //
            // REFRACTION
            //
            #ifdef DEBUG
            // Compute refraction for the Line of Sight only
            if (REFRACd && (!le || (le && FFSd && ph->nint==0))) {
            #else
            // Compute refraction everywhere
            if (REFRACd)  {
            #endif
                float3 uu = normalize(cross(no, ph->v)); // unit vector around which one turns
                // We update photon direction at the interface due to refraction
                // 1. sin_i just to verify if refraction occurs
                float s1    = sqrt(1. - vzn*vzn);
                if (s1 > 1.) s1=1.;
                // 2. determine old and new refraction indices from old and new layer indices
                float nind  = __fdividef(prof_atm[i_layer_fw+ilam].n, prof_atm[i_layer_bh+ilam].n);
                float i2, alpha = 0.; // emergent direction, deviation angle
                if (s1!=0. && nind!=1.) { // in case of refraction
	              if((s1 <= nind) || (nind > 1.)) {
                      i2 = __fdividef(s1, nind);
                      if (i2 > 1.) i2=1.;
                      i2 = asin(i2); 
                      alpha   = fabs(i2 - asin(s1));
                  }
                  else alpha=0.F;
                  // we rotate around uu which lies in the vertical plane, by the deviation angle
                  float3x3 R=rotation3D(alpha, uu);
                  // new photon direction 
                  float3 v2 = normalize(mul(R, ph->v));
                  ph->v = v2;
                  vzn = dot(ph->v, no);
                } // no refraction computation necessary
            } // No Refraction

            #ifndef ALIS
            // in the general case absorption is computed until new position if BEER
            if (BEERd == 1) ph->weight *= __expf(-( h_cur_abs));
            #else
            // for ALIS case record cumulative distances and scatering corrections
            float tau;
            ph->cdist_atm[ph->layer] += d;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                tau = abs(get_OD(1,prof_atm[i_layer_bh + k*DL*(NATMd+1)]) 
                        - get_OD(1,prof_atm[i_layer_fw + k*DL*(NATMd+1)]));
			    ph->weight_sca[k] *= __expf(-(tau-tau_cur)*AMF);
            }
            #endif

            ph->layer -= sign_direction;
        } // photon advances to next layer

    } // while loop

    // update u vector
    #ifdef DEBUG
    if (REFRACd && (!le || (le && FFSd && ph->nint==0))) {
    //if (REFRACd && ph->nint==0 && !le) {
    #else
    if (REFRACd)  {
    #endif
        float psi;
        // Update photon u vector
        ComputePsiLE(u0, v0, ph->v, &psi, &ph->u); 

        if (psi!=0.F) {
		 rotateStokes(ph->stokes, psi, &ph->stokes);
         #ifdef BACK
		 float4x4 L;
		 rotationM(-psi,&L);
		 ph->M   = mul(ph->M,L);
         #endif
        }
    }

    // LE mode used to compute transmission
    if (le) {
        // if the transmission to be evaluated is toward TOA and the photon is not in SPACE
        // it has been stopped somewhere (hidden) and thus transmission is zero (weight=0)
        // same reasoning for a transmission toward the surface and the photon location not at the surface
        // we also update taumax of the photon to be eventually used in the Forced First Scattering mode
        if (( (count_level==UPTOA)  && (ph->loc==SPACE ) ) || 
            ( (count_level==DOWN0P) && (ph->loc==SURF0P) )) ph->weight *= __expf(-hph);
        else ph->weight = 0.;
        if (ph->loc==SPACE || ph->loc==SURF0P) ph->taumax = hph;
    }

    // in case of propagation, if the photon is still in the atmosphere
    // compute the absorption using Single Scattering Albedo if BEER=0
    if ((BEERd == 0) && (ph->loc == ATMOS)) ph->weight *= prof_atm[ph->layer+ilam].ssa;
}
#endif // SPHERIQUE



/*--------------------------------------------------------------------------------------------------*/
/*                      MOVE PHOTONS                                                                */
/*--------------------------------------------------------------------------------------------------*/
/*                      2) Plane Parrallel 1D Standard mode                                         */
/*                      (OCEAN or ATMOS)
/*--------------------------------------------------------------------------------------------------*/
#ifdef ALT_PP
 #ifndef OPT3D // 1D
__device__ void move_pp2(Photon* ph, struct Profile *prof_atm, 
                        struct Profile *prof_oc, int le, int count_level,
                        struct RNG_State *rngstate) {


    if (!le && ph->scatterer == THERMAL_EM) return;

    float tauRdm;
    float hph = 0.;  // cumulative optical thickness
    float vzn, h_cur, tau_cur, epsilon, AMF;
    #ifndef ALIS
    float h_cur_abs, tau_cur_abs;
    #endif
    float d;
    int sign_direction;
    int i_layer_fw, i_layer_bh; // index or layers forward and behind the photon
    int ilam; 
    struct Profile *prof;
    int  NL;
    
    // use profile corresponding to photon location
    if (ph->loc==OCEAN) {
        NL   = NOCEd+1;
        prof = prof_oc;
    }
    if (ph->loc==ATMOS) {
        NL   = NATMd+1;
        prof = prof_atm;
    }
    ilam = ph->ilam*NL;  // wavelength offset in optical thickness table

    if (ph->layer == 0) ph->layer = 1;

    // Random Optical Thickness to go through
    if (!le) tauRdm = -logf(1.F-RAND);
    // if called with le mode, it serves to compute the transmission
    // from photon last intercation position to TOA, thus 
    // photon is forced to exit upward or downward and tauRdm is chosen to be an upper limit
    else tauRdm = 1e6;

    vzn = ph->v.z;

    // a priori value for sign_direction:
    // sign_direction may change sign from -1 to +1 if the photon does not
    // cross lower layer
    if (vzn <= 0) sign_direction = -1;
    else sign_direction = 1;
    int count=0;

    while (1) {

        //
        // stopping criteria
        //
        // 1. Photon in atmosphere
        if (ph->loc == ATMOS) {
         if (ph->layer == NATMd+1) {
            // photon has reached surface
            ph->loc = SURF0P;
            ph->layer -= 1;  // next time photon enters , it's at layers NATM
            break;
         }
         if (ph->layer <= 0) {
            // photon has reached TOA
            ph->loc = SPACE;
            break;
         }
        } 

        // 2. Photon in ocean
        if (ph->loc == OCEAN) {
         if (ph->layer == NOCEd+1) {
            // photon has reached seafloor
            ph->loc = SEAFLOOR;
            ph->layer -= 1;  // next time photon enters , it's at layers NOCE
            break;
         }
         if (ph->layer <= 0) {
            // photon has reached surface
            if (SIMd!=3) ph->loc = SURF0M;
            else ph->loc = SPACE;
            ph->layer= 0;
            break;
         }
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

        //
        // calculate the distance d to the fw layer
        // from the current position
        d   = __fdividef(abs(ph->pos.z - prof[i_layer_fw].z), fabs(ph->v.z));
        AMF = __fdividef(d, abs(prof[i_layer_bh].z - prof[i_layer_fw].z)); // Air Mass Factor

        //
        // calculate the optical thicknesses h_cur and h_cur_abs to the next layer
        // We compute the layer extinction coefficient of the layer DTau/Dz 
        // and multiply by the distance within the layer
        //
        tau_cur = abs(get_OD(BEERd,prof[i_layer_bh+ilam]) 
                    - get_OD(BEERd,prof[i_layer_fw+ilam]));
        h_cur   = tau_cur * AMF;
        #ifndef ALIS
        tau_cur_abs = abs(prof[i_layer_bh+ilam].OD_abs - prof[i_layer_fw+ilam].OD_abs);
        h_cur_abs = tau_cur_abs * AMF;
        #endif

        //
        // update photon position
        // (See comments as for move_sp)
        //
        // 1. photon stops within the layer
        if (hph + h_cur > tauRdm) {
            if (h_cur !=0.F) epsilon = (tauRdm - hph)/h_cur;
            else epsilon =1.F;
            d *= epsilon;
            AMF*= epsilon;
            ph->pos = operator+(ph->pos, ph->v*d);
            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( epsilon * h_cur_abs));
            #else
            float tau;
            if (ph->loc==ATMOS) ph->cdist_atm[ph->layer] += d;
            if (ph->loc==OCEAN) ph->cdist_oc[ ph->layer] += d;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                tau = abs(get_OD(1,prof[i_layer_bh + k*DL*NL]) 
                        - get_OD(1,prof[i_layer_fw + k*DL*NL]));
			    ph->weight_sca[k] *= exp(-(tau-tau_cur)*AMF);
            }
            #endif
            break;

        } 
        
        // 2. photon advances to the next layer
        else {
            hph += h_cur;
            ph->pos = operator+(ph->pos, ph->v*d);

            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( h_cur_abs));
            #else
            float tau;
            if (ph->loc==ATMOS) ph->cdist_atm[ph->layer] += d;
            if (ph->loc==OCEAN) ph->cdist_oc[ ph->layer] += d;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                tau = abs(get_OD(1,prof[i_layer_bh + k*DL*NL]) 
                        - get_OD(1,prof[i_layer_fw + k*DL*NL]));
			    ph->weight_sca[k] *= __expf(-(tau-tau_cur)*AMF);
            }
            #endif

            ph->layer -= sign_direction;
            count++;
        } // photon advances to next layer

    } // while loop

    // if the transmission to be evaluated is toward TOA and the photon is not in SPACE
    // it has been stopped somewhere (hidden) and thus transmission is zero (weight=0)
    // same reasoning for a transmission toward the surface 0+ and the photon location not at the surface 0+
    // same reasoning for a transmission toward the surface 0- and the photon location not at the surface 0-
    // same reasoning for a transmission toward the seafloor  and the photon location not at the seafloor
    if (le) {
        if (( (count_level==UPTOA)  && (ph->loc==SPACE ) ) || 
            ( (count_level==DOWN0P) && (ph->loc==SURF0P) ) ||
            ( (count_level==UP0M)   && (ph->loc==SURF0M) ) ||
            ( (count_level==UP0M)   && (ph->loc==SPACE) && (SIMd==3) ) ||
            ( (count_level==DOWNB)  && (ph->loc==SEAFLOOR) ) ) 
            ph->weight *= __expf(-hph);
        else ph->weight = 0.;
    }

    // in case of propagation, if the photon is still in the atmosphere or ocean
    // compute the absorption using Single Scattering Albedo if BEER=0
    if ((BEERd == 0) && ((ph->loc == ATMOS) || (ph->loc == OCEAN))) {
        ph->weight *= prof[ph->layer+ilam].ssa;
    }
}



/*--------------------------------------------------------------------------------------------------*/
/*                      MOVE PHOTONS                                                                */
/*--------------------------------------------------------------------------------------------------*/
/*                      3) Plane Parrallel 3D Standard mode                                         */
/*                      (OCEAN or ATMOS)                                                            */
/*--------------------------------------------------------------------------------------------------*/
#else// OPT3D
__device__ void move_pp2_bak(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc, 
                         struct Cell *cell_atm, struct Cell *cell_oc,
                        int le, int count_level, struct RNG_State *rngstate) {

    float tauRdm;    // Sample optical thickness
    float hph = 0.;  // Cumulative optical thickness
    float vzn, h_cur, coef_cur, epsilon;
    #ifndef ALIS
    float h_cur_abs, tau_cur_abs;
    #endif
    float d;
    int ilam; 
    struct Profile *prof;
    struct Cell *cell;
    int  NL;
	float intTime0=0., intTime1=0.;
	float3 intersectPoint = make_float3(-1., -1., -1.);
	bool intersectBox;
    float3 pmin, pmax ;
	//int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);


    if (ph->loc==OCEAN) {
        NL   = NOCEd+1;
        prof = prof_oc;
        cell = cell_oc;
    }
    if (ph->loc==ATMOS) {
        NL   = NATMd+1;
        prof = prof_atm;
        cell = cell_atm;
    }
    ilam = ph->ilam*NL;  // wavelength offset in optical thickness table

    // Random Optical Thickness to go through
    if (!le) tauRdm = -logf(1.F-RAND);
    // if called with le mode, it serves to compute the transmission
    // from photon last intercation position to TOA, thus 
    // photon is forced to exit upward or downward and tauRdm is chosen to be an upper limit
    else tauRdm = 1e6;

    // Init photon position
    int count=0;
    int next_layer=ph->layer;
    pmin = make_float3(cell[ph->layer].pminx, cell[ph->layer].pminy, cell[ph->layer].pminz);
    pmax = make_float3(cell[ph->layer].pmaxx, cell[ph->layer].pmaxy, cell[ph->layer].pmaxz);
    BBox Box_ini(pmin, pmax);
    ph->pos = Box_ini.RoundAlmostInside(ph->pos);
    //if (!Box_ini.AlmostInside(ph->pos) && ph->layer>=0 ) {
    //if (!Box_ini.Inside(ph->pos) && ph->layer>=0 ) {
        //if (idx==0) printf("%d %d %d %d %d %g %g %g %g %g %f %f %f %g %g\n",idx, ph->loc, ph->layer, next_layer, 
         //     count, (double)intTime0, (double)intTime1, ph->pos.x, ph->pos.y, ph->pos.z
          //    ,ph->v.x, ph->v.y,ph->v.z,pmin.z, pmax.z);
         //ph->loc = REMOVED;
         //return;
    //}   

    while (1) {
        // avoid infinite loop
        if (count >= 500) {
            ph->loc=REMOVED; 
            break;}
        if (ph->loc == REMOVED) break;

        // Update photon location if exit and cell number
        if (next_layer == BOUNDARY_ABS){
            ph->loc = ABSORBED;
            break;
        }
        if (ph->loc == ATMOS) {
         if (next_layer == BOUNDARY_0P) {
            ph->loc = SURF0P;
            break;
         }
         else if (next_layer == BOUNDARY_TOA) {
            ph->loc = SPACE;
            break;
         }
         else {
             ph->layer = next_layer;
         }
        } 
        if (ph->loc == OCEAN) {
         if (next_layer == BOUNDARY_FLOOR) {
            ph->loc = SEAFLOOR;
            break;
         }
         else if (next_layer == BOUNDARY_0M) {
            ph->loc = SURF0M;
            break;
         }
         else {
             ph->layer = next_layer;
         }
        }

        // Intersection with current cell boundaries
		Ray Ray_cur(ph->pos, ph->v, 0);
        pmin = make_float3(cell[ph->layer].pminx, cell[ph->layer].pminy, cell[ph->layer].pminz);
        pmax = make_float3(cell[ph->layer].pmaxx, cell[ph->layer].pmaxy, cell[ph->layer].pmaxz);
        BBox Box_cur(pmin, pmax);
		intersectBox = Box_cur.IntersectP(Ray_cur, &intTime0, &intTime1);

        //if (intTime1 !=0) { // the photon is not already on a boundary
        if (intersectBox) { // the photon is not already on a boundary
        //intersectPoint = Box_cur.RoundAlmostInside(operator+(ph->pos, ph->v * intTime1));
        intersectPoint = operator+(ph->pos, ph->v * intTime1);
        //if ((!Box_cur.AlmostInside(ph->pos) || (intTime0 == intTime1)) &&
        //if ((!Box_cur.Inside(ph->pos) || (intTime0 == intTime1)) &&
         //       (ph->layer>=0)) {
                //(ph->layer>=0)) {
            //if (idx==0 && count>=0) printf("%d %d %d %d %d %g %g %f %f %f %f %f %f\n",idx, ph->loc, ph->layer, next_layer, 
             //      count, (double)intTime0, (double)intTime1, ph->pos.x, ph->pos.y, ph->pos.z
              //    ,ph->v.x, ph->v.y,ph->v.z);
          //  ph->loc = REMOVED;
           // break;
        //}
        
        //
        // calculate the optical thicknesses h_cur and h_cur_abs to the next layer
        // We get the layer extinction coefficient and multiply by the distance within the layer
        //
        coef_cur = get_OD(BEERd,prof[cell[ph->layer].iopt+ilam]);
        h_cur    = coef_cur * intTime1;

        #ifndef ALIS
        h_cur_abs = prof[cell[ph->layer].iopt+ilam].OD_abs * intTime1;
        #endif

        //
        // update photon position
        //
        if (hph + h_cur > tauRdm) {
            // photon stops within the box
            epsilon = (tauRdm - hph)/h_cur;
            intTime1 *= epsilon;
            ph->pos = operator+(ph->pos, ph->v * intTime1);
            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( epsilon * h_cur_abs));
            #else
            float coef;
            if (ph->loc==ATMOS) ph->cdist_atm[cell[ph->layer].iabs] += intTime1;
            if (ph->loc==OCEAN) ph->cdist_oc[ cell[ph->layer].iabs] += intTime1;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                coef = get_OD(1,prof[cell[ph->layer].iopt + k*DL*NL]);
			    ph->weight_sca[k] *= exp(-(coef-coef_cur)*intTime1);
            }
            #endif
            break;

        } else {
            // photon advances to the next layer
            hph += h_cur;
            ph->pos = intersectPoint;

            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( h_cur_abs));
            #else
            float coef;
            if (ph->loc==ATMOS) ph->cdist_atm[cell[ph->layer].iabs] += intTime1;
            if (ph->loc==OCEAN) ph->cdist_oc[ cell[ph->layer].iabs] += intTime1;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                coef = get_OD(1,prof[cell[ph->layer].iopt + k*DL*NL]);
			    ph->weight_sca[k] *= __expf(-(coef-coef_cur)*intTime1);
            }
            #endif

            // determine the index of the next potential box
            //
            float3 p=ph->pos;
            operator-=(p, operator+(pmin*0.5, pmax*0.5));
            p.x = __fdividef(p.x, fabs(pmax.x-pmin.x));
            p.y = __fdividef(p.y, fabs(pmax.y-pmin.y));
            p.z = __fdividef(p.z, fabs(pmax.z-pmin.z));
            int ind;
            GetFaceIndex(p, &ind);
            switch(ind)
            {
                 case 0: next_layer = cell[ph->layer].neighbour1; break;
                 case 1: next_layer = cell[ph->layer].neighbour2; break;
                 case 2: next_layer = cell[ph->layer].neighbour3; break;
                 case 3: next_layer = cell[ph->layer].neighbour4; break;
                 case 4: next_layer = cell[ph->layer].neighbour5; break;
                 case 5: next_layer = cell[ph->layer].neighbour6; break;
                 default: ph->loc = REMOVED;
            }
            count++;
            //if (idx==0 && count>=0) printf("YES %d %d %d %d %d %g %g %f %f %f %f %f %f\n",idx, ph->loc, ph->layer, next_layer, 
             //     count, (double)intTime0, (double)intTime1, ph->pos.x, ph->pos.y, ph->pos.z
              //    ,ph->v.x, ph->v.y,ph->v.z);
            // in case of periodic boundaries
            /*if (next_layer >= 0) {
                GetFaceMiddlePoint(ind, pmin, pmax, &p);
                float3 pmin_next = make_float3(cell[next_layer].pminx, cell[next_layer].pminy, cell[next_layer].pminz);
                float3 pmax_next = make_float3(cell[next_layer].pmaxx, cell[next_layer].pmaxy, cell[next_layer].pmaxz);
                float3 p_next;
                GetFaceMiddlePoint(_FLIP(ind), pmin_next, pmax_next, &p_next);
                if (idx==0) printf("Apres %d %d %d %d %f %d %d %f %f %f %f %f %f %f %f %f\n",ph->nint, ph->loc, ph->layer, next_layer, ph->v.z, ind, 
                    _FLIP(ind), p_next.x, p_next.y, p_next.z, 
                    ph->pos.x, ph->pos.y, ph->pos.z,  p.x, p.y, p.z);
                // translation of the photon to the next layer
                operator+=(ph->pos, operator-(p_next,p));
            }*/
        } // photon advances to next layer

        } // Intersection True

        else { //InterSection False
            // determine the index of the next potential box
            float3 p=ph->pos;
            operator-=(p, operator+(pmin*0.5, pmax*0.5));
            p.x = __fdividef(p.x, fabs(pmax.x-pmin.x));
            p.y = __fdividef(p.y, fabs(pmax.y-pmin.y));
            p.z = __fdividef(p.z, fabs(pmax.z-pmin.z));
            int ind;
            GetFaceIndex(p, &ind);
            switch(ind)
            {
                 case 0: next_layer = cell[ph->layer].neighbour1; break;
                 case 1: next_layer = cell[ph->layer].neighbour2; break;
                 case 2: next_layer = cell[ph->layer].neighbour3; break;
                 case 3: next_layer = cell[ph->layer].neighbour4; break;
                 case 4: next_layer = cell[ph->layer].neighbour5; break;
                 case 5: next_layer = cell[ph->layer].neighbour6; break;
                 default: ph->loc = REMOVED;
            }
            count++;
            //if (idx==0 && count>=0) printf("NO  %d %d %d %d %d %g %g %f %f %f %f %f %f\n",idx, ph->loc, ph->layer, next_layer, 
             //     count, (double)intTime0, (double)intTime1, ph->pos.x, ph->pos.y, ph->pos.z
              //    ,ph->v.x, ph->v.y,ph->v.z);
        }

    } // while loop

    if (le) {
        if (( (count_level==UPTOA)  && (ph->loc==SPACE ) ) || 
            ( (count_level==DOWN0P) && (ph->loc==SURF0P) ) ||
            ( (count_level==UP0M)   && (ph->loc==SURF0M) ) ||
            ( (count_level==DOWNB)  && (ph->loc==SEAFLOOR) ) ) 
            ph->weight *= __expf(-hph);
        else ph->weight = 0.;
    }

    if ((BEERd == 0) && ((ph->loc == ATMOS) || (ph->loc == OCEAN))) {
        ph->weight *= prof[cell[ph->layer].iopt+ilam].ssa;
    }
}
 #endif // 3D
#endif // ALT_PP

#ifdef ALT_PP
#ifdef OPT3D
//!!!!!!!!!!!!!!!!!!!! DEV !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__device__ void move_pp2(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc, 
                         struct Cell *cell_atm, struct Cell *cell_oc,
                        int le, int count_level, struct RNG_State *rngstate) {

    float tauRdm;    // Sample optical thickness
    float hph = 0.;  // Cumulative optical thickness
    float vzn, h_cur, coef_cur, epsilon;
    #ifndef ALIS
    float h_cur_abs, tau_cur_abs;
    #endif
    float d;
    int ilam; 
    struct Profile *prof;
    struct Cell *cell;
    int  NL;
	float intTime0=0., intTime1=0.;
	float3 intersectPoint;
	bool intersectBox;
    float3 pmin, pmax ;
	int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);

    if (ph->loc==OCEAN) {
        NL   = NOCEd+1;
        prof = prof_oc;
        cell = cell_oc;
    }
    if (ph->loc==ATMOS) {
        NL   = NATMd+1;
        prof = prof_atm;
        cell = cell_atm;
    }
    ilam = ph->ilam*NL;  // wavelength offset in optical thickness table

    // Random Optical Thickness to go through
    if (!le) tauRdm = -logf(1.F-RAND);
    // if called with le mode, it serves to compute the transmission
    // from photon last intercation position to TOA, thus 
    // photon is forced to exit upward or downward and tauRdm is chosen to be an upper limit
    else tauRdm = 1e6;

    // Init photon cell
    int count=0;
    int next_layer=ph->layer;

    while (1) {
        // avoid infinite loop
        if (count >= 500) ph->loc=REMOVED; 

        // stop propagating useless photons
        if (ph->loc == REMOVED) break;

        // identify absorbed photons and stop propagating them
        if (next_layer == BOUNDARY_ABS){
            ph->loc = ABSORBED;
            break;
        }

        // Update photon location, and cell number if still in atmosphere
        if (ph->loc == ATMOS) {
         if (next_layer == BOUNDARY_0P) {
            ph->loc = SURF0P;
            break;
         }
         else if (next_layer == BOUNDARY_TOA) {
            ph->loc = SPACE;
            break;
         }
         else {
             ph->layer = next_layer;
         }
        } 

        // Update photon location, and cell number if still in ocean
        if (ph->loc == OCEAN) {
         if (next_layer == BOUNDARY_FLOOR) {
            ph->loc = SEAFLOOR;
            break;
         }
         else if (next_layer == BOUNDARY_0M) {
            ph->loc = SURF0M;
            break;
         }
         else {
             ph->layer = next_layer;
         }
        }

        // Intersection with current cell boundaries
		Ray Ray_cur(ph->pos, ph->v, 0);
        pmin = make_float3(cell[ph->layer].pminx, cell[ph->layer].pminy, cell[ph->layer].pminz);
        pmax = make_float3(cell[ph->layer].pmaxx, cell[ph->layer].pmaxy, cell[ph->layer].pmaxz);
        BBox Box_cur(pmin, pmax);
		intersectBox = Box_cur.IntersectP(Ray_cur, &intTime0, &intTime1);

        if (intersectBox) { // the photon is not already on a boundary
        //
        intersectPoint = operator+(ph->pos, ph->v * intTime1);
        // calculate the optical thicknesses h_cur and h_cur_abs to the next layer
        // We get the layer extinction coefficient and multiply by the distance within the layer
        //
        coef_cur = get_OD(BEERd,prof[cell[ph->layer].iopt+ilam]);
        h_cur    = coef_cur * intTime1;

        #ifndef ALIS
        h_cur_abs = prof[cell[ph->layer].iopt+ilam].OD_abs * intTime1;
        #endif

        //
        // update photon position
        //
        if (hph + h_cur > tauRdm) {
            // photon stops within the box
            epsilon = (tauRdm - hph)/h_cur;
            intTime1 *= epsilon;
            ph->pos = operator+(ph->pos, ph->v * intTime1);
            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( epsilon * h_cur_abs));
            #else
            float coef;
            if (ph->loc==ATMOS) ph->cdist_atm[cell[ph->layer].iabs] += intTime1;
            if (ph->loc==OCEAN) ph->cdist_oc[ cell[ph->layer].iabs] += intTime1;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                coef = get_OD(1,prof[cell[ph->layer].iopt + k*DL*NL]);
			    ph->weight_sca[k] *= exp(-(coef-coef_cur)*intTime1);
            }
            #endif
            break;

        } else {
            // photon advances to the next layer
            hph += h_cur;
            ph->pos = intersectPoint;

            #ifndef ALIS
            if (BEERd == 1) ph->weight *= __expf(-( h_cur_abs));
            #else
            float coef;
            if (ph->loc==ATMOS) ph->cdist_atm[cell[ph->layer].iabs] += intTime1;
            if (ph->loc==OCEAN) ph->cdist_oc[ cell[ph->layer].iabs] += intTime1;
            int DL=(NLAMd-1)/(NLOWd-1);
            for (int k=0; k<NLOWd; k++) {
                coef = get_OD(1,prof[cell[ph->layer].iopt + k*DL*NL]);
			    ph->weight_sca[k] *= __expf(-(coef-coef_cur)*intTime1);
            }
            #endif

            // determine the index of the next potential box
            //
            float3 p=ph->pos;
            operator-=(p, operator+(pmin*0.5, pmax*0.5));
            p.x = __fdividef(p.x, fabs(pmax.x-pmin.x));
            p.y = __fdividef(p.y, fabs(pmax.y-pmin.y));
            p.z = __fdividef(p.z, fabs(pmax.z-pmin.z));
            int ind;
            GetFaceIndex(p, &ind);
            switch(ind)
            {
                 case 0: next_layer = cell[ph->layer].neighbour1; break;
                 case 1: next_layer = cell[ph->layer].neighbour2; break;
                 case 2: next_layer = cell[ph->layer].neighbour3; break;
                 case 3: next_layer = cell[ph->layer].neighbour4; break;
                 case 4: next_layer = cell[ph->layer].neighbour5; break;
                 case 5: next_layer = cell[ph->layer].neighbour6; break;
                 default: ph->loc = REMOVED;
            }
            count++;
        } // photon advances to next layer

        } // Intersection True

        else { //InterSection False
            // determine the index of the next potential box
            float3 p=ph->pos;
            operator-=(p, operator+(pmin*0.5, pmax*0.5));
            p.x = __fdividef(p.x, fabs(pmax.x-pmin.x));
            p.y = __fdividef(p.y, fabs(pmax.y-pmin.y));
            p.z = __fdividef(p.z, fabs(pmax.z-pmin.z));
            int ind;
            GetFaceIndex(p, &ind);
            switch(ind)
            {
                 case 0: next_layer = cell[ph->layer].neighbour1; break;
                 case 1: next_layer = cell[ph->layer].neighbour2; break;
                 case 2: next_layer = cell[ph->layer].neighbour3; break;
                 case 3: next_layer = cell[ph->layer].neighbour4; break;
                 case 4: next_layer = cell[ph->layer].neighbour5; break;
                 case 5: next_layer = cell[ph->layer].neighbour6; break;
                 default: ph->loc = REMOVED;
            }
            count++;
        }

    } // while loop

    if (le) {
        if (( (count_level==UPTOA)  && (ph->loc==SPACE ) ) || 
            ( (count_level==DOWN0P) && (ph->loc==SURF0P) ) ||
            ( (count_level==UP0M)   && (ph->loc==SURF0M) ) ||
            ( (count_level==DOWNB)  && (ph->loc==SEAFLOOR) ) ) 
            ph->weight *= __expf(-hph);
        else ph->weight = 0.;
    }

    if ((BEERd == 0) && ((ph->loc == ATMOS) || (ph->loc == OCEAN))) {
        ph->weight *= prof[cell[ph->layer].iopt+ilam].ssa;
    }
}
 #endif // 3D
#endif // ALT_PP
//!!!!!!!!!!!!!!!!!!!! DEV !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


/*--------------------------------------------------------------------------------------------------*/
/*                      MOVE PHOTONS                                                                */
/*--------------------------------------------------------------------------------------------------*/
/*                      4) Plane Parrallel 1D Fast mode                                             */
/*                      (OCEAN or ATMOS)                                                            */
/*--------------------------------------------------------------------------------------------------*/
#if !defined(ALT_PP) && !defined(SPHERIQUE)
__device__ void move_pp(Photon* ph, struct Profile *prof_atm, struct Profile *prof_oc,
                        struct RNG_State *rngstate
						#ifdef OBJ3D
						, IGeo *geoS, struct IObjets *myObjets, struct GObj *myGObj, void *tabObjInfo
						#endif
	) {

    // tau is the main vertical coordinate in this mode
	float delta_i=0.f, delta=0.f, epsilon;
	float tauR, tauBis, phz; //rdist
    int ilayer;
	
    #ifdef ALIS
    // ALIS scattering correction
    float dsca_dl, dsca_dl0=-ph->tau ;
    int DL=(NLAMd-1)/(NLOWd-1);
    #else
    float ab;
    #endif

	#ifdef OBJ3D
	float prev_tau;
	prev_tau = ph->tau;        // previous value of tau photon
    #endif
  
    // Optical depth sampling
	tauR = -logf(1.f - RAND);
	ph->tau += (tauR*ph->v.z); // the value of tau is updated

    // 1. OCEAN case
        // partial geometrical thickness in the layer (from top) : 
        // epsilon = (z[i-1]-z)/(z[i-1]-z[i]) ; 0<epsilon<1
        // optical thickness of the layer delta[i] = OD[i-1] - OD[i], delta[i] >0
        // partial optical thickness in the layer (from top) : epsilon * delta[i]
        // tau(z) = tau[i-1] - epsilon * delta[i]
        // tau[i-1] = OD[i-1]
        // **********************************************************
        // **** tau(z) = OD[i-1] - epsilon * (OD[i-1] - OD[i])  *****
        // **********************************************************

	if (ph->loc == OCEAN){  
        // If tau>0 photon is reaching the surface 
        if (ph->tau > 0) {
            ph->layer = 1;
            epsilon   = 0.F;
            #ifndef ALIS
            // in the general case absorption using BEER law
            if (BEERd == 1) {// absorption between start and stop
                ab =  0.F; // absorption OD at the surface
                // absorption between current photon tau_abs and final value ab
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            // for ALIS scattering correction between 
            // old an new position for the reference (propagation) wavelength
            // this is included in dsca_dl0
            // for other wavelengths it is dsca_dl[k]
            // scattering correction is the differential attenuation between
            // wavelengths (weight_sca[k])
            // finally update of the spectral tau_sca to the surface values : here 0 
            dsca_dl0 += 0.F;
            for (int k=0; k<NLOWd; k++) {
                dsca_dl =- ph->tau_sca[k]; 
                dsca_dl += 0.F;
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl)-fabs(dsca_dl0),  fabs(ph->v.z)));
                ph->tau_sca[k] = 0.F;
            }
            #endif

            // update of the reference wavelength photon tau and tau_abs values to the surface values : here 0 
            ph->tau = 0.F;
            ph->tau_abs = 0.F;
            // location update
            ph->loc = SURF0M;
            if (SIMd == OCEAN_ONLY){
              ph->loc = SPACE;
            }

            // move the photon forward up to the surface
            // the linear distance is ph->z/ph->vz
            operator+=(ph->pos, ph->v * fabs(ph->pos.z/ph->v.z));

            #ifdef ALIS 
            // complete the records for this event
            ph->nevt++;
            ph->layer_prev[ph->nevt] = -ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            #endif
           return;
        } // Surface

        // If tau<TAU_OCEAN photon is reaching the sea bottom
        else if( ph->tau < get_OD(BEERd, prof_oc[NOCEd + ph->ilam *(NOCEd+1)]) ){
            ph->layer = NOCEd;
            epsilon   = 1.F;
            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                // at the seafloor ab=TAU_OCEAN_ABS
                ab = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].OD_abs;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            // at the seafloor tau_sca=TAU_OCEAN
            dsca_dl0 += get_OD(1,prof_oc[NOCEd + ph->ilam*(NOCEd+1)]) ; 
            for (int k=0; k<NLOWd; k++) {
                dsca_dl =- ph->tau_sca[k]; 
                dsca_dl += get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]);
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) - fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = get_OD(1,prof_oc[NOCEd + k*DL*(NOCEd+1)]);
            }
            #endif

            // as usual updates
            ph->loc = SEAFLOOR;
            ph->tau = get_OD(BEERd, prof_oc[NOCEd + ph->ilam *(NOCEd+1)]);
            ph->tau_abs = prof_oc[NOCEd + ph->ilam *(NOCEd+1)].OD_abs;

			// move the photon forward down to the seafloor
            operator+=(ph->pos, ph->v * fabs( (ph->pos.z - prof_oc[NOCEd].z) /ph->v.z));

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = -ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            #endif
            return;
        }   // Seafloor 
        
        // Photon is still in the OCEAN
        else {
            //computing photons final layer number
            ilayer = 1;
            while (( get_OD(BEERd, prof_oc[ilayer+ ph->ilam *(NOCEd+1)]) > (ph->tau)) && (ilayer < NOCEd)) {
                ilayer++;
            }
            ph->layer = ilayer;

            // computing fine position within the layer
            // layer Optical thickness
            delta_i= fabs(get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]) 
                        - get_OD(BEERd, prof_oc[ilayer  +ph->ilam*(NOCEd+1)]));
            // optical thickness between position andd top of layer
            delta= fabs(ph->tau - get_OD(BEERd, prof_oc[ilayer-1+ph->ilam*(NOCEd+1)])) ;
            // fractional optical thickness
            epsilon = __fdividef(delta,delta_i);
            
            #ifndef ALIS
            // General case absorption using Single scattering albedo
            if (BEERd == 0) ph->weight *= prof_oc[ph->layer+ph->ilam*(NOCEd+1)].ssa;
            else { // We compute the cumulated absorption OT at the new postion of the photon
                // photon new position in the layer
                ab = prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs 
                     - epsilon * 
                     (prof_oc[ilayer-1+ph->ilam*(NOCEd+1)].OD_abs 
                    - prof_oc[ilayer  +ph->ilam*(NOCEd+1)].OD_abs);
                // absorption between start and stop
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
                // update photon absorption tau
                ph->tau_abs = ab;
            }
            #else
            // ALIS specific computations
            ph->nevt++;
            ph->layer_prev[ph->nevt] = -ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            // cumulated scattering OD at reference wavelength
            //dsca_dl0 += get_OD(1,prof_oc[NOCEd + ph->ilam*(NOCEd+1)]) - 
             //   (epsilon * (get_OD(1,prof_oc[ilayer+ph->ilam*(NOCEd+1)]) - get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)])) +
              //  get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]));
            dsca_dl0 +=   get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)])
                        - epsilon * 
                         (get_OD(1,prof_oc[ilayer-1+ph->ilam*(NOCEd+1)]) 
                        - get_OD(1,prof_oc[ilayer  +ph->ilam*(NOCEd+1)])); 
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = - ph->tau_sca[k]; 
                float tautmp =   get_OD(1,prof_oc[ilayer-1+k*DL*(NOCEd+1)]) 
                               - epsilon * 
                                (get_OD(1,prof_oc[ilayer-1+k*DL*(NOCEd+1)]) 
                               - get_OD(1,prof_oc[ilayer  +k*DL*(NOCEd+1)])) ;
                dsca_dl += tautmp;
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) -fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = tautmp;
            }
            #endif

            // calculate new photon position
            phz =  prof_oc[ilayer-1].z + epsilon * ( prof_oc[ilayer].z - prof_oc[ilayer-1].z); 
            // move the photon to new position
            operator+=(ph->pos, ph->v * fabs( (ph->pos.z - phz) / ph->v.z));
        } // photon still in ocean
    } // Ocean



    if (ph->loc == ATMOS) {
        float rdist;
	    #ifdef OBJ3D
		// ========================================================================================================
		// Here geometry modification in the function move_pp
		// ========================================================================================================
		float timeT;                                 // the time from the parametric form of a ray
		bool mytest = false;                         // initiate the boolean of the intersection test
		float3 phit=make_float3(0.f, 0.f, 0.f);      // initiate the intersection point 


		// Launch the function geoTest to see if there are an intersection with the 
		// geometry, return true/false and give the position phit of the intersection
		// nObj = le nombre d'objets, si = 0 alors le test n'est pas nécessaire.
	    if (nObj > 0){
			mytest = geoTest(ph->pos, ph->v, &phit, geoS, myObjets, myGObj);
			if (!mytest && LMODEd == 1 &&  ph->pos.z >= (120.F-VALMIN5) && ph->direct == 0) {ph->loc=NONE; return;}
			if (mytest && phit.z > -VALMIN5 && phit.z < (120.F+VALMIN5) && IsAtm == 0)
			{
				ph->tau = 0.F;
				ph->loc = OBJSURF;
				ph->pos = phit;
				return;
			}
		}

		// if mytest = true (intersection with the geometry) and the position of the intersection is in
		// the atmosphere (0 < Z < 120), then: Begin to analyse is there is really an intersection
		if(mytest && phit.z > -VALMIN5 && phit.z < (120.F+VALMIN5) && IsAtm == 1)
		{
	        // if phit.z < 0 then correct the value to 0 (there is no object below the surface)
	        //if (phit.z < 0) phit.z =0;
			float tauHit = 0.f; // Optical depth distance (from the initial position of the photon to phit)
			int ilayer2 = ph->layer;
			if (ilayer2==0) {ilayer2=1;} // Be sure that we're not out of the atmosphere

			if((phit.z >= prof_atm[ilayer2].z) && (phit.z < prof_atm[ilayer2-1].z)) // 1 layer case: n = 1
			{
				// delta_i is: Delta(tau)1 = |tau(i-1) - tau(i)|
				delta_i = fabs(get_OD(BEERd, prof_atm[ilayer2+ph->ilam*(NATMd+1)]) - 
							   get_OD(BEERd, prof_atm[ilayer2-1+ph->ilam*(NATMd+1)]));
				// tauHit = (Delat(D1)/Delat(Z1))*delta_i
				tauHit += (length(ph->pos, phit)/fabs(prof_atm[ilayer2-1].z - prof_atm[ilayer2].z))*delta_i;
			}
			else // several layers case: n >= 2
			{
				// Find the layer where there is intersection
				ilayer2 = 1;
				while(prof_atm[ilayer2].z > phit.z && prof_atm[ilayer2].z > 0.F)
				{
					ilayer2 ++;
				} 

				float3 newP, oldP;
				bool higher = false;

				ilayer = ph->layer;          // initialise with the actual layer
				if (ilayer==0) {ilayer=1;}   // be sure that we're not out of the atmosphere
				oldP = ph->pos;                // initialise with the actual position

				// check if the photon come from higher or lower layers
				if(ilayer < ilayer2) // true if the photon come from higher layers
					higher =  true;

				while(ilayer != ilayer2)
				{
					if(higher){timeT = fabs(prof_atm[ilayer].z - oldP.z)/fabs(ph->v.z);}
					else{timeT = fabs(prof_atm[ilayer-1].z - oldP.z)/fabs(ph->v.z);}
					newP = oldP + timeT*ph->v;
					delta_i = fabs(get_OD(BEERd, prof_atm[ilayer+ph->ilam*(NATMd+1)]) - 
								   get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
					tauHit += (length(newP, oldP)/fabs(prof_atm[ilayer - 1].z - prof_atm[ilayer].z))*delta_i;
					
					// the photon come from higher layers
					if(higher){ilayer++;}
					// the photon come from lower layers
					else{ilayer--;}
					oldP = newP; //Update the position of the photon
				}

				// Calculate and add the last tau distance when ilayer is equal to ilayer2
				delta_i = fabs(get_OD(BEERd, prof_atm[ilayer2+ph->ilam*(NATMd+1)]) - 
							   get_OD(BEERd, prof_atm[ilayer2-1+ph->ilam*(NATMd+1)]));
				tauHit += (length(phit, oldP)/fabs(prof_atm[ilayer2 - 1].z - prof_atm[ilayer2].z))*delta_i;
			}


			// if tauHit (optical distance to hit the geometry) < tauR, then: there is interaction.
			if (tauHit < tauR)
			{
				ph->layer = ilayer2;
				if (BEERd == 0) ph->weight *= prof_atm[ph->layer+ph->ilam*(NATMd+1)].ssa;
				else
				{ // We compute the cumulated absorption OT at the new postion of the photon
					// see move photon paper eq 11
					tauBis =  get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)]) - (prev_tau + tauHit * ph->v.z);
					delta_i= fabs(get_OD(BEERd, prof_atm[ilayer2+ph->ilam*(NATMd+1)]) - get_OD(BEERd, prof_atm[ilayer2-1+ph->ilam*(NATMd+1)]));
					delta= fabs(tauBis - get_OD(BEERd, prof_atm[ilayer2-1+ph->ilam*(NATMd+1)])) ;
					epsilon = __fdividef(delta, delta_i);
					
					float ab = prof_atm[NATMd+ph->ilam*(NATMd+1)].OD_abs - 
						(epsilon * (prof_atm[ilayer2+ph->ilam*(NATMd+1)].OD_abs - prof_atm[ilayer2-1+ph->ilam*(NATMd+1)].OD_abs) +
						 prof_atm[ilayer2-1+ph->ilam*(NATMd+1)].OD_abs);
					// absorption between start and stop
					ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
					ph->tau_abs = ab;
					//ph->weight *= exp(-fabs(tauHit));
					//prof_atm[NATMd + ph->ilam *(NATMd+1)].OD_sca;
					
				}
				
				ph->loc = OBJSURF;                      // update of the loc of the photon 
				ph->tau = prev_tau + tauHit * ph->v.z;  // update the value of tau photon
				ph->pos = phit;                         // update the position of the photon
				return;
			}
		} // End of mytest = true
		
		// Case where atm is false in special case with objetcs
		// if there is not an intersect with an objet we have a special treatment
		if (nObj > 0 && !mytest && IsAtm == 0)
		{
			BBox boite(make_float3(-12000., -12000., 0.F), make_float3(12000.F, 12000.F, 120.F));
			Ray Rayon(ph->pos, ph->v, 0);
			float intTime0=-10.F, intTime1=-10.F;
			bool intersectBox;
			float3 intersectPoint = make_float3(-1.F, -1.F, -1.F);
			
			intersectBox = boite.IntersectP(Rayon, &intTime0, &intTime1);		 
			
			if (!intersectBox) {printf("error1 in move_pp geo!! \n"); return;}
			
			intersectPoint = Rayon(intTime1);
			
			if (intersectPoint.z >= (120-VALMIN))
			{			
				ph->loc = SPACE;
				ph->layer = 0;
				return;
			}
			else if (intersectPoint.z <= VALMIN)
			{
				ph->loc = SURF0P;
				ph->tau = 0.F;
				ph->tau_abs = 0.F;
				ph->pos.x = intersectPoint.x;
				ph->pos.y = intersectPoint.y;
				ph->pos.z = 0.F;
				ph->layer = NATMd;
				return;
			}
			else {ph->loc = ABSORBED;return;}
		}
		// ========================================================================================================
        #endif //END OBJ3D



		// partial geometrical thickness in the layer (from bottom) : 
        // epsilon = (z-z[i])/(z[i-1]-z[i]) ; 0<epsilon<1
        // optical thickness of the layer delta[i] = OD[i] - OD[i-1] delta[i]>0
        // partial optical thickness in the layer (from bottom) : epsilon * delta[i]
        // tau(z) = tau[i] + epsilon * delta[i]
        // tau[i] = OD[NATM] - OD[i]
        // **********************************************************
        // * tau(z) = OD[NATM] - OD[i] + epsilon * (OD[i] - OD[i-1])*
        // **********************************************************

        // If tau<0 photon is reaching the surface 
        if(ph->tau < 0.F){
            ph->layer = NATMd;
            epsilon = 0.F;
            #ifndef ALIS
            if (BEERd == 1) {// absorption between start and stop
                ab =  0.F;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += 0.F;
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = - ph->tau_sca[k]; 
                dsca_dl += 0.F;
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
            ph->pos.z = 0.F;

            #ifdef ALIS 
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            #endif
        return;
        }


        // If tau>TAUATM photon is reaching space
        else if( ph->tau > get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)]) ){
            ph->layer = 0;
            epsilon = 0.F;
            #ifndef ALIS
		    if (BEERd == 1) {// absorption between start and stop
                ab = prof_atm[NATMd + ph->ilam *(NATMd+1)].OD_abs;
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
            }
            #else
            dsca_dl0 += get_OD(1,prof_atm[NATMd + ph->ilam*(NATMd+1)]) ; 
            for (int k=0; k<NLOWd; k++) {
                dsca_dl = - ph->tau_sca[k]; 
                dsca_dl += get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]);
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) - fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = get_OD(1,prof_atm[NATMd + k*DL*(NATMd+1)]);
            }
            #endif

            ph->loc = SPACE;

            #ifdef ALIS
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            #endif

            return;
        }


        // Photon is still in the ATMOSPHERE
        else {
            tauBis =  get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)]) - ph->tau;
            ilayer = 1;
            
            while (( get_OD(BEERd, prof_atm[ilayer+ ph->ilam *(NATMd+1)]) < (tauBis)) && (ilayer < NATMd)) {
                ilayer++;
            }
            
            ph->layer = ilayer;

            delta_i= fabs(get_OD(BEERd, prof_atm[ilayer  +ph->ilam*(NATMd+1)]) 
                        - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
            //delta= fabs(tauBis - get_OD(BEERd, prof_atm[ilayer-1+ph->ilam*(NATMd+1)])) ;
            delta= ph->tau - (get_OD(BEERd, prof_atm[NATMd + ph->ilam *(NATMd+1)])
                            - get_OD(BEERd, prof_atm[ilayer + ph->ilam *(NATMd+1)]));
            epsilon = __fdividef(delta,delta_i);


            #ifndef ALIS
            if (BEERd == 0) ph->weight *= prof_atm[ph->layer+ph->ilam*(NATMd+1)].ssa;
            else { // We compute the cumulated absorption OT at the new postion of the photon
                // photon new position in the layer
                ab = prof_atm[NATMd+ ph->ilam*(NATMd+1)].OD_abs - 
                     prof_atm[ilayer+ph->ilam*(NATMd+1)].OD_abs +
                     epsilon * 
                     (prof_atm[ilayer  +ph->ilam*(NATMd+1)].OD_abs 
                    - prof_atm[ilayer-1+ph->ilam*(NATMd+1)].OD_abs) ;
                // absorption between start and stop
                ph->weight *= exp(-fabs(__fdividef(ab-ph->tau_abs, ph->v.z)));
                ph->tau_abs = ab;
            }

            #else
            ph->nevt++;
            ph->layer_prev[ph->nevt] = ph->layer;
            ph->vz_prev[ph->nevt] = ph->v.z;
            ph->epsilon_prev[ph->nevt] = epsilon;
            // cumulated scattering OD at reference wavelength
            //dsca_dl0 += get_OD(1,prof_atm[NATMd + ph->ilam*(NATMd+1)]) - 
            //   (epsilon * (get_OD(1,prof_atm[ilayer+ph->ilam*(NATMd+1)]) - get_OD(1,prof_atm[ilayer-1+ph->ilam*(NATMd+1)])) +
            //  get_OD(1,prof_atm[ilayer-1+ph->ilam*(NATMd+1)]));
            dsca_dl0 += get_OD(1,prof_atm[NATMd+ ph->ilam*(NATMd+1)]) - 
                        get_OD(1,prof_atm[ilayer+ph->ilam*(NATMd+1)]) +
                        epsilon * 
                        (get_OD(1,prof_atm[ilayer  +ph->ilam*(NATMd+1)]) 
                       - get_OD(1,prof_atm[ilayer-1+ph->ilam*(NATMd+1)])) ;
            for (int k=0; k<NLOWd; k++) {
            // cumulated scattering relative OD wrt reference wavelength
                dsca_dl  = - ph->tau_sca[k]; 
                float tautmp = get_OD(1,prof_atm[NATMd+ k*DL*(NATMd+1)]) - 
                               get_OD(1,prof_atm[ilayer+k*DL*(NATMd+1)]) +
                               epsilon * 
                               (get_OD(1,prof_atm[ilayer  +k*DL*(NATMd+1)]) 
                              - get_OD(1,prof_atm[ilayer-1+k*DL*(NATMd+1)])) ;
                dsca_dl += tautmp; 
                ph->weight_sca[k] *= exp(-__fdividef(fabs(dsca_dl) -fabs(dsca_dl0), fabs(ph->v.z)));
                ph->tau_sca[k] = tautmp;
            }
            #endif

            // calculate new photon position
            phz = epsilon * (prof_atm[ilayer].z - prof_atm[ilayer-1].z) + prof_atm[ilayer-1].z; 
            rdist=  fabs(__fdividef(phz-ph->pos.z, ph->v.z));
            operator+= (ph->pos, ph->v*rdist);
            ph->pos.z = phz;
        } // photon still in atmosphere

    } //ATMOS

}
#endif // move_pp Fast Move Mode




/*--------------------------------------------------------------------------------------------------*/
/*                      SCATTER                                                                     */
/*--------------------------------------------------------------------------------------------------*/
/*                      (ATMOS OR OCEAN)
/*--------------------------------------------------------------------------------------------------*/

__device__ void scatter(Photon* ph,
       struct Profile *prof_atm, struct Profile *prof_oc,
        #ifdef OPT3D
        struct Cell *cell_atm, struct Cell *cell_oc,
        #endif
        struct Phase *faer, struct Phase *foce,
        int le, float refrac_angle,
        float* tabthv, float* tabphi, int count_level,
        struct RNG_State *rngstate) {

	float cTh=0.f;
	float zang=0.f, theta=0.f;
	int iang, ilay, ipha;
	float psi, sign=1.F;
	struct Phase *func;
	float P11, P12, P22, P33, P43, P44;
	//int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
	#ifdef OBJ3D
	ph->direct += 1;
	ph->S += 1;
	#endif
    
    #if defined(THERMAL) && defined(BACK)
    // Absorption thermal in backward mode: end of photon s life    
    if (ph->scatterer == THERMAL_EM) {
	    if(ph->loc!=OCEAN){
            float tabs= fabs(prof_atm[ph->layer-1+ph->ilam*(NATMd+1)].OD_abs - prof_atm[ph->layer+ph->ilam*(NATMd+1)].OD_abs); 
            ph->weight= (2*DEUXPI) * tabs *  BPlanck(ph->wavel*1e-9, prof_atm[ph->layer].T); 
        }
	    else{
            float tabs= fabs(prof_oc[ph->layer+1+ph->ilam*(NOCEd+1)].OD_abs - prof_oc[ph->layer+ph->ilam*(NOCEd+1)].OD_abs); 
            ph->weight= (2*DEUXPI) * tabs *  BPlanck(ph->wavel*1e-9, prof_oc[ph->layer].T); 
        }
        ph->loc = SOURCE;
        return;
    }
    #endif

    if (le){
        /* in case of LE the photon units vectors, scattering angle and Psi rotation angles
         are determined by output zenith and azimuth angles*/
        float thv, phi;
        float3 v;

        if (count_level==DOWN0P || count_level==DOWNB) sign = -1.0F;
        phi = tabphi[ph->iph];
        thv = tabthv[ph->ith];
        v.x = cosf(phi) * sinf(thv);
        v.y = sinf(phi) * sinf(thv);
        v.z = sign * cosf(thv);
        if (refrac_angle != 0.F) {
          float3 no = normalize(ph->pos);
          float3 u  = normalize(cross(v, no));
          float3x3 R=rotation3D(refrac_angle, u);
          /* update the virtual photon direction to compensate for refraction*/
          v = mul(R, v);
        }
        theta = ComputeTheta(ph->v, v);
        cTh = __cosf(theta);
		if (cTh < -1.0) cTh = -1.0;
		if (cTh >  1.0) cTh =  1.0;
        ComputePsiLE(ph->u, ph->v, v, &psi, &ph->u); 
        ph->v = v;
    }

    /* Scattering in atmosphere */
	if(ph->loc!=OCEAN){

            #ifndef OPT3D
			ilay = ph->layer + ph->ilam*(NATMd+1); // atm layer index
            #else
			ilay = cell_atm[ph->layer].iopt + ph->ilam*(NATMd+1); // atm layer index
            #endif
			func = faer; // atm phases
			
			/************************************/
			/* Rayleigh or ptcle scattering */
			/************************************/
			if( ph->scatterer == RAY ){ipha  = 0;}   // Rayleigh index
			else if(ph->scatterer == PTCLE ){ipha  = prof_atm[ilay].iphase + 2;} // particle index
		
		}
	/* Scattering in ocean */
	else {

            #ifndef OPT3D
			ilay = ph->layer + ph->ilam*(NOCEd+1); // oce layer index
            #else
			ilay = cell_oc[ph->layer].iopt + ph->ilam*(NOCEd+1); // oce layer index
            #endif
			func = foce; // oce phases
			
			if (ph->scatterer == RAY){ipha  = 0;}	// Rayleigh index
			else if(ph->scatterer == VRS ){ ipha  = 1;} // VRS index
			else if(ph->scatterer == PTCLE ){ ipha  = prof_oc[ilay].iphase + 2;} // particle index

    }


	if ( (ph->scatterer == RAY) || (ph->scatterer == PTCLE) || (ph->scatterer == VRS)){

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
			P11 = (1-zang)*func[ipha*NF+iang].p_P11 + zang*func[ipha*NF+iang+1].p_P11;
			P12 = (1-zang)*func[ipha*NF+iang].p_P12 + zang*func[ipha*NF+iang+1].p_P12;
			P22 = (1-zang)*func[ipha*NF+iang].p_P22 + zang*func[ipha*NF+iang+1].p_P22;
			P33 = (1-zang)*func[ipha*NF+iang].p_P33 + zang*func[ipha*NF+iang+1].p_P33;
			P43 = (1-zang)*func[ipha*NF+iang].p_P43 + zang*func[ipha*NF+iang+1].p_P43;
			P44 = (1-zang)*func[ipha*NF+iang].p_P44 + zang*func[ipha*NF+iang+1].p_P44;

            #ifndef BIAS
			/////////////
			//  Get Psi
			//  Rejection method for sampling psi 
			float fpsi_cond=0.F; 
			float fpsi=0.F; 
			float gamma=0.F; 
			float Q = ph->stokes.x - ph->stokes.y;
			float U = ph->stokes.z;
			float DoLP = __fdividef(sqrtf(Q*Q+U*U), ph->stokes.x + ph->stokes.y);
			float K = __fdividef(P11-P22,P11+P22+2*P12);
			if (abs(Q) > 0.F) gamma   = 0.5F * atan2(-(double)U,(double)Q);
			float fpsi_cond_max = (1.F + DoLP * fabs(K) )/DEUXPI;
			int niter=0;
			while (fpsi >= fpsi_cond)
				{
					niter++;
					psi = RAND * DEUXPI;	
					fpsi= RAND * fpsi_cond_max;
					fpsi_cond = (1.F + DoLP * K * cosf(2*(psi-gamma)) )/DEUXPI;
					if (niter >= 100) {
						ph->loc = NONE;
						break;
					}
				}

            #else
			/////////////
			//  Get Phi
			//  Biased sampling scheme for psi 1)
			psi = RAND * DEUXPI;	
            #endif


		}else {
	
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
		rotationM(-psi,&L);
		ph->M   = mul(ph->M,mul(L,P_scatter));
		//float4x4 Lf;
		//rotationM(psi,&Lf);
		//ph->Mf  = mul(mul(P_scatter,Lf),ph->Mf);
        #endif

		if (!le){
			float debias = 1.F;
            #ifdef BIAS
			// Bias sampling scheme 2): Debiasing and normalizing
			debias = __fdividef(2.F, P11 + P22 + 2*P12 ); // Debias is equal to the inverse of the phase function
            #else
			debias = __fdividef(1.F, ph->stokes.x + ph->stokes.y);
            #endif

			operator*=(ph->stokes, debias); 
            #ifdef BACK
			ph->M  = mul(ph->M ,   make_diag_float4x4(debias)); // Bias sampling scheme only for backward mode
			//ph->Mf = mul(ph->Mf ,  make_diag_float4x4(debias));
            #endif
		}

		else {
			ph->weight /= 4.F; // Phase function normalization
		}

	}

	else if ((ph->scatterer == CHLFLUO) || (ph->scatterer == THERMAL_EM)){ 

		/////////////////
		// Fluorescence

		if (!le){

			// isotropic point source
			// see Leathers, R. A.; Downes, T. V.; Davis, C. O. & Davis, C. D. Monte Carlo Radiative Transfer Simulations for Ocean Optics: A Practical Guide Naval Research Laboratory, 2004, section 5.1.3
			float phi;
			float sTh;
			cTh = 1.0-2.0*RAND;
			phi = RAND*DEUXPI;
			sTh = sqrtf(1.F - cTh*cTh);
			ph->v.x   = cosf(phi)*sTh;
			ph->v.y   = sinf(phi)*sTh;
			ph->v.z   = cTh;
			// Initialization of the orthogonal vector to the propagation
			ph->u.x   = cosf(phi)*cTh;
			ph->u.y   = sinf(phi)*cTh;
			ph->u.z   = -sTh;
			
		}else{
			ph->weight /= 4.0  ;    // Phase function normalization	
		}


		// Depolarisation du Photon
		float4x4 L = make_float4x4(
								   0.5F, 0.5F, 0.F, 0.F,
								   0.5F, 0.5F, 0.F, 0.F,
								   0.0F, 0.0F, 0.F, 0.F,
								   0.0F, 0.0F, 0.F, 0.F 
								   );
		ph->stokes = mul(L,ph->stokes);
	}

    #ifdef ALIS
	if ( (ph->scatterer == RAY) or (ph->scatterer == PTCLE) or (ph->scatterer == VRS)){
        Profile *prof;                                             
        #ifdef OPT3D
        Cell *cell;
        #endif
        int layer_end;                                             

        if(ph->loc == ATMOS){                                      
             layer_end = NATMd;                                     
             prof = prof_atm;                                       
             #ifdef OPT3D
             cell = cell_atm;
             #endif
        }                                                          
        if(ph->loc == OCEAN){                                      
             layer_end = NOCEd;                                     
             prof = prof_oc;                                        
             #ifdef OPT3D
             cell = cell_oc;
             #endif
        } 

		int DL=(NLAMd-1)/(NLOWd-1);
		float P11_aer_ref, P11_ray, P22_aer_ref, P22_ray, P12_aer_ref, P12_ray, P_ref;
        #ifndef OPT3D
		float pmol= prof[ph->layer+ ph->ilam*(layer_end+1)].pmol;
        #else
		float pmol= prof[cell[ph->layer].iopt+ ph->ilam*(layer_end+1)].pmol;
        #endif
       
		if (pmol < 1.) {
			zang = theta * (NF-1)/PI ;
			iang = __float2int_rd(zang);
			zang = zang - iang;
            #ifndef OPT3D
			int ipharef = prof[ph->layer+ph->ilam*(layer_end+1)].iphase + 2; 
            #else
			int ipharef = prof[cell[ph->layer].iopt+ph->ilam*(layer_end+1)].iphase + 2; 
            #endif
			// Phase functions of particles and molecules, and mixture of both at reference wavelength
			P11_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P11 + zang*func[ipharef*NF+iang+1].a_P11;
			P11_ray     = (1-zang)*func[0      *NF+iang].a_P11 + zang*func[0      *NF+iang+1].a_P11;
			P22_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P22 + zang*func[ipharef*NF+iang+1].a_P22;
			P22_ray     = (1-zang)*func[0      *NF+iang].a_P22 + zang*func[0      *NF+iang+1].a_P22;
			P12_aer_ref = (1-zang)*func[ipharef*NF+iang].a_P12 + zang*func[ipharef*NF+iang+1].a_P12;
			P12_ray     = (1-zang)*func[0      *NF+iang].a_P12 + zang*func[0      *NF+iang+1].a_P12;
			P_ref       = (P11_ray+P22_ray+2.F*P12_ray) * pmol + (P11_aer_ref+P22_aer_ref+2.F*P12_aer_ref) * (1.-pmol);
		}

		for (int k=0; k<NLOWd; k++) {
            #ifndef OPT3D
			ph->weight_sca[k] *= __fdividef(get_OD(1,prof[ph->layer   + k*DL*(layer_end+1)]) - 
                                            get_OD(1,prof[ph->layer-1 + k*DL*(layer_end+1)]) , 
											get_OD(1,prof[ph->layer   + ph->ilam*(layer_end+1)]) - 
                                            get_OD(1,prof[ph->layer-1 + ph->ilam*(layer_end+1)]));
            #else
			ph->weight_sca[k] *= __fdividef(get_OD(1,prof[cell[ph->layer].iopt   + k*DL*(layer_end+1)]) , 
                                            get_OD(1,prof[cell[ph->layer].iopt   + ph->ilam*(layer_end+1)]));
            #endif
			if (pmol < 1.) {
                #ifndef OPT3D
				int iphak    = prof[ph->layer + k*DL*(layer_end+1)].iphase + 2; 
				float pmol_k = prof[ph->layer + k*DL*(layer_end+1)].pmol;
                #else
				int iphak    = prof[cell[ph->layer].iopt + k*DL*(layer_end+1)].iphase + 2; 
				float pmol_k = prof[cell[ph->layer].iopt + k*DL*(layer_end+1)].pmol;
                #endif
				// Phase functions of particles  at other wavelengths, molecular is supposed to be constant with wavelength
				float P11_aer = (1-zang)*func[iphak*NF+iang].a_P11 + zang*func[iphak*NF+iang+1].a_P11;
				float P22_aer = (1-zang)*func[iphak*NF+iang].a_P22 + zang*func[iphak*NF+iang+1].a_P22;
				float P12_aer = (1-zang)*func[iphak*NF+iang].a_P12 + zang*func[iphak*NF+iang+1].a_P12;
				// Phase functions of the mixture of particles and molecules at other wavelengths
				float P_k   = (P11_ray+P22_ray+2.F*P12_ray) * pmol_k + (P11_aer+P22_aer+2.F*P12_aer) * (1.-pmol_k);
				ph->weight_sca[k] *= __fdividef(P_k, P_ref);
			}
		}
	}

    #endif

	if (!le){
		if (RRd==1){
			/** Russian roulette for propagating photons **/
			if( ph->weight < WEIGHTRRd ){
				if( RAND < __fdividef(ph->weight,WEIGHTRRd) ){ph->weight = WEIGHTRRd;}
				else{ph->loc = ABSORBED;}
			}
		}
		if ((ph->scatterer != CHLFLUO) && (ph->scatterer != THERMAL_EM)) { modifyUV( ph->v, ph->u, cTh, psi, &ph->v, &ph->u) ;}
	}
    else {
        if (HORIZd) ph->weight /= fabs(ph->v.z); 
    }

	if(!le) ph->scatterer = UNDEF;
	
}

#ifdef SIF
__device__ void choose_emitter(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
		struct Spectrum *spectrum,
        struct RNG_State *rngstate) {

  
	float pfluo;
    float fAPAR=0.5;
    float QY=0.1; //Fluoresecnce yield
    float fluo_p=1.F; // Fluo power normalization
	
	if(ph->loc==SURF0P){
		/* SIF probability */
		if ( (fAPAR*QY) > RAND && (ph->nsif==0)){
			ph->emitter = SIF_EM; // SIF index
            ph->nsif = 1;
		    /* Compute fluorescence power*/
            // PAR
            float PAR=0.F;
            int il=0;
            while ( spectrum[il].lambda <= 700.){
                if (spectrum[il].lambda >= 400.) PAR += ph->weight_sca[il];
                il++;
            }
	        ph->weight *= PAR * fluo_p;
            // reinitialization
            ph->nint = 0;
            ph->nenv = 0;
            ph->nref = 0;
            ph->nrrs = 0;
            ph->nvrs = 0;
            for (int k=0; k<NLOWd; k++) ph->weight_sca[k] = 1.F;
            for (int k=0; k<NATM_ABSd+1; k++) ph->cdist_atm[k] = 0.F;
            //for (int k=0; k<NATMd+1; k++) ph->cdist_atm[k] = 0.F;
            //for (int k=0; k<NOCEd+1; k++) ph->cdist_oc[k] = 0.F;
            for (int k=0; k<NOCE_ABSd+1; k++) ph->cdist_oc[k] = 0.F;
		} else {
			ph->emitter = SOLAR_REF; // SOLAR reflection index
            if (ph->env) ph->nenv +=1;
            else ph->nref += 1;
            ph->nint += 1;
		}	
	}
}
#endif



__device__ void choose_scatterer(Photon* ph,
        struct Profile *prof_atm, struct Profile *prof_oc,
        #ifdef OPT3D
        struct Cell *cell_atm, struct Cell *cell_oc,
        #endif
		struct Spectrum *spectrum,
        struct RNG_State *rngstate) {

	//int idx = (blockIdx.x * YGRIDd + blockIdx.y) * XBLOCKd * YBLOCKd + (threadIdx.x * YBLOCKd + threadIdx.y);
    
    #ifndef THERMAL
    ph->nint += 1;
    #else
    // in forward thermal mode, A thermal emission is not counted as interaction and choose_scatterer does not do anything
    // #ifndef BACK
    if (ph->scatterer!=THERMAL_EM) ph->nint +=1;
    else return;
    #endif
    //#endif
  
	float pmol;
	float pine;
	
	if(ph->loc!=OCEAN){
        /* Absorption in atmosphere */
        #if defined(THERMAL) && defined(BACK)
        float ssa = prof_atm[ph->layer+ph->ilam*(NATMd+1)].ssa;
		if ( ssa < RAND ) {
            ph->scatterer = THERMAL_EM;
            ph->nint -=1;
            return;
        }
        #endif

		/* Scattering in atmosphere */
        #ifndef OPT3D
		pmol = 1.f - prof_atm[ph->layer+ph->ilam*(NATMd+1)].pmol;
        #else
		pmol = 1.f - prof_atm[cell_atm[ph->layer].iopt+ph->ilam*(NATMd+1)].pmol;
        #endif
		/* Elastic scattering    */
		if ( pmol < RAND ){
			ph->scatterer = RAY; // Rayleigh index
            if (RAND >fRRS_air(ph->wavel, 50.F)) ph->nrrs +=1; //  RRS probability, averaged directional
		} else {
			ph->scatterer = PTCLE;	; // particle index
		}	
	}else{
        /* Absorption in ocean */
        #if defined(THERMAL) && defined(BACK)
        float ssa = prof_oc[ph->layer+ph->ilam*(NOCEd+1)].ssa;
		if ( ssa < RAND ) ph->scatterer = THERMAL_EM;
        #endif

		/* Scattering in ocean */
        #ifndef OPT3D
		pmol = 1.f - prof_oc[ph->layer+ph->ilam*(NOCEd+1)].pmol;
		pine = 1.f - prof_oc[ph->layer+ph->ilam*(NOCEd+1)].pine;
        #else
		pmol = 1.f - prof_oc[cell_oc[ph->layer].iopt+ph->ilam*(NOCEd+1)].pmol;
		pine = 1.f - prof_oc[cell_oc[ph->layer].iopt+ph->ilam*(NOCEd+1)].pine;
        #endif

		if (pine  < RAND){
			/* Fluorescence    */
            ph->scatterer =CHLFLUO ;
		}else{
			/* Molecular scattering    */
			if ( pmol < RAND ){
				ph->scatterer = RAY; // Rayleigh index
                if (RAND >fVRS(ph->wavel)) { //  VRS probability
			        //ph->scatterer = VRS ;
                    ph->nvrs +=1;
                }
			} else {
				ph->scatterer = PTCLE;	; // particle index
			}
		}

	}
	

	if (ph->scatterer == CHLFLUO){

		////////////////////
		// Chl Fluorescence
		
		/* Wavelength change */
		float sigmac   = 10.6;
		float lambdac0 = 685.0; 
		float new_wavel;
		int new_ilam;
		float rand1 = RAND;
		float rand2 = RAND;
		new_wavel = lambdac0 + sigmac * sqrtf(-2.0*logf(RAND)) * cosf(DEUXPI * rand2);

		ph->weight /= ph->wavel / new_wavel;
		ph->wavel = new_wavel;

		if (ph->wavel > spectrum[NLAMd-1].lambda ){

			ph->weight = 0.0;
			ph->loc = ABSORBED;

		}else if (ph->wavel < spectrum[0].lambda ) {

			ph->weight = 0.0;
			ph->loc = ABSORBED;

		} else {
			
			// get new lamb index
			//new_ilam = __float2int_rd(__fdividef( (ph->wavel -  spectrum[0].lambda)* NLAMd, spectrum[NLAMd-1].lambda - spectrum[0].lambda ));
            new_ilam=0;
            while ((ph->wavel>spectrum[new_ilam].lambda) && (new_ilam<NLAMd)) new_ilam++;
			// update tau photon coordinates according to its new wavelength
			ph->tau_abs = ph->tau_abs * prof_oc[ph->layer + new_ilam *(NOCEd+1)].OD_abs / prof_oc[ph->layer + ph->ilam *(NOCEd+1)].OD_abs;				
			ph->tau     = ph->tau     * get_OD(BEERd, prof_oc[ph->layer + new_ilam *(NOCEd+1)]) /  get_OD(BEERd, prof_oc[ph->layer + ph->ilam *(NOCEd+1)]);
			ph->ilam = new_ilam;
		}

	}
	
}


__device__ void surfaceWaterRough(Photon* ph, int le,
                              float* tabthv, float* tabphi, int count_level,
                              struct RNG_State *rngstate) {

	if( SIMd == ATM_ONLY){ // Atmosphere only, surface absorbs all. 
		ph->loc = ABSORBED;
		return;
	}
    ph->nint += 1;

    #ifdef OBJ3D
	ph->E += 1;
	#endif

	// Reflection rough dioptre
	float theta;	// reflection polar angle[rad]
	float psi;		// reflection azimuth angle [rad]
	float cTh, sTh;	// cos and sin of photon incident angle on the dioptre
	
	float sig, sig2  ;
	float beta = 0.F;	// polar angle of the facet normal 
	float sBeta;
	float cBeta;
	
	float alpha ;	// azimuth angle of the facet normal
	
	float nind; // relative index of refraction 
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
	float rat  ;	    // Reflection coefficient for unpolarized light
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

    /* Compute the photon v vector in the local frame */
    v_l.x = dot(ph->v,Nx);
    v_l.y = dot(ph->v,Ny);
    v_l.z = dot(ph->v,Nz);

    #else
    v_l = ph->v;
    #endif

    if (ph->loc==SURF0M) v_l = ph->v;

	/** **/
    //  Estimation of the probability P of interaction of the photon with zentih angle theta with a facet of slope beta and azimut alpha	
    //  P_alpha_beta : Probability of occurence of a given azimuth and slope
    //  P_alpha_beta = P_Cox_Munk(beta) * P(alpha | beta), conditional probability, for normal incidence, independent variables and P(alpha|beta)=P(alpha)=1/2pi
    //  following Plass75:
    //  Pfacet : Probability of occurence of a facet
    //  Pfacet = projected area of the facet divided by unit area of the possible interaction surface * P_alpha_beta
    //  Pfacet = P_alpha_beta / cos(beta)
    //  for non normal incident angle, the probability of interaction between the photon and the facet is proportional to the surface of the facet seen by the photon so
    //  that is cosine of incident angle of photon on the facet theta_inc=f(alpha,beta,theta)
    //  P # Pfacet * cos(theta_inc) for cos(theta_inc) >0
    //  P = 0 for cos(theta_inc)<=0
    //  for having a true probability, one has to normalize this to 1. The A normalization factor depends on theta and is the sum on all alpha and beta with the condition
    //  cos(theta_inc)>0 (visible facet)
    //  A = Sum_0_2pi Sumr_0_pi/2 P_alpha_beta /cos(beta) cos(theta_inc) dalpha dbeta
    //  Finally P = 1/A * P_alpha_beta  /cos(beta) cos(theta_inc)


    sig2 = 0.003F + 0.00512f *WINDSPEEDd;
    sig  = sqrtf(sig2);

    /* SAMPLING */

    if (!le) {
	 if( DIOPTREd !=0 ){
        // Rough surface

        theta = DEMIPI;
        //  Computation of P_alpha_beta = P_Cox_Munk(beta) * P(alpha | beta)
        //  we draw beta first according to Cox_Munk isotropic and then draw alpha, conditional probability
        //  rejection method: to exclude unphysical azimuth (leading to incident angle theta >=PI/2)
        //  we continue until acceptable value for alpha


        while (theta >= DEMIPI) {
           iter++;
           if (iter >= 100) {
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
        } // while

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

	// temp = dot(cross(ph->v,ph->u),normalize(cross(ph->v,no)));
    // Simplification :
	temp = __fdividef(dot(no, ph->u), sTh);
	psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

	if( dot(no, cross(ph->u, ph->v)) <0 ){
		psi = -psi;
	}

    rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L = make_diag_float4x4 (1.F);
    rotationM(-psi,&L);
    float tpar_b, tper_b, tpar2_b, tper2_b, tparper_b;
    float rpar2_b, rper2_b, rat_b;
    float cTh_b;
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
        #ifdef BIAS
		rat =  __fdividef(rpar2 + rper2, 2.F);
        #else
        rat =  __fdividef(ph->stokes.x*rpar2 + ph->stokes.y*rper2, ph->stokes.x+ph->stokes.y);
        #endif
		ReflTot = 0;
        #ifdef BACK
        // in backward mode, nind -> 1/nind and incidence angle <-> emergence angle
        cTh_b = cot;
        float cot_b = cTh;
        float nind_b= 1.F/nind;
        float ncTh_b  = nind_b * cTh_b;
        float ncot_b  = nind_b * cot_b;
		rpar2_b = __fdividef(ncTh_b - cot_b,ncTh_b  + cot_b);
        rpar2_b *= rpar2_b;
		rper2_b = __fdividef(cTh_b - ncot_b,cTh_b + ncot_b);
        rper2_b *= rper2_b;
        rat_b = __fdividef(rpar2_b + rper2_b, 2.F);
		tpar_b = __fdividef( 2.F*cTh_b,ncTh_b+ cot_b);
		tper_b = __fdividef( 2.F*cTh_b,cTh_b+ ncot_b);
        tpar2_b= tpar_b * tpar_b;
        tper2_b= tper_b * tper_b;
        tparper_b = tpar_b * tper_b;
        #endif
	}
	else{
		cot = 0.f;
		rpar = 1.f;
		rper = 1.f;
        rat = 1.f;
		rpar2 = rpar*rpar;
		rper2 = rper*rper;
        rparper = __fdividef(2.*sTh*sTh*sTh*sTh, 1.-(1.+nind * nind)*cTh*cTh) - 1.; //  Mobley 2015
        rparper_cross = -__fdividef(2.*cTh*sTh*sTh*sqrtf(sTh*sTh-nind*nind), 1.-(1.+nind * nind)*cTh*cTh); //  Mobley 2015
        tpar = 0.;
        tper = 0.;
        tpar2 =0.;
        tper2 =0.;
        tparper =0.;
		ReflTot = 1;
	}

    // Weighting
    float p,qv,LambdaS,LambdaR,jac;

    // Lambda shadowing Source direction
    LambdaS  =  LambdaM(avz,sig2*0.5);

    //
    // Local Estimate part
    //
    if (le && (DIOPTREd!=0)) {
     // The weight depends on the normalized VISIBLE interaction PDF qv (Ross 2005) 
     // Compute p 
     float cBeta2 = cBeta*cBeta;
     p =  __fdividef( __expf(-(1.F-cBeta2)/(cBeta2*sig2)) , cBeta2*cBeta * sig2); 

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
                #ifndef BACK
                jac = __fdividef(nind*nind * cot, (ncot - cTh)*(ncot - cTh)); // See Zhai et al., 2010
                #else
                jac = __fdividef(cTh, nind*nind * (cTh/nind - cot)*(cTh/nind - cot));
                #endif
            }
            else qv = 0.F;
     }

     // Reflected/Refracted direction, Normalization of qv
     LambdaR  =  LambdaM(fabs(v.z),sig2*0.5);

     float norma;
     if (WAVE_SHADOWd) norma = 1. + LambdaS + LambdaR;
     else norma = 1. + LambdaS;
     qv /= norma;

     // apply the BRDF to the weight
     ph->weight *= __fdividef(qv * jac , avz);

    } /*le */

    //
    // Propagation part
    //

    // 1. Reflection
    int condR=1;
    if (!le) condR = (SURd==3)&&(RAND<rat);
	if (  (!le && (condR || (SURd==1) || ReflTot) )
       || ( le && (ph->loc==SURF0M) && (count_level == DOWN0M) )
       || ( le && (ph->loc==SURF0P) && (count_level == UP0P)   )
       ){	// Reflection

	    R= make_float4x4(
		    rpar2, 0., 0., 0.,
		    0., rper2, 0., 0.,
		    0., 0.,  rparper, rparper_cross,
		    0., 0., -rparper_cross, rparper
		    );

        ph->stokes = mul(R,ph->stokes);
        #ifndef BIAS
        ph->weight *= ph->stokes.x + ph->stokes.y;
		operator/=(ph->stokes, ph->stokes.x + ph->stokes.y); 
        #endif

        #ifdef BACK
        ph->M   = mul(ph->M,mul(L,R));
        #endif
		
        if (le) { ph->v = v; }
        else { operator+=(ph->v, (2.F*cTh)*no); }

		ph->u = operator/(operator-(no, cTh*ph->v), sTh);	

        //  Normalization of the reflexion matrix
        //  the reflection coefficient is taken into account:
        //  once in the random selection (Rand < rat)
        //  once in the reflection matrix multiplication
        //  so twice and thus we normalize by rat (Xun 2014).
        //  not to be applied for forced reflection (SUR=1 or total reflection) where there is no random selection
		if (SURd==3 && !ReflTot && !le) {
            ph->weight /=rat;
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
                if( SIMd==SURF_ONLY || SIMd==OCEAN_SURF ){
                    ph->loc = SPACE;
                } else{
                    ph->loc = ATMOS;
                    ph->layer = NATMd;
                }
            } // else, no change of location
            else if (SINGLEd) ph->loc = REMOVED;
        } else {
            if (vzn < 0) {  // avoid multiple reflexion under the surface
               // SURF0M becomes OCEAN or ABSORBED
               if( SIMd==SURF_ATM ){
                  ph->loc = ABSORBED;
               } else{
                  ph->loc = OCEAN;
                  ph->layer = 0;
               }
            } // else, no change of location
            else if (SINGLEd) ph->loc = REMOVED;
        }

		#ifdef OBJ3D
		if (ph->loc == OBJSURF)
		{
			if (vzn > 0) {ph->loc = ATMOS;}
			else {ph->loc = OCEAN;}
		}
		#endif

     } // Reflection


    // 2. Transmission
    else if (  (!le && !condR) 
        || ( le && (ph->loc==SURF0M) && (count_level == UP0P  ) && !ReflTot )
        || ( le && (ph->loc==SURF0P) && (count_level == DOWN0M) && !ReflTot )
        ){	// Transmission

        T= make_float4x4(
        tpar2, 0., 0., 0.,
        0., tper2, 0., 0.,
        0., 0., tparper, 0.,
        0., 0., 0., tparper
        );
        ph->stokes = mul(T,ph->stokes);

        #ifndef BACK
        float geo_trans_factor = nind* cot/cTh; // DR Mobley 2015 OK , see Xun 2014, Zhai et al 2010
        #else
        float geo_trans_factor = 1./nind* cTh/cot;
        #endif
        ph->weight *= geo_trans_factor;

        #ifndef BIAS
        ph->weight *= ph->stokes.x + ph->stokes.y;
	    operator/=(ph->stokes, ph->stokes.x + ph->stokes.y); 
        #endif

        #ifdef BACK
        /* for reciprocity of transmission function see Walter 2007 */
        float4x4 T_b= make_float4x4(
        tpar2_b, 0., 0., 0.,
        0., tper2_b, 0., 0.,
        0., 0., tparper_b, 0.,
        0., 0., 0., tparper_b
        );
        ph->M   = mul(ph->M,mul(L,T_b));
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


        // Normalization of the transmission matrix
        // the transmission coefficient is taken into account:
        // once in the random selection (Rand > rat)
        // once in the transmission matrix multiplication
        // so we normalize by (1-rat) (Xun 2014).
        // Not to be applied for forced transmission (SUR=2)
        if ( (SURd == 3 ) && !le) 
            #ifndef BACK
            ph->weight /= (1-rat);
            #else
            ph->weight /= (1-rat_b);
            #endif

        //
        // photon next location
        //
        if (ph->loc == SURF0M) {
         if (vzn > 0) {
            // SURF0P becomes ATM or SPACE
            if( SIMd==SURF_ONLY || SIMd==OCEAN_SURF ){
                ph->loc = SPACE;
            } else{
                ph->loc = ATMOS;
                ph->layer = NATMd;
            }
         } else {
            // multiple transmissions (vz<0 after water->air transmission)
            ph->loc = SURF0P;
            if (SINGLEd) ph->loc = REMOVED;
         }
        } else {
           if (vzn < 0) {  // avoid multiple reflexion under the surface
              // SURF0M becomes OCEAN or ABSORBED
              if( SIMd==SURF_ONLY || SIMd==SURF_ATM ){
                ph->loc = ABSORBED;
              } else{
                ph->loc = OCEAN;
                ph->layer = 0;
              }
           } else {
              // multiple transmissions (vz<0 after water->air transmission)
              // (for symmetry, but should not happen)
              ph->loc = SURF0M;
              if (SINGLEd) ph->loc = REMOVED;
           }
        }
	} // Transmission

    LambdaR  =  LambdaM(fabs(ph->v.z),sig2*0.5);

    if (!le) {
        if (WAVE_SHADOWd) ph->weight *= __fdividef(fabs(cTh), cBeta * (1.F + LambdaS + LambdaR) * avz );
        else              ph->weight *= __fdividef(fabs(cTh), cBeta * (1.F + LambdaS) * avz );
        // Ross et al 2005, Ross and Dion, 2007, Zeisse 1995
        // Slope sampling bias correction using the normalized interaction PDF q
        // weight has to be multiplied by q/p, where p is the slope PDF
        // Coefficient Lambda for normalization of q taking into acount slope shadowing and hiding
        // Including wave shadows is performed at the end after the outgoing direction is calculated

		if (RRd==1){
			/* Russian roulette for propagating photons **/
			if( ph->weight < WEIGHTRRd ){
				if( RAND < __fdividef(ph->weight,WEIGHTRRd) ){ph->weight = WEIGHTRRd;}
				else{ph->loc = ABSORBED;}
			}
		}
    }

}

/* Fresnel Reflection Matrix*/
__device__ float4x4 FresnelR(float3 vi, float3 vr) {

    float cTh, sTh, cot, ncTh, ncot, theta, temp;
    float rpa, rpe, rpape, rpa2, rpe2, rpape_c;
    float4x4 R;
    // Determination of the relative refractive index
    // a: air, b: water , Mobley 2015 nind = nba = nb/na
    // and sign for further computation
    float nind = NH2Od;
    // vector equation for determining the half direction h = sign(i dot o) (i + o)
	float3 no = operator-(vr, vi);
    // Normalization of the half direction vector
    no=normalize(no);
    // Incidence angle in the local frame
    cTh   = fabs(dot(no, vi));
    theta = acosf(fmin(1.F-VALMIN, fmax(-(1.F-VALMIN), cTh)));
    sTh   = sinf(theta);
    // Fresnel coefficients
	temp    = __fdividef(sTh, nind);
	cot     = sqrtf(1.0F - temp*temp);
	ncTh    = nind*cTh;
	ncot    = nind*cot;
	rpa     = __fdividef(ncTh - cot, ncTh  + cot); // DR Mobley 2015 sign convention
	rpe     = __fdividef(cTh - ncot, cTh + ncot);
	rpa2    = rpa*rpa;
	rpe2    = rpe*rpe;
    rpape   = rpa*rpe;
    rpape_c = 0.F;

	R   = make_float4x4(
	          rpa2, 0.  , 0.     , 0.,
	          0.  , rpe2, 0.     , 0.,
	          0.  , 0.  , rpape  , rpape_c,
	          0.  , 0.  ,-rpape_c, rpape
	          );

    return R;
}

/* Surface BRDF */
__device__ void surfaceBRDF(Photon* ph, int le,
                              float* tabthv, float* tabphi, int count_level,
                              struct RNG_State *rngstate) {
	
	if( SIMd == ATM_ONLY){ // Atmosphere only, surface absorbs all
		ph->loc = ABSORBED;
		return;
	}
    ph->nint += 1;

    #ifdef OBJ3D
	ph->E += 1;
    #endif

	// Réflexion sur le dioptre agité
	float theta;	// Angle de deflection polaire de diffusion [rad]
	float psi;		// Angle azimutal de diffusion [rad]
	float cTh, sTh;	//cos et sin de l'angle d'incidence du photon sur le dioptre
	
	float sig2;
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
    #endif

    sig2 = 0.003F + 0.00512f *WINDSPEEDd;

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
    cTh = fabs(dot(no, ph->v));
    theta = acosf(fmin(1.F-VALMIN, fmax(-(1.F-VALMIN), cTh)));

    #ifdef SPHERIQUE
    // facet slope
    cBeta = fabs(dot(no, Nz));
    #else
    cBeta = fabs(no.z);
    #endif


    #ifdef SPHERIQUE
    // avz is the projection of V on the local vertical
	float avz = fabs(dot(Nz, ph->v));
    #else
    float avz = fabs(ph->v.z);
    #endif

	// Rotation of Stokes parameters
	sTh  = __sinf(theta);
    temp = __fdividef(dot(no, ph->u), sTh);
	psi  = acosf(fmin(1.F, fmax(-1.F, temp)));	

	if( dot(no, cross(ph->u, ph->v)) <0 ){
		psi = -psi;
	}

    rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L = make_diag_float4x4(1.F);
    rotationM(-psi,&L);
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

    // BRDF Weighting
    float cBeta2 = cBeta*cBeta;
    ph->weight *=  __fdividef( __expf(-(1.F-cBeta2)/(cBeta2*sig2)), 4.F * cBeta2*cBeta2 * avz * fabs(v.z) * sig2);

	R= make_float4x4(
	   rpar2, 0., 0., 0.,
	   0., rper2, 0., 0.,
	   0., 0., rparper, rparper_cross,
	   0., 0., -rparper_cross, rparper
	   );

    ph->stokes = mul(R,ph->stokes);
    #ifndef BIAS
    ph->weight *= ph->stokes.x + ph->stokes.y;
	operator/=(ph->stokes, ph->stokes.x + ph->stokes.y); 
    #endif

    #ifdef BACK
    ph->M   = mul(ph->M,mul(L,R));
    #endif
		
    ph->v = v;
	ph->u = operator/(operator-(no, cTh*ph->v), sTh);	

        // photon next location
    if( SIMd==SURF_ONLY || SIMd==OCEAN_SURF ){
        ph->loc = SPACE;
    } else {
          ph->loc = ATMOS;
          ph->layer = NATMd;
    }

    if (WAVE_SHADOWd) {
        // Add Wave shadowing
        // compute wave shadow outgoing photon
        float LambdaR, LambdaS;
        LambdaS  =  LambdaM(avz,sig2*0.5);
        LambdaR  =  LambdaM(fabs(v.z),sig2*0.5);
        ph->weight *= __fdividef(1.F, 1.F + LambdaR + LambdaS);
    }

    if (!le) {
		if (RRd==1){
			// Russian roulette for propagating photons 
			if( ph->weight < WEIGHTRRd ){
				if( RAND < __fdividef(ph->weight,WEIGHTRRd) ){ph->weight = WEIGHTRRd;}
				else{ph->loc = ABSORBED;}
			}
		}
    }

}

/* Surface Lambert */
__device__ void surfaceLambert(Photon* ph, int le,
                              float* tabthv, float* tabphi, struct Spectrum *spectrum,
                              struct RNG_State *rngstate) {
	
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if( SIMd == ATM_ONLY){ // Atmosphere only, surface absorbs all
		ph->loc = ABSORBED;
		return;
	}

	#ifdef OBJ3D
	if(IsAtm == 0 && LMODEd == 1){ph->loc = NONE; return;}
	ph->direct += 1;
	ph->E += 1;
	#endif
  	
    float thv, phi;
	float3 v_n, u_n; // photon outgoing direction in the LOCAL frame
    float3 v0=ph->v;

    #ifdef SPHERIQUE
    // define 3 vectors Nx, Ny and Nz in cartesian coordinates which define a
    // local orthonormal basis at the impact point.
    // Nz is the local vertical direction, the direction of the 2 others does not matter
    // because the azimuth is chosen randomly
	float3 Nx, Ny, Nz;
    MakeLocalFrame(ph->pos, &Nx, &Ny, &Nz);
    #endif

    /***************************************************/
    /* Computation of outgoing direction */
    /***************************************************/
    if (le) {
     // Outgoing direction in GLOBAL frame
     phi = tabphi[ph->iph];
     thv = tabthv[ph->ith];
     DirectionToUV(thv, phi, &ph->v, &ph->u);

     #ifdef SPHERIQUE
     float weight = dot(ph->v, Nz);
     if ((weight <= 0.) && (ph->loc != SEAFLOOR)) { /*[Eq. 40]*/
         ph->loc = ABSORBED;
         return;
     }
     else {
         if (HORIZd) ph->weight *= weight/fabs(ph->v.z); /*[Eq. 39]*/
         else ph->weight *= weight;
     }
     #endif
      
    }
    else {
     // Cosine of the LOCAL zenith angle sampling for Lambertian reflector
	 phi = RAND*DEUXPI;
	 thv = acosf(sqrtf(RAND));
     DirectionToUV(thv, phi, &v_n, &u_n);

     // Computation of the outgoing direction in GLOBAL frame
     #ifdef SPHERIQUE
     if (ph->loc == SEAFLOOR) {
         ph->v = v_n;
         ph->u = u_n;
     }
     else {
         /* LOCAL to GLOBAL frame */
         ph->v = LocalToGlobal(Nx, Ny, Nz, v_n);
         ph->u = LocalToGlobal(Nx, Ny, Nz, u_n);
     }
     #else
     ph->v = v_n;
     ph->u = u_n;
     #endif
    }

    /***************************************************/
    /* Update of Stokes vector  */
    /***************************************************/
    // Reflection Matrix
    float4x4 RL = make_float4x4(
                    0.5F, 0.5F, 0.F, 0.F,
                    0.5F, 0.5F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F 
            );
    ph->stokes = mul(RL, ph->stokes); /*[Eq. 15,39]*/

    #ifdef BACK
    ph->M = mul(ph->M, RL);
    #endif

	/***************************************************/
	/* Update of photon location and weight */
	/***************************************************/
	if (ph->loc == SURF0P){
		bool test_s = ( SIMd == SURF_ONLY);
		ph->loc = SPACE*test_s + ATMOS*(!test_s);
        #ifndef OPT3D
		ph->layer = NATMd;
        #endif
        #ifndef SIF
        if (ph->env) {
            ph->nenv +=1;
            ph->weight *= spectrum[ph->ilam].alb_env;
        }
        else        {
            ph->nref += 1;
            ph->weight *=  BRDF(ph->ilam, -v0, ph->v, spectrum);
            if (abs(ENVd)!=2) ph->weight *= spectrum[ph->ilam].alb_surface;  /*[Eq. 16,39]*/
            if (ENVd==2) ph->weight *= gauss_albedo(ph->pos, X0d, Y0d) * (spectrum[ph->ilam].alb_surface - spectrum[ph->ilam].alb_env) +
                    spectrum[ph->ilam].alb_env;
        }
        ph->nint+=1;
        #else
        if (ph->emitter==SOLAR_REF){
            if (ph->env) {
                ph->nenv +=1;
                ph->weight *= spectrum[ph->ilam].alb_env;
            }
            else        {
                ph->nref += 1;
                ph->weight *=  BRDF(ph->ilam, -v0, ph->v, spectrum);
                if (abs(ENVd)!=2) ph->weight *= spectrum[ph->ilam].alb_surface;  /*[Eq. 16,39]*/
                if (ENVd==2) ph->weight *= gauss_albedo(ph->pos, X0d, Y0d) * (spectrum[ph->ilam].alb_surface - spectrum[ph->ilam].alb_env) +
                    spectrum[ph->ilam].alb_env;
            }
            ph->nint+=1;
        }
        #endif
	}
	else
	{
		ph->loc = OCEAN;
        #ifndef OPT3D
		ph->layer = NOCEd; 
        #endif
		ph->weight *= spectrum[ph->ilam].alb_seafloor; /*[Eq. 16,39]*/
        ph->nsfl+=1;
        ph->nint+=1;
	}

    #ifdef SIF
	if(!le) ph->emitter = UNDEF;
    #endif
} //surfaceLambert


__device__ int checkerboard(float3 pos, float X0, float Y0) {
        float dis = ENV_SIZEd;
        int freq  = 2;
        int testx = abs(__float2int_rn((pos.x-X0)/dis)%freq);
        int testy = abs(__float2int_rn((pos.y-Y0)/dis)%freq);
        return ( (testx==0) && (testy==1)  ) || ( (testx==1) && (testy==0)  );
}

__device__ float gauss_albedo(float3 pos, float X0, float Y0) {
        float dis2 = (pos.x-X0)*(pos.x-X0) + (pos.y-Y0)*(pos.y-Y0);
        return expf(-dis2/ENV_SIZEd);
}


/* Surface BRDF */
__device__ void surfaceBRDF_new(Photon* ph, int le,
							float* tabthv, float* tabphi, int count_level,
							struct RNG_State *rngstate) {
	if( SIMd == ATM_ONLY){ // Atmosphere only, surface absorbs all
		ph->loc = ABSORBED;
		return;
	}
    ph->nint += 1;
	
    float thv, phi, psi;
    float sig2, temp;
	float3 vr; // photon outgoing direction in the LOCAL frame
	float3 vi; // photon ingoing  direction in the LOCAL frame
	float3 v , u;  // photon outgoing direction in the GLOBAL frame
    float3 no_n, no; // normal to the facet LOCAL and GLOBAL frame
    //float3 w_ne, w_ol;
    float cBeta2; //facet slope squared
    float4x4 R; // Fresnel Reflection Matrix

    #ifdef SPHERIQUE
    // define 3 vectors Nx, Ny and Nz in cartesian coordinates which define a
    // local orthonormal basis at the impact point.
    // Nz is the local vertical direction, the direction of the 2 others does not matter
    // because the azimuth is chosen randomly
	float3 Nx, Ny, Nz;
    //float weight;
    MakeLocalFrame(ph->pos, &Nx, &Ny, &Nz);
    /* Transformation of ingoing direction in the local frame*/
    vi = GlobalToLocal(Nx, Ny, Nz, ph->v);
    #else
    vi = ph->v;
    #endif

    /***************************************************/
    /* Computation of outgoing direction */
    /***************************************************/
    if (le) {
     // Outgoing direction in GLOBAL frame
     phi = tabphi[ph->iph];
     thv = tabthv[ph->ith];
     DirectionToUV(thv, phi, &v, &u);
     #ifdef SPHERIQUE
     // Test if outgoing direction is in Earth s shadow
     if ((dot(v, Nz) <= 0.) && (ph->loc != SEAFLOOR)) { /*[Eq. 40]*/
         ph->loc = ABSORBED;
         return;
     }
     /*Transformation in the local frame*/
     vr = GlobalToLocal(Nx, Ny, Nz, v);
     #else
     vr = v;
     #endif
    }

    else {
     // Cosine of the LOCAL zenith angle sampling for Lambertian reflector
	 phi = RAND*DEUXPI;
	 thv = acosf(sqrtf(RAND));
     DirectionToUV(thv, phi, &vr, &u);
    }

    // Computation of the outgoing direction in GLOBAL frame
    #ifdef SPHERIQUE
    if (ph->loc == SEAFLOOR) {
         v = vr;
    }
    else {
         /* LOCAL to GLOBAL frame */
         v = LocalToGlobal(Nx, Ny, Nz, vr);
    }
    #else
    v = vr;
    #endif

    /***************************************************/
    /* Computation of slope and weight */
    /***************************************************/
	no_n   = operator-(vr, vi);
    no_n   = normalize(no_n);
    cBeta2 = no_n.z * no_n.z;
    sig2   = 0.003F + 0.00512F * WINDSPEEDd;
    if (le) ph->weight *= __fdividef( __expf(-(1.F-cBeta2)/(cBeta2*sig2)), 4.F * cBeta2*cBeta2 * fabs(vi.z) * fabs(v.z)  * sig2);
    else    ph->weight *= __fdividef( __expf(-(1.F-cBeta2)/(cBeta2*sig2)), 4.F * cBeta2*cBeta2 * fabs(vi.z) * fabs(vr.z) * sig2);

    if (WAVE_SHADOWd) {
        // Add wave shadowing computed in the local frame
        float LambdaR, LambdaS;
        LambdaS = LambdaM(fabs(vi.z), sig2*0.5);
        LambdaR = LambdaM(fabs(vr.z), sig2*0.5);
        ph->weight *= __fdividef(1.F, 1.F + LambdaR + LambdaS);
    }


    /***************************************************/
    /* Update of Stokes vector  */
    /***************************************************/
	// Psi determination
    #ifdef SPHERIQUE
    no   = LocalToGlobal(Nx, Ny, Nz, no_n);
    #else
    no   = no_n;
    #endif
    float cTh = fabs(dot(no, ph->v));
    float theta = acosf(fmin(1.F-VALMIN, fmax(-(1.F-VALMIN), cTh)));
    float sTh = sinf(theta);
    temp = __fdividef(dot(no, ph->u), sTh);
	psi  = acosf(fmin(1.F, fmax(-1.F, temp)));	
	if(dot(no, cross(ph->u, ph->v)) < 0 ) psi = -psi;

    /*w_ne = normalize(cross(ph->v, no));
    w_ol = cross(ph->v, ph->u);
    temp = dot(w_ol, w_ne);
	psi  = acosf(fmin(1.F, fmax(-1.F, temp)));	
	if(dot(no, cross(ph->u, ph->v)) < 0 ) psi = -psi;*/

    // Stokes rotation
    rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L;
    rotationM(-psi, &L);
    #endif

    /*float3x3 M = rotation3D(psi, ph->v);
    u = mul(M, ph->u);
    M = rotation3D(acosf(dot(vr, -vi)), w_ne);
    u = mul(M, u);*/

    // Fresnel Matrix Multiplication
    R = FresnelR(vi, vr);
    ph->stokes = mul(R, ph->stokes);

    #ifdef BACK
    ph->M = mul(ph->M, mul(L, R));
    #endif

    /***************************************************/
    /* Update of photon direction, location  */
    /***************************************************/
	u     = operator/(operator-(no, cTh*ph->v), sTh);	
    ph->v = v;
    ph->u = u;

    if (SIMd==SURF_ONLY || SIMd==OCEAN_SURF){
        ph->loc   = SPACE;
    } else {
        ph->loc   = ATMOS;
        ph->layer = NATMd;
    }

    // Russian roulette for propagating photons 
    if (!le && RRd==1) {
		if (ph->weight < WEIGHTRRd){
			if (RAND < __fdividef(ph->weight,WEIGHTRRd)) ph->weight = WEIGHTRRd;
			else ph->loc = ABSORBED;
		}
    }

} //surfaceBRDF

#ifdef OBJ3D
__device__ void surfaceLambert3D(Photon* ph, int le, float* tabthv, float* tabphi,
									  struct Spectrum *spectrum, struct RNG_State *rngstate, IGeo* geoS)
{
	ph->nint += 1;

	if (geoS->type == HELIOSTAT)
	{
		#ifdef DOUBLE
		if (  isBackward( make_double3(geoS->normalBase.x, geoS->normalBase.y, geoS->normalBase.z),
						  make_double3(ph->v.x, ph->v.y, ph->v.z) )  ) // AV
		#else
		if (isBackward(geoS->normalBase, ph->v))
		#endif
		{ ph->H += 1; }
		else { ph->E += 1; } // AR traité comme environnement
        if (ph->direct == 0) {ph->v_i = make_float3(-ph->v.x, -ph->v.y, -ph->v.z);}
	}
	else if ( geoS->type == RECEIVER)
	{ ph->E += 1; }


	float thv, phi;

	/***************************************************/
    /* Computation of outgoing direction */
    /***************************************************/
    if (le)
	{
		// Outgoing direction in GLOBAL frame
		phi = tabphi[ph->iph];
		thv = tabthv[ph->ith];
		DirectionToUV(thv, phi, &ph->v, &ph->u);

		float weight = dot(ph->v, geoS->normalBase);
		if (weight <= 0.)
		{
			ph->loc = ABSORBED;
			return;
		}
		ph->weight *= weight/fabs(ph->v.z);
    }
    else
	{
		float3 v_n, u_n; // photon outgoing direction in the LOCAL frame

		// Cosine of the LOCAL zenith angle sampling for Lambertian reflector
		phi = RAND*DEUXPI;
		thv = acosf(sqrtf(RAND));
		DirectionToUV(thv, phi, &v_n, &u_n);

		float3 vecX, vecY, vecZ;
		vecX=make_float3(1., 0., 0.);
		vecY=make_float3(0., 1., 0.);
		vecZ=make_float3(0., 0., 1.);
   
		Transform transfo=geoS->mvTF;
		vecX=transfo(Vectorf(vecX)); vecX=normalize(vecX);
		vecY=transfo(Vectorf(vecY)); vecY=normalize(vecY);
		vecZ=transfo(Vectorf(vecZ)); vecZ=normalize(vecZ);

		float4x4 M = make_float4x4(
			vecX.x, vecY.x, vecZ.x, 0.f,
			vecX.y, vecY.y, vecZ.y, 0.f,
			vecX.z, vecY.z, vecZ.z, 0.f,
			0.f , 0.f , 0.f , 1.f
			);

		float4x4 tM = transpose(M);	
		Transform wTo(M, tM);

		// apply the transformation
		v_n = wTo(Vectorf(v_n));
		u_n = wTo(Vectorf(u_n));

		if ( (isnan(v_n.x)) || (isnan(v_n.y)) || (isnan(v_n.z)) || 
			 (isnan(u_n.x)) || (isnan(u_n.y)) || (isnan(u_n.z)) )
		{
			ph->loc = REMOVED;
			return;
		}
	
		// Update the value of u and v of the photon	
		ph->v = v_n;
		ph->u = u_n;
    }


    /***************************************************/
    /* Update of Stokes vector  */
    /***************************************************/
    // Reflection Matrix
	float4x4 RL = make_float4x4(
                    0.5F, 0.5F, 0.F, 0.F,
                    0.5F, 0.5F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F,
                    0.0F, 0.0F, 0.F, 0.F 
            );
    ph->stokes = mul(RL, ph->stokes); /*[Eq. 15,39]*/

    #ifdef BACK
    ph->M = mul(ph->M, RL);
    #endif
	
	/***************************************************/
	/* Update of photon location and weight */
	/***************************************************/
	ph->locPrev = OBJSURF;
	ph->loc = ATMOS;

	ph->weight *= geoS->reflectivity;
		
    // if (!le)
	// {
	// 	if (RRd==1){
	// 		/* Russian roulette for propagating photons **/
	// 		if( ph->weight < WEIGHTRRd ){
	// 			if( RAND < __fdividef(ph->weight,WEIGHTRRd) ){ph->weight = WEIGHTRRd;}
	// 			else{ph->loc = ABSORBED;}
	// 		}
	// 	}
	// } // not le
} // Function lamb3D

__device__ void surfaceRugueuse3D(Photon* ph, IGeo* geoS, struct RNG_State *rngstate)
{
    ph->nint += 1;

	if (geoS->type == 1)
	{
		#ifdef DOUBLE
		if (  isBackward( make_double3(geoS->normalBase.x, geoS->normalBase.y, geoS->normalBase.z),
						  make_double3(ph->v.x, ph->v.y, ph->v.z) )  ) // AV
		#else
		if (isBackward(geoS->normalBase, ph->v))
		#endif
		{ ph->H += 1; }
		else { ph->E += 1; } // AR traité comme environnement
	}
	else if ( geoS->type == 2)
	{ ph->E += 1; }
	
	float3 u_n, v_n;	// Vecteur du photon après reflexion

	v_n = ph->v;
	u_n = ph->u;

	// Rotation of Stokes parameters
	float temp, psi;
	
	temp = dot(cross(ph->v, ph->u),normalize(cross(ph->v, geoS->normal)));
	psi = acosf( fmin(1.00F, fmax( -1.F, temp ) ));	

	if( dot(geoS->normal, cross(ph->u, ph->v)) <0 ){
		psi = -psi;
	}
	
	rotateStokes(ph->stokes, psi, &ph->stokes);
    #ifdef BACK
    float4x4 L = make_diag_float4x4 (1.F);
    rotationM(-psi,&L);
    #endif
	
	float4x4 R = make_float4x4(
		1., 0.,  0.,  0.,
		0., 1.,  0.,  0.,
		0., 0., -1.,  0.,
		0., 0.,  0., -1. 
		);
	
	ph->stokes = mul(R, ph->stokes);
	
    #ifdef BACK
	ph->M   = mul(ph->M,mul(L,R));
    #endif
	Transform transfo, invTransfo, aRot;
	
	transfo = geoS->mvTF;
	aRot = aRot.RotateZ(180);
	invTransfo = transfo.Inverse(transfo);
	
	v_n = invTransfo(Vectorf(v_n));
	u_n = invTransfo(Vectorf(u_n));

	v_n = aRot(Vectorf(v_n));
	u_n = aRot(Vectorf(u_n));

	v_n = make_float3(-v_n.x, -v_n.y, -v_n.z);
	u_n = make_float3(-u_n.x, -u_n.y, -u_n.z);

	v_n = transfo(Vectorf(v_n));
	u_n = transfo(Vectorf(u_n));
	
	if ( (isnan(v_n.x)) || (isnan(v_n.y)) || (isnan(v_n.z)))
	{
		ph->loc = REMOVED;
		return;
	}
	
	// Update the value of u and v of the photon	
	ph->locPrev = OBJSURF;
	ph->loc = ATMOS;
	ph->v = normalize(v_n);
	ph->u = normalize(u_n);
	
	ph->weight *= geoS->reflectivity;
	
	// if (RRd==1){
	// 	/* Russian roulette for propagating photons **/
	// 	if( ph->weight < WEIGHTRRd ){
	// 		if( RAND < __fdividef(ph->weight,WEIGHTRRd) ){ph->weight = WEIGHTRRd;}
	// 		else{ph->loc = ABSORBED;}
	// 	}
	// }
	
} // FUNCTION SURFACEAGITE3D


__device__ void Obj3DRoughSurf(Photon* ph, int le, float* tabthv, float* tabphi, IGeo* geoS, struct RNG_State *rngstate)
{
    ph->nint += 1;
	
	float3 v_i = ph->v;                    // - direction of the incoming photon
	float3 u_i = ph->u;                    // - perp direction of the incoming photon
	float3 v_o, u_o;                       // outcoming directions
	float cTheta_i;                        // - cos of the angle between normal m and the
	                                       //   direction of the incoming photon v
	float sTheta_i;                        // - sin of theta_i
	float3 macroFnormal_n = geoS->normal;  // - the (big) macrofacet normal n
	float alpha = geoS->roughness;         // - Slope error or rugosity parameter
	float3 microFnormal_m;                 // - the (small) random microfacet normal m
	float theta_m, phi_m;                  // - theta_m and phi_m respectively the azi
	                                       //   and zen angles between normals n and m
	float tTheta_m;                        // - tan(theta_m)
	float cTheta_m, sTheta_m;              // - cos and sin of theta_m
	float cPhi_m, sPhi_m;                  // - cos and sin of phi_m
	float nind = geoS->nind;               // - relative refractive index air/obj
	float3 h_r;                            // half-direction for reflection used in LE
	float thv, phi;                        // used only in LE

	// Find if the photon come from the front(1) or back(-1) of the obj surface
	// geoS->normalBase is the obj normal of the front surface
	int sign = (isBackward(macroFnormal_n, v_i)) ? 1 : -1;
    float avz;
	
	if (geoS->type == HELIOSTAT)
	{
		if (sign > 0) { ph->H += 1; }
		else { ph->E += 1; } // Back heliostat surface considered as environnement
	}
	else if ( geoS->type == RECEIVER)
	{ ph->E += 1; }

	if (le == 0)
	{
		// The initial normal of the obj before transfo is colinear to the z axis,
		// then create the transfo inverse for the direction v_i and simplify sampling
		Transform transfo=geoS->mvTF, invTransfo;
		invTransfo = transfo.Inverse(transfo);
		
		// Incident direction v if obj normal colinear to z axis
		float3 v_iInv = normalize(invTransfo(Vectorf(v_i)));
		
		// ***********************************************************************************
		// Beckmann sampling of theta_m, phi_m to get cTheta_i, sTheta_i -> Walter et al. 2007
		// ***********************************************************************************
		if (alpha > VALMIN) // Roughness > zero --> roughness surface
		{
			int iter=0; float rand1, rand2;
			cTheta_i = -1.F;
			while (cTheta_i <= 0)
			{
				iter++;
				if (iter >= 100) {ph->loc = NONE; return;}
				rand1 = RAND; rand2 = RAND;
				if (geoS->dist == DIST_GGX) {tTheta_m = fdividef(alpha*sqrtf(rand1), sqrtf(1 - rand1));}
				else {tTheta_m = alpha*sqrtf(-__logf(rand1) );} //Beckmann
				theta_m = atanf(tTheta_m);
				phi_m   = DEUXPI * rand2;
		
				// find the normal of the microfacet m thanks to thata_m and phi_m sampling
				cTheta_m = __cosf(theta_m); sTheta_m = __sinf(theta_m);
				cPhi_m = __cosf(phi_m); sPhi_m = __sinf(phi_m);
				microFnormal_m = make_float3( sTheta_m*cPhi_m,  sTheta_m*sPhi_m, cTheta_m );
				microFnormal_m *= sign; microFnormal_m = normalize(microFnormal_m);
				cTheta_i = -dot( microFnormal_m, v_iInv);
				cTheta_i = clamp(cTheta_i, -1.F, 1.F);
			}
		}
		else // Roughness equal zero (or very very close) --> perfect flat surface
		{
			microFnormal_m = make_float3(0.F, 0.F, 1.F);
			microFnormal_m *= sign;
			cTheta_i = -dot( microFnormal_m, v_iInv);
			cTheta_i = clamp(cTheta_i, -1.F, 1.F);
		}
	
		// Inverse transfo has been used in sampling then come back to "real basis"
		//microFnormal_m = normalize(transfo(Normalf(microFnormal_m)));
		microFnormal_m = normalize(transfo(Vectorf(microFnormal_m)));
	} // end le==0 
	else // else if le==1 -->
	{
		phi = tabphi[ph->iph]; thv = tabthv[ph->ith];
		v_o.x = cosf(phi) * sinf(thv);
		v_o.y = sinf(phi) * sinf(thv);
		v_o.z = cosf(thv);
		// If macro normal isn't in the same hemisphere as v_o=-vSun, then no contribution
		if(isBackward(macroFnormal_n, v_o)) {ph->weight=0.; return;};
		h_r = v_o - v_i;
	    h_r = normalize(h_r);
		microFnormal_m = h_r;
		// Exclude facets whose normal are not on the same side as incoming photons
		if(isForward(microFnormal_m, v_i)) {ph->loc=REMOVED; return;};
		//cTheta_i = dot(microFnormal_m, -v_i);
        cTheta_i = fabs(-dot(microFnormal_m, v_i));
		cTheta_m = fabs(dot(macroFnormal_n, microFnormal_m));
        avz = fabs(dot(macroFnormal_n, v_i));
	} // end le==1

	// Less expensive than find theta from arcos and apply the sin
	sTheta_i = sqrtf( fmaxf(VALMIN,  1.F-(cTheta_i*cTheta_i)) );

	// ***********************************************************************************

	// ********************************************************
	// Rotation of the stokes vector
	// ********************************************************
	// Compute psi angle and rotation matrix L in (For/Back)ward mode
	float psi; float4x4 LF;
	psi = computePsiFN(u_i, v_i, microFnormal_m, sTheta_i);
	rotationM(psi, &LF); // fill the rotation matrix LF (forward mode)
	
	#ifdef BACK
	float4x4 LB;
	float4x4 L = make_diag_float4x4 (1.F);
	rotationM(-psi, &LB); // fill the rotation matrix LB (backward mode)
	#endif

	// Reflection matrix R
	float4x4 R; float sR;
	if (nind == PERFECT_MIRROR) {R = perfect_mirrorRF(); sR = 1.F;}
	else
    {
		//R = computeRefMat(1.33, cTheta_i, sTheta_i);
		refMat(nind, cTheta_i, sTheta_i, &R, &sR);
	}
	
	// Muller matrix for reflection (forward mode)
	float4x4 Mu_r; // Mu_r = (1/scaR)*R*L -> Ramon et al. 2019
	Mu_r = mul(R, LF); // relfection totale scaR = 1

    #ifndef BIAS
	ph->weight *= ph->stokes.x + ph->stokes.y;
	ph->stokes /= ph->stokes.x + ph->stokes.y; 
    #endif

    #ifdef BACK
    ph->M = mul(ph->M, mul(LB, R));
	#endif
	
	// update the stokes vector (forward mode)
	ph->stokes = mul(Mu_r, ph->stokes); //S_l = Mu_r*S_l-1
	// ********************************************************

	// ********************************************
	// Reflection of the direction v and perp dir u
	// ********************************************
	if (le == 0) { v_o = specularFNC(v_i, microFnormal_m, cTheta_i);}
	u_o = (microFnormal_m-cTheta_i*v_o)/sTheta_i;
	v_o = normalize(v_o); u_o = normalize(u_o);
	// ********************************************

	// ********************************************************************
	// Shadowing-Masking Function G2
	// ********************************************************************
	float G2, dotViN, dotViM;
	dotViN = dot(v_i, macroFnormal_n);
	dotViM = dot(v_i, microFnormal_m);
	if (geoS->shadow) // if we consider Shadowing-Masking effect
	{
		float G1_i, cTheta_iN=-dotViN, sTheta_iN, tTheta_iN;
		sTheta_iN = sqrtf(fmaxf( 0.F,  1.F - (cTheta_iN*cTheta_iN) ));
		tTheta_iN = __fdividef(sTheta_iN, cTheta_iN);
		int xsiPi, xsiPo; // positive characteristic function	
		xsiPi = ( __fdividef(dotViM, dotViN) > 0) ? 1 : 0;
		if (xsiPi == 0 and le == 0) { ph->loc = ABSORBED; return; }
		if (geoS->dist == DIST_GGX){G1_i = G1GGX(alpha*alpha, tTheta_iN*tTheta_iN);}
		else {G1_i = G1B(alpha, tTheta_iN);} // Beckmann
		
		float G1_o, dotVoN, cTheta_oN, sTheta_oN, tTheta_oN;
		dotVoN = dot(v_o, macroFnormal_n);
		cTheta_oN = dotVoN; //dot( microFnormal_m, v_o);
		sTheta_oN = sqrtf(fmaxf( 0.F,  1.F - (cTheta_oN*cTheta_oN) ));
		tTheta_oN = __fdividef(sTheta_oN, cTheta_oN);	
		xsiPo = ( __fdividef(dot(v_o, microFnormal_m), dotVoN) > 0) ? 1 : 0;
		if (xsiPo == 0 and le == 0) { ph->loc = ABSORBED; return; }
		if (geoS->dist == DIST_GGX){G1_o = G1GGX(alpha*alpha, tTheta_oN*tTheta_oN);}
		else{G1_o = G1B(alpha, tTheta_oN);} // Beckmann
		
		// Several methods possible, we choose to compute from G1
		// Bellow best approximation than G2 = G1_i*G1_o
		// G2 = 1/(1 + LambO + LambI) and G1_[i or o] = 1/(1 + Lamb[i or o]) then
		// G2 = 1/(1/G1_i + 1/G1_o - 1) -->
		float invG1_i, invG1_o;
		invG1_i = fdividef(1.F, G1_i);
		invG1_o = fdividef(1.F, G1_o);
		G2 = fdividef(1.F, invG1_i + invG1_o - 1.F);

		// Be sure that G2 is <= 1
		G2 = fminf(G2, 1.F);
	}
	else { G2 = 1.F; }
	// ********************************************************************

	if (!le)
	{
		// ***************************************************************************************************
		// Weighting according to Walter et al. 2007
		// ***************************************************************************************************
		ph->weight *= __fdividef(fabsf(dotViM)*G2, fabsf(dotViN)*fabsf(dot(microFnormal_m, macroFnormal_n)));
	} // end le==0
	else // le==1
	{
		float cTheta2_m, alph2, sTheta_m, tTheta2_m, p_m;
		int xsiP_mn;

		cTheta2_m = cTheta_m*cTheta_m;
		//cTheta_o = fabs(dot(v_o, macroFnormal_n));
		sTheta_m = sqrtf(fmaxf(0.F, 1.F-cTheta2_m));
		alph2 = alpha*alpha;
		tTheta2_m = sTheta_m/fmaxf(VALMIN, cTheta_m);
		tTheta2_m = tTheta2_m*tTheta2_m;
		xsiP_mn = ( cTheta_m > 0) ? 1 : 0; //( (pSca_mn) > 0) ? 1 : 0;

		

		if (geoS->dist == DIST_GGX and xsiP_mn)
		{
			// pdf without the pi, but be careful no validation with GGX!!
			p_m = __fdividef(alph2, cTheta2_m*cTheta2_m*(alph2 + tTheta2_m)*(alph2 + tTheta2_m));
		}
		else if (geoS->dist == DIST_BECKMANN and xsiP_mn)
		{
			// pdf without the pi (mistake in ramon et al 2019 formulation?)
			//p_m = __fdividef( __expf(-(1.F-cTheta2_m)/(cTheta2_m*alph2)) , cTheta2_m*cTheta_m * alph2);
			p_m = __fdividef( __expf(-(tTheta2_m/alph2)) , cTheta2_m*cTheta2_m * alph2);
		}
		else // if xsiP_mn == 0 --> p_m=0 then:
		{
			ph->weight = 0.F; ph->locPrev = OBJSURF; ph->loc = ATMOS; return;
		}

		float jac, p_o;
		
		jac = __fdividef(1.F, 4.F*fabs(cTheta_i));
        //p_o = __fdividef(p_m * fabs(cTheta_i), cTheta_m * fabs(cosf(tabthv[ph->ith])));
        p_o = __fdividef(p_m * fabs(cTheta_i), fabs(cosf(tabthv[ph->ith])));
		p_o *= G2; // normalization
		ph->weight *= fdividef(p_o*jac, avz);

        /* jac = __fdividef(1.F, 4.F*fabs(dot(v_o, microFnormal_m)));
        p_o = __fdividef(p_m * fabs(dot(microFnormal_m, macroFnormal_n)) * jac, cTheta_o);
		p_o *= G2; // normalization
        ph->weight *= p_o; */

		//ph->weight *= __fdividef(p_m*G2, 4.F*fabs(cTheta_o)*fabs(cTheta_i));
	} //end le==1

	// **********************************************
	// update photon directions u, v, location and albedo
	// **********************************************
	ph->v = v_o;
	ph->u = u_o;
    ph->weight *= geoS->reflectivity;
	if (ph->loc == OBJSURF)
	{
		//if (isForward(geoS->normalBase, ph->v))
        if (isForward(geoS->normal, ph->v))
		{
			ph->locPrev = OBJSURF;
			ph->loc = ATMOS;
		}
		else if (true) //SINGLEd) for now always SINGLE = True
			ph->loc = REMOVED;
	}
	else
		ph->loc = REMOVED;
	// **********************************************

    if (!le)
    {
        // Russian roulette for propagating photons
		if (RRd==1)
        {
		    if( ph->weight < WEIGHTRRd )
            {
		 		if( RAND < __fdividef(ph->weight, WEIGHTRRd) ){ ph->weight = WEIGHTRRd; }
		 		else { ph->loc = ABSORBED; }
		 	}
		}
    }
} // FUNCTION OBJ3DROUGHSURF

__device__ void countLoss(Photon* ph, IGeo* geoS, void *wPhLoss, void *wPhLoss2)
{
	#ifdef DOUBLE
	double *wPhLossC; double *wPhLossC2; double2 stokes; double w_I, w_rhoM;
	if (IsAtm == 1)stokes = make_double2(ph->stokes.x, ph->stokes.y);
	else stokes = make_double2(0.5, 0.5);
	wPhLossC = (double*)wPhLoss; wPhLossC2 = (double*)wPhLoss2;      // - table comprinsing different weights
	w_I = double(ph->weight_loss[0])*double(stokes.x + stokes.y);    // - incident flux weight after cos effect
	w_rhoM = double(1-geoS->reflectivity)*w_I;                       // - flux weight lost due to reflectivity
	#ifndef BACK
	double w_rhoP, w_SM, w_SP, w_BM, w_BP;
	w_rhoP = double(geoS->reflectivity)*w_I;                         // - reflected flux weight
	w_BM = double(ph->weight_loss[1])*double(stokes.x + stokes.y);   // - flux weight lost due to blocking effect
	w_BP = w_rhoP-w_BM;
	w_SP = double(ph->weight_loss[2])*double(stokes.x + stokes.y);
	w_SM = double(ph->weight_loss[3])*double(stokes.x + stokes.y);   // - flux weight lost due to spillage
	#endif
    #else // FLOAT
	float *wPhLossC; float *wPhLossC2; float2 stokes; float w_I, w_rhoM;
	if (IsAtm == 1)stokes = make_float2(ph->stokes.x, ph->stokes.y);
	else stokes = make_float2(0.5, 0.5);
	wPhLossC = (float*)wPhLoss; wPhLossC2 = (float*)wPhLoss2;	
	w_I = ph->weight_loss[0]*float(stokes.x + stokes.y);
	w_rhoM = float(1-geoS->reflectivity)*w_I;
	#ifndef BACK
	float w_rhoP, w_SM, w_SP, w_BM, w_BP;
	w_rhoP = float(geoS->reflectivity)*w_I;
	w_BM = ph->weight_loss[1]*(stokes.x + stokes.y);   // - flux weight lost due to blocking effect
	w_BP = w_rhoP-w_BM;
	w_SP = ph->weight_loss[2]*(stokes.x + stokes.y);
	w_SM = ph->weight_loss[3]*(stokes.x + stokes.y);   // - flux weight lost due to spillage	
	#endif
	#endif
    
	if (ph->H < 2) // If this is the first time that a photon is reaching a heliostat
	{
		#if !defined(DOUBLE) || (defined(DOUBLE) && (__CUDA_ARCH__ >= 600))
	    atomicAdd(wPhLossC, w_I); atomicAdd(wPhLossC2, w_I*w_I);
		atomicAdd(wPhLossC+1, w_rhoM); atomicAdd(wPhLossC2+1, w_rhoM*w_rhoM);
		#ifndef BACK
		atomicAdd(wPhLossC+2, w_rhoP); atomicAdd(wPhLossC2+2, w_rhoP*w_rhoP);
		atomicAdd(wPhLossC+3, w_BM); atomicAdd(wPhLossC2+3, w_BM*w_BM);
		atomicAdd(wPhLossC+4, w_BP); atomicAdd(wPhLossC2+4, w_BP*w_BP);
		atomicAdd(wPhLossC+5, w_SM); atomicAdd(wPhLossC2+5, w_SM*w_SM);
		atomicAdd(wPhLossC+6, w_SP); atomicAdd(wPhLossC2+6, w_SP*w_SP);
        #endif // END BACK
        #else // double and old nvidia card
		DatomicAdd(wPhLossC, w_I); DatomicAdd(wPhLossC2, w_I*w_I);
		DatomicAdd(wPhLossC+1, w_rhoM); DatomicAdd(wPhLossC2+1, w_rhoM*w_rhoM);
		#ifndef BACK
		DatomicAdd(wPhLossC+2, w_rhoP); DatomicAdd(wPhLossC2+2, w_rhoP*w_rhoP);
		DatomicAdd(wPhLossC+3, w_BM); DatomicAdd(wPhLossC2+3, w_BM*w_BM);
		DatomicAdd(wPhLossC+4, w_BP); DatomicAdd(wPhLossC2+4, w_BP*w_BP);
		DatomicAdd(wPhLossC+5, w_SM); DatomicAdd(wPhLossC2+5, w_SM*w_SM);
		DatomicAdd(wPhLossC+6, w_SP); DatomicAdd(wPhLossC2+6, w_SP*w_SP);
		#endif // END BACK
        #endif // END !defined(DOUBLE) || (defined(DOUBLE) && (__CUDA_ARCH__ >= 600))
	}
}

__device__ void countPhotonObj3D(Photon* ph, int le, void *tabObjInfo, IGeo* geoS, unsigned long long *nbPhCat,
		void *wPhCat, void *wPhCat2, struct Profile *prof_atm, void *wPhLoss, void *wPhLoss2)
{
	int indI = 0; int indJ = 0;
	float3 p_t; float sizeX = nbCx*TCd; float sizeY = nbCy*TCd;

    // test single scattering or photons removed
    if (ph->loc==REMOVED || ph->loc==ABSORBED || ph->nint>SMAXd || ph->nint<SMINd) { return; }

    // don't count photons with nof finite results
    if (!isfinite(ph->stokes.x) || !isfinite(ph->stokes.y) ||
        !isfinite(ph->stokes.z) || !isfinite(ph->stokes.w) ||
        !isfinite(ph->weight) )  {
        //printf("%d %g\n",ph->ilam, ph->weight);    
        return;
    }

    // don't count the photons directly transmitted
    if (ph->nint == 0 && DIRECTd==0) {
        return;
    }
	
    #if defined(BACK)	
	if (LMODEd == 4 and le == 0 )
	{
	    // Use also double precison operations due to simple precison limit 
		double cosANGD, cosPHSUN;
		double3 vecSUN = normalize(make_double3(-DIRSXd, -DIRSYd, -DIRSZd));
		double3 vecPH = normalize(make_double3(ph->v.x, ph->v.y, ph->v.z));
		
		cosANGD = cos( radiansd( SUN_DISCd )  );
		cosPHSUN = dot(vecSUN, vecPH);
		p_t = ph->posIni;
        
        // Below an orther method normally for efficient with small angle. Verification is needed
        /* double angleBis = atan2(length(cross(vecSUN, vecPH)), cosPHSUN)*180./CUDART_PI;
        if(angleBis > double(SUN_DISCd)) { return; } */

		if (cosPHSUN < cosANGD) {return;}
		if (ph->direct == 0){countLoss(ph, geoS, wPhLoss, wPhLoss2);}
	}
    else if (LMODEd == 4 and le == 1)
    {
        p_t = ph->posIni;
    }
	else
	{
		Transform transfo, invTransfo;
		p_t = ph->pos;
		transfo = geoS->mvTF;
		invTransfo = transfo.Inverse(transfo);
		p_t = invTransfo(Pointf(p_t));
	}
	#else
	//In order to be sure to not consider the photons coming behind the receiver
	if (!isBackward(geoS->normalBase, ph->v)) return;
	#ifdef DOUBLE
	if (   isForward(  make_double3(geoS->normalBase.x, geoS->normalBase.y, geoS->normalBase.z),
				  make_double3(ph->v.x, ph->v.y, ph->v.z)  )   )
	#else
	if (isForward(geoS->normalBase, ph->v))
	#endif
		return;
	
	Transform transfo, invTransfo;
	p_t = ph->pos;
	transfo = geoS->mvTF;
	invTransfo = transfo.Inverse(transfo);
	p_t = invTransfo(Pointf(p_t));
	#endif
	
    // x (axis from bot to the top ^); y (axis from right to the left <--)
	indJ = floorf( (-(p_t.y/TCd)) + (sizeY/(2*TCd)) );
	indI = floorf( (-(p_t.x/TCd)) + (sizeX/(2*TCd)) );
	if (indJ == nbCy) indJ -= 1;
	if (indI == nbCx) indI -= 1;
	
    #ifdef DOUBLE
	double *tabCountObj, *wPhCatC, *wPhCatC2;
	double2 stokes; double weight, weight2;
	if (IsAtm == 1) stokes = make_double2(ph->stokes.x, ph->stokes.y);
	else stokes = make_double2(0.5, 0.5);
	tabCountObj = (double*)tabObjInfo;
	wPhCatC = (double*)wPhCat; wPhCatC2 = (double*)wPhCat2;
	weight = double(ph->weight) * double(stokes.x + stokes.y);
    #else // If not DOUBLE
	float *tabCountObj, *wPhCatC, *wPhCatC2;
	float2 stokes; float weight, weight2;
	if (IsAtm == 1) stokes = make_float2(ph->stokes.x, ph->stokes.y);
	else stokes = make_float2(0.5, 0.5);
	tabCountObj = (float*)tabObjInfo;
	wPhCatC = (float*)wPhCat; wPhCatC2 = (float*)wPhCat2;
	weight = ph->weight * float(stokes.x + stokes.y);
	#endif

	if (le == 1)
	{
        // Normally count only case where count level is UPTOA
        int layer_le;
        float tau_le;
        Profile *prof;
        int layer_end;

        // define correct start and end layers and profiles for LE
        layer_le = 0; 
        layer_end= NATMd;
        prof = prof_atm;
        
        // First get the extinction optical depth at the counting level
        tau_le = prof[(layer_end-layer_le) + ph->ilam *(layer_end+1)].OD;
        // if BEER=0, photon variable tau corresponds to extinction
        if (BEERd == 0) weight *= expf(-fabs(__fdividef(tau_le - ph->tau, ph->v.z))); // LE attenuation to count_level
        // if BEER=1, photon variable tau corresponds to scattering only, need to add photon absorption variable
        else weight *= expf(-fabs(__fdividef(tau_le - (ph->tau+ph->tau_abs), ph->v.z))); // LE attenuation to count_level
	}

    weight2 = weight * weight;
	if(isnan(weight)){printf("Care weight is nan !! \n");return;}

    #if !defined(DOUBLE) || (defined(DOUBLE) && __CUDA_ARCH__ >= 600)
	// All the beams reaching a receiver
	atomicAdd(tabCountObj+(nbCy*indI)+indJ, weight);	

	// Les huit catégories
	if (ph->H == 0 && ph->E == 0 && ph->S == 0) 
	{ // CAT 1 : aucun changement de trajectoire avant de toucher le R.
		atomicAdd(wPhCatC, weight); atomicAdd(wPhCatC2, weight2);// comptage poids
		atomicAdd(nbPhCat, 1);     // comptage nombre de photons
		atomicAdd(tabCountObj+(nbCy*nbCx)+(nbCy*indI)+indJ, weight); // distri
	}
	else if ( ph->H > 0 && ph->E == 0 && ph->S == 0)
	{ // CAT 2 : only H avant de toucher le R.
		atomicAdd(wPhCatC+1, weight); atomicAdd(wPhCatC2+1, weight2);
		atomicAdd(nbPhCat+1, 1);
		atomicAdd(tabCountObj+(2*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H == 0 && ph->E > 0 && ph->S == 0)
	{ // CAT 3 : only E avant de toucher le R.
		atomicAdd(wPhCatC+2, weight); atomicAdd(wPhCatC2+2, weight2);
		atomicAdd(nbPhCat+2, 1);
		atomicAdd(tabCountObj+(3*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H == 0 && ph->E == 0 && ph->S > 0)
	{ // CAT 4 : only S avant de toucher le R.
		atomicAdd(wPhCatC+3, weight); atomicAdd(wPhCatC2+3, weight2);
		atomicAdd(nbPhCat+3, 1);
		atomicAdd(tabCountObj+(4*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H > 0 && ph->E == 0 && ph->S > 0)
	{ // CAT 5 : 2 proc. H et S avant de toucher le R.
		atomicAdd(wPhCatC+4, weight); atomicAdd(wPhCatC2+4, weight2);
		atomicAdd(nbPhCat+4, 1);
		atomicAdd(tabCountObj+(5*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H > 0 && ph->E > 0 && ph->S == 0)
	{ // CAT 6 : 2 proc. H et E avant de toucher le R.
		atomicAdd(wPhCatC+5, weight); atomicAdd(wPhCatC2+5, weight2);
		atomicAdd(nbPhCat+5, 1);
		atomicAdd(tabCountObj+(6*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
		//printf("H = %d, E = %d, S = %d", ph->H, ph->E, ph->S);
		//=(%f,%f)
	}
	else if ( ph->H == 0 && ph->E > 0 && ph->S > 0)
	{ // CAT 7 : 2 proc. E et S avant de toucher le R.
		atomicAdd(wPhCatC+6, weight); atomicAdd(wPhCatC2+6, weight2);
		atomicAdd(nbPhCat+6, 1);
		atomicAdd(tabCountObj+(7*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}	
	else if ( ph->H > 0 && ph->E > 0 && ph->S > 0)
	{ // CAT 8 : 3 proc. H, E et S avant de toucher le R.
		atomicAdd(wPhCatC+7, weight); atomicAdd(wPhCatC2+7, weight2);
		atomicAdd(nbPhCat+7, 1);
		atomicAdd(tabCountObj+(8*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}		
    #else // If DOUBLE and not a new nvidia card
	DatomicAdd(tabCountObj+(nbCy*indI)+indJ, weight);

	// Les huit catégories
	if (ph->H == 0 && ph->E == 0 && ph->S == 0) 
	{ // CAT 1 : aucun changement de trajectoire avant de toucher le R.
		DatomicAdd(wPhCatC, weight); DatomicAdd(wPhCatC2, weight2);// comptage poids
		atomicAdd(nbPhCat, 1);     // comptage nombre de photons
		DatomicAdd(tabCountObj+(nbCy*nbCx)+(nbCy*indI)+indJ, weight); // distri
	}
	else if ( ph->H > 0 && ph->E == 0 && ph->S == 0)
	{ // CAT 2 : only H avant de toucher le R.
		DatomicAdd(wPhCatC+1, weight); DatomicAdd(wPhCatC2+1, weight2);
		atomicAdd(nbPhCat+1, 1);
		DatomicAdd(tabCountObj+(2*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
		//printf("H = %d, E = %d, S = %d", ph->H, ph->E, ph->S);
	}
	else if ( ph->H == 0 && ph->E > 0 && ph->S == 0)
	{ // CAT 3 : only E avant de toucher le R.
		DatomicAdd(wPhCatC+2, weight); DatomicAdd(wPhCatC2+2, weight2);
		atomicAdd(nbPhCat+2, 1);
		DatomicAdd(tabCountObj+(3*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H == 0 && ph->E == 0 && ph->S > 0)
	{ // CAT 4 : only S avant de toucher le R.
		DatomicAdd(wPhCatC+3, weight); DatomicAdd(wPhCatC2+3, weight2);
		atomicAdd(nbPhCat+3, 1);
		DatomicAdd(tabCountObj+(4*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H > 0 && ph->E == 0 && ph->S > 0)
	{ // CAT 5 : 2 proc. H et S avant de toucher le R.
		DatomicAdd(wPhCatC+4, weight); DatomicAdd(wPhCatC2+4, weight2);
		atomicAdd(nbPhCat+4, 1);
		DatomicAdd(tabCountObj+(5*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
	else if ( ph->H > 0 && ph->E > 0 && ph->S == 0)
	{ // CAT 6 : 2 proc. H et E avant de toucher le R.
		DatomicAdd(wPhCatC+5, weight); DatomicAdd(wPhCatC2+5, weight2);
		atomicAdd(nbPhCat+5, 1);
		DatomicAdd(tabCountObj+(6*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
		//printf("H = %d, E = %d, S = %d", ph->H, ph->E, ph->S);
	}
	else if ( ph->H == 0 && ph->E > 0 && ph->S > 0)
	{ // CAT 7 : 2 proc. E et S avant de toucher le R.
		atomicAdd(nbPhCat+6, 1);
		DatomicAdd(tabCountObj+(7*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}	
	else if ( ph->H > 0 && ph->E > 0 && ph->S > 0)
	{ // CAT 8 : 3 proc. H, E et S avant de toucher le R.
		DatomicAdd(wPhCatC+7, weight); DatomicAdd(wPhCatC2+7, weight2);
		atomicAdd(nbPhCat+7, 1);
		DatomicAdd(tabCountObj+(8*nbCy*nbCx)+(nbCy*indI)+indJ, weight);
	}
    #endif // End of !defined(DOUBLE) || (defined(DOUBLE) && defined(NEW_CARDS))
}
#endif // End OBJ3D


__device__ void countPhoton(Photon* ph, struct Spectrum *spectrum,
        struct Profile *prof_atm, struct Profile *prof_oc,
        float *tabthv, float *tabphi,
        int count_level,
		unsigned long long *errorcount,
        void *tabPhotons, void *tabDist, void *tabHist, unsigned long long *NPhotonsOut
        ) {


    // test single scattering or photons removed
    if (count_level < 0 || ph->loc==REMOVED || ph->loc==ABSORBED || ph->nint>SMAXd || ph->nint<SMINd) {
    //if (count_level < 0 || ph->loc==REMOVED || ph->loc==ABSORBED) {
        // don't count anything
        return;
    }

    // don't count photons with nof finite results
    if (!isfinite(ph->stokes.x) || !isfinite(ph->stokes.y) ||
        !isfinite(ph->stokes.z) || !isfinite(ph->stokes.w) ||
        !isfinite(ph->weight) )  {
        //printf("%d %g\n",ph->ilam, ph->weight);    
        return;
    }

    // don't count the photons directly transmitted
    if (ph->nint == 0 && DIRECTd==0) {
        return;
    }
    
    // Declaration for double
    #ifdef DOUBLE 
     double *tabCount;                   // pointer to the "counting" array:
     double dweight;
	 double4 ds;                         // Stokes vector casted to double 
     #ifdef ALIS
      double dwsca, dwabs;                // General ALIS variables 
      #if ( defined(SPHERIQUE) || defined(ALT_PP) )
       double *tabCount2;                  // Specific ALIS counting array pointer for path implementation (cumulative distances)
      #endif
     #endif

    // Declaration for single
    #else                              
     float *tabCount; 
     #if ( defined(SPHERIQUE) || defined(ALT_PP) ) && defined(ALIS)
      float *tabCount2;
     #endif
    #endif

    #if ( defined(SPHERIQUE) || defined(ALT_PP) ) && defined(ALIS)
     float *tabCount3; // Specific ALIS counting array pointer for path implementation (distances histograms)
    #endif

    // We dont count UPTOA photons leaving in boxes outside SZA range
    if ((LEd==0) && (count_level==UPTOA) && (acosf(ph->v.z) > (SZA_MAXd*90./DEMIPI))) return;

    float theta = acosf(fmin(1.F, fmax(-1.F, ph->v.z)));

	float psi=0.;
	int ith=0, iphi=0, il=0, is=ph->is;
    float4 st; // replace s1, s2, s3, s4
    unsigned long long II, JJ, JJJ;


    if ((theta != 0.F) && (theta!= acosf(-1.F))) {
       ComputePsi(ph, &psi, theta);
    }
    else {
       if (LEd == 0) {
          atomicAdd(errorcount+ERROR_THETA, 1);
		  // Permet de visualiser les photons aux zenith dans le cas où il y a au moins un obj
		  #ifdef OBJ3D
		  return;
		  #endif
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
    rotationM(-psi,&L);
	ph->M = mul(ph->M,L);
    stback = mul(ph->M, stback);
    st = stback;
    /*float4 stforw = make_float4(0.5F, 0.5F, 0., 0.);
    rotationM(psi,&L);
	ph->Mf = mul(L,ph->Mf);
    stforw = mul(ph->Mf,stforw);
    st = stforw;*/
    #endif

	float weight = ph->weight;
    #ifdef ALIS
        float weight_sca[MAX_NLOW];
        for (int k=0; k<NLOWd; k++) {
            weight_sca[k] = ph->weight_sca[k];
        }
    #endif

	// Compute Box for outgoing photons in case of cone sampling
	if (LEd == 0) { 
        // if compute box returns 0, it excluded the photon (outside sun disc for example), so we dont count it
        if (!ComputeBox(&ith, &iphi, &il, ph, errorcount, count_level)) return;
    }


    // For virtual (LE) photons the direction is stored within photon structure
    // Moreover we compute also final attenuation for LE 
    else {
        ith = ph->ith;
        if (!ZIPd) iphi= ph->iph;
        il  = ph->ilam;

        if (!(   (SIMd==SURF_ONLY) 
              || (NATMd==0 && (count_level==UPTOA || count_level==UP0P)) 
              || (NOCEd==0 && count_level==UP0P)
             )
           ){

        // Computation of final attenutation only in fast PP
        #if !defined(SPHERIQUE) && !defined(ALT_PP)
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
            if ((ph->loc == ATMOS) || (ph->loc == SURF0M) || (ph->loc == SURF0P)
				#ifdef OBJ3D
				|| (ph->loc == OBJSURF)
				#endif
				) {
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
        // ph->layer_prev[ph->nevt] = ph->layer;
        if (ph->loc == ATMOS) ph->layer_prev[ph->nevt]   = ph->layer;
        if (ph->loc == OCEAN || ph->loc == SURF0M) ph->layer_prev[ph->nevt]   = -ph->layer;
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
        #endif // NOT SPHERIQUE && NOT ALT_PP
     } // SIMd  

    }   //LE
	
    float weight_irr = fabs(ph->v.z);
    // In Forward mode, and in case of spherical flux, update the weight
	if (FLUXd==2 && LEd==0 & weight_irr > 0.001f) weight /= weight_irr;
    //if (count_level == UPTOA && HORIZd == 0 && LEd == 1) weight *= weight_irr;
    //if (count_level == UPTOA && HORIZd == 0) weight *= weight_irr;
    if (FLUXd==3 && LEd==0) {
        float mu = sqrtf(2.F)/2.F;
        float3 tilted = make_float3(1.F, 0.F, 0.F);
        //float3 tilted = make_float3(0.F, 0.F, 1.F);
        if (dot(ph->v, tilted)>=0 ) weight=0.F;
    }

    II = NBTHETAd*NBPHId*NLAMd*NSENSORd;
    JJJ= NPSTKd*II;

    // Regular counting procedure
    #ifndef ALIS //=========================================================================================================
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (il >= 0) && (il < NLAMd) && (!isnan(weight)))
	{
      JJ = is*NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi;

      //int idx = blockIdx.x * blockDim.x + threadIdx.x;
      //if(idx==0 && count_level==UPTOA) printf("%g %g %g\n", weight, st.x, st.y, JJ);

      #ifdef DOUBLE 
      // select the appropriate level (count_level)
      tabCount = (double*)tabPhotons + count_level*JJJ;
      dweight = (double)weight;
      ds = make_double4(st.x, st.y, st.z, st.w);

	  #if __CUDA_ARCH__ >= 600
	  // If GTX 1000 or more recent use native double atomic add
      atomicAdd(tabCount+(0*II+JJ), dweight*(ds.x+ds.y));
      atomicAdd(tabCount+(1*II+JJ), dweight*(ds.x-ds.y));
      atomicAdd(tabCount+(2*II+JJ), dweight*ds.z);
      atomicAdd(tabCount+(3*II+JJ), dweight*ds.w);
      #else
      DatomicAdd(tabCount+(0*II+JJ), dweight*(ds.x+ds.y));
	  DatomicAdd(tabCount+(1*II+JJ), dweight*(ds.x-ds.y));
	  DatomicAdd(tabCount+(2*II+JJ), dweight*ds.z);
	  DatomicAdd(tabCount+(3*II+JJ), dweight*ds.w);
	  #endif

      #else
      tabCount = (float*)tabPhotons + count_level*JJJ;
      atomicAdd(tabCount+(0*II+JJ), weight * (st.x+st.y));
      atomicAdd(tabCount+(1*II+JJ), weight * (st.x-st.y));
      atomicAdd(tabCount+(2*II+JJ), weight * st.z);
      atomicAdd(tabCount+(3*II+JJ), weight * st.w);
      #endif

      atomicAdd(NPhotonsOut + (((count_level*NSENSORd + is)*NLAMd + il)*NBTHETAd + ith)*NBPHId + iphi, 1);
	}
	else
	{
	  atomicAdd(errorcount+ERROR_CASE, 1);
	}
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #else //ALIS ===========================================================================================================
    int DL=(NLAMd-1)/(NLOWd-1);
	if(((ith >= 0) && (ith < NBTHETAd)) && ((iphi >= 0) && (iphi < NBPHId)) && (!isnan(weight)))
    {
     if(HISTd==0) {
      // For all wavelengths
      for (il=0; il<NLAMd; il++) {
          float wabs = 1.0f;
          JJ = is*NBTHETAd*NBPHId*NLAMd + il*NBTHETAd*NBPHId + ith*NBPHId + iphi;

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
        
          #if !defined(SPHERIQUE) && !defined(ALT_PP)
          Profile *prof;
          // Computation of the absorption along photon history with heights and direction cosines 
          for (int n=0; n<ph->nevt; n++){
              //Computing absorption optical depths form start to stop for all segments
              float tau_abs1, tau_abs2;
              int ilayer, NL;
              if (ph->layer_prev[n+1] == 0) tau_abs2 = 0.;
              else {
               if (ph->layer_prev[n+1] < 0)  { 
                  prof = prof_oc;
                  ilayer=-ph->layer_prev[n+1];
                  NL = NOCEd+1;
               }
               if (ph->layer_prev[n+1] > 0)  { 
                  prof = prof_atm;
                  ilayer=ph->layer_prev[n+1];
                  NL = NATMd+1;
               }
                tau_abs2 = (prof[ilayer   + il *NL].OD_abs -
                               prof[ilayer-1 + il *NL].OD_abs) *
                               ph->epsilon_prev[n+1] + prof[ilayer-1 + il *NL].OD_abs;
              }


              if (ph->layer_prev[n]   == 0) tau_abs1 = 0.;
              else {
               if (ph->layer_prev[n] < 0)  { 
                  prof = prof_oc;
                  ilayer=-ph->layer_prev[n];
                  NL = NOCEd+1;
               }
               if (ph->layer_prev[n] > 0)  { 
                  prof = prof_atm;
                  ilayer=ph->layer_prev[n];
                  NL = NATMd+1;
               }
               tau_abs1 = (prof[ilayer   + il *NL].OD_abs -
                               prof[ilayer-1 + il *NL].OD_abs) *
                               ph->epsilon_prev[n+1] + prof[ilayer-1 + il *NL].OD_abs;
              }

              wabs *= exp(-fabs(__fdividef(tau_abs2 - tau_abs1 , ph->vz_prev[n+1])));
          }

          #else // fast 1D

          // Computation of the absorption along photon history with cumulative distances in layers
          wabs = 0.F;

          #ifndef OPT3D // in 1D cumulative absorption OD
          for (int n=0; n<NATMd; n++){
              wabs += abs(__fdividef(prof_atm[n+1   + il*(NATMd+1)].OD_abs -
                                     prof_atm[n + il*(NATMd+1)].OD_abs,
                                     prof_atm[n+1].z  - prof_atm[n].z) ) * ph->cdist_atm[n+1];
          }
          for (int n=0; n<NOCEd; n++){
              wabs += abs(__fdividef(prof_oc[n+1   + il*(NOCEd+1)].OD_abs -
                                     prof_oc[n + il*(NOCEd+1)].OD_abs,
                                     prof_oc[n+1].z  - prof_oc[n].z) ) * ph->cdist_oc[n+1];
          }

          #else // in 3D absorption coefficient
          /*for (int n=0; n<(NATMd); n++){
              wabs += prof_atm[n+1 + il*(NATMd+1)].OD_abs * ph->cdist_atm[n];
          }*/
          for (int n=0; n<(NATM_ABSd); n++){
              wabs += prof_atm[n+1 + il*(NATMd+1)].OD_abs * ph->cdist_atm[n];
          }
          /*for (int n=0; n<(NOCEd); n++){
              wabs += prof_oc[ n+1 + il*(NOCEd+1)].OD_abs * ph->cdist_oc[n];
          }*/
          for (int n=0; n<(NOCE_ABSd); n++){
              wabs += prof_oc[ n+1 + il*(NOCEd+1)].OD_abs * ph->cdist_oc[n];
          }
          #endif

          wabs = exp(-wabs);

          #endif // alt 1D

          // reflection on surface (main)
          if (ph->nref !=0 && spectrum[ph->ilam].alb_surface != 0.F) 
              wabs *= pow(__fdividef(spectrum[il].alb_surface,
                                 spectrum[ph->ilam].alb_surface),
                      (float)ph->nref);
          // reflection on surface (environment)
          if (ph->nenv !=0 && spectrum[ph->ilam].alb_env != 0.F) 
              wabs *= pow(__fdividef(spectrum[il].alb_env,
                                 spectrum[ph->ilam].alb_env),
                      (float)ph->nenv);
          // reflection on seafloor
          if (ph->nsfl !=0 && spectrum[ph->ilam].alb_seafloor != 0.F) 
              wabs *= pow(__fdividef(spectrum[il].alb_seafloor,
                                 spectrum[ph->ilam].alb_seafloor),
                      (float)ph->nsfl);

          #ifdef DOUBLE 
          tabCount = (double*)tabPhotons + count_level*JJJ;
          dweight = (double)weight;
          ds = make_double4(st.x, st.y, st.z, st.w);
          dwsca=(double)wsca;
          dwabs=(double)wabs;

		  #if __CUDA_ARCH__ >= 600
          atomicAdd(tabCount+(0*II+JJ), dweight * dwsca * dwabs * (ds.x+ds.y));
          atomicAdd(tabCount+(1*II+JJ), dweight * dwsca * dwabs * (ds.x-ds.y));
		  atomicAdd(tabCount+(2*II+JJ), dweight * dwsca * dwabs * ds.z);
          atomicAdd(tabCount+(3*II+JJ), dweight * dwsca * dwabs * ds.w);
		  #else
		  // If GTX 1000 or more recent use native double atomic add
          DatomicAdd(tabCount+(0*II+JJ), dweight * dwsca * dwabs * (ds.x+ds.y));
          DatomicAdd(tabCount+(1*II+JJ), dweight * dwsca * dwabs * (ds.x-ds.y));
          DatomicAdd(tabCount+(2*II+JJ), dweight * dwsca * dwabs * ds.z);
          DatomicAdd(tabCount+(3*II+JJ), dweight * dwsca * dwabs * ds.w);
		  #endif		  

          #else
          tabCount = (float*)tabPhotons + count_level*JJJ;
          atomicAdd(tabCount+(0*II+JJ), weight * wsca * wabs * (st.x+st.y));
          atomicAdd(tabCount+(1*II+JJ), weight * wsca * wabs * (st.x-st.y));
          atomicAdd(tabCount+(2*II+JJ), weight * wsca * wabs * st.z);
          atomicAdd(tabCount+(3*II+JJ), weight * wsca * wabs * st.w);
          #endif    

          atomicAdd(NPhotonsOut + (((count_level*NSENSORd +is)*NLAMd + il)*NBTHETAd + ith)*NBPHId + iphi, 1);
      } // wavelength loop 
     } //  if HISTd==0

     #if ( defined(SPHERIQUE) || defined(ALT_PP) )
     unsigned long long K   = NBTHETAd*NBPHId*NSENSORd; /* number of potential output per photon*/
     unsigned long long LL;
     if (HISTd==1 && count_level==UPTOA) { // Histories stored for absorption computation afterward (only spherical or alt_pp)
          //int idx = blockIdx.x * blockDim.x + threadIdx.x;
          unsigned long long counter2;
          counter2=atomicAdd(NPhotonsOut, 1);
          if (counter2 >= MAX_HIST) return;
          unsigned long long KK2 = NATM_ABSd+NOCE_ABSd+4+NLOWd+6; /* Number of information per local estmate photon (Record length)*/
          //unsigned long long KK2 = K*(NATM_ABSd+NOCE_ABSd+4+NLOWd+5); /* Number of information per local estmate photon (Record length)*/
          //unsigned long long KK2 = K*(NATMd+NOCEd+4+NLOWd+5); /* Number of information per local estmate photon (Record length)*/
          //unsigned long long KKK2= KK2 * MAX_HIST; /* Number of individual information per vertical Level (Number of Records)*/
          unsigned long long LL2;
          tabCount3   = (float*)tabHist     ; /* we position the pointer at the good vertical level*/
          //tabCount3   = (float*)tabHist     + count_level*KKK2; /* we position the pointer at the good vertical level*/
          for (int n=0; n<NOCE_ABSd; n++){
                /* The offset is the number of previous writing (counter2) * Record Length
                   + the offset of the individual information  + the place of the physical quantity */
                LL2 = counter2*KK2 +  n;
                //LL2 = counter2*KK2 +  n*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
                tabCount3[LL2]= ph->cdist_oc[n+1];
          }
          //for (int n=0; n<NATMd; n++){
          for (int n=0; n<NATM_ABSd; n++){
                LL2 = counter2*KK2 +  n+NOCE_ABSd;
                //LL2 = counter2*KK2 +  (n+NOCE_ABSd)*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
                tabCount3[LL2]= ph->cdist_atm[n+1];
          }
          LL2 = counter2*KK2 +  NATM_ABSd+NOCE_ABSd+0;
          //LL2 = counter2*KK2 +  (NATM_ABSd+NOCE_ABSd+0)*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
          tabCount3[LL2]= weight * (st.x+st.y);
          LL2 = counter2*KK2 +  NATM_ABSd+NOCE_ABSd+1;
          tabCount3[LL2]= weight * (st.x-st.y);
          LL2 = counter2*KK2 +  NATM_ABSd+NOCE_ABSd+2;
          tabCount3[LL2]= weight * (st.z);
          LL2 = counter2*KK2 +  NATM_ABSd+NOCE_ABSd+3;
          tabCount3[LL2]= weight * (st.w);

          for (int n=0; n<NLOWd; n++){
                LL2 = counter2*KK2 + n+NATM_ABSd+NOCE_ABSd+4;
                tabCount3[LL2]= weight_sca[n];
          }
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4;
          tabCount3[LL2]= (float)(ph->nrrs>=1);
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4+1;
          tabCount3[LL2]= (float)(ph->nref);
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4+2;
          tabCount3[LL2]= (float)(ph->nsif);
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4+3;
          tabCount3[LL2]= (float)(ph->nvrs>=1);
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4+4;
          tabCount3[LL2]= (float)(ph->nenv);
          LL2 = counter2*KK2 +  NLOWd+NATM_ABSd+NOCE_ABSd+4+5;
          tabCount3[LL2]= (float)(ph->ith);
       } // HISTd==1

       unsigned long long KK  = K*(NATM_ABSd+NOCE_ABSd);
       #ifdef DOUBLE
          tabCount2   = (double*)tabDist     + count_level*KK;
          for (int n=0; n<NOCE_ABSd; n++){
            LL = n*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
            #if __CUDA_ARCH__ >= 600
            atomicAdd(tabCount2+LL, (double)ph->cdist_oc[n+1]);
            #else
            DatomicAdd(tabCount2+LL, (double)ph->cdist_oc[n+1]);
            #endif
          }
          for (int n=0; n<NATM_ABSd; n++){
            LL = (n+NOCE_ABSd)*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
            #if __CUDA_ARCH__ >= 600
            atomicAdd(tabCount2+LL, (double)ph->cdist_atm[n+1]);
            #else
            DatomicAdd(tabCount2+LL, (double)ph->cdist_atm[n+1]);
            #endif
          }
       #else
          tabCount2   = (float*)tabDist     + count_level*KK;
          for (int n=0; n<NOCE_ABSd; n++){
            LL = n*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
            atomicAdd(tabCount2+LL, ph->cdist_oc[n+1]);
          }
          for (int n=0; n<NATM_ABSd; n++){
            LL = (n+NOCE_ABSd)*K + is*NBPHId*NBTHETAd + ith*NBPHId + iphi;
            atomicAdd(tabCount2+LL, ph->cdist_atm[n+1]);
          }
       #endif 
      #endif // SPHERIQUE or ALT_PP

    } // correct output box
	else
	{
		atomicAdd(errorcount+ERROR_CASE, 1);
	}
    #endif //ALIS
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

// Rotation Matrix of angle theta around unit vector u
__device__ float3x3 rotation3D(float theta, float3 u)
{
    // Rodrigues rotation formula
    float ct=cosf(theta);
    float st=sinf(theta);
    float3x3 A, B, C, R;
    A = make_diag_float3x3(1.F);
    B = make_float3x3( 0.F,-u.z, u.y,
                       u.z, 0.F,-u.x,
                      -u.y, u.x, 0.F
                     );
    C = mul(B, B);
    R = add(add(A, mul(B, st)), mul(C, 1.F-ct)); 

    return R;
}

// Rotation Matrix of angle theta around unit vector u
__device__ double3x3 rotation3D(double theta, double3 u)
{
    // Rodrigues rotation formula
    double ct=cos(theta);
    double st=sin(theta);
    double3x3 I, K, K2, R;
    I = make_double3x3(1., 0., 0.,
                       0., 1., 0.,
                       0., 0., 1.
                      );
    K = make_double3x3(0.,-u.z, u.y,
                       u.z, 0.,-u.x,
                      -u.y, u.x, 0.
                      );
    K2 = mul(K, K);
    R  = add(add(I, mul(K, st)), mul(K2, 1.-ct));

    return R;
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
        float eps=1e-2;
        //float eps=1e-4;

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
*/
__device__ int ComputeBox(int* ith, int* iphi, int* il,
                           Photon* photon, unsigned long long *errorcount, int count_level)
{
	// vxy est la projection du vecteur vitesse du photon sur (x,y)
	float vxy = sqrtf(photon->v.x * photon->v.x + photon->v.y * photon->v.y);

	// Calcul de la valeur de ithv
	// _rn correspond à round to the nearest integer
    #ifndef SPHERIQUE
	*ith = __float2int_rd(__fdividef(acosf(fabsf(photon->v.z)) * NBTHETAd, DEMIPI));
    #else
    //if (count_level==UPTOA) *ith = __float2int_rd(__fdividef((acosf(photon->v.z) + (90.F-SZA_MAXd)*DEMIPI/90.) * NBTHETAd, SZA_MAXd/90.*DEMIPI));
    if (count_level==UPTOA) *ith = __float2int_rd(__fdividef((acosf(photon->v.z) - 0.F)                            * NBTHETAd, SZA_MAXd/90.*DEMIPI));
    else                    *ith = __float2int_rd(__fdividef( acosf(fabsf(photon->v.z)) * NBTHETAd, DEMIPI));
    #endif


	// Calcul de la valeur de il
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
    if (SUN_DISCd <= 0) return 1;	

    float phi = *iphi * __fdividef(2.F*PI, NBPHId);
    float dth =  __fdividef(SZA_MAXd * PI, 180.F * NBTHETAd);
    float cth = cosf((*ith + 0.5F) * dth);
    float sth = sqrtf(1.F - cth*cth);
    float3 center_dir = make_float3(cosf(phi)*sth, sinf(phi)*sth, cth);
    if ((abs(acosf(dot(photon->v, center_dir)))*180.F/PI) >  SUN_DISCd ) {

        //int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x * blockDim.y + (threadIdx.x * blockDim.y + threadIdx.y);
        //if (idx==0 && count_level==UPTOA) printf("center %f %f %f, phot %f %f %f \n", center_dir.x, center_dir.y, center_dir.z, photon->v.x, photon->v.y, photon->v.z);
        return 0;
    }
    return 1;
}

#ifdef VERBOSE_PHOTON
__device__ void display(const char* desc, Photon* ph) {
    //
    // display the status of the photon (only for thread 0)
    //
	int idx = blockIdx.x *blockDim.x + threadIdx.x;

    if (idx == 0) {
		
		printf("%16s %4i X=(%9.4f,%9.4f,%9.4f) V=(%6.3f,%6.3f,%6.3f) U=(%6.3f,%6.3f,%6.3f) S=(%6.3f,%6.3f,%6.3f,%6.3f) tau=%8.3f tau_abs=%8.3f wvl=%6.3f weight=%11.3e scat=%4i",
               desc,
			   ph->nint,
               ph->pos.x, ph->pos.y, ph->pos.z,
               ph->v.x,ph->v.y,ph->v.z,
			   ph->u.x,ph->u.y,ph->u.z,
               ph->stokes.x, ph->stokes.y,
               ph->stokes.z, ph->stokes.w,
               ph->tau,ph->tau_abs, ph->wavel, ph->weight, ph->scatterer
               );

        switch(ph->scatterer) {
            case -1: printf("scatterer =   UNDEF"); break;
            case 0: printf("scatterer =     RAY"); break;
            case 1: printf("scatterer =   PTCLE"); break;
            case 2: printf("scatterer = CHLFLUO"); break;
            case 3: printf("scatterer = VRS"); break;
            case 4: printf("scatterer = THERMAL_EM"); break;
            default:
                    printf("scatterer =   UNDEF");
        }
        switch(ph->loc) {
            case 0: printf(" loc=   SPACE"); break;
            case 1: printf(" loc=   ATMOS"); break;
            case 2: printf(" loc=  SURF0P"); break;
            case 3: printf(" loc=  SURF0M"); break;
            case 4: printf(" loc=ABSORBED"); break;
            case 5: printf(" loc=    NONE"); break;
            case 6: printf(" loc=   OCEAN"); break;
            case 7: printf(" loc=SEAFLOOR"); break;
            case 8: printf(" loc= OBJSURF"); break;
		    case 9: printf(" loc= REMOVED"); break;
		    case 10: printf("loc= SOURCE"); break;
            default:
                    printf(" loc=UNDEFINED");
        }
        #ifdef ALIS
        printf(" wsca=");
        for (int k=0; k<NLOWd; k++) printf("%7.5f ",ph->weight_sca[k]);
        #if !defined(ALT_PP) && !defined(SPHERIQUE)
        printf(" nevt=%2d",ph->nevt);
        printf(" dtausca=");
        for (int k=0; k<NLOWd; k++) printf("%7.5f ",ph->tau_sca[k]);
        printf(" layers=");
        for (int k=0; k<ph->nevt+1; k++) printf("%3d ",ph->layer_prev[k]);
        printf(" vz=");
        for (int k=0; k<ph->nevt+1; k++) printf("%7.5f ",ph->vz_prev[k]);
        printf(" delta=");
        for (int k=0; k<ph->nevt+1; k++) printf("%7.5f ",ph->epsilon_prev[k]);
        #endif
        #ifdef SIF
        switch(ph->emitter) {
            case -1: printf("emitter =   UNDEF"); break;
            case 0: printf("emitter =     SIF"); break;
            case 1: printf("emitter =   SOLAR_REF"); break;
            default:
                    printf("emitter =   UNDEF");
        }
        #endif
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
	//float EPS6 = 1e-5;	
	float EPS6 = 1e-4;	
	float3 w0, w1;

	// compute former w
	w0 = cross(u0, v0); // w : cross product entre l'ancien vec u et l'ancien vec v du photon
	w1 = cross(v1, v0);	// compute the normal to the new scattering plan i.e. new w vector

	den = length(w1); // Euclidean length also called L2-norm
	if (den < EPS6) {
		prod_scal =  dot(v0, v1);
		/*if (prod_scal < 0.0)*/
		if (prod_scal > 0.0)
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

#ifdef OBJ3D
__device__ void copyIGeo(IGeo* strucG, IGeo* strucG_le)
{
    strucG_le->normal = strucG->normal;
    strucG_le->normalBase = strucG->normalBase;
    strucG_le->material = strucG->material;
    strucG_le->reflectivity = strucG->reflectivity;
    strucG_le->roughness = strucG->roughness;
    strucG_le->shadow = strucG->shadow;
    strucG_le->nind = strucG->nind;
    strucG_le->dist = strucG->dist;
    strucG_le->mvTF = strucG->mvTF;
    strucG_le->type = strucG->type;
    strucG_le->mvR = strucG->mvR;
}
#endif

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
	ph_le->scatterer=ph->scatterer;
	ph_le->nrrs=ph->nrrs;
	ph_le->nvrs=ph->nvrs;
    ph_le->pos = ph->pos; // float3
    ph_le->nint = ph->nint;
    ph_le->nref = ph->nref;
    ph_le->nsfl = ph->nsfl;
    ph_le->nenv = ph->nenv;
    ph_le->env = ph->env;
    ph_le->is = ph->is;
    ph_le->iph = ph->iph;
    ph_le->ith = ph->ith;
    #ifdef SPHERIQUE
    ph_le->radius = ph->radius;
    #endif

    #ifdef ALIS
    int k; 
    #if !defined(ALT_PP) && !defined(SPHERIQUE)
    int kmax=ph->nevt+1;
    ph_le->nevt = ph->nevt;
    for (k=0; k<kmax; k++) ph_le->layer_prev[k] = ph->layer_prev[k];
    for (k=0; k<kmax; k++) ph_le->vz_prev[k] = ph->vz_prev[k];
    for (k=0; k<kmax; k++) ph_le->epsilon_prev[k] = ph->epsilon_prev[k];
    for (k=0; k<NLOWd; k++) ph_le->tau_sca[k] = ph->tau_sca[k];
    #else
    for (k=0; k<(NATM_ABSd+1); k++) ph_le->cdist_atm[k] = ph->cdist_atm[k];
    //for (k=0; k<(NATMd+1); k++) ph_le->cdist_atm[k] = ph->cdist_atm[k];
    //for (k=0; k<(NOCEd+1); k++) ph_le->cdist_oc[k]  = ph->cdist_oc[k];
    for (k=0; k<(NOCE_ABSd+1); k++) ph_le->cdist_oc[k]  = ph->cdist_oc[k];
    #endif
    for (k=0; k<NLOWd; k++) ph_le->weight_sca[k] = ph->weight_sca[k];
    ph_le->nsif = ph->nsif;
    #ifdef SIF
    ph_le->emitter = ph->emitter;
    #endif
    #endif

    #ifdef BACK
    int ii,jj;
    for (ii=0; ii<4; ii++){
        for (jj=0; jj<4; jj++) {
            ph_le->M[ii][jj] = ph->M[ii][jj];
        }
    }
    /*for (ii=0; ii<4; ii++){
        for (jj=0; jj<4; jj++) {
            ph_le->Mf[ii][jj] = ph->Mf[ii][jj];
        }
    }*/
    #endif
	
	#ifdef OBJ3D
	ph_le->direct = ph->direct;
	ph_le->H = ph->H; ph_le->E = ph->E; ph_le->S = ph->S;
	for (int k=0; k<4; k++) ph_le->weight_loss[k] = ph->weight_loss[k];
	#endif
	
    #if defined(BACK) && defined(OBJ3D)
	ph_le->posIni = ph->posIni;
    #endif
}

__device__ float get_OD(int BEERd, struct Profile prof) {  
    if (BEERd == 1) return prof.OD_sca;
    else            return prof.OD;
}


__device__ float Lambda(float avz, float sig) {
    float l;
    if (avz == 1.F) l = 0.F;
    else {
        float nu = __fdividef(1.F, tanf(acosf(avz))*(sqrtf(2.) * sig));
        l = __fdividef(__expf(-nu*nu) - nu * sqrtf(PI) * erfcf(nu),2.F * nu * sqrtf(PI));
    }
    return l;
}

__device__ float G1B(float alpha, float tanTheta) {
	// approx proposed by Walter et al. 2007 for Beckmann dist
	float a, a2;
	a = __fdividef(1.F, alpha*tanTheta);
	a2 = a*a;

	if (a >= 1.6f) return 1.F;
	return ( (3.535F*a + 2.181F*a2)/(1 + 2.276F*a + 2.577F*a2) );
}

__device__ float G1GGX(float alpha2, float tan2Theta) {
	// proposed by Walter et al. 2007 for GGX dist
	return fdividef(2, 1 + sqrtf(1 + alpha2*tan2Theta ));
}

__device__ float LambB(float alpha, float tanTheta) {
	// Beckmann Lambda approx proposed by Eric Heitz 2014
	float a, a2;
	a = __fdividef(1.F, alpha*fabsf(tanTheta));
	a2 = a*a;

	if (a >= 1.6F) return 0.F;
	return (1 - 1.259F*a + 0.396F*a2) / (3.535F*a + 2.181F*a2);
}

__device__ float LambdaM(float avz, float sig2) {
    // Mischenko implementation
    float l;
    if (avz == 1.F) l = 0.;
    else {
        float s1,s2,s3,xi,xxi,dcot,t1,t2;
        s1 = __fsqrt_rn(2.*sig2/PI);
        s3 = __frcp_rn(__fsqrt_rn(2.*sig2));
        s2 = s3*s3;
        xi = avz;
        xxi=xi*xi;
        dcot =  xi *__frcp_rn(__fsqrt_rn(1.-xxi));
        t1 = exp(-dcot*dcot*s2);
        t2 = erfc(dcot*s3);
        l  = 0.5*(s1*t1/dcot-t2);
    }
    return l;
}

__device__ void DirectionToUV(float th, float phi, float3* v, float3* u) {
     *v = make_float3(cosf(phi) * sinf(th),
                      sinf(phi) * sinf(th),
                      cosf(th));  
	 *u = make_float3(cosf(phi) * cosf(th),
	                  sinf(phi) * cosf(th),
	                  -sinf(th));
}

__device__ float3 LocalToGlobal(float3 Nx, float3 Ny, float3 Nz, float3 v) {
     float3x3 B = make_float3x3(
                Nx.x, Ny.x, Nz.x,
                Nx.y, Ny.y, Nz.y,
                Nx.z, Ny.z, Nz.z
                );
     return mul(B, v);
}

__device__ float3 GlobalToLocal(float3 Nx, float3 Ny, float3 Nz, float3 v) {
     float3x3 B = make_float3x3(
                Nx.x, Nx.y, Nx.z,
                Ny.x, Ny.y, Ny.z,
                Nz.x, Nz.y, Nz.z
                );
     return mul(B, v);
}

__device__ void MakeLocalFrame(float3 pos, float3* Nx, float3* Ny, float3* Nz) {
	*Nz = normalize(pos); // Nz is the vertical at the impact point
	*Ny = normalize(cross(*Nz, make_float3(1.0, 0.0, 0.0)));
	*Nx = normalize(cross(*Ny, *Nz));
}


/*Functions for RRS*/

//Bates, Planel. Space Sa., Vol.32, No.6, pp. 785-790. 1984 
__device__ float Fk_N2(float lam){
    /*
    lam in nm
    */
    return 1.034 + __fdividef(3.170*1e-4, lam*lam*1e-6);
}

__device__ float Epsilon_N2(float lam){
    /*
    lam in nm
    */
    return (Fk_N2(lam) -1.F) * 4.5;;
}

__device__ float Fk_O2(float lam){
    /*
    lam in nm
    */
    return 1.096 + __fdividef(1.385*1e-3, lam*lam*1e-6)  
                 + __fdividef(1.448*1e-4, lam*lam*lam*lam*1e-12) ;
}

__device__ float Epsilon_O2(float lam){
    /*
    lam in nm
    */
    return (Fk_O2(lam) -1.F) * 4.5;;
}

__device__ float Epsilon_air(float lam){
    /*
    lam in nm
    */
    return Epsilon_N2(lam) * X_N2 + Epsilon_O2(lam) * X_O2;
}

// Kattawar, Astrophysical Journalo, Part 1, vol. 243, Feb. 1, 1981, p. 1049-1057.
__device__ float fRRS_air(float lam, float theta){
    /*
    lam in nm
    theta in deg
    */
    float eps = Epsilon_air(lam);
    float ct  = cosf(theta*DEUXPI/360.);
    float num = (180.+13.*eps) + (180.+eps)*ct*ct;
    float den = (180.+52.*eps) + (180.+4.*eps)*ct*ct;
    return __fdividef(num, den);
}


// Zhai et al., 2015, Optics Express. , ratio of b_VRS over bw
__device__ float fVRS(float lam){
    /*
    lam in nm
    */
    float num = 2.7e-4 * pow(488.F/lam, 5.3F);
    float den = 1.93e-3 * pow(550.F/lam, 4.32F);
    return __fdividef(num, den);
}

/*---------------*/

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
#endif

#if defined(DOUBLE) && !(__CUDA_ARCH__ >= 600)
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


/**********************************************************
*	> Functions linked to the creation of geometries
***********************************************************/
#ifdef OBJ3D
/* geoTest
* Check if there is an intersection with at least an object, in case with intersections with
* several objects return the intersection information of the object with the smallest traveled distance
*/
__device__ bool geoTest(float3 o, float3 dir, float3* phit, IGeo *GeoV, struct IObjets *ObjT, struct GObj *myGObj)
{
	// Initialization of the ray for the intersection study
	Ray R1(o, dir, 0.00025); // 0.0001 -> ray begin 10cm further in direction "dir"

	// ******************interval of study******************
	BBox interval(make_float3(Pmin_x-VALMIN, Pmin_y-VALMIN, Pmin_z-VALMIN),
				  make_float3(Pmax_x+VALMIN, Pmax_y+VALMIN, Pmax_z+VALMIN));
	if (!interval.IntersectP(R1))
		return false;
	// *****************************************************
	
	// ***************common to all objects*****************
	float myT = CUDART_INF_F; // myT = time
	bool myB = false;
	DifferentialGeometry myDg;
	float3 tempPhit; // Temporary variable of Phit
    // *****************************************************
	
	// *************Specific to plane objects***************
	int vi[6] = {0, 1, 2,  // vertices index for triangle 1
				 2, 3, 1}; // vertices index for triangle 2
	// *****************************************************

	for (int i = 0; i < nGObj; ++i)
	{	
		int IND = myGObj[i].index;  // Index
		int NE  = myGObj[i].nObj;   // Number of entity in the group i

		// We test firstly the bounding box of the group
		BBox bboxG(make_float3(myGObj[i].bPminx, myGObj[i].bPminy, myGObj[i].bPminz),
				   make_float3(myGObj[i].bPmaxx, myGObj[i].bPmaxy, myGObj[i].bPmaxz));

			
		// If the test with bboxG is ok then perform intersection test with all the obj inside
		if (bboxG.IntersectP(R1))
		{
			for (int j = 0; j < NE; ++j)
			{
				float myTj = CUDART_INF_F;
				bool myBj = false;
				DifferentialGeometry myDgj;
				// *****************************First Step********************************
				// Consider all the transformation of object (j)
				Transform Tj, invTj; // Declaration of the tranform and its inverse

				/* !!! We note that it is crucial to begin with the translation because if there
				   is a rotation then the coordinate system change (x or y or z axis) !!! */

				// If a value in x, y or z is diff of 0 then there is a translation
				if ( (ObjT[IND+j].mvTx>VALMIN and ObjT[IND+j].mvTx<-VALMIN) or
					 (ObjT[IND+j].mvTy>VALMIN and ObjT[IND+j].mvTy>-VALMIN) or
					 (ObjT[IND+j].mvTz>VALMIN and ObjT[IND+j].mvTz>-VALMIN)) {
					Transform TmT;
					TmT = Tj.Translate(make_float3(ObjT[IND+j].mvTx, ObjT[IND+j].mvTy,
												   ObjT[IND+j].mvTz));
					Tj = TmT; }

				// Add rotation tranformations
				Tj = addRotAndParseOrder(Tj, ObjT[IND+j]); //see the function
				invTj = Tj.Inverse(Tj); // inverse of the transform

				// ******************************Second Step******************************
				// See if there is an intersection with object(j)
				if (ObjT[IND+j].geo == 1) // Case with a spherical object
				{
					Sphere myObject(&Tj, &invTj, ObjT[IND+j].myRad, ObjT[IND+j].z0,
									ObjT[IND+j].z1, ObjT[IND+j].phi);
		
					BBox myBBox = myObject.WorldBoundSphere();

					if (myBBox.IntersectP(R1))
						myBj = myObject.Intersect(R1, &myTj, &myDgj);
				}
				else if (ObjT[IND+j].geo == 2) // Case with a plane object
				{
					// declaration of a table of float3 which contains P0, P1, P2, P3
					float3 Pvec[4] = {make_float3(ObjT[IND+j].p0x, ObjT[IND+j].p0y, ObjT[IND+j].p0z),
									  make_float3(ObjT[IND+j].p1x, ObjT[IND+j].p1y, ObjT[IND+j].p1z),
									  make_float3(ObjT[IND+j].p2x, ObjT[IND+j].p2y, ObjT[IND+j].p2z),
									  make_float3(ObjT[IND+j].p3x, ObjT[IND+j].p3y, ObjT[IND+j].p3z)};
					
					// Create the triangleMesh (2 = number of triangle ; 4 = number of vertices)
					TriangleMesh myObject(&Tj, &invTj, 2, 4, vi, Pvec);
				
					BBox myBBox = myObject.WorldBoundTriangleMesh();
					if (myBBox.IntersectP(R1))
						myBj = myObject.Intersect2(R1, &myTj, &myDgj);
					if(myBj)
						myDgj.nn = faceForward(make_float3(ObjT[IND+j].nBx, ObjT[IND+j].nBy, ObjT[IND+j].nBz), -1.*R1.d);
				}

				// ******************************third Step*******************************
				// If there are intersection with several objects -> this insure that we
				// keep the nearest object from the initial point of the photon
				if (myBj & (myT > myTj))
				{
					tempPhit = R1(myTj);
					myB = true;
					myT = myTj;
					myDg = myDgj;
					GeoV->normal = faceForward(myDg.nn, -1.*R1.d);
					GeoV->normalBase = make_float3(ObjT[IND+j].nBx, ObjT[IND+j].nBy, ObjT[IND+j].nBz);
					if(  isBackward( make_double3(GeoV->normalBase.x, GeoV->normalBase.y, GeoV->normalBase.z),
									 make_double3(dir.x, dir.y, dir.z) )  )
					{
						GeoV->material = ObjT[IND+j].materialAV;
						GeoV->reflectivity = ObjT[IND+j].reflectAV;
						GeoV->roughness = ObjT[IND+j].roughAV;
						GeoV->shadow = ObjT[IND+j].shdAV;
						GeoV->nind = ObjT[IND+j].nindAV;
						GeoV->dist = ObjT[IND+j].distAV;
					}
					else
					{
						GeoV->material = ObjT[IND+j].materialAR; //AR
						GeoV->reflectivity = ObjT[IND+j].reflectAR;
						GeoV->roughness = ObjT[IND+j].roughAR;
						GeoV->shadow = ObjT[IND+j].shdAR;
						GeoV->nind = ObjT[IND+j].nindAR;
						GeoV->dist = ObjT[IND+j].distAR;
					}
					*(phit) = tempPhit;
					GeoV->mvTF = Tj;
					GeoV->type = ObjT[IND+j].type;
					GeoV->mvR = make_float3(ObjT[IND+j].mvRx, ObjT[IND+j].mvRy, ObjT[IND+j].mvRz);
				}
			} // END FOR j LOOP (traveling entity objects)
		} // END intersection test with the group bounding box
		// ***********************************************************************
	} // END FOR i LOOP (traveling object groups)
	if (myB) return true; // If there is an intersection with at least one object
	else return false; // If there is not a single intersection
} // END OF THE FUNCTION geoTest()

__device__ bool geoTestMir(float3 o, float3 dir, struct IObjets *ObjT, struct GObj *myGObj)
{
	// Initialization of the ray for the intersection study
	Ray R1(o, dir, 0.00025); // 0.0001 -> ray begin 10cm further in direction "dir"
	
	// *************Specific to plane objects***************
	int vi[6] = {0, 1, 2,  // vertices index for triangle 1
				 2, 3, 1}; // vertices index for triangle 2
	// *****************************************************
	
	for (int i = 0; i < nGObj; ++i)
	{	
		int IND = myGObj[i].index;  // Index
		int NE  = myGObj[i].nObj;   // Number of entity in the group i

		// We test firstly the bounding box of the group
		BBox bboxG(make_float3(myGObj[i].bPminx, myGObj[i].bPminy, myGObj[i].bPminz),
				   make_float3(myGObj[i].bPmaxx, myGObj[i].bPmaxy, myGObj[i].bPmaxz));
		
		// If the test with bboxG is ok then perform intersection test with all the obj inside
		if (bboxG.IntersectP(R1))
		{
			for (int j = 0; j < NE; ++j)
			{
				bool myBj = false;
				// *****************************First Step********************************
				// Consider all the transformation of object (j)
				Transform Tj, invTj; // Declaration of the tranform and its inverse

				/* !!! We note that it is crucial to begin with the translation because if there
				   is a rotation then the coordinate system change (x or y or z axis) !!! */

				// If a value in x, y or z is diff of 0 then there is a translation
				if ( (ObjT[IND+j].mvTx>VALMIN and ObjT[IND+j].mvTx<-VALMIN) or
					 (ObjT[IND+j].mvTy>VALMIN and ObjT[IND+j].mvTy>-VALMIN) or
					 (ObjT[IND+j].mvTz>VALMIN and ObjT[IND+j].mvTz>-VALMIN)) {
					Transform TmT;
					TmT = Tj.Translate(make_float3(ObjT[IND+j].mvTx, ObjT[IND+j].mvTy,
												   ObjT[IND+j].mvTz));
					Tj = TmT; }

				// Add rotation tranformations
				Tj = addRotAndParseOrder(Tj, ObjT[IND+j]); //see the function
				invTj = Tj.Inverse(Tj); // inverse of the transform

				// ******************************Second Step******************************
				// See if there is an intersection with object(j)
			    if (ObjT[IND+j].geo == 2 and ObjT[IND+j].type == HELIOSTAT)
				{
					// Declaration of a table of float3 which contains P0, P1, P2, P3
					float3 Pvec[4] = {make_float3(ObjT[IND+j].p0x, ObjT[IND+j].p0y, ObjT[IND+j].p0z),
									  make_float3(ObjT[IND+j].p1x, ObjT[IND+j].p1y, ObjT[IND+j].p1z),
									  make_float3(ObjT[IND+j].p2x, ObjT[IND+j].p2y, ObjT[IND+j].p2z),
									  make_float3(ObjT[IND+j].p3x, ObjT[IND+j].p3y, ObjT[IND+j].p3z)};
					
					// Create the triangleMesh (2 = number of triangle ; 4 = number of vertices)
					TriangleMesh myObject(&Tj, &invTj, 2, 4, vi, Pvec);
				
					BBox myBBox = myObject.WorldBoundTriangleMesh();
					if (myBBox.IntersectP(R1)) myBj = myObject.IntersectP2(R1);
					if(myBj) return true;
				}
			}// END FOR j LOOP
		}// END BBOX TEST
	}// END FOR i LOOP
	return false;
} // END OF THE FUNCTION geoTestMir()

__device__ bool geoTestRec(float3 o, float3 dir, struct IObjets *ObjT)
{
	// Initialization of the ray for the intersection study
	Ray R1(o, dir, 0.00025); // 0.0001 -> ray begin 10cm further in direction "dir"
	// ******************interval of study******************
	BBox interval(make_float3(Pmin_x, Pmin_y, Pmin_z),
				  make_float3(Pmax_x, Pmax_y, Pmax_z));
	
	if (!interval.IntersectP(R1))
		return false;
	// *****************************************************
	
	// *************Specific to plane objects***************
	int vi[6] = {0, 1, 2,  // vertices index for triangle 1
				 2, 3, 1}; // vertices index for triangle 2
	// *****************************************************

	for (int i = 0; i < nRObj; ++i)
	{
		bool myBi = false;
		// *****************************First Step********************************
		// Consider all the transformation of object (i)
		Transform Ti, invTi; // Declaration of the tranform and its inverse

		/* !!! We note that it is crucial to begin with the translation because if there
		   is a rotation then the coordinate system change (x or y or z axis) !!! */

		// If a value in x, y or z is diff of 0 then there is a translation
		if ( (ObjT[i].mvTx>VALMIN and ObjT[i].mvTx<-VALMIN) or
			 (ObjT[i].mvTy>VALMIN and ObjT[i].mvTy>-VALMIN) or
			 (ObjT[i].mvTz>VALMIN and ObjT[i].mvTz>-VALMIN)) {
			Transform TmT;
			TmT = Ti.Translate(make_float3(ObjT[i].mvTx, ObjT[i].mvTy,
										   ObjT[i].mvTz));
			Ti = TmT; }

		// Add rotation tranformations
		Ti = addRotAndParseOrder(Ti, ObjT[i]); //see the function
		invTi = Ti.Inverse(Ti); // inverse of the transform
		// ***********************************************************************
		
		// ******************************Second Step******************************
	    // See if there is an intersection with an object
		if (ObjT[i].geo == 2 and ObjT[i].type == RECEIVER)
		{
			// Declaration of a table of float3 which contains P0, P1, P2, P3
			float3 Pvec[4] = {make_float3(ObjT[i].p0x, ObjT[i].p0y, ObjT[i].p0z),
							  make_float3(ObjT[i].p1x, ObjT[i].p1y, ObjT[i].p1z),
							  make_float3(ObjT[i].p2x, ObjT[i].p2y, ObjT[i].p2z),
							  make_float3(ObjT[i].p3x, ObjT[i].p3y, ObjT[i].p3z)};
			
			// Create the triangleMesh (2 = number of triangle ; 4 = number of vertices)
			TriangleMesh myObject(&Ti, &invTi, 2, 4, vi, Pvec);
			
			BBox myBBox = myObject.WorldBoundTriangleMesh();
			if (myBBox.IntersectP(R1))
				myBi = myObject.IntersectP2(R1);	
			if (myBi) return true;
		}
		// ***********************************************************************
	} // END FOR LOOP
	return false;
} // END OF THE FUNCTION geoTestRec()

__device__ Transform addRotAndParseOrder(Transform Ti, IObjets object)
{
	// Add the rotation tranformartions + Consider the rotation order
	switch (object.rotOrder)
	{
	case XYZ:
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		break;
	case XZY:
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		break;
	case YXZ:
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		break;
	case YZX:
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
		break;
	case ZXY:
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		break;
	case ZYX:
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Ti = Ti*Ti.RotateZ(object.mvRz);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Ti = Ti*Ti.RotateY(object.mvRy);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Ti = Ti*Ti.RotateX(object.mvRx);
	default:
		break;
	}
	return Ti;
} // END OF THE FUNCTION addRotAndParseOrder()

__device__ Transformd DaddRotAndParseOrder(Transformd Tid, IObjets object)
{
	// Add the rotation tranformartions + Consider the rotation order
	switch (object.rotOrder)
	{
	case XYZ:
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		break;
	case XZY:
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		break;
	case YXZ:
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		break;
	case YZX:
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
		break;
	case ZXY:
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		break;
	case ZYX:
		if (object.mvRz != 0) // si diff de 0 alors il y a une rot en z
			Tid = Tid*Tid.RotateZ(object.mvRz);
		if (object.mvRy != 0) // si diff de 0 alors il y a une rot en y
			Tid = Tid*Tid.RotateY(object.mvRy);
		if (object.mvRx != 0) // si diff de 0 alors il y a une rot en x
			Tid = Tid*Tid.RotateX(object.mvRx);
	default:
		break;
	}
	return Tid;
} // END OF THE FUNCTION DaddRotAndParseOrder()
#endif

#ifdef OPT3D
__device__ void GetFaceMiddlePoint(int ind, float3 pmin, float3 pmax, float3 *p)
{
    if (ind==0) *p = make_float3(pmax.x, (pmin.y+pmax.y)/2., (pmin.z+pmax.z)/2.);
    if (ind==1) *p = make_float3(pmin.x, (pmin.y+pmax.y)/2., (pmin.z+pmax.z)/2.);
    if (ind==2) *p = make_float3((pmin.x+pmax.x)/2., pmax.y, (pmin.z+pmax.z)/2.);
    if (ind==3) *p = make_float3((pmin.x+pmax.x)/2., pmin.y, (pmin.z+pmax.z)/2.);
    if (ind==4) *p = make_float3((pmin.x+pmax.x)/2., (pmin.y+pmax.y)/2., pmax.z);
    if (ind==5) *p = make_float3((pmin.x+pmax.x)/2., (pmin.y+pmax.y)/2., pmin.z);
}



__device__ void GetFaceIndex(float3 pos, int *index)
// en.wikipedia.org/wiki/Cube_mapping
{
  float absX = fabs(pos.x);
  float absY = fabs(pos.y);
  float absZ = fabs(pos.z);
  
  int isXPositive = pos.x > 0 ? 1 : 0;
  int isYPositive = pos.y > 0 ? 1 : 0;
  int isZPositive = pos.z > 0 ? 1 : 0;
  
  // POSITIVE X
  if (isXPositive && absX >= absY && absX >= absZ) {
    *index = 0;
  }
  // NEGATIVE X
  if (!isXPositive && absX >= absY && absX >= absZ) {
    *index = 1;
  }
  // POSITIVE Y
  if (isYPositive && absY >= absX && absY >= absZ) {
    *index = 2;
  }
  // NEGATIVE Y
  if (!isYPositive && absY >= absX && absY >= absZ) {
    *index = 3;
  }
  // POSITIVE Z
  if (isZPositive && absZ >= absX && absZ >= absY) {
    *index = 4;
  }
  // NEGATIVE Z
  if (!isZPositive && absZ >= absX && absZ >= absY) {
    *index = 5;
  }
}

#endif

//########## Ross Thick Li-Sparse  ##############"

__device__ float F1_rtls(float ths, float thv, float phi ){  //  rossthick-lisparse, only F1
    if (phi < 0.) phi += DEUXPI; 
    if (phi > PI) phi = DEUXPI - phi; 
    float cos_xi = cos(ths) * cos(thv) + sin(ths) * sin(thv) * cos(phi);
    float xi = acos(cos_xi);
    float mm = 1./cos(thv) + 1./cos(ths);
    float tthv = tan(thv);
    float tths = tan(ths);

    float cos_t = 2./mm * sqrt(tthv*tthv + tths*tths - 2.*tthv*tths * cos(phi) + pow(tthv *tths * sin(phi),2) );
    cos_t = min(cos_t, 1.);
    float t = acos(cos_t);
    float sin_t = sin(t);
    float big_O = mm*(t - sin_t*cos_t)/PI;
            
    // geometric kernel
    float F1 = big_O-(1./cos(thv) + 1./cos(ths)) + (1. + cos_xi)/(cos(thv)*cos(ths))/2.;

    return F1;
}


__device__ float F2_rtls(float ths, float thv, float phi ){  //  rossthick-lisparse, only F2
    if (phi < 0.) phi += DEUXPI; 
    if (phi > PI) phi = DEUXPI - phi; 
    float cos_xi = cos(ths) * cos(thv) + sin(ths) * sin(thv) * cos(phi);
    float xi = acos(cos_xi);

    // volume-scattering kernel
    float F2 = (((PI/2. -xi)*cos_xi + sin(xi))/(cos(thv) + cos(ths))) - PI/4.;

    return F2;
}

// ################ BRDF ##############################

__device__ float BRDF(int ilam, float3 v0, float3 v1, struct Spectrum *spectrum ){  //  general BRDF
    float wbrdf = 1.;
	//int idx = blockIdx.x *blockDim.x + threadIdx.x;
    if (DIOPTREd>3) {
        float dph = 0.;
        float th0 = acos(fabs(v0.z));
        float th1 = acos(fabs(v1.z));
        float2 v0xy  = make_float2(v0);
        float2 v1xy  = make_float2(v1);
        v0xy/=length(v0xy);
        v1xy/=length(v1xy);
        if (th0!=0. && th1!=0.) dph = acos(dot(v0xy,v1xy));
        switch(DIOPTREd) {
            case 4: // RossThick Li-Sparse
                wbrdf = 1. + spectrum[ilam].k1p_surface*F1_rtls(th0,th1,dph) +
                    spectrum[ilam].k2p_surface*F2_rtls(th0,th1,dph);
                break;

            default:
                wbrdf = 1.;
        }
        //if (wbrdf==0.) printf("%f %f %f %f %f\n",th0,th1,dph,F1_rtls(th0,th1,dph),F2_rtls(th0,th1,dph));
    }
    return wbrdf;
}

//##########" Planck Radiance ################"
__device__ float BPlanck(float wav, float T) {
    float a = 2.0*PLANCK*pow(SPEED_OF_LIGHT, 2);
    float b = PLANCK*SPEED_OF_LIGHT/(wav*BOLTZMANN*T);
    float intensity = a/((pow(wav, 5)) * (exp(b) - 1.0));
return intensity;
}


//#########################################################################
//                         KERNEL 2
//#########################################################################

extern "C" {
__global__ void reduce_absorption_gpu(unsigned long long NPHOTON, unsigned long long NLAYER, unsigned long long NWVL, 
        unsigned long long NTHREAD, unsigned long long NGROUP, unsigned long long NBUNCH, 
        unsigned long long NP_REST, unsigned long long NWVL_LOW, unsigned long long NBTHETA,
        double *res, double *res_sca, double *res_rrs, double *res_sif, double *res_vrs, float *ab, float *al, float *cd, float *S, float *weight, 
        unsigned char *nrrs, unsigned char *nref, unsigned char *nsif, unsigned char *nvrs, unsigned char *nenv, unsigned char *ith, 
        unsigned char *iw_low, float *ww_low)
{
  const unsigned long long idx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned long long n,nstart,nstop,ns;
  unsigned long long iw,ig,l,s;
  unsigned long long nl,li;
  double wabs, walb0, walb1,walb; // absorption and albedo high resolution weigths
  float wsca1,wsca2,wsca;
  unsigned long long iw1,iw2;
  unsigned long long offset;

  if (idx<NTHREAD) {
    iw = idx%NWVL ; // index of current wavelength
    ig = idx/NWVL ;   // index of current group
    nstart = ig    *NBUNCH; // Start index of the photon's stack
    nstop  = (ig+1)*NBUNCH + (ig==(NGROUP-1))*NP_REST; // end index of photon's stack
                                    // last group has some remaining phton's
    iw1 = iw_low[iw];    // bracketing indices of low resolution wavelength grid
    iw2 = iw1+1;
    walb0= (double)al[iw]; // albedo of surface
    walb1= (double)al[iw +NWVL]; // albedo of surface (environment)

    for (n=nstart;n<nstop;n++) { // Loop on photon number
        //interpolating scattering 'low resolution' weights 
        wsca1 = weight[iw1+NWVL_LOW*n]; 
        wsca2 = weight[iw2+NWVL_LOW*n]; 
        wsca  = ww_low[iw] * (wsca2-wsca1) + wsca1;
        //start the computation of absorption weights
        wabs = 0.;
        for (l=0;l<NLAYER;l++) { // Loop on vertical layer
            nl = l   + NLAYER*n;
            li = iw  + NWVL*l;
            wabs += (double)cd[nl] * (double)ab[li];
        }
        walb  = pow(walb0, (double)nref[n]);
        walb *= pow(walb1, (double)nenv[n]);
        for (s=0;s<4;s++) {
            ns = s + 4*n;
            //offset = iw + NWVL*s + NWVL*NBTHETA*ith[n]; 
            offset = ith[n] + NBTHETA*iw + NWVL*NBTHETA*s;
	    #if defined(DOUBLE) && !(__CUDA_ARCH__ >= 600)
	    if (!nsif[n])             DatomicAdd(res    +offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if (!nsif[n])             DatomicAdd(res_sca+offset, (double)S[ns] *              (double)wsca * walb);
            if ( nrrs[n] && !nsif[n]) DatomicAdd(res_rrs+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nsif[n])             DatomicAdd(res_sif+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nvrs[n] && !nsif[n]) DatomicAdd(res_vrs+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
	    #else
            if (!nsif[n])             atomicAdd(res    +offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if (!nsif[n])             atomicAdd(res_sca+offset, (double)S[ns] *              (double)wsca * walb);
            if ( nrrs[n] && !nsif[n]) atomicAdd(res_rrs+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nsif[n])             atomicAdd(res_sif+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
            if ( nvrs[n] && !nsif[n]) atomicAdd(res_vrs+offset, (double)S[ns] * exp(-wabs) * (double)wsca * walb);
	    #endif
        }
    }
  }
}
}
