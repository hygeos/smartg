
	  ///////////////////////
	 // PROTOTYPES DEVICE //
	///////////////////////

__global__ void lancementKernel(Random* , Constantes* , Progress* , unsigned long long*
		#ifdef TABNBPHOTONS
		, unsigned long long*
		#endif
		#ifdef TRAJET
		, Evnt*
		#endif
			       );
__device__ void init(Photon* , Constantes*
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );
__device__ void surfac(Photon* , Random*
		#ifdef TRAJET
		, int, Evnt*
		#endif
		      );
__device__ void exit(Photon* , Constantes* , unsigned int*
		#ifdef PROGRESSION
		, unsigned int*
		#endif
		, Progress* , unsigned long long*
		#ifdef TABNBPHOTONS
		, unsigned long long*
		#endif
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );
__device__ void move(Photon* , Constantes* , Random*
		#ifdef TRAJET
		, int, Evnt*
		#endif
		    );
__device__ void scatter(Photon* , Random*
		#ifdef TRAJET
		, int, Evnt*
		#endif
		       );
__device__ void calculsPhoton(float, Constantes*, Photon*);
__device__ void calculCase(int*, int*, Photon*, Progress*);
__device__ float rand_MWC_co(unsigned long long*,unsigned int*);
__device__ float rand_MWC_oc(unsigned long long*,unsigned int*);
__device__ void atomicAddULL(unsigned long long* address, unsigned int add);
