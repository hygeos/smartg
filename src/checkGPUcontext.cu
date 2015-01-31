/**********************************************************
*
*                   checkGPUcontext.cu
*
*	> Verifie la presence d'un environnement GPU conforme
*
***********************************************************/

/**********************************************************
*	> Includes
***********************************************************/
#include "checkGPUcontext.h"


/**********************************************************
*	> Variable interne a checkGPUcontext
***********************************************************/


/**********************************************************
*	> Methodes internes
***********************************************************/

/* PrintDevicesProperties
* affiche les proprietes des cartes installees
*/
#ifdef DEBUG
static CrMCCUDA PrintDevicesProperties(){
    cudaError_t CodeRetourGPU;

    int dCount;
    struct cudaDeviceProp deviceProp;
    int device;
    int driverVersion = 0, runtimeVersion = 0;

    CodeRetourGPU = cudaGetDeviceCount( &dCount );
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceCount = %s", cudaGetErrorString(CodeRetourGPU));
        #endif
        return MCCUDA_KO;
    }

    if (dCount == 1){
        //soit il n'y a qu'un seul device disponible soit il n'y en a point (mode emulation)
        CodeRetourGPU = cudaGetDeviceProperties( &deviceProp, 0 );
        if ( CodeRetourGPU != cudaSuccess ){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceProperties = %s", cudaGetErrorString(CodeRetourGPU));
            #endif
            return MCCUDA_KO;
        }
        if (deviceProp.major == 9999){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => Pas de carte graphique disponible (seul le mode emulation est possible)");
            #endif
            return MCCUDA_ENVIRONNEMENT_GPU_NON_COMPATIBLE;
        }
    }

    printf("\n\n\t\t\tCUDA Device Query (Runtime API) version (CUDART static linking)\n\n\n");
    for ( device = 0; device < dCount; ++device )
    {
        CodeRetourGPU = cudaGetDeviceProperties( &deviceProp, device );
        if ( CodeRetourGPU != cudaSuccess ){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceProperties = %s", cudaGetErrorString(CodeRetourGPU));
            #endif
            return MCCUDA_KO;
        }

        printf( "\n%d - name:                    %s\n" ,device ,deviceProp.name );
        #if CUDART_VERSION >= 2020
            cudaDriverGetVersion(&driverVersion);
            printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
        #endif
        printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
        printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
        printf("  Total amount of global memory:                 %u bytes\n", deviceProp.totalGlobalMem);

        #if CUDART_VERSION >= 2000
            printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
            printf("  Number of cores:                               %d\n", nGpuArchCoresPerSM[deviceProp.major] * deviceProp.multiProcessorCount);
        #endif
        printf("  Total amount of constant memory:               %u bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        #if CUDART_VERSION >= 2000
            printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
        #endif
        #if CUDART_VERSION >= 2020
            printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
            printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
            printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
            printf("  >>Compute mode:                                %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
                "Default (multiple host threads can use this device simultaneously)" :
                        deviceProp.computeMode == cudaComputeModeExclusive ?
                "Exclusive (only one host thread at a time can use this device)" :
                        deviceProp.computeMode == cudaComputeModeProhibited ?
                        "Prohibited (no host thread can use this device)" : "Unknown");
        #endif
    }
    printf("\n\n\n");

    return MCCUDA_OK;
}
#endif


/**********************************************************
*	> Methodes externes
***********************************************************/

/* CheckGPUContext
* verifie la presence d'un environnement GPU conforme
* le cas echeant, affiche les attributs de la carte
* success: return id of attributed device
* failure: returns -1
*/
int CheckGPUContext(int device_selected){
    
#ifdef DEBUG
    PrintDevicesProperties();
#endif

    cudaError_t CodeRetourGPU;

    int     deviceCount =   0;
    int     device      =   0;

    // Verifie qu'aucune erreur n'est anterieure
    CodeRetourGPU = cudaGetLastError();
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => erreur GPU anterieure a la procedure = %s", cudaGetErrorString(CodeRetourGPU));
        #endif
        return -1;
    }

    // Verification de la disponibilite de l'environnement GPU
    CodeRetourGPU = cudaGetDeviceCount(&deviceCount);
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => erreur cudaGetDeviceCount = %s", cudaGetErrorString(CodeRetourGPU));
        #endif
        return -1;
    }

    if (device_selected >= 0) {
        // we want a specific device
        if (select_device(device_selected)) {
            return -1;
        }

        return device_selected;

    } else {
        printf("Selecting device...\n");
        // select the first available device
        for (device=0;device<deviceCount;device++) {
            if (select_device(device)) {
                continue;
            } else {
                break;
            }
        }

        return device;
    }
}


// switch to a device
// return 0 on success, 1 on failure
int select_device(int device) {
    struct  cudaDeviceProp properties;

    if (cudaGetDeviceProperties(&properties, device)) {
        printf("Error in cudaGetDeviceProperties (device %d)\n", device);
        cudaDeviceReset();
        return 1;
    }
    printf("Device %d, '%s' ", device, properties.name);
    if( properties.major == 9999  ) {
        printf("[incompatible, emulation mode]\n");
        return 1;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        printf("Error in cudaSetDevice (device %d)\n", device);
        cudaDeviceReset();
        return 1;
    }
    if (cudaFree(0) != cudaSuccess) {
        printf("[busy]\n");
        cudaDeviceReset();
        return 1;
    }

    printf("[OK]\n");

    return 0;
}

// display which device has been used
void message_end(int device) {

    struct cudaDeviceProp deviceProp;

    if (cudaGetDeviceProperties( &deviceProp, device) == cudaSuccess) {

        printf("Done (used device %d, '%s')\n", device, deviceProp.name);

    } else {
        printf("Done (used device %d)\n", device);

    }
}
