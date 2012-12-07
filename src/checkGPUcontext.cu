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
static CrMCCUDA PrintDevicesProperties(){
    CrMCCUDA    CodeRetour      = MCCUDA_KO;
    cudaError_t CodeRetourGPU;

    int dCount;
    struct cudaDeviceProp deviceProp;
    int device;
    int driverVersion = 0, runtimeVersion = 0;

    CodeRetourGPU = cudaGetDeviceCount( &dCount );
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceCount = %s", GetGPUErrorString(CodeRetourGPU));
        #endif
        CodeRetour = MCCUDA_KO;
        goto ERREUR;
    }

    if (dCount == 1){
        //soit il n'y a qu'un seul device disponible soit il n'y en a point (mode emulation)
        CodeRetourGPU = cudaGetDeviceProperties( &deviceProp, 0 );
        if ( CodeRetourGPU != cudaSuccess ){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceProperties = %s", GetGPUErrorString(CodeRetourGPU));
            #endif
            CodeRetour = MCCUDA_KO;
            goto ERREUR;
        }
        if (deviceProp.major == 9999){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => Pas de carte graphique disponible (seul le mode emulation est possible)");
            #endif
            CodeRetour = MCCUDA_ENVIRONNEMENT_GPU_NON_COMPATIBLE;
            goto ERREUR;
        }
    }

    printf("\n\n\t\t\tCUDA Device Query (Runtime API) version (CUDART static linking)\n\n\n");
    for ( device = 0; device < dCount; ++device )
    {
        CodeRetourGPU = cudaGetDeviceProperties( &deviceProp, device );
        if ( CodeRetourGPU != cudaSuccess ){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::PrintDevicesProperties() => erreur cudaGetDeviceProperties = %s", GetGPUErrorString(CodeRetourGPU));
            #endif
            CodeRetour = MCCUDA_KO;
            goto ERREUR;
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

    CodeRetour = MCCUDA_OK;
    ERREUR:
            return CodeRetour;
}


/**********************************************************
*	> Methodes externes
***********************************************************/

/* CheckGPUContext
* verifie la presence d'un environnement GPU conforme
* le cas echeant, affiche les attributs de la carte
*/
extern CrMCCUDA CheckGPUContext(){
    CrMCCUDA    CodeRetour      = MCCUDA_KO;
    cudaError_t CodeRetourGPU;

    int     deviceCount =   0;
    int     device      =   0;
    struct  cudaDeviceProp properties;

    // Verifie qu'aucune erreur n'est anterieure
    CodeRetourGPU = cudaGetLastError();
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => erreur GPU anterieure a la procedure = %s", GetGPUErrorString(CodeRetourGPU));
        #endif
        CodeRetour = MCCUDA_KO;
        goto ERREUR;
    }

    // Verification de la disponibilite de l'environnement GPU
    CodeRetourGPU = cudaGetDeviceCount(&deviceCount);
    if ( CodeRetourGPU != cudaSuccess ){
        #ifdef DEBUG
            printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => erreur cudaGetDeviceCount = %s", GetGPUErrorString(CodeRetourGPU));
        #endif
        CodeRetour = MCCUDA_KO;
        goto ERREUR;
    }
    if (deviceCount == 1){
        //dans ce cas soit il n'y a qu'une seule carte disponible soit il n'y en a aucune (mode emulation)
        CodeRetourGPU = cudaGetDeviceProperties(&properties, 0);
        if ( CodeRetourGPU != cudaSuccess ){
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => erreur cudaGetDeviceProperties = %s", GetGPUErrorString(CodeRetourGPU));
            #endif
            CodeRetour = MCCUDA_KO;
            goto ERREUR;
        }
        if ( properties.major == 9999 ){
            //mode emulation seulement
            #ifdef DEBUG
                printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => mode emulation non accepte");
            #endif
            CodeRetour = MCCUDA_ENVIRONNEMENT_GPU_NON_COMPATIBLE;
            goto ERREUR;
        }
        else{
            #ifdef DEBUG
            CodeRetour = PrintDevicesProperties();
            if ( CodeRetour != MCCUDA_OK ){
                goto ERREUR;
            }
            #endif
        }
    }
    else{
        //affiche les proprietes des devices installes
        #ifdef DEBUG
        CodeRetour = PrintDevicesProperties();
        if ( CodeRetour != MCCUDA_OK ){
            return CodeRetour;
        }
        #endif
    }

    // Association manuelle MCCUDA avec un contexte GPU
    // -> si plusieurs devives sont presents, et le device #0 n'est pas compatible
    // on tente "d'accrocher" le device suivant
    cudaGetDeviceCount( &deviceCount );

    for(device=0;device<deviceCount;device++){
        //On force la creation du contexte ce device
        CodeRetourGPU = cudaSetDevice(device);

        //On teste la validite du context
        CodeRetourGPU = cudaFree(0);

        //Si le test reussit: on a un device valide
        if (CodeRetourGPU == cudaSuccess){
            //On recupere les  proprietes du device
            CodeRetourGPU = cudaGetDeviceProperties( &properties, device );

            //on verifie qu'on a pas un device en mode emu
            if( (properties.major != 9999 ) ){
                //on initialise tout et on return
                // => si les caracteristiques du device devaient etre recuperees dans une structure,
                //    il faudrait le faire ici
                CodeRetour = MCCUDA_OK;
                goto QUITTER;
            }
        }
        //Sinon on libère le contexte avant de boucler
        else{
            cudaDeviceReset();
        }
    }
    //si on sort ici, c'est qu'on a pas reussi a initialiser un device
    //en effet, le seul test effectue dans la boucle a ce jour ( (deviceProp.major != 9999) )
    //pourrait etre enrichi (test de la double precision, test du nombre de coeurs etc.), aussi,
    //les cartes "valides" pourraient ne pas resister a ces tests.
    #ifdef DEBUG
    printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() : Impossible d'initialiser un device CUDA valide :\n");
    if(CodeRetourGPU != cudaSuccess){
        printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => cudaLastError    = %s\n", cudaGetErrorString(CodeRetourGPU));
    }else{
        printf("!!MCCUDA Erreur!! >> checkGPUcontext::CheckGPUContext() => (des gpus ont ete trouves mais aucun n'est valide, les criteres de selections ne sont pas passes avec succes)\n");
    }
    #endif
    CodeRetour = MCCUDA_ENVIRONNEMENT_GPU_NON_COMPATIBLE;
    goto ERREUR;

    ERREUR:
            return CodeRetour;
    QUITTER:
            return CodeRetour;
}

/* getGPUErrorString
* recuperation du message d'erreur GPU,
* la methode cudaGetErrorString est "enroulee" ici
* (getGPUErrorString pourrait etre enrichie)
*/
extern const char* GetGPUErrorString( cudaError_t CodeRetourGPU )
{
    return cudaGetErrorString(CodeRetourGPU);
}
