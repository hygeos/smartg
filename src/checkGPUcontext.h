#ifndef CHECKGPUCONTEXT_H
#define CHECKGPUCONTEXT_H

/**********************************************************
*
*                   checkGPUcontext.h
*
*	> Verifie la presence d'un environnement GPU conforme
*
***********************************************************/
/**********************************************************
*	> Includes
***********************************************************/

#include <cuda.h>
#include "communs.h"

/**********************************************************
*	> Variable interne a checkGPUcontext
***********************************************************/
//cores numbers by SM
static  int nGpuArchCoresPerSM[] = { -1, 8, 32 };

/**********************************************************
*	> Methodes externes
***********************************************************/

/* checkGPUContext
* verifie la presence d'un environnement GPU conforme
* le cas echeant, affiche les attributs de la carte
*/
extern CrMCCUDA CheckGPUContext();


/* getGPUErrorString
* recuperation du message d'erreur GPU,
* la methode cudaGetErrorString est "enroulee" ici
* (getGPUErrorString pourrait etre enrichie)
*/
extern const char* GetGPUErrorString( cudaError_t CodeRetourGPU );

#endif
