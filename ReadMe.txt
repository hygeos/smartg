
MCCuda 
======



1) Compilation
   -----------

La compilation est réalisée par la commande 'make'.
Ajuster au préalable les chemins vers la librairie cuda dans 'makefile'.

Dépendances:
    * librairie cuda (testé avec cuda 4.1 et 4.2)
    * librairie hdf4 (testé avec hdf4-4.2.6)
      (écriture des fichiers témoin et sortie au format hdf4)


2) Usage
   -----

mccuda <fichier parametres>

Voir le fichier exemple Parametres.txt




3) Remarques sur le code
   ---------------------


*** Ajouter un paramètres dans le fichier Paramètres.txt ***

Si l'on ajoute un paramètre, il faut le définir en tant que variable externe
dans le programme et l'initialiser dans le host. Pour cela :
  - Le définir dans le fichier commun.h en utilisant le mot clé extern (le nom
    de la variable est en majuscule)
  - Le définir également dans le fichier main.h sans le mot clé extern
  - L'initialiser en le lisant dans Paramètres.txt grâce à la fonction
    initConstantesHost() du fichier host.cu

S'il faut maintenant copier cette valeur dans le device (GPU), définir dans
device.h la variable avec le même nom plus un suffixe 'd'. Puis effectuer la
copie du host vers le device dans la fonction initConstantesDevice().


*** Paramètres NBLOOP ***

Si la valeur de NBLOOP est trop importante, on observe des erreurs dans le résultats de sortie.
Ceci vient probablement du fait que sommer la valeur pour un photon au cumul de la valeur pour tous les photons crée des erreurs d'arrondis important.
C'est pourquoi il est INDISPENSABLE de stocker dans le CPU les valeurs intermédiaires de la simulation lors de chaque appel du kernel.
Les erreurs s'observent aussi lorsque le NBLOOP est trop important. On les observe facilement pour NBLOOp=18000 et légérement pour NBLOOP=9000.
La dernière valeur choisie est 5000, elle semble apporter des résultats
satisfaisant.

