################################################
# Details concernant l'implémentation des RNGs #
################################################

*****************************************
* 1) Option de compilation -DRANDCUDA : *

-> RNG ?
	- XorWow (PRNG)
	- integre dans CURAND de Nvidia
-> Initialisation ?
	- allocation d'un etat de 40 octets par thread lance
	- lancement d'un kernel pour initialiser les 40 octets pour chaque thread
-> Caractéristiques ?
	- période de 2^190
	- la séquence est divisee par portion de 2^67
-> Limites ?
	- 2^67 nombres par threads
	- 2^123 threads par grille
-> Impact sur le kernel ?
	- lecture pour chaque thread d'un etat en memoire globale de 40 octets
	- stockage en registre pour chaque thread d'un etat de 40 octets

*****************************************

**************************************************
* 2) Option de compilation -DRANDCURANDSOBOL32 : *

-> RNG ?
	- Sobol32 (QRNG)
	- integre dans CURAND de Nvidia
-> Initialisation ?
	- allocation d'un etat de 136 octets par thread lance
	- lancement d'un kernel pour initialiser les 136 octets pour chaque thread
-> Caractéristiques ?
	- période de 2^32
	- 20 000 dimensions au maximum (Joe et Kuo, voir Rapport d'Etudes)
-> Limites ?
	- 2^32 nombres par threads
	- 20 000 threads par grille 
-> Impact sur le kernel ?
	- lecture pour chaque thread d'un etat en memoire globale de 136 octets
	- stockage en registre pour chaque thread d'un etat de 136 octets

**************************************************

***********************************************************
* 3) Option de compilation -DRANDCURANDSCRAMBLEDSOBOL32 : *

-> RNG ?
	- ScrambledSobol32 (QPRNG)
	- integre dans CURAND de Nvidia
	- le "scrambling" est tire de B.Owen (voir Rapport d'Etudes)
-> Initialisation ?
	- allocation d'un etat de 140 octets par thread lance
	- lancement d'un kernel pour initialiser les 140 octets pour chaque thread
-> Caractéristiques ?
	- cf. 2) Caractéristiques
-> Limites ?
	- cf. 2) Limites
-> Impact sur le kernel ?
	- lecture pour chaque thread d'un etat en memoire globale de 140 octets
	- stockage en registre pour chaque thread d'un etat de 140 octets

**************************************************

***************************************
* 4) Option de compilation -DRANDMT : *

-> RNG ?
	- MT (PPRNG) (version ?)
-> Initialisation ?
	- ?
-> Caractéristiques ?
	- ?
-> Limites ?
	- ?

***************************************

****************************************
* 5) Option de compilation -DRANDMWC : *

-> RNG ?
	- Multiply With Carry (PPRNG)
-> Initialisation ?
	- allocation d'un etat de 12 octets par thread lance
	- lecture sur CPU d'un etat de 12 octets par thread lance
	- copie d'un etat de 12 octets sur GPU par thread lance
-> Caractéristiques ?
	- periode de 2^64
	- 150 000 jeux de parametres disponibles dans un fichier externe
-> Limites ?
	- 2^64 nombres par threads
	- 150 000 threads par grille 
-> Impact sur le kernel ?
	- lecture pour chaque thread d'un etat en memoire globale de 12 octets
	- stockage en registre pour chaque thread d'un etat de 12 octets

****************************************

*************************************************
* 6) Option de compilation -DRANDPHILOX4x32_7 : *

-> RNG ?
	- Philox-4x32-7 (PRNG) (J.K Salmon et al. voir Rapport d'Etudes)
-> Initialisation ?
	- allocation d'un etat de 4 octets par thread lance + 4 octets
	- lecture sur CPU d'un etat de 4 octets par thread lance + 4 octets
	- lancement d'un kernel pour initialiser les 4 octets pour chaque thread
-> Caractéristiques ?
	- periode de 2^32 (possibilite de passer a 2^128)
	- 2^32 generateurs possibles (possibilite de passer a 2^64)
-> Limites ?
	- 2^32 nombres par threads
	- 2^32 threads par grille 
-> Impact sur le kernel ?
	- lecture pour chaque thread d'un etat en memoire globale de 8 octets
	- stockage en registre pour chaque thread d'un etat de 24 octets

*************************************************
