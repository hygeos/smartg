
MCCuda 
======


1) Compilation
   -----------

La compilation est réalisée par la commande 'make'.
Ajuster au préalable les chemins vers la librairie cuda dans 'makefile'.

Dépendances:
    * librairie cuda (testé avec cuda 4.1, 4.2 et 5.0)
    * librairie hdf4 (testé avec hdf4-4.2.6)
      (écriture des fichiers témoin et sortie au format hdf4)


2) Usage
   -----

mccuda <fichier parametres>

Voir le fichier exemple Parametres.txt



3) Script de test
   --------------

Le script run_all.sh permet de :
    - compiler le code en mode plan parallèle (PP) et sphérique (SP)
      -> deux exécutables sont générés
    - lancer une série de tests (correspondants aux fichiers paramètres
      localisés dans resultat/*.inp)
    - tracer les figures de visualisation des fichiers générés

Pour référence, les fichiers résultat et les figures correspondantes générés à
Hygeos sont placés dans le répertoire resultat:
    - les fichiers de sortie: resultat/*reference.hdf
    - les sorties stdout: resultat/*reference.out
    - les fichiers de visualisation: resultat/*reference.png
