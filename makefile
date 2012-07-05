CC = /usr/local/cuda/bin/nvcc
EXEC = Prog

#=============Options============#  (en fonction de la carte graphique utilisee)
# CFLAGS = -g -G -arch=sm_10 -O3 -Xptxas -v
CFLAGS = -g -G -arch=sm_20 -O3 # --ptxas-options=-v
# CFLAGS += -gencode arch=compute_20,code=sm_20
# CFLAGS += -m64
IFLAGS = -I /usr/local/cuda/include
IFLAGS += -I /usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc
IFLAGS += -I /usr/include/hdf/

# IIFLAGS = $(wildcard /usr/lib/libdf.a /usr/lib64/hdf/libdf.a)
# IIFLAGS += $(wildcard /usr/lib/libmfhdf.a /usr/lib64/hdf/libmfhdf.a)
IIFLAGS = -ldf -lmfhdf -ljpeg

LFLAGS =
LFLAGS += -L /usr/local/NVIDIA_GPU_Computing_SDK/C/common/lib/linux
LFLAGS += -L /usr/local/cuda/lib64
LFLAGS += -L /usr/local/cuda/lib
LFLAGS += -L /usr/local/cuda/bin
LFLAGS += -L /usr/local/NVIDIA_GPU_Computing_SDK/C/lib
LFLAGS += -lcuda -lcudart

#===== Caractéristiques majeures =====#
DFLAGS =
# DFLAGS += -DSPHERIQUE	# Pour utiliser une atmosphère sphérique
DFLAGS += -DFLAGOCEAN	# À utiliser si l'océan fait parti de la simulation
DFLAGS += -DNOMAUTO	# Créer automatiquement le nom du fichier résultat


#============== Options ===============#
DFLAGS += -DPARAMETRES 	# Affichage des parametres initiaux
DFLAGS += -DRANDMWC 	# Utilisation du random MWC (Multiply-With-Carry)
# DFLAGS += -DRANDCUDA	# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDMT	# Utilisation du random MT (Mersenne Twister)


#============== Debogage ==============#
DFLAGS += -DPROGRESSION # Calcul et affichage de la progression de la simulation
# DFLAGS += -DTRAJET 	# Calcul et affichage des premiers evenements d'un thread
# DFLAGS += -DTEMPS 	# Affichage du temps passé dans chaque fonctions pour un thread
# DFLAGS += -DTABRAND 	# Affichage des premiers nombre aleatoires generes
# DFLAGS += -DDEBUG 	# Ajout de tests intermédiaires utilisés lors du débugage
#####################################################################################

all: init $(EXEC)

$(EXEC): obj/main.o obj/host.o obj/device.o
	$(CC) $^ $(IFLAGS) $(IIFLAGS) $(LFLAGS) -o $(EXEC)

obj/main.o: src/main.cu src/main.h src/communs.h src/host.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/host.o: src/host.cu src/host.h src/communs.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/device.o: src/device.cu src/device.h src/communs.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

init:
	mkdir -p obj

clean:
	rm -f obj/* src/*~ *~ $(EXEC)

mrproper: clean
	rm -rf tmp/* out_prog/* out_scripts/* $(EXEC)
	
#=============Debogage===========#
suppr:
	rm -f out_prog/Resultats.hdf tmp/Temoin.hdf
	
rebuild: suppr clean all

