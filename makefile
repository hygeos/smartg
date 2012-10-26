CUDA_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/CUDA_V4.0.13
CUDA_BIN=${CUDA_HOME}/bin/
CUDA_LIB=${CUDA_HOME}/lib/
CUDA_INC=${CUDA_HOME}/inc/

HDF_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/HDF_V4.2r5
HDF_BIN=${HDF_HOME}/bin/
HDF_LIB=${HDF_HOME}/lib/
HDF_INC=${HDF_HOME}/include/

JPEG_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/JPEG_V6b
JPEG_BIN=${JPEG_HOME}/bin/
JPEG_LIB=${JPEG_HOME}/lib/
JPEG_INC=${JPEG_HOME}/include/

CC = ${CUDA_BIN}/nvcc
#CC = /opt/cuda42/cuda/bin/nvcc
EXEC = mccuda


#=============Options============#  (en fonction de la carte graphique utilisee)
# CFLAGS = -g -G -arch=sm_10 -O3 -Xptxas -v
CFLAGS=-O3 -maxrregcount=32 -use_fast_math -Xptxas=-v -D_CUDA -arch=sm_20
#CFLAGS = -g -G -arch=sm_20 -O3 # --ptxas-options=-v
## CFLAGS += -gencode arch=compute_20,code=sm_20
## CFLAGS += -m64
#IFLAGS = -I /opt/cuda42/cuda/include
IFLAGS = -I ${CUDA_INC} -I ${HDF_INC} -I ${JPEG_INC} -I ./src/incRNGs/Random123/

# IIFLAGS = $(wildcard /usr/lib/libdf.a /usr/lib64/hdf/libdf.a)
# IIFLAGS += $(wildcard /usr/lib/libmfhdf.a /usr/lib64/hdf/libmfhdf.a)
#IIFLAGS = -ldf -lmfhdf -ljpeg

#LFLAGS =
#LFLAGS += -L /opt/cuda42/lib64
#LFLAGS += -L /opt/cuda42/lib
#LFLAGS += -L ${CUDA_LIB}
LFLAGS = -L ${HDF_LIB} -ldf -lmfhdf \
	-L ${JPEG_LIB} -ljpeg \
	-L ${CUDA_LIB} -lcuda -lcudart -lcurand

#===== Caractéristiques majeures =====#
DFLAGS =
# DFLAGS += -DSPHERIQUE	# Pour utiliser une atmosphère sphérique
DFLAGS += -DFLAGOCEAN	# À utiliser si l'océan fait parti de la simulation
DFLAGS += -DNOMAUTO	# Créer automatiquement le nom du fichier résultat


#============== Options ===============#
DFLAGS += -DPARAMETRES 				# Affichage des parametres initiaux
# DFLAGS += -DRANDMWC 				# Utilisation du random MWC (Multiply-With-Carry)
 DFLAGS += -DRANDCUDA				# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDCURANDSOBOL32			# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDCURANDSCRAMBLEDSOBOL32	# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDMT				# Utilisation du random MT (Mersenne Twister)
# DFLAGS += -DRANDPHILOX4x32_7			# Utilisation du random Philox-4x32-7


#============== Debogage ==============#
DFLAGS += -DPROGRESSION # Calcul et affichage de la progression de la simulation
# DFLAGS += -DTRAJET 	# Calcul et affichage des premiers evenements d'un thread
# DFLAGS += -DTEMPS 	# Affichage du temps passé dans chaque fonctions pour un thread
# DFLAGS += -DTABRAND 	# Affichage des premiers nombre aleatoires generes
# DFLAGS += -DDEBUG 	# Ajout de tests intermédiaires utilisés lors du débugage
ifeq ("$(PERFO_TIMER)","ON")
DFLAGS += -D_PERF
endif
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

