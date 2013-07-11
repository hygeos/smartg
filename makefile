# CUDA_HOME=/opt/cuda42/cuda/
CUDA_HOME=/opt/cuda50/
# CUDA_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/CUDA_V4.0.13
CUDA_BIN=${CUDA_HOME}/bin/
CUDA_LIB=${CUDA_HOME}/lib64/
CUDA_INC=${CUDA_HOME}/include/

# HDF_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/HDF_V4.2r5
HDF_HOME=
HDF_LIB=${HDF_HOME}/lib/
HDF_INC=${HDF_HOME}/include/

# JPEG_HOME=/home/gpubti/Loic/PROJETS/gpubti/zone_try/QPEC_MEDICIS/MEDICIS/BIBEXT/XEON/JPEG_V6b
JPEG_HOME=
JPEG_LIB=${JPEG_HOME}/lib/
JPEG_INC=${JPEG_HOME}/include/

CC = ${CUDA_BIN}/nvcc
EXEC = mccuda

# switch spherical/plane parallel
SPH=no


#=============Options============#  (en fonction de la carte graphique utilisee)
CFLAGS=-O3 -maxrregcount=32 -use_fast_math -Xptxas=-v -D_CUDA -arch=sm_20

IFLAGS = -I ${CUDA_INC} -I ${HDF_INC} -I ${JPEG_INC} -I ./src/incRNGs/Random123/ 

LFLAGS =
LFLAGS += -L ${HDF_LIB} -ldf -lmfhdf
LFLAGS += -L ${JPEG_LIB} -ljpeg
LFLAGS += -L ${CUDA_LIB} -lcuda -lcudart -lcurand -L /usr/lib64/nvidia/

#===== Caractéristiques majeures =====#
DFLAGS =
ifeq ("$(SPH)","yes")
	DFLAGS += -DSPHERIQUE	# atmosphère sphérique
endif
# DFLAGS += -DFLAGOCEAN	# À utiliser si l'océan fait partie de la simulation


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
# DFLAGS += -DDEBUG 	# Ajout de tests intermédiaires utilisés lors du débugage
ifeq ("$(PERFO_TIMER)","ON")
DFLAGS += -D_PERF
endif
#####################################################################################

all: init $(EXEC)

$(EXEC): obj/main.o obj/host.o obj/device.o obj/checkGPUcontext.o
	$(CC) $^ $(IFLAGS) $(LFLAGS) -o $(EXEC)

obj/%.o: src/%.cu
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

