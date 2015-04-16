CUDA_HOME=/opt/cuda65/
CUDA_BIN=${CUDA_HOME}/bin/
CUDA_LIB=${CUDA_HOME}/lib64/
CUDA_INC=${CUDA_HOME}/include/

HDF_HOME=
HDF_LIB=${HDF_HOME}/lib/
HDF_INC=${HDF_HOME}/include/

JPEG_HOME=
JPEG_LIB=${JPEG_HOME}/lib/
JPEG_INC=${JPEG_HOME}/include/

CC = ${CUDA_BIN}/nvcc

EXEC_PP = SMART-G-PP
EXEC_SP = SMART-G-SP


#=============Options============#  (en fonction de la carte graphique utilisee)
CFLAGS=-O3 -maxrregcount=32 -use_fast_math -Xptxas=-v -D_CUDA -arch=sm_20 

IFLAGS = -I ${CUDA_INC} -I ${HDF_INC} -I ${JPEG_INC} -I ./src/incRNGs/Random123/ 

LFLAGS =
LFLAGS += -L ${HDF_LIB} -ldf -lmfhdf
LFLAGS += -L ${JPEG_LIB} -ljpeg
LFLAGS += -L ${CUDA_LIB} -lcuda -lcudart -lcurand -L /usr/lib64/nvidia/ -L /usr/lib64/nvidia-304xx/


#============== Options ===============#
DFLAGS += -DPARAMETRES 				# Affichage des parametres initiaux
# DFLAGS += -DRANDMWC 				# Utilisation du random MWC (Multiply-With-Carry)
# DFLAGS += -DRANDCUDA				# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDCURANDSOBOL32			# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDCURANDSCRAMBLEDSOBOL32	# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDMT				# Utilisation du random MT (Mersenne Twister)
DFLAGS += -DRANDPHILOX4x32_7			# Utilisation du random Philox-4x32-7


#============== Debogage ==============#
DFLAGS += -DPROGRESSION # Calcul et affichage de la progression de la simulation
# DFLAGS += -DDEBUG 	# Ajout de tests intermédiaires utilisés lors du débugage
ifeq ("$(PERFO_TIMER)","ON")
DFLAGS += -D_PERF
endif
#####################################################################################

all: pp sp

$(EXEC_PP): obj/pp/main.o obj/pp/host.o obj/pp/device.o obj/pp/checkGPUcontext.o
	$(CC) $^ $(IFLAGS) $(LFLAGS) -o $(EXEC_PP)

$(EXEC_SP): obj/sp/main.o obj/sp/host.o obj/sp/device.o obj/sp/checkGPUcontext.o
	$(CC) $^ $(IFLAGS) $(LFLAGS) -o $(EXEC_SP)

obj/pp/%.o: src/%.cu
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/sp/%.o: src/%.cu
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -DSPHERIQUE -o $@

init_pp:
	@echo
	@echo MAKING PP...
	@mkdir -p obj/pp

init_sp:
	@echo
	@echo MAKING SP...
	@mkdir -p obj/sp

clean:
	rm -f obj/pp/* obj/sp/* src/*~ *~ $(EXEC_PP) $(EXEC_SP)

rebuild: clean all

sp: init_sp $(EXEC_SP)

pp: init_pp $(EXEC_PP)

