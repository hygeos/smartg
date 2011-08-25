CC = nvcc
EXEC = Prog

#=============Options============#  (en fonction de la carte graphique utilisee)
CFLAGS = -g -G -arch=sm_11 -O3 -Xptxas -v
# CFLAGS = -g -G -arch=sm_20 -O3 # -Xptxas -v

IFLAGS = -I /opt/cuda/include
IFLAGS += -I /opt/NVIDIA_GPU_Computing_SDK/C/common/inc
IFLAGS += -I /usr/include/hdf/

IIFLAGS = $(wildcard /usr/lib/libdf.a /usr/lib64/hdf/libdf.a)
IIFLAGS += $(wildcard /usr/lib/libmfhdf.a /usr/lib64/hdf/libmfhdf.a)
IIFLAGS += -ljpeg

LFLAGS =
LFLAGS += -L /opt/NVIDIA_GPU_Computing_SDK/C/common/lib/linux
LFLAGS += -L /opt/cuda/lib64
LFLAGS += -L /opt/cuda/lib
LFLAGS += -L /opt/cuda/bin
LFLAGS += -L /opt/NVIDIA_GPU_Computing_SDK/C/lib
LFLAGS += -lcuda -lcudart

#=============Options============#
DFLAGS =
DFLAGS += -DPARAMETRES # Affichage des parametres initiaux
DFLAGS += -DRANDMWC 	# Utilisation du random MWC (Multiply-With-Carry)
# DFLAGS += -DRANDCUDA	# Utilisation du random CUDA (Fonction fournie par cuda)
# DFLAGS += -DRANDMT	# Utilisation du random MT (Mersenne Twister)

#=============Debogage===========#
DFLAGS += -DPROGRESSION # Calcul et affichage de la progression de la simulation
# DFLAGS += -DTRAJET # Calcul et affichage des premiers evenements d'un threads
# DFLAGS += -DTABRAND # Affichage des premiers nombre aleatoires generes

#####################################################################################

all: $(EXEC)

$(EXEC): obj/main.o obj/host.o obj/device.o
	$(CC) $^ $(IFLAGS) $(IIFLAGS) $(LFLAGS) -o $(EXEC)

obj/main.o: src/main.cu src/main.h src/communs.h src/host.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/host.o: src/host.cu src/host.h src/communs.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/device.o: src/device.cu src/device.h src/communs.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

clean:
	rm -f obj/* src/*~ *~

mrproper: clean
	rm -rf tmp/* out_prog/* out_scripts/* $(EXEC)
	
#=============Debogage===========#
suppr:
	rm -f out_prog/Resultats.hdf tmp/Temoin.hdf
	
rebuild: suppr clean all

