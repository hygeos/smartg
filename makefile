CC = nvcc
EXEC = LancePhotons

CFLAGS = -g -G -arch=sm_11 -O3 -Xptxas -v

IFLAGS = -I /opt/cuda/include
IFLAGS += -I /home/tristan/NVIDIA_GPU_Computing_SDK/C/common/inc
IFLAGS += -I /usr/include/hdf/

IIFLAGS = $(wildcard /usr/lib/libdf.a /usr/lib64/hdf/libdf.a)
IIFLAGS += $(wildcard /usr/lib/libmfhdf.a /usr/lib64/hdf/libmfhdf.a)
IIFLAGS += -ljpeg

LFLAGS = -L/home/tristan/NVIDIA_GPU_Computing_SDK/C/common/lib/linux
LFLAGS += -L /opt/cuda/lib64
LFLAGS += -L /opt/cuda/lib
LFLAGS += -L /opt/cuda/bin
LFLAGS += -L /home/tristan/NVIDIA_GPU_Computing_SDK/C/lib
LFLAGS += -lcuda -lcudart -lcutil_x86_64

# Options de sortie
DFLAGS =
# DFLAGS += -DPARAMETRES # Affichage des parametres initiaux
DFLAGS += -DQUART # Calcul et creation-hdf du tableau final reporté sur un quart de sphère

# Options pour debugage
DFLAGS += -DPROGRESSION # Calcul et affichage de la progression de la simulation
# DFLAGS += -DTRAJET # Calcul et affichage des premiers evenements d'un threads
# DFLAGS += -DTABSTOKES # Affichage des tableaux finaux pour chaque nombre de Stokes
# DFLAGS += -DTABFINAL # Affichage du tableau final
# DFLAGS += -DCOMPARAISON # Calcul et creation hdf du tableau de comparaison des 2 quarts de sphère
# DFLAGS += -DRANDMWC
DFLAGS += -DRANDCUDA
# DFLAGS += -DRANDMT
# DFLAGS += -DTABRAND

#####################################################################################

all: $(EXEC)

LancePhotons: obj/main.o obj/host.o obj/device.o
	$(CC) $^ $(IFLAGS) $(IIFLAGS) $(LFLAGS) -o $(EXEC)

obj/main.o: src/main.cu src/main.h src/communs.h src/host.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/host.o: src/host.cu src/host.h src/communs.h src/device.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

obj/device.o: src/device.cu src/device.h src/communs.h
	$(CC) -c $< $(CFLAGS) $(IFLAGS) $(DFLAGS) -o $@

clean:
	rm -f obj/*.o *~ src/*~

mrproper: clean
	rm -f $(EXEC) out/*.hdf
