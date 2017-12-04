CC=gcc
LD=gcc
EXEC=mgbc
CUDAHOME=<path>
CUDACC=$(CUDAHOME)/bin/nvcc 
CUDA_ARCH=-arch=sm_35
CUDACFLAGS=-m64 -c -O3  --ptxas-options=-v -DCUDA

MPI_DIR=/usr/lib/openmpi
CFLAGS=-g -W -Wall -Wno-unused-function -Wno-unused-parameter -c -O3 -DMPI -DCUDA  -L$(MPI_DIR)/lib -lmpi  -I$(CUDAHOME)/include
LDFLAGS = -L$(MPI_DIR)/lib -lmpi ../generator/libgraph_generator_mpi.a -L$(CUDAHOME)/lib64 -lcudart -lstdc++ -lm 
MPI_LIB=$(MPI_DIR)/lib


OBJ=bc2d.o cuda_kernels.o adtp.o


bc2d: ${OBJ}
	mkdir -p ../bin
	${LD} -o ../bin/$(EXEC) ${OBJ} ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS) -I$(MPI_DIR)/include    $< -o $@

%.o: %.cu        
	$(CUDACC) $(CUDACFLAGS) $(CUDA_ARCH) -I$(MPI_DIR)/include -lm $<
clean:
	rm -rf *.o ../bin/$(EXEC)
init:
	cp ../bin/$(EXEC) $(HOME)/test_src
