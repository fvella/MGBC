CC=gcc
LD=gcc
EXEC=mgbc

#CUDA_PATH use CUDATOOLKIT_HOME for CRAY
CUDA_HOME=
MPI_HOME=
GRAPH_GEN_LIB=
CUB_INCLUDE=


CUDACC=$(CUDA_HOME)/bin/nvcc 
CUDA_ARCH=-arch=sm_37
CUDACFLAGS=-m64 -c -O3  --ptxas-options=-v -DCUDA  -DONEDEGREE --use_fast_math


CFLAGS=-g -W -Wall -Wno-unused-function -Wno-unused-parameter -c -O3 -DMPI -DCUDA  -L$(MPI_HOME)/lib64 -lmpi   -I$(CUDA_HOME)/include -I$(CUB_INCLUDE)
LDFLAGS = -L$(MPI_HOME)/lib64 -lmpi -L$(CUDA_HOME)/lib64 -lcudart -lstdc++ ${GRAPH_GEN_LIB}/libgraph_generator_mpi.a  -lm -Wno-deprecated-gpu-targets

OBJ=bc2d.o cuda_kernels.o adtp.o

# https://stackoverflow.com/questions/2734719/how-to-compile-a-static-library-in-linux
axbc: ${OBJ}
	${LD} -o $(DEST_DIR)$(EXEC) ${OBJ} ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS) -I$(MPI_HOME)/include  -I$(CUB_INCLUDE)  $< -o $@

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) $(CUDA_ARCH) -I$(MPI_HOME)/include -I$(CUB_INCLUDE)  $<
clean:
	rm -rf *.o $(DEST_DIR)$(EXEC)

