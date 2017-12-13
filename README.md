# MGBC
### Description ###

Betweenness Centrality (BC) is steadily growing in popularity as a metrics of the influence of a vertex in a graph. 
The exact BC computation for a large scale graph is an extraordinary challenging and requires high performance computing techniques to provide results in a reasonable amount of time. 
We provide MGBC to speed-up the computation of the BC on Multi-GPU systems. 
MGBC supports 2-D and 3-D decomposition of the graph and multi-level parallelism.
MGBC supports degree-1 reduction (2D and 3D) and DMF algorithms heuristics (only single GPU). 
Experimental results show that the proposed techniques are well suited to compute BC scores in graphs which are too large to fit in single GPU memory. 
The computation time of a 234 million edges graph is reduced to less than 2 hours.

[CK Demo for reproducibility on single GPU](https://github.com/ctuning/ck-bc)

### How do I get set up? ###

Dependencies Tested: MPI implementation for CRAY, cuda-6.5+ and compute capability >= 3.5

```
cd <path>/generator

make

cd ../src

make -f Makefile
```

EXEC: <path>/bin

Usage: 
./mgbc -h

         $> bin/mgbc -p RxC  [-o outfile] [-D] [-d] [-m] [-N <# of serarch>] [-H 0,1] 

         -> to visit a graph read from file: 

                 -f <graph file> -n <# vertices> [-r <start vert>] 

         -> to visit an RMAT graph: 

                 -S <scale> [-E <edge factor>] 

**Params description**

                 -RxC  grid of processors

                 -D to ENABLE debug information

                 -d to DUMP RMAT generated graph to file

                 -m to DISABLE mono GPU optimization

                 -o file_scores_bc

                 -H 1-degree reduction off (0)

### Test it ###

**MGBC-2D** 
R-MAT EXAMPLE
graph scale 20; 
graph ef 16;
degree-1 ON;
4 processors arranged in 2x2 mesh;
bc rounds 10000;
```
$> mpirun -np 4 mgbc -p 2x2 -S 20 -E 16 -N 10000 -H 1
```
**MGBC-3D**

R-MAT EXAMPLE
graph scale 20
graph ef 16
degree-1 ON
processors 8 arranged in 2 subclusters 2x2.
bc rounds 10000 BC round.
```
$> mpirun -np 8 mgbc -p 2x2 -S 20 -E 16 -N 10000 -H 1
```

### References  ###
* Massimo Bernaschi, Giancarlo Carbone, and Flavio Vella. 2015. Betweenness centrality on Multi-GPU systems. In Proceedings of the 5th Workshop on Irregular Applications: Architectures and Algorithms (IA3 '15). ACM, New York, NY, USA, , Article 12 , 4 pages. DOI: https://doi.org/10.1145/2833179.2833192
* Massimo Bernaschi, Giancarlo Carbone, and Flavio Vella. 2016. Scalable betweenness centrality on multi-GPU systems. In Proceedings of the ACM International Conference on Computing Frontiers (CF '16). ACM, New York, NY, USA, 29-36. DOI: https://doi.org/10.1145/2903150.2903153
* Flavio Vella, Giancarlo Carbone, Massimo Bernaschi. Algorithms and heuristics for scalable betweenness centrality computation on multi-GPU systems. https://arxiv.org/abs/1602.00963
