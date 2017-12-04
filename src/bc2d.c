#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdarg.h>
#include <mpi.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "bc2d.h"
#define GRAPH_GENERATOR_MPI
#include "../generator/make_graph.h"


#define VTAG(t) (  0*ntask+(t))
#define HTAG(t) (100*ntask+(t))
#define PTAG(t) (200*ntask+(t))

#define _TIMINGS 1
//#define PRINTSTATS 1
#define BIN_NUM 16

#define LIMIT 1
//LOCINT N=0;   /* number of vertices: N */
uint64_t N=0;   /* number of vertices: N */
LOCINT  row_bl; /* adjacency matrix rows per block: N/(RC) */
LOCINT  col_bl; /* adjacency matrix columns per block: N/C */
LOCINT  row_pp; /* adjacency matrix rows per proc: N/(RC) * C = N/R */
uint64_t degree_reduction_time = 0;
uint64_t overlap_time = 0;
uint64_t two_degree_reduction_time = 0;
uint64_t sort_time = 0;
int C=0;
int R=0;
int gmyid;
int myid;
int gntask;
int ntask;
int mono = 1;
int undirected = 1;
int analyze_degree = 0;
LOCINT *tlvl = NULL;
LOCINT *tlvl_v1 = NULL;

int heuristic=0;

int myrow;
int mycol;
int pmesh[MAX_PROC_I][MAX_PROC_J];
MPI_Comm MPI_COMM_CLUSTER;

LOCINT flag = 0;

LOCINT *reach=NULL;
FILE * outdebug = NULL;
LOCINT loc_count = 0;
STATDATA *mystats = NULL;
unsigned int outId;
char strmesh[10];
char cmdLine[256];

MPI_Comm Row_comm, Col_comm;

static void freeMem (void* p) {
	if (p) free(p);
}

static void prexit(const char *fmt, ...) {

	int myid;
	va_list ap;

	va_start(ap, fmt);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if (0 == myid) vfprintf(stderr, fmt, ap);
	MPI_Finalize();
	exit(EXIT_FAILURE);
}

void *Malloc(size_t sz) {

	void *ptr;

	ptr = (void *) malloc(sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	memset(ptr, 0, sz);
	return ptr;
}

void writeStats()
{
	FILE *fout = NULL;
	char fname[256];

	LOCINT i;
	int j;

	if (loc_count == 0) return;

	snprintf(fname, 256, "stats_%s_%d_%d.txt", strmesh, outId, gmyid);
	fprintf(stdout, "Statistical Information written to file %s\n", fname);
	fout = fopen(fname, "w+");
	if (fout == NULL) {
		fprintf(stderr, "in function %s: error opening %s\n", __func__, fname);
		exit(EXIT_FAILURE);
	}

	fprintf(fout, "Command Line: %s\n", cmdLine);

	fflush(stdout);

	/*
	uint64_t id; 		// Vertex identifier
		LOCINT degree;      // Vertex degree
		int lvl;		    // Levels explored
		uint64_t msgSize;   // MPI message size
		uint64_t visited;   // Vertices visited
		LOCINT nfrt[256];  // For each level how many elements in the frontier
		uint64_t upw[2];    // Communication and Computation time
		uint64_t dep[2];    // Communication and Computation time
		uint64_t upd[2];    // Communication and Computation time
		uint64_t over;      // overlap time
		uint64_t tot;
	*/

	if (loc_count>0)
		fprintf(fout,"N.\tID\tDegree\tLevel\tVisited\tTot_T\tTot_CUDA\tTot_MPI\tUpw_C\tUpw_M"
			     	 "\tDep_C\tDep_M\tUpd_C\tUpd_M\tOverL\t\tNFRT\n");
	else fprintf(fout,"No vertex processed!");

	for (i = 0; i < loc_count; i++) {
			// Count, Vertex id, Degree, Level, Visited, Tot_Time, Tot_CUDA, Tot_MPI, Upw_Cuda, Upw_Mpi, Dep_Cuda, Dep_Mpi, Upd_Cuda, Upd_Mpi, OverLap
		   fprintf(fout,"%d\t%"PRIu64"\t%d\t%d\t%"PRIu64"\t%"PRIu64
						"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64
						"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64
						"\t%"PRIu64"\t",
						 	i,
					mystats[i].id,
					mystats[i].degree,
					mystats[i].lvl,
					mystats[i].visited,
					mystats[i].tot,
					mystats[i].compu,
					mystats[i].commu,
					mystats[i].upw[0],
					mystats[i].upw[1],
					mystats[i].dep[0],
					mystats[i].dep[1],
					mystats[i].upd[0],
					mystats[i].upd[1],
					mystats[i].over
			);

			for (j=0; j< mystats[i].lvl; j++) {
				fprintf(fout,"\t%d",mystats[i].nfrt[j]);
			}

			fprintf(fout,"\n");
	}

	fclose(fout);

}

// root node, degree, nvisited, num_lvl
void setStats(uint64_t v0, LOCINT degree)
{
	if (VERT2PROC(v0) != myid) return;
	mystats[loc_count].id = v0;
	mystats[loc_count].degree = degree;
}

// root node, degree, nvisited, num_lvl
void addStats(uint64_t v0, int lvl, uint64_t nfrt)
{
	if (VERT2PROC(v0) != myid) return;

	mystats[loc_count].nfrt[lvl] = nfrt;

}

// root node, degree, nvisited, num_lvl
void upStats(uint64_t v0, int visited, int lvl, uint64_t msgSize,
			 uint64_t upwc_time, uint64_t upwm_time,
			 uint64_t depc_time, uint64_t depm_time,
			 uint64_t updc_time, uint64_t updm_time,
			 uint64_t ovrl_time,
			 uint64_t compu_time, uint64_t commu_time,
			 uint64_t total_time)
{
	if (VERT2PROC(v0) != myid) return;

	mystats[loc_count].visited = visited;
	mystats[loc_count].lvl = lvl;
	mystats[loc_count].msgSize = msgSize;

	mystats[loc_count].upw[0] = upwc_time; // Cuda
	mystats[loc_count].upw[1] = upwm_time; // Mpi
	mystats[loc_count].dep[0] = depc_time;
	mystats[loc_count].dep[1] = depm_time;
	mystats[loc_count].upd[0] = updc_time;
	mystats[loc_count].upd[1] = updm_time;
	mystats[loc_count].over = ovrl_time;
	mystats[loc_count].compu = compu_time;
	mystats[loc_count].commu = commu_time;
	mystats[loc_count].tot = total_time;

	loc_count++;
}
/*
 * Print statistics
 */
void prstat(uint64_t val, const char *msg, int det) {

	int      myid, ntask, i, j, w1, w2, min, max;
	uint64_t t, *v = NULL;
	double   m, s;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
	MPI_Comm_size(MPI_COMM_CLUSTER, &ntask);

	if (myid == 0)
		v = (uint64_t *)Malloc(ntask*sizeof(*v));

	MPI_Gather(&val, 1, MPI_UNSIGNED_LONG_LONG, v, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_CLUSTER);

	if (myid == 0) {
		t = 0;
		m = s = 0.0;
		min = max = 0;
		for(i = 0; i < ntask; i++) {
			if (v[i] < v[min]) min = i;
			if (v[i] > v[max]) max = i;
			t += v[i];
			m += (double)v[i];
			s += (double)v[i] * (double)v[i];
		}
		m /= ntask;
		s = sqrt((1.0/(ntask-1))*(s-ntask*m*m));

		fprintf(stdout, "%s", msg);
		if (det) {
			for(w1 = 0, val =  ntask; val; val /= 10, w1++);
			for(w2 = 0, val = v[max]; val; val /= 10, w2++);

			fprintf(stdout, "\n");
			for(i = 0; i < R; i++) {
				fprintf(stdout, " ");
				for(j = 0; j < C; j++) {
					fprintf(stdout, "%*d: %*"PRIu64"  ", w1, i*C+j, w2, v[i*C+j]);
				}
				fprintf(stdout, "\n");
			}
		}
		fprintf(stdout,
				" [total=%"PRIu64", mean=%.2lf, stdev=%.2lf, min(%d)=%"PRIu64", max(%d)=%"PRIu64"]\n",
				t, m, s, min, v[min], max, v[max]);

		free(v);
	}
	return;
}

static void *Realloc(void *ptr, size_t sz) {

	void *lp;

	lp = (void *) realloc(ptr, sz);
	if (!lp && sz) {
		fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

	FILE *fp = NULL;
	fp = fopen(path, mode);
	if (!fp) {
		fprintf(stderr, "Cannot open file %s...\n", path);
		exit(EXIT_FAILURE);
	}
	return fp;
}

static off_t get_fsize(const char *fpath) {

	struct stat st;
	int rv;

	rv = stat(fpath, &st);
	if (rv) {
		fprintf(stderr, "Cannot stat file %s...\n", fpath);
		exit(EXIT_FAILURE);
	}
	return st.st_size;
}

static uint64_t getFsize(FILE *fp) {

	int rv;
	uint64_t size = 0;

	rv = fseek(fp, 0, SEEK_END);
	if (rv != 0) {
		fprintf(stderr, "SEEK END FAILED\n");
		if (ferror(fp)) fprintf(stderr, "FERROR SET\n");
		exit(EXIT_FAILURE);
	}

	size = ftell(fp);
	rv = fseek(fp, 0, SEEK_SET);

	if (rv != 0) {
		fprintf(stderr, "SEEK SET FAILED\n");
		exit(EXIT_FAILURE);
	}

	return size;
}
/*
 * Duplicates vertices to make graph undirected
 *
 */
static uint64_t *mirror(uint64_t *ed, uint64_t *ned) {

	uint64_t i, n;

	if (undirected == 1) {
	ed = (uint64_t *)Realloc(ed, (ned[0]*4)*sizeof(*ed));

	n = 0;
	for(i = 0; i < ned[0]; i++) {
		if (ed[2*i] != ed[2*i+1]) {
			ed[2*ned[0]+2*n] = ed[2*i+1];
			ed[2*ned[0]+2*n+1] = ed[2*i];
			n++;
		}
	}
	ned[0] += n;
	}
	return ed;
}

/*
 * Read graph data from file
 */
static uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge) {
#define ALLOC_BLOCK     (2*1024)

	uint64_t *ed=NULL;
	uint64_t i, j;
	uint64_t n, nmax;
	uint64_t size;
	int64_t  off1, off2;

	int64_t  rem;
	FILE     *fp;
	char     str[MAX_LINE];

	fp = Fopen(fpath, "r");

	size = getFsize(fp);
	rem = size % ntask;
	off1 = (size/ntask)* myid    + (( myid    > rem)?rem: myid);
	off2 = (size/ntask)*(myid+1) + (((myid+1) > rem)?rem:(myid+1));

	if (myid < (ntask-1)) {
		fseek(fp, off2, SEEK_SET);
		fgets(str, MAX_LINE, fp);
		off2 = ftell(fp);
	}
	fseek(fp, off1, SEEK_SET);
	if (myid > 0) {
		fgets(str, MAX_LINE, fp);
		off1 = ftell(fp);
	}

	n = 0;
	nmax = ALLOC_BLOCK; // must be even
	ed = (uint64_t *)Malloc(nmax*sizeof(*ed));
	uint64_t lcounter = 0;
	uint64_t nedges = -1;
	int comment_counter = 0;

	/* read edges from file */
	while (ftell(fp) < off2) {

		// Read the whole line
		fgets(str, MAX_LINE, fp);

		// Strip # from the beginning of the line
		if (strstr(str, "#") != NULL) {
			//fprintf(stdout, "\nreading line number %"PRIu64": %s\n", lcounter, str);
			if (strstr(str, "Nodes:")) {
				sscanf(str, "# Nodes: %"PRIu64" Edges: %"PRIu64"\n", &i, &nedges);
				fprintf(stdout, "N=%"PRIu64" E=%"PRIu64"\n", i, nedges);
			}
			comment_counter++;
		} else if (str[0] != '\0') {
			lcounter ++;
			// Read edges
			sscanf(str, "%"PRIu64" %"PRIu64"\n", &i, &j);

			if (i >= N || j >= N) {
				fprintf(stderr,
						"[%d] In file %s line %"PRIu64" found invalid edge in %s for N=%"PRIu64": (%"PRIu64", %"PRIu64")\n",
						myid, fpath, lcounter, str, N, i, j);
				exit(EXIT_FAILURE);
			}

			if (n >= nmax) {
				nmax += ALLOC_BLOCK;
				ed = (uint64_t *)Realloc(ed, nmax*sizeof(*ed));
			}
			ed[n]   = i;
			ed[n+1] = j;
			n += 2;
		}
	}
	// Check the number of edges against the number of lines, if there were
	// comments
	//nedges += comment_counter;
	/*
	if ((comment_counter > 0) && (lcounter != nedges)) {
		fprintf(stderr, "Error reading the input file %s: the number of lines differ from the number of edges in the header\n", fpath);
		fprintf(stderr, "lcounter = %"PRIu64" nedges = %"PRIu64"\n", lcounter, nedges);
		exit(EXIT_FAILURE);
	}
	 */
	fclose(fp);

	n /= 2; // number of ints -> number of edges
	*edge = mirror(ed, &n);
	return n;
#undef ALLOC_BLOCK
}

/*
 *  select one root vertex randomly
 */
static uint64_t select_root(LOCINT *col) {

	int      pi, pj, p;
	LOCINT   ld, gd;
	uint64_t lv, gv, c;

	struct timeval time;

	gettimeofday(&time, NULL);
	srand48(getpid()+time.tv_sec+time.tv_usec);

	lv = lrand48();
	MPI_Allreduce(&lv, &gv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_CLUSTER);
	p = gv%ntask;
	pi = p/C;
	pj = p%C;
	//if (myid == p) fprintf(stderr, "Bfs will start from processor %d\n", p);
	do {
		if (pi == myrow && pj == mycol) {
				c = LOCI2GI(mycol*row_bl + lrand48()%row_bl);
				//fprintf(stderr, "chosen vertex %"PRIu64", global degree: ", c);
		}
		MPI_Bcast(&c, 1, MPI_UNSIGNED_LONG_LONG, p, MPI_COMM_CLUSTER);

		if (pj == mycol) ld = col[GJ2LOCJ(c)+1]-col[GJ2LOCJ(c)];
		else             ld = 0;

		MPI_Allreduce(&ld, &gd, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
		//if (myid == p) fprintf(stderr, "%"LOCPRI"\n", gd);
	} while (!gd);
	/*
	if (myid == p)
		fprintf(stdout, "[%d] selected root node %"PRIu64
						", belonging to processor %d\n",
						myid, c, VERT2PROC(c));
	*/
	return c;
}

static uint64_t select_root1(LOCINT *col) {

	LOCINT  ld, gd;
	int64_t lv, gv;

	static int ftime=1;

	if (ftime) {
		struct timeval time;
		//uint64_t seed = 4343543;
		//srand48(seed + myid);
		gettimeofday(&time, NULL);
		srand48(getpid()+time.tv_sec+time.tv_usec);
		ftime = 0;
	}
	do {
		lv = lrand48();
		MPI_Allreduce(&lv, &gv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_CLUSTER);
		gv %= N;

		if (mycol == GJ2PJ(gv)) ld = col[GJ2LOCJ(gv)+1]-col[GJ2LOCJ(gv)];
		else                    ld = 0;

		MPI_Allreduce(&ld, &gd, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
	} while (!gd);

	return gv;
}

static uint64_t select_root2(LOCINT *col) {

	LOCINT  ld, gd;
	int64_t lv, gv;

	static int ftime=1;

	if (ftime) {
		struct timeval time;
		//uint64_t seed = 4343543;
		//srand48(seed + myid);
		gettimeofday(&time, NULL);
		srand48(getpid()+time.tv_sec+time.tv_usec);
		ftime = 0;
		//if (myid == 0)
		//      fprintf(stdout, "Seeding random generator with %d+myid\n", seed);
	}
		lv = lrand48();
		MPI_Allreduce(&lv, &gv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_CLUSTER);
		gv %= N;
	return gv;
}

static uint64_t select_root3(LOCINT *col) {

	LOCINT  ld, gd;
	int64_t lv, gv;

	static int ftime=1;

	if (ftime) {
		struct timeval time;
		uint64_t seed = 4343543;
		srand48(seed + myid);
		//gettimeofday(&time, NULL);
		//srand48(getpid()+time.tv_sec+time.tv_usec);
		ftime = 0;
	}
	do {
		lv = lrand48();
		MPI_Allreduce(&lv, &gv, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_CLUSTER);
		gv %= N;

		if (mycol == GJ2PJ(gv)) ld = col[GJ2LOCJ(gv)+1]-col[GJ2LOCJ(gv)];
		else                    ld = 0;

		MPI_Allreduce(&ld, &gd, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
	} while (!gd);

	return gv;
}


/*
 * Generate RMAT graph calling make_graph function
 */
static uint64_t gen_graph(int scale, int edgef, uint64_t **ed) {

	uint64_t ned;
	double   initiator[4] = {.57, .19, .19, .05};

	make_graph(scale, (((int64_t)1)<<scale)*edgef, 23, 24, initiator, (int64_t *)&ned, (int64_t **)ed, MPI_COMM_CLUSTER);
	*ed = mirror(*ed, &ned);

	return ned;
}

static void dump_edges(uint64_t *ed, uint64_t nedge, const char *desc) {

	uint64_t i;
	fprintf(outdebug, "%s - %ld\n",desc, nedge);

	for (i = 0; i < nedge ; i++)
		fprintf(outdebug, "%"PRIu64"\t%"PRIu64"\n", ed[2*i], ed[2*i+1]);

	fprintf(outdebug, "\n");
	return;
}

static int cmpedge_1d(const void *p1, const void *p2) {
	  uint64_t *l1 = (uint64_t *) p1;
	  uint64_t *l2 = (uint64_t *) p2;



	  if (l1[0] < l2[0]) return -1;
	  if (l1[0] > l2[0]) return 1;

	  if (l1[1] < l2[1]) return -1;
	  if (l1[1] > l2[1]) return 1;

	  return 0;
}
static void  dump_rmat(uint64_t *myedges, uint64_t myned, int scale, int edgef){

	FILE *fout = NULL;
	char fname[256];

	uint64_t i;
	uint64_t max = 0;

	qsort(myedges, myned, sizeof(uint64_t[2]), cmpedge_1d);


	snprintf(fname, 256, "rmat_S%d_EF%d.txt", scale, edgef);
	fprintf(stdout, "DUMP THE GENERATED RMAT FILE IN %s\n", fname);
	fout = fopen(fname, "w+");
	if (fout == NULL) {
		fprintf(stderr, "in function %s: error opening %s\n", __func__, fname);
		exit(EXIT_FAILURE);
	}

	max = N;

	fflush(stdout);
	//fprintf(fout, "# Directed RMAT Scale=%d Edgefactor=%d\n", scale, edgef);
	//fprintf(fout, "#\n");
	//fprintf(fout, "# Nodes: %"PRIu64" Edges: %"PRIu64"\n", max, myned);
	//fprintf(fout, "# NodeId\tNodeId\n");

	for (i = 0; i < myned; i++){
		fprintf(fout,"%"PRIu64"\t%"PRIu64"\n", myedges[2*i], myedges[2*i+1]);
	}

	fclose(fout);
}

/*
 * Graph Partitioning
 *
 * MPI exchange edges based on 2-D partitioning
 */
static uint64_t part_graph(int myid, int ntask, uint64_t **ed, uint64_t nedge, int part_mode) {

	uint64_t i;

	uint64_t *s_ed=NULL;
	uint64_t *r_ed=*ed;

	uint64_t totrecv;

	uint64_t *soff=NULL;
	uint64_t *roff=NULL;
	uint64_t *send_n=NULL;
	uint64_t *recv_n=NULL;

	int *pmask=NULL;

	int n, p;
	MPI_Status *status;
	MPI_Request *request;

	/* compute processor mask for edges */
	pmask = (int *)Malloc(nedge*sizeof(*pmask));
	send_n = (uint64_t *)Malloc(ntask*sizeof(*send_n));
	for(i = 0; i < nedge; i++) {
		if (part_mode == 1){
			pmask[i] = r_ed[2*i] % ntask; // 1D partiorning

		}
		else pmask[i] = EDGE2PROC(r_ed[2*i], r_ed[2*i+1]);
		send_n[pmask[i]]++;
	}

	/* sort edges by owner process (recv_n is used as a tmp) */
	soff = (uint64_t *)Malloc(ntask*sizeof(*soff));
	soff[0] = 0;
	for(p = 1; p < ntask; p++)
		soff[p] = soff[p-1] + send_n[p-1];

	recv_n = (uint64_t *)Malloc(ntask*sizeof(*recv_n));
	memcpy(recv_n, soff, ntask*sizeof(*soff));

	s_ed = (uint64_t *)Malloc(2*nedge*sizeof(*s_ed));
	for(i = 0; i < nedge; i++) {
		s_ed[2*recv_n[pmask[i]]]   = r_ed[2*i];
		s_ed[2*recv_n[pmask[i]]+1] = r_ed[2*i+1];
		recv_n[pmask[i]]++;
	}
	/* to proc k must be send send_n[k] edges starting at s_ei[soff[k]] */
	MPI_Alltoall(send_n, 1, MPI_UNSIGNED_LONG_LONG, recv_n, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_CLUSTER);
	if (send_n[myid] != recv_n[myid]) {
		fprintf(stderr, "[%d] Error in %s:%d\n", myid, __func__, __LINE__);
		exit(EXIT_FAILURE);
	}

	roff = (uint64_t *)Malloc(ntask*sizeof(*roff));
	roff[0] = 0;
	totrecv = recv_n[0];
	for(p = 1; p < ntask; p++) {
		totrecv += recv_n[p];
		roff[p] = roff[p-1] + recv_n[p-1];
	}
	r_ed = (uint64_t *)Realloc(r_ed, 2*totrecv*sizeof(*r_ed));

	status  = (MPI_Status  *)Malloc(ntask*sizeof(*status));
	request = (MPI_Request *)Malloc(ntask*sizeof(*request));

	/* post RECVs */
	for(p = 0, n = 0; p < ntask; p++) {
		if (recv_n[p] == 0 || p == myid) continue;
		MPI_Irecv(r_ed + 2*roff[p], 2*recv_n[p], MPI_UNSIGNED_LONG_LONG, p, PTAG(p), MPI_COMM_CLUSTER, request+n);
		n++;
	}
	/* do the SENDs */
	memcpy(r_ed+2*roff[myid], s_ed+2*soff[myid], 2*send_n[myid]*sizeof(*s_ed));
	for(p = 0; p < ntask; p++) {
		if (send_n[p] == 0 || p == myid) continue;
		MPI_Send(s_ed + 2*soff[p], 2*send_n[p], MPI_UNSIGNED_LONG_LONG, p, PTAG(myid),   MPI_COMM_CLUSTER);
	}
	MPI_Waitall(n, request, status);

	free(s_ed);
	free(send_n);
	free(soff);
	free(roff);
	free(recv_n);
	free(pmask);
	free(status);
	free(request);

	*ed = r_ed;
	return totrecv;
}

/*
 * Compare Edges
 *
 * Compares Edges p1 (a,b) with p2 (c,d) according to the following algorithm:
 * First compares the nodes where edges are assigned, if they are on the same processor than compares
 * tail and than head
 */
static int cmpedge(const void *p1, const void *p2) {

	uint64_t *l1 = (uint64_t *) p1;
	uint64_t *l2 = (uint64_t *) p2;

	if (EDGE2PROC(l1[0], l1[1]) < EDGE2PROC(l2[0], l2[1]) ) return -1;
	if (EDGE2PROC(l1[0], l1[1]) > EDGE2PROC(l2[0], l2[1]) ) return 1;

	if (l1[0] < l2[0]) return -1;
	if (l1[0] > l2[0]) return 1;

	if (l1[1] < l2[1]) return -1;
	if (l1[1] > l2[1]) return 1;

	return 0;
}

static LOCINT degree_reduction(int myid, int ntask, uint64_t **ed, uint64_t nedge,
		                       uint64_t **edrem, LOCINT *ner) {
      uint64_t u=-1, v=-1, next_u= -1, next_v= -1, prev_u = -1;
      uint64_t i,j;
      uint64_t *n_ed=NULL;  // new edge list
      uint64_t *o_ed=NULL;  // removed edge list
      uint64_t *r_ed=NULL;   // input edge list

	uint64_t ncouple = 0;

      uint64_t nod=0, ne=0, pnod = 0, skipped=0; // vrem=0;

      //Graph partitioning 1-D
      if (ntask > 1) nedge = part_graph(myid, ntask, ed, nedge, 1);

      // Sort Edges (u,v) by u
      qsort(*ed, nedge, sizeof(uint64_t[2]), cmpedge_1d);
      r_ed = *ed;

      //dump_edges(r_ed, nedge,"Degree Reduction Edges 1-D");

      n_ed = (uint64_t *)Malloc(2*nedge*sizeof(*n_ed));
      o_ed = (uint64_t *)Malloc(2*nedge*sizeof(*o_ed));

	fprintf(stdout, "[rank %d] Memory allocated\n", myid);

    for (i = 0; i < nedge-1; i++){
        u = r_ed[2*i];
        v = r_ed[2*i+1]; // current is ( u,v )    next pair is next_u, next_t based on index j
        j = 2*i + 2;
        next_u = r_ed[j];
        next_v = r_ed[j+1];

        if ((u==v) || ( (u == next_u) && (v == next_v))) { // Skip
			skipped++;
			prev_u = u;
			continue;
        }

        if ((u != next_u) && (u != prev_u)){
	    // This is a 1-degre remove
	        o_ed[2*nod] = v;
			o_ed[2*nod+1] = u;
			nod++;
        }
        else { // this is a first of a series or within a series
        	n_ed[2*ne] = u;
			n_ed[2*ne+1] = v;
			ne++;
        }
        prev_u = u;
    }
    // Check last edge
    u = r_ed[2*nedge-2];
    v = r_ed[2*nedge-1];
    if (u==prev_u) {
        n_ed[2*ne] = u;
        n_ed[2*ne+1] = v;
        ne++;
    } else if (u!=v) { // 1-degree store v before u
        o_ed[2*nod] = v;
        o_ed[2*nod+1] = u;
        nod++;
    }

    fprintf(stdout, "[rank %d] Edges removed during fist step %lu\n", myid, nod);

    // 1-degree vertices (nod )removed
    // HERE n_ed contains the new edges list
    // o_ed contains removed edges list
    if (ntask > 1) nod = part_graph(myid, ntask, &o_ed, nod, 1); // partition removed edges

    // sort partitioned edges
    // Sort Edges (u,v) by u
    qsort(o_ed, nod, sizeof(uint64_t[2]), cmpedge_1d);

    //dump_edges(o_ed, nod, "edges removed");

    // remove edges for vertices removed in the previous step
    if (nod > 0){
        // Number of edges left after 1-degree removal
        nedge = ne;
        // nod is the number of edges we need to remove after exchange
        // nedge are the number of edges we
        ne=0;
		for (i = 0; i < nedge; i++) {
			// This is required to solve the case when two vertices are connected between them but
			// disconnected from all the others
			while ((n_ed[2*i] > o_ed[2*pnod]) && (pnod < nod)) {
				ncouple++;
				pnod++;
			}

			if ((n_ed[2*i] == o_ed[2*pnod]) && (n_ed[2*i+1] == o_ed[2*pnod+1])) {
				pnod++;
				// skip this
				continue;
			} else {
				// save this in the remaining edges
				r_ed[2*ne] = n_ed[2*i];
				r_ed[2*ne+1] = n_ed[2*i+1];
				ne++;
			}
		}
    } else memcpy(r_ed, n_ed, 2*ne*sizeof(r_ed));

    fprintf(stdout, "[rank %d] Edges removed during second step %lu\n", myid, pnod);
    fprintf(stdout, "[rank %d] Couple of edges removed %lu\n", myid, ncouple);

    //dump_edges(r_ed,ne, "GRAPH FOR CSC");
    *ner = pnod;    // How many vertices have been removed
    *edrem = o_ed;  // Array of removed edges
    //o_ed = NULL;  // ATTENZIONE !!
    free(n_ed);
    //free(o_ed);   //ATTENZIONE !! ???
    return ne;
}


/*
 *
 * ed   array with edges
 * ned  number of edges
 * deg  array with degrees
 *
 */

static uint64_t norm_graph(uint64_t *ed, uint64_t ned, LOCINT *deg) {

	uint64_t l, n;

	if (ned == 0) return 0;

	qsort(ed, ned, sizeof(uint64_t[2]), cmpedge);
	// record degrees considering multiple edges
	// and self-loop and remove them from edge list
	if (deg != NULL) deg[GI2LOCI(ed[0])]++;
	for(n = l = 1; n < ned; n++) {

		if (deg != NULL) deg[GI2LOCI(ed[0])]++;
		if (((ed[2*n]   != ed[2*(n-1)]  )  ||   // Check if two consecutive heads are different
			 (ed[2*n+1] != ed[2*(n-1)+1])) &&   // Check if two consecutive tails are different
			 (ed[2*n] != ed[2*n+1])) {          // It is not a "cappio"

			ed[2*l]   = ed[2*n];                // since it is not a "cappio" and is not a duplicate edge, copy it in the final edge array
			ed[2*l+1] = ed[2*n+1];
			l++;
		}
	}
	return l;
}

// probably unneeded
static int verify_32bit_fit(uint64_t *ed, uint64_t ned) {

	uint64_t i;

	for(i = 0; i < ned; i++) {
		uint64_t v;

		v = GI2LOCI(ed[2*i]);
		if (v >> (sizeof(LOCINT)*8)) {
			fprintf(stdout, "[%d] %"PRIu64"=GI2LOCI(%"PRIu64") won't fit in a 32-bit word\n", myid, v, ed[2*i]);
			return 0;
		}
		v = GJ2LOCJ(ed[2*i+1]);
		if (v >> (sizeof(LOCINT)*8)) {
			fprintf(stdout, "[%d] %"PRIu64"=GJ2LOCJ(%"PRIu64") won't fit in a 32-bit word\n", myid, v, ed[2*i+1]);
			return 0;
		}
	}
	return 1;
}


static void init_bc_1degree(uint64_t *edrem, uint64_t nedrem, uint64_t nverts, LOCINT * reach)
{

	uint64_t i;
	LOCINT ur = 0;

	for (i = 0; i < nedrem; i++){
		// Edrem are edges (u,v) where v is a 1-degree vertex removed
		ur = GI2LOCI(edrem[2*i]); // this is local row
		// We need to use the number of vertices in the connected component
		reach[ur]++;
	}
}

/*
 * Build compressed sparse row
 *
 */

static void build_csc(uint64_t *ed, uint64_t ned, LOCINT **col, LOCINT **row) {

	LOCINT *r, *c, *tmp, i;

	/* count edges per col */
	tmp = (LOCINT *)Malloc(col_bl*sizeof(*tmp));
	for(i = 0; i < ned; i++)
		tmp[GJ2LOCJ(ed[2*i+1])]++;  // Here we have the local degree (number of edges for each local row)

	/* compute csc col[] vector with nnz in last element */
	c = (LOCINT *)Malloc((col_bl+1)*sizeof(*c));
	c[0] = 0;
	for(i = 1; i <= col_bl; i++)
		c[i] = c[i-1] + tmp[i-1];  // Sum to the previous index the local degree.

	/* fill csc row[] vector */
	memcpy(tmp, c, col_bl*sizeof(*c)); /* no need to copy last int (nnz) */

	r = (LOCINT *)Malloc(ned*sizeof(*r));
	for(i = 0; i < ned; i++) {
		r[tmp[GJ2LOCJ(ed[2*i+1])]] = GI2LOCI(ed[2*i]);
		tmp[GJ2LOCJ(ed[2*i+1])]++;
	}
	free(tmp);

	*row = r;
	*col = c;

	return;
}

/*
 * Compare unsigned local values
 */
static int cmpuloc(const void *p1, const void *p2) {

	LOCINT l1 = *(LOCINT *)p1;
	LOCINT l2 = *(LOCINT *)p2;

	if (l1 < l2) return -1;
	if (l1 > l2) return  1;
	return 0;
}

/*
 *
 *
 */
uint64_t compact(uint64_t *v, uint64_t ld, int *vnum, int n) {

	int      i, j;
	uint64_t cnt = vnum[0];

	for(i = 1; i < n; i++)
		for(j = 0; j < vnum[i]; j++)
			v[cnt++] = v[i*ld + j];
	return cnt;
}



static void check_recv_cols(LOCINT *rbuf, LOCINT ld, int *rnum, int n) {

	int     myid;
	int64_t i, j;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);

	/* sanity check */
	for(i = 0; i < n; i++) {
		for(j = 0; j < rnum[i]; j++) {

			if (rbuf[i*ld+j] >= col_bl) {
				fprintf(stderr,
						"[%d] V-Received vertex %"LOCPRI" whose column does not belong to me!\n",
						myid, rbuf[i*ld+j]);
				exit(EXIT_FAILURE);
			}
		}
	}
	return;
}

static void check_recv_rows(LOCINT *rbuf, LOCINT ld, int *rnum, int n) {

	int     myid;
	int64_t i, j;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);

	/* sanity check */
	for(i = 0; i < n; i++) {
		for(j = 0; j < rnum[i]; j++) {

			if (rbuf[i*ld+j] < mycol*row_bl || rbuf[i*ld+j] >= (mycol*row_bl+row_bl)) {
				fprintf(stderr,
						"[%d] H-Received vertex %"LOCPRI" whose row does not belong to me!\n",
						myid, rbuf[i*ld+j]);
				exit(EXIT_FAILURE);
			}
		}
	}
	return;
}

static inline void exchange_vert4x2(LOCINT *frt, LOCINT *frt_sig, int nfrt, LOCINT *rbuf, LOCINT ld, int *rnum,
                                                  MPI_Request *request, MPI_Status *status, int post) {
	int i, p;

	ld = ld*2;

	// Receive vertices from the processors in the same column except myself
	// There are R processors in the same column
	// Here I receive the frontiers from all other processors in the same column
	for(i = 1; i < R; i++) {
		p = (myrow+i)%R;
		MPI_Irecv(rbuf + p*ld, ld, LOCINT_MPI, pmesh[p][mycol], VTAG(pmesh[p][mycol]), MPI_COMM_CLUSTER, request+i-1);
	}

	// Copy in the receiving buffer my frontier
	memcpy(rbuf+myrow*ld, frt, nfrt*sizeof(*frt));
	// Copy in the receiving buffer sigma values right after the frontier
	memcpy(rbuf+myrow*ld+nfrt, frt_sig, nfrt*sizeof(*frt_sig));

	// Store in rnum number of vertices in the new frontier found on this processor
	rnum[myrow] = nfrt;

	// Send the frontier to all processors in the same Column.
	// Here we send all vertices in the froniter to all processors on the column
	for(i = 1; i < R; i++) {
		p = (myrow+i)%R;
		MPI_Send(rbuf+myrow*ld, 2*nfrt, LOCINT_MPI, pmesh[p][mycol], VTAG(myid), MPI_COMM_CLUSTER);
	}
	// Wait for IRecv to complete
	MPI_Waitall(R-1, request, status);
	for(i = 1; i < R; i++) {
		// Get how many vertices have been received from each processor and store the value in rnum[]
		p = (myrow+i)%R;
		MPI_Get_count(status+i-1, LOCINT_MPI, rnum+p);
		// Receive both vertices and their sigma value
		if (rnum[p]>0) rnum[p] = rnum[p] / 2;  // In this way rnum contains the number of couples V,S
	}

	return;
}


static inline void exchange_horiz4x2(LOCINT *sbuf, LOCINT sld, int *snum, LOCINT *rbuf, LOCINT rld,
								     int *rnum, MPI_Request *request, MPI_Status *status, int post) {
	int i, p;

	rld = rld * 2;
	sld = sld * 2;

	// Post the IRecv for all processes in the same row
	for(i = 1; i < C; i++) {
		p = (mycol+i)%C;
		MPI_Irecv(rbuf + p*rld, rld, LOCINT_MPI, pmesh[myrow][p], HTAG(pmesh[myrow][p]), MPI_COMM_CLUSTER, request+i-1);
	}

	rnum[mycol] = 0;

	for(i = 1; i < C; i++) {
		// Send data to other processes
		p = (mycol+i)%C;
		MPI_Send(sbuf + p*sld, snum[p]*2, LOCINT_MPI, pmesh[myrow][p], HTAG(myid), MPI_COMM_CLUSTER);
	}
	MPI_Waitall(C-1, request, status);
	for(i = 1; i < C; i++) {
		// Get the real number of data sent
		MPI_Get_count(status+i-1, LOCINT_MPI, rnum+(mycol+i)%C);
		if (rnum[(mycol+i)%C] > 0) rnum[(mycol+i)%C] = rnum[(mycol+i)%C] / 2;
	}

	return;
}


static void dump_lvl(int *lvl, LOCINT min, LOCINT max) {

	FILE     *fp=NULL;
	char     name[MAX_LINE];
	int      myid;
	uint64_t i;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
	snprintf(name, MAX_LINE, "lvl_%d", myid);
	fp = Fopen(name, "w");

	for(i = min; i < max; i++)
			fprintf(fp, "%"PRIu64" %d\n", LOCI2GI(i), lvl[i]);

	fclose(fp);
	return;
}

static void dump_deg(LOCINT *deg, LOCINT *deg_count, int n) {

	FILE     *fp=NULL;
	char     name[MAX_LINE];
	int      myid;
	int i;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
	snprintf(name, MAX_LINE, "degree_%d", myid);
	fp = Fopen(name, "a");

	for (i = 0; i < n ; i++)
	   fprintf(fp, " %d (%d),", deg[i], deg_count[i]);

	fprintf(fp, "\n");
	fclose(fp);
	return;

}

static void dump_array(const char *name, LOCINT *arr, int n) {

	FILE     *fp=NULL;
	char     fname[MAX_LINE];
	int      myid;
	int i;

	MPI_Comm_rank(MPI_COMM_CLUSTER, &myid);
	snprintf(fname, MAX_LINE, "%s_%d", name, myid);
	fp = Fopen(fname, "a");

	for (i = 0; i < n ; i++)
	   fprintf(fp, " %d,", arr[i]);

	fprintf(fp, "\n");
	fclose(fp);
	return;

}


static void analyze_deg(LOCINT *deg, int n) {

	int i, curr_index = 0, new_count;
	char fname[256];

	LOCINT * deg_unique = (LOCINT *) Malloc(col_bl * sizeof(*deg_unique));
	LOCINT * deg_count = (LOCINT *) Malloc(col_bl * sizeof(*deg_count));

	memcpy(deg_unique, deg, col_bl * sizeof(*deg_unique));
	qsort(deg_unique, n, sizeof(LOCINT), cmpuloc);

	deg_count[curr_index] = 1;

	for (i = 1; i < n; i++) {
		if (deg_unique[i] == deg_unique[curr_index]) {
			deg_count[curr_index]++;
		} else {
			curr_index++;
			deg_unique[curr_index] = deg_unique[i];
			deg_count[curr_index] = 1;
		}
	}
	new_count = curr_index + 1;

	int bin_count = 16;
	LOCINT bin_limits[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000, 10000000 };
	//int bin_count = sizeof(bin_limits);

	LOCINT * bins = NULL;
	bins = (LOCINT *) Malloc(bin_count * sizeof(*bins));

	memset(bins, 0, bin_count * sizeof(*bins));

	i = new_count - 1;
	int curr_bin = (bin_count - 1);

	while (i > -1) {
		if (deg_unique[i] >= bin_limits[curr_bin]) {
			bins[curr_bin] += deg_count[i];
			i--;
		} else {
			curr_bin--;
			if (curr_bin < 0)
				break;
		}
	}

	//dump_deg(deg_unique, deg_count, new_count);

	snprintf(fname, 256, "degree_stats_%d", outId);
	dump_array(fname,bins, bin_count);

	freeMem(deg_unique);
	freeMem(deg_count);
}

static double bc_func(LOCINT *row, LOCINT *col, LOCINT *frt_all, LOCINT* frt, int* hFnum,
                      LOCINT *msk, int *lvl, LOCINT *deg,
                      LOCINT *sigma, LOCINT *frt_sig, float *delta, LOCINT rem_ed,
                      uint64_t v0, LOCINT *vRbuf, int *vRnum, LOCINT *hSbuf, int *hSnum, LOCINT *hRbuf,
                      int *hRnum, float *hSFbuf, float *hRFbuf, LOCINT *reach,
					  MPI_Request *vrequest, MPI_Request *hrequest, MPI_Status *status,
					  uint64_t* total_time, uint64_t* compu_time, uint64_t* commu_time, int dump) {

	int      level = 0, ncol;
	int64_t  i, j;
	uint64_t n=1, ned, nfrt=0;
	uint64_t nvisited = 1;
	double   teps=0;

	LOCINT *frt_all_start;

	TIMER_DEF(0);

#ifdef _TIMINGS
        TIMER_DEF(1);
#ifdef _FINE_TIMINGS
        uint64_t msgSize = 0, anfrt = 0;
        uint64_t upwm_time=0;   // UPWard Mpi
        uint64_t upwc_time=0;   // UPWard Cuda
        uint64_t depm_time=0;  	// DEPendency Mpi
        uint64_t depc_time=0;   // DEPendency Cuda
        uint64_t updm_time=0;   // UPDate bc Mpi
        uint64_t updc_time=0;   // UPDate bc Cuda
        uint64_t overlap_time = 0;
#endif
        uint64_t cuda_time = 0, mpi_time = 0;
#endif
	*total_time=0;
	*compu_time=0;
	*commu_time=0;

	memset(frt_all, 0,MAX(col_bl, row_pp)*sizeof(*frt_all));
	memset(hFnum, 0, row_bl*sizeof(*hFnum));

	memset((LOCINT *)sigma, 0, row_pp*sizeof(*sigma));
	memset((float*)delta,0,row_bl*sizeof(*delta));
	memset((int *)lvl, 0, row_pp*sizeof(*lvl));
	memset((float *)hSFbuf, 0, MAX(col_bl, row_pp)*sizeof(*hSFbuf));
	memset((float *)hRFbuf, 0, MAX(col_bl, row_pp)*sizeof(*hRFbuf));

	frt_all_start = frt_all;

	/* search for start edge */
	if (VERT2PROC(v0) == myid) {
		// Add root vertex to the Frontier

		// Get local value for vertex
		LOCINT lv0 = GI2LOCI(v0);  // Row index
		// Set BFS level
		lvl[lv0] = 0;
		// Update Bit Mask
		MSKSET(msk,lv0);
		// Add root vertex to frontier
		frt[nfrt] = MYLOCI2LOCJ(lv0);   // Col index
		// Set sigma
		sigma[lv0] = 1;
		frt_sig[nfrt] = 1;
		nfrt++;

		set_mlp_cuda(lv0, 0, 1);
	}
	// START BFS
	MPI_Barrier(MPI_COMM_CLUSTER);
	TIMER_START(0);

//      if (myid==0) fprintf(stdout, "Root_vertex = %lu\n", v0);
        //MPI_Pcontrol(1);
	while(1) {

#ifdef _FINE_TIMINGS
		addStats(v0, level, n);
#endif
			// We start a new BFS level
		level++;

        //dump_array2(&level,1,"BFS_LEVEL");
		// col-send frt
		// col-recv frts in vRbuf[0:R-1]
#ifdef _TIMINGS
		TIMER_START(1);
#endif
		// Exchange vertices in the frontier by column - EXPAND
		// For each vertex in the frontier send sigma as well
		// Here we use the old offset since we are sending Current Frontier
		exchange_vert4x2(frt, frt_sig, nfrt, vRbuf, row_bl, vRnum, vrequest, status, 1);
		ncol = 0;
		/* HERE WE NEED TO COPY ALL FRONTIERS INTO frt_all */ //the error is here
		for(i = 0; i < R; i++) {
			if (vRnum[i] > 0) {
			   memcpy(frt_all + ncol, vRbuf+2*i*row_bl, vRnum[i]*sizeof(*vRbuf));
			   ncol +=vRnum[i];
			}
		}
		hFnum[level] = hFnum[level-1] + ncol;
		frt_all += ncol;

		// Now we have in vRbuf all vertices from all frontiers on the same Column
		// vRnum is an array of size R containing how many elements are in VRbuf for each processor
		// NOTE: we cannot have duplicates in vRbuf since each processor can send only nodes that it owns

#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		upwm_time += mpi_time;
#endif
		*commu_time += mpi_time;
		TIMER_START(1);
#endif

		// Get neightbour vertices that have not been visited yet and put them into hSbuf
		/*
		 *  vRbuf contains the frontier according to the following pattern
		 *  each vRbuf[i] is long row_bl and contains vRnum[i] elements
		 *  Va,Vb,Vc,Sa,Sb,Sc
		 */

		nfrt = scan_col_csc_cuda(vRbuf, row_bl, vRnum, R, hSbuf, hSnum, frt, frt_sig, level);
		//printf("SCAN_COL_CSC: NFRT %d PROCESSOR %d\n",nfrt,myid);
		// Here we have hSbuf containing Vertices and Sigmas in the following format
		/*
		 * P 0 -> Va,Vb,Vc,Sa,Sb,Sc
		 * P 1 -> Vd,Ve,Sd,Se
		 */

#ifdef _TIMINGS
        TIMER_STOP(1);
        cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
        anfrt = nfrt;
        upwc_time += cuda_time;
#endif
        *compu_time += cuda_time;
        TIMER_START(1);
#endif
		// WE NEED TO SEND SIGMA
		// row-send hSbuf[0:C-1]
		// row-recv hRbuf[0:C-1]
		exchange_horiz4x2(hSbuf, row_bl, hSnum, hRbuf, row_bl, hRnum, hrequest, status, 1);
#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		upwm_time += mpi_time;
#endif
		*commu_time += mpi_time;
		TIMER_START(1);
#endif

		// WE NEED TO UPDATE SIGMA
		nfrt = append_rows_cuda(hRbuf, row_bl, hRnum, C, frt, frt_sig, nfrt, level);
        //printf("APPEND_ROWS: NFRT %d PROCESSOR %d\n",nfrt,myid);

		//dump_uarray2(frt , nfrt, "Frontier Append");
		//dump_uarray2(frt_sig , nfrt, "Frontier Sigma Append");

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		upwc_time += cuda_time;
#endif
		*compu_time += cuda_time;
		TIMER_START(1);
#endif
			// Get in n the total number of vertices in the NLFS
		MPI_Allreduce(&nfrt, &n, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_CLUSTER);

#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		upwm_time += mpi_time;
#endif
		*commu_time += mpi_time;
#endif
		// if (myid == 0) fprintf(stdout, "[%7.3lf%%] %"PRIu64"\n", (n*100.0)/N, n);
		if (n == 0) break; // Exit from the loop since we do not have new vertices to visit
		nvisited += n;

	} // While(1)

	//fprintf(outdebug, "BFS_done-visited=%d\n",nvisited);
	// Copy sigma value from device to CPU
	//dump_array2(lvl, row_pp, "Final Level");
	//dump_uarray2(frt_all_start, hFnum[level], "All Verticals Frontier");
	//dump_array2(hFnum, level, "Final hFnum");

	// Get Sigma and Level by row

	if (ntask > 1){
		MPI_Barrier(MPI_COMM_CLUSTER);

#ifdef _FINE_TIMINGS
		TIMER_START(1);
#endif

#ifdef OVERLAP
		set_get_overlap(sigma,lvl);
#else
		get_sigma(sigma);
		get_lvl(lvl);
		//sync cudaEventSynchronize( get_sigma_event );
		//MPI_Allreduce(MPI_IN_PLACE, sigma, row_pp, LOCINT_MPI, MPI_SUM, Row_comm);
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sigma, row_bl, LOCINT_MPI, Row_comm);
		set_sigma(sigma);
		//sync cudaEventSynchronize( get_lvl_event );
		//MPI_Allreduce(MPI_IN_PLACE, lvl,   row_pp, MPI_INT,    MPI_SUM, Row_comm);
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, lvl, row_bl, MPI_INT, Row_comm);
		set_lvl(lvl);
#endif
		MPI_Barrier(MPI_COMM_CLUSTER);

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
        overlap_time += TIMER_ELAPSED(1);
#endif
	}

    // SIGMA e LEVEL SONO INDICIZZATI PER RIGA
	// Copy back sigma value from device to CPU
	int depth = level - 2;

	frt = frt_all_start;
	ncol = 0;
	do {
		//fprintf(outdebug, "Depth %d\n", depth);
		//dump_array2(&depth, 1, "-- DEPTH --");
		// exchange delta values calculated in the previous round
		//MPI_Allreduce(MPI_IN_PLACE, hSRbuf, ncol, MPI_FLOAT, MPI_SUM, Col_comm);

#ifdef _TIMINGS
		TIMER_START(1);
#endif

		ncol = hFnum[depth+1] - hFnum[depth]; // number of vertices to process
		frt = frt_all_start + hFnum[depth];
		// Here I calculate hSRbuf for each vertex in the frontier
		// using the same index of the frontier
		scan_frt_csc_cuda(frt, ncol, depth, hRFbuf);

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		depc_time += cuda_time;
#endif
		*compu_time += cuda_time;
		TIMER_START(1);
#endif

		//hSRbuf contains the contribute to delta
		// Instead of exchanging all hSRbuf values I could send only to those
		// processors that own it. To do that hSRbuf should be referenced using the
		// same index of the CSC. Now instead we use the frontier index
		// After an additional thought I understood that if we want to send only the
		// local aggregated accumulation we have to use the frontier index since that is shared among
		// processors in the same column
		if (R>1)
			MPI_Allreduce(MPI_IN_PLACE, hRFbuf, ncol, MPI_FLOAT, MPI_SUM, Col_comm);

#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		depm_time += mpi_time;
#endif
		*commu_time += mpi_time;
		TIMER_START(1);
#endif
        //fprintf(outdebug, "wdc\n");
        // Copy back hSRbuf into device memory and calculate delta
		// Call device function to update delta values
		// We can have all delta values into DEVICE memory only
		write_delta_cuda(ncol, hRFbuf, hSFbuf);
		// Copy delta values into host buffer for sending by column

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		depc_time += cuda_time;
#endif
		*compu_time += cuda_time;
		TIMER_START(1);
#endif

		//fprintf(outdebug, "wdc - 2\n");
		// SEND BACK DELTA BY ROW!!!
		//dump_farray2(hSFbuf, row_pp, "hSFbuf BEFORE");

		if (C>1)
			//MPI_Allreduce(MPI_IN_PLACE, hSFbuf, row_pp, MPI_FLOAT, MPI_SUM, Row_comm);
			MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hSFbuf, row_bl, MPI_FLOAT, Row_comm);
		//dump_farray2(hSFbuf, row_pp, "hSFbuf AFTER");

#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		depm_time += mpi_time;
#endif
		*commu_time += mpi_time;
		TIMER_START(1);
#endif
		//fprintf(outdebug, "sdc\n");
		//Put delta into device memory
		set_delta_cuda(hSFbuf,row_pp);

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		depc_time += cuda_time;
#endif
		*compu_time += cuda_time;
#endif

		depth--;

	} while (depth > 0);
	// All delta values have been calculated and stored into device buffer we need to sum all them up
	// only on the processors on column 0 i processor may be more then 1

	LOCINT all = 0;
	if (mycol == 0){
// 		if nvisited + rem_ed == N the graph is connected otherwise I have more connected components so I have to take into account only the reach of the vertices in the current connected compoent.
//              if ((nvisited + rem_ed) != (N) ){
//              pre_update_bc_cuda(lvl, reach, v0, &all);
//              if (R>1)
//                      MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, Col_comm);
//                      printf("TOTAL %d ;  visited  %d ;  edge_rem %d :ONLY scc %d\n",N, nvisited, rem_ed, );
//              }else all = rem_ed;

#ifdef _TIMINGS
		TIMER_START(1);
#endif

		pre_update_bc_cuda(reach, v0, &all);

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		updc_time += cuda_time;
#endif
		*compu_time += cuda_time;
		TIMER_START(1);
#endif

		if (R>1)
			MPI_Allreduce(MPI_IN_PLACE, &all, 1, LOCINT_MPI, MPI_SUM, Col_comm);

#ifdef _TIMINGS
		TIMER_STOP(1);
		mpi_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		updm_time += mpi_time;
#endif
		*commu_time += mpi_time;
		TIMER_START(1);
#endif

		if (GI2PI(v0) == myrow) all += reach[GI2LOCI(v0)]; // questo serve?
		update_bc_cuda(v0, row_pp, nvisited+all);

#ifdef _TIMINGS
		TIMER_STOP(1);
		cuda_time = TIMER_ELAPSED(1);
#ifdef _FINE_TIMINGS
		updc_time += cuda_time;
#endif
		*compu_time += cuda_time;
#endif

	}

	MPI_Barrier(MPI_COMM_CLUSTER);
	TIMER_STOP(0);
	*total_time = TIMER_ELAPSED(0);

#ifdef _FINE_TIMINGS
	upStats(v0, nvisited, level, msgSize,
				upwc_time, upwm_time,
				depc_time, depm_time,
				updc_time, updm_time,
				overlap_time, *compu_time, *commu_time, *total_time);
#endif

/*
	get_msk(msk);

	// compute teps
	n = 0;
	for(j = 0; j < row_pp; j++)
			n += (!!MSKGET(msk,j)) * deg[j];

	MPI_Reduce(&n, &ned, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_CLUSTER);
	ned >>= 1; // ??
	teps = ((double)ned)/(*total_time/1.0E+6);

      if (0 == myid) {
                fprintf(stdout, "\n\n\n\nElapsed time: %f secs\n", *total_time/1.0E+6);
                fprintf(stdout, "Traversed edges: %"PRIu64"\n", ned);
                fprintf(stdout, "Measured TEPS: %lf\n", teps);
        }
*/
        /*
        if (dump) {
                dump_lvl(lvl, mycol*row_bl, mycol*row_bl+row_bl);
                dump_prd(prd, mycol*row_bl, mycol*row_bl+row_bl, hSbuf, row_bl, hSnum, msk);
        }
    */
	return teps;
}

//prd qui era un int ora e' LOCINT ma non dovrebbe essere usato
static double bc_func_mono(LOCINT *row, LOCINT *col, LOCINT *frt_all, LOCINT* frt, int* hFnum,
					       LOCINT *msk, int *lvl, LOCINT *deg,
						   LOCINT *sigma, LOCINT *frt_sig, float *delta, LOCINT rem_ed,
						   uint64_t v0, LOCINT *vRbuf, int *vRnum, LOCINT *hSbuf, int *hSnum, LOCINT *hRbuf,
						   int *hRnum, float *hSFbuf, float *hRFbuf, LOCINT *reach,
						   MPI_Request *vrequest, MPI_Request *hrequest, MPI_Status *status,
						   uint64_t* total_time, int dump) {

	int 	 level = 0, ncol;
	uint64_t nfrt=0;
	uint64_t nvisited = 1;
	double	 teps=0;


	TIMER_DEF(0);

#ifdef _FINE_TIMINGS
	TIMER_DEF(1);
	uint64_t msgSize = 0;
	uint64_t upwc_time=0;   // UPWard Cuda
	uint64_t depc_time=0;   // DEPendency Cuda
	uint64_t updc_time=0;   // UPDate bc Cuda
#endif
	*total_time=0;

	memset((LOCINT *)sigma, 0, row_pp*sizeof(*sigma));
	memset((float*)delta,0,row_bl*sizeof(*delta));
	memset((int *)lvl, 0, row_pp*sizeof(*lvl));
	memset(hFnum, 0, row_bl*sizeof(*hFnum));
	memset((LOCINT *)frt, 0, row_pp*sizeof(*frt));

	/* LOCINT *cbuf;
	cbuf = (LOCINT *)Malloc(row_pp*sizeof(LOCINT));
	memset((LOCINT *)cbuf, 0, row_pp*sizeof(*cbuf));
	get_frt(frt);
	dump_uarray2(frt, row_pp, "FRT-0");
	get_cbuf(cbuf);
	dump_uarray2(cbuf, row_pp, "CBUF-0");
    */

	LOCINT lv0 = GI2LOCI(v0);
	nfrt++;
	set_mlp_cuda(lv0, 0, 1);

	// START UPWARD BC
	TIMER_START(0);

#ifdef _FINE_TIMINGS
		TIMER_START(1);
#endif

	while(1) {
#ifdef _FINE_TIMINGS
//		fprintf(stdout, "BFS_LEVEL=%d\n",level);
		addStats(v0, level, nfrt);
#endif
		// We start a new BFS level
        level++;
		hFnum[level] = hFnum[level-1] + nfrt;

		// Now we have in vRbuf all vertices from all frontiers on the same Column
		// vRnum is an array of size R containing how many elements are in VRbuf for each processor
		// NOTE: we cannot have duplicates in vRbuf since each processor can send only nodes that it owns

		// Get neightbour vertices that have not been visited yet and put them into hSbuf
		/*
		 *  vRbuf contains the frontier according to the following pattern
		 *  each vRbuf[i] is long row_bl and contains vRnum[i] elements
		 *  Va,Vb,Vc,Sa,Sb,Sc
		 */
		nfrt = scan_col_csc_cuda_mono(nfrt, level);
		if (!nfrt) break; // Exit from the loop since we do not have new vertices to visit
		nvisited += nfrt;

		//get_sigma(sigma);
		//dump_uarray2(sigma, row_pp, "Sigma");

	} // While(1)

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		upwc_time += TIMER_ELAPSED(1);
		TIMER_START(1);
#endif

	//fprintf(outdebug, "BFS_done-visited=%d\n",nvisited);
	// Copy sigma value from device to CPU

	//get_sigma(sigma);
	//dump_uarray2(sigma, row_pp, "Final Sigma");

	//get_lvl(lvl);
	//dump_array2(lvl, row_pp, "Final Level");

	//dump_uarray2(frt_all_start, hFnum[level], "All Verticals Frontier");
	//dump_array2(hFnum, level, "Final hFnum");
	//dump_array2(lvl, row_pp, "Final Level after All reduce");


    // SIGMA e LEVEL SONO INDICIZZATI PER RIGA
	// Copy back sigma value from device to CPU
	int depth = level - 2;
	//frt = frt_all_start;
	ncol = 0;

	do {
		//fprintf(outdebug, "Depth %d\n", depth);
		//dump_array2(&depth, 1, "-- DEPTH --");
		// exchange delta values calculated in the previous round

	    ncol = hFnum[depth+1] - hFnum[depth]; // number of vertices to process
	    //frt = frt_all_start + hFnum[depth];
	    // Here I calculate hSRbuf for each vertex in the frontier
	    // using the same index of the frontier
	    scan_frt_csc_cuda_mono(hFnum[depth], ncol, depth);

		depth--;

	} while (depth > 0);

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		depc_time += TIMER_ELAPSED(1);
		TIMER_START(1);
#endif

	// All delta values have been calculated and stored into device buffer
	LOCINT all = 0;
	//printf("heuristic %d\n", heuristic);
	if (heuristic == 1 || heuristic == 3){
		pre_update_bc_cuda(reach, v0, &all);
	}
	all += reach[GI2LOCI(v0)]; // questo serve?
    	update_bc_cuda(v0, row_pp, nvisited+all);

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		updc_time += TIMER_ELAPSED(1);
#endif

	TIMER_STOP(0);
	*total_time = TIMER_ELAPSED(0);

#ifdef _FINE_TIMINGS
	upStats(v0, nvisited, level, msgSize,
			upwc_time, 0,
			depc_time, 0,
			updc_time, 0,
			0, 0, 0, *total_time);
#endif

	//get_msk(msk);

	// compute teps
	/*
	n = 0;
	for(j = 0; j < row_pp; j++)
		n += (!!MSKGET(msk,j)) * deg[j];

// 	MPI_Reduce(&n, &ned, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	n >>= 1; // ??
	teps = ((double)n)/(*total_time/1.0E+6);

	if (0 == myid) {
		fprintf(stdout, "\n\n\n\nElapsed time: %f secs\n", *total_time/1.0E+6);
		fprintf(stdout, "Traversed edges: %"PRIu64"\n", ned);
		fprintf(stdout, "Measured TEPS: %lf\n", teps);
	}
*/
	/*
	if (dump) {
		dump_lvl(lvl, mycol*row_bl, mycol*row_bl+row_bl);
		dump_prd(prd, mycol*row_bl, mycol*row_bl+row_bl, hSbuf, row_bl, hSnum, msk);
	}
    */
	return teps;
}

static double bc_func_mono_2degree(LOCINT *row, LOCINT *col, LOCINT *frt_all, LOCINT* frt, int* hFnum, LOCINT *frt_all_v1, LOCINT *frt_v1, int *hFnum_v1, LOCINT *msk, int *lvl, LOCINT *deg, LOCINT *sigma, LOCINT *frt_sig, float *delta, LOCINT rem_ed,  uint64_t v0, uint64_t v1, uint64_t v2, LOCINT *reach, uint64_t* total_time, int dump) {


	//memory duplication in order to avoid mistake
	int level_v1 = 0, ncol_v1;
	int level_v2 = 0, ncol_v2;

	//uint64_t n_v1, n_v2, n_v0; // probably n_v0 is not needed
	uint64_t nfrt_v1 = 0;
	uint64_t nfrt_v2 = 0;

	uint64_t nvisited_v1 = 1;
	uint64_t nvisited_v2 = 1;

	int depth_v1 = 0;
	int depth_v2 = 0;
	int depth = 0;

	//printf("2_degree[%s] (%ld) -> %ld %ld\n", __func__ ,v0, v1, v2);
	// init 2_degree set zero vectors... setted by set_cuda_degree....
	TIMER_DEF(0); // for upward

#ifdef _FINE_TIMINGS
	TIMER_DEF(1);
	uint64_t msgSize = 0;
	uint64_t upwc_time=0;   // UPWard Cuda
	uint64_t depc_time=0;   // DEPendency Cuda
	uint64_t updc_time=0;   // UPDate bc Cuda
#endif
	*total_time=0;

	memset(hFnum, 0, row_bl*sizeof(*hFnum)); // for v2
    	memset(hFnum_v1, 0, row_bl*sizeof(*hFnum));
	memset((LOCINT *)frt_v1, 0, row_pp*sizeof(*frt));

	LOCINT lv1 = GI2LOCI(v1);
	LOCINT lv2 = GI2LOCI(v2);
    	// v1 computation must be return sigma_v1 lvl_v0
	set_mlp_cuda(lv1, 0, 1);
    	nfrt_v1++;
	// START SSSP ON V1
	TIMER_START(0);

#ifdef _FINE_TIMINGS
		TIMER_START(1);
#endif

	while(1) {
#ifdef _FINE_TIMINGS
		addStats(v0, level_v1, nfrt_v1);
#endif
		// We start a new BFS level
       		level_v1++;
		hFnum_v1[level_v1] = hFnum_v1[level_v1-1] + nfrt_v1;
		nfrt_v1 = scan_col_csc_cuda_mono(nfrt_v1, level_v1);
		if (!nfrt_v1) break; 
		nvisited_v1 += nfrt_v1;
	} // While(1)

    //  get_sigma(sigma);
    //  dump_uarray2(sigma, row_pp, "Final Sigma");
	// Copy sigma value from device to CPU
	//get_sigma(sigma);
	//dump_uarray2(sigma, row_pp, "Final Sigma");
	//get_lvl(lvl);
	//dump_array2(lvl, row_pp, "Final Level");
	//dump_uarray2(frt_all_start, hFnum[level], "All Verticals Frontier");
	//dump_array2(hFnum, level, "Final hFnum");
   	// SIGMA e LEVEL SONO INDICIZZATI PER RIGA
	// Copy back sigma value from device to CPU

	set_mlp_cuda_2degree(lv2, 0, 1);
	nfrt_v2++;
	//START SINGLE SSSP ON V2
	while(1){
#ifdef _FINE_TIMINGS
		//addStats(v2, level_v2, nfrt_v2);
#endif
        level_v2++;
		hFnum[level_v2] = hFnum[level_v2-1] + nfrt_v2;
		nfrt_v2 = scan_col_csc_cuda_mono(nfrt_v2, level_v2);
		if (!nfrt_v2) break; 
		nvisited_v2 += nfrt_v2;
	}

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		upwc_time += TIMER_ELAPSED(1);
		TIMER_START(1);
#endif
        /*
  	depth_v1 = level_v1 - 2;
        depth_v2 = level_v2 - 2;
	ncol_v1 = 0;
        ncol_v2 = 0;
        int test = 1;
        if (test == 0 ){
	do {
	    	ncol_v1 = hFnum_v1[depth_v1+1] - hFnum_v1[depth_v1]; //number of nodes to process
                scan_frt_csc_cuda_mono_2degree(hFnum_v1[depth_v1], ncol_v1, depth_v1,0);
		depth_v1--;
	} while (depth_v1 > 0);

        do {
	    	ncol_v2 = hFnum[depth_v2+1] - hFnum[depth_v2]; //number of nodes to process
	   	scan_frt_csc_cuda_mono_2degree(hFnum[depth_v2], ncol_v2, depth_v2,1);
		depth_v2--;
	} while (depth_v2 > 0);
        }
        else{*/
        //printf("MIXED DEP\n"); 

        depth_v1 = level_v1 - 2;
        depth_v2 = level_v2 - 2;
        ncol_v1 = 0;
        ncol_v2 = 0;
        depth = MAX(depth_v1, depth_v2); //printf("max depth is %d\n", depth);
        do{
		  if (depth_v1 == depth){
			  ncol_v1 = hFnum_v1[depth_v1+1] - hFnum_v1[depth_v1];
			  scan_frt_csc_cuda_mono_2degree(hFnum_v1[depth_v1], ncol_v1, depth_v1,0, v0);
			  depth_v1--;
		  }

		  if (depth_v2 == depth){
			  ncol_v2 = hFnum[depth_v2+1] - hFnum[depth_v2];
			  scan_frt_csc_cuda_mono_2degree(hFnum[depth_v2], ncol_v2, depth_v2,1,v0);
			  depth_v2--;
		  }
		  depth--;
        }while (depth >= 0);

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		depc_time += TIMER_ELAPSED(1);
		TIMER_START(1);
#endif

	// All delta values have been calculated and stored into device buffer        
	LOCINT all_v0, all_v1, all_v2 = 0;
        // UPDATE BC nota lvl qui non e' usato
	pre_update_bc_cuda(reach, v0, &all_v0); //va fatta per 2-degree una sola volta
        all_v1 = all_v2 = all_v0;
        //infatti i visited dei v0 v1 v2 sono sempre gli stessi.
	all_v0 += reach[GI2LOCI(v0)];
    	all_v1 += reach[GI2LOCI(v1)];
    	all_v2 += reach[GI2LOCI(v2)];

    update_bc_cuda_2degree(v0, v1, v2, row_pp, nvisited_v1+all_v0,nvisited_v1+all_v1,nvisited_v1+all_v2);

#ifdef _FINE_TIMINGS
		TIMER_STOP(1);
		updc_time += TIMER_ELAPSED(1);
#endif

	TIMER_STOP(0);
	*total_time = TIMER_ELAPSED(0);

#ifdef _FINE_TIMINGS
	upStats(v0, nvisited_v1+2, MAX(level_v1,level_v2)+1, 0,
				upwc_time, 0,
				depc_time, 0,
				updc_time, 0,
				0, 0, 0, *total_time);
#endif
   return 1;
}










enum {s_minimum,
      s_firstquartile,
      s_median,
      s_thirdquartile,
      s_maximum,
      s_mean,
      s_std,
      s_LAST};

static int compare_doubles(const void* a, const void* b) {

	double aa = *((const double *)a);
	double bb = *((const double *)b);

	return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

static void get_statistics(const double x[], int n, double r[s_LAST]) {

	double temp;
	int i;

	/* Compute mean. */
	temp = 0;
	for(i = 0; i < n; ++i) temp += x[i];
	temp /= n;
	r[s_mean] = temp;

	/* Compute std. dev. */
	temp = 0;
	for(i = 0; i < n; ++i)
			temp += (x[i] - r[s_mean])*(x[i] - r[s_mean]);
	temp /= n-1;
	r[s_std] = sqrt(temp);
	r[s_std] /= (r[s_mean]*r[s_mean]*sqrt(n-1));

	/* Sort x. */
	double* xx = (double*)Malloc(n*sizeof(double));
	memcpy(xx, x, n*sizeof(double));
	qsort(xx, n, sizeof(double), compare_doubles);

	/* Get order statistics. */
	r[s_minimum] = xx[0];
	r[s_firstquartile] = (xx[(n-1)/4] + xx[n/4]) * .5;
	r[s_median] = (xx[(n-1)/2] + xx[n/2]) * .5;
	r[s_thirdquartile] = (xx[n-1-(n-1)/4] + xx[n-1-n/4]) * .5;
	r[s_maximum] = xx[n-1];

	/* Clean up. */
	free(xx);
}

static void print_stats(double *teps, int n) {

	int i;
	double stats[s_LAST];

	for(i = 0; i < n; i++) teps[i] = 1.0/teps[i];

	get_statistics(teps, n, stats);

	fprintf(stdout, "TEPS statistics:\n");
	fprintf(stdout, "\t   harm mean: %lf\n", 1.0/stats[s_mean]);
	fprintf(stdout, "\t   harm stdev: %lf\n", stats[s_std]);
	fprintf(stdout, "\t   median: %lf\n", 1.0/stats[s_median]);
	fprintf(stdout, "\t   minimum: %lf\n", 1.0/stats[s_maximum]);
	fprintf(stdout, "\t   maximum: %lf\n", 1.0/stats[s_minimum]);
	fprintf(stdout, "\tfirstquartile: %lf\n", 1.0/stats[s_firstquartile]);
	fprintf(stdout, "\tthirdquartile: %lf\n", 1.0/stats[s_thirdquartile]);
	return;
}

void usage(const char *pname) {

	prexit("Usage:\n"
			"\t %1$s -p RxC [-d dev0,dev1,...] [-o outfile] [-D] [-d] [-m] [-N <# of searches>]\n"
		    "\t -> to visit a graph read from file:\n"
		   "\t\t -f <graph file> -n <# vertices> [-r <start vert>]\n"
		   "\t -> to visit an RMAT graph:\n"
		   "\t\t -S <scale> [-E <edge factor>]\n"
			"\t Where:\n"
			"\t\t -D to ENABLE debug information\n"
			"\t\t -d to DUMP RMAT generated graph to file\n"
			"\t\t -m to DISABLE mono GPU optimization\n"
			"\t\t -U DO NOT make graph Undirected\n"
			"\t\t -a perform degree analysis\n"
			"\n", pname);
	return;
}

int main(int argc, char *argv[]) {

	int s, t, gread=-1;
	int scale=21, edgef=16; // 1 1-degree reduction
	short dump = 0, debug = 0;
	int64_t  i, j;
	uint64_t nbfs=1, ui;
	LOCINT n, l, ned, rem_ed = 0;

	uint64_t *edge=NULL;
	uint64_t *rem_edge=NULL;

	LOCINT  *col=NULL;
	LOCINT  *row=NULL;
	LOCINT  *frt=NULL;
	LOCINT  *frt_all=NULL;
	int *hFnum = NULL; /* offsets of ventices in the frontier for each level */

	LOCINT  *frt_v1=NULL;
	LOCINT  *frt_all_v1=NULL;
	int *hFnum_v1 = NULL; /* offsets of ventices in the frontier for each level */

	LOCINT *degree=NULL; // Degree for all vertices in the same column
	LOCINT *sigma=NULL;  // Sigma (number of SP)
	LOCINT *frt_sigma=NULL;  // Sigma (number of SP)

	float   *hRFbuf=NULL;
	float   *hSFbuf=NULL;
	float   *delta=NULL;
	float   *bc_val=NULL;
	LOCINT  *msk=NULL;
	int     *lvl=NULL; //, level;
	LOCINT  *deg=NULL;

	LOCINT  *vRbuf=NULL;
	int     *vRnum=NULL; /* type int (MPI_Send()/Recv() assumes int counts) */

	LOCINT  *hSbuf=NULL;
	LOCINT  *hRbuf=NULL;
	int     *hSnum=NULL;
	int     *hRnum=NULL;
	LOCINT  *prd=NULL; /* predecessors array */
	LOCINT	*bc_order=NULL;
	MPI_Status  *status;
	MPI_Request *vrequest;
	MPI_Request *hrequest;
	MPI_Comm MPI_COMM_COL;

	uint64_t v0 = 0, startv = 0;
	uint64_t v1=0;
	uint64_t v2=0;

	int rootset = 0;

	int cntask;
	char *gfile=NULL, *p=NULL, c, *ofile=NULL;

	TIMER_DEF(0);

	double *teps=NULL;

	int random = 0;

	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);

//      struct timespec mytime;
//      clock_gettime( CLOCK_REALTIME, &mytime);

	MPI_Comm_rank(MPI_COMM_WORLD, &gmyid);
	MPI_Comm_size(MPI_COMM_WORLD, &gntask);

	if (argc == 1) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	outId = time(NULL);

	int wr, pl=0, limit = sizeof(cmdLine);
	wr = snprintf (cmdLine+pl, limit, "MPI Tasks %d\n", gntask);
	if (wr<0) exit(EXIT_FAILURE);
	limit -= wr; pl +=wr;
	for (i = 0; i < argc; i++) {
		wr = snprintf (cmdLine+pl, limit," %s", argv[i]);
		if (wr<0) exit(EXIT_FAILURE);
		limit -= wr; pl +=wr;
	}
	snprintf (cmdLine+pl, limit,"\n");

	if (gmyid==0) {
		fprintf(stdout,"%s\n",cmdLine);
	}

	while((c = getopt(argc, argv, "o:p:amhDR:dUf:n:r:S:E:N:H:")) != EOF) {
#define CHECKRTYPE(exitval,opt) {\
		if (exitval == gread) prexit("Unexpected option -%c!\n", opt);\
				else gread = !exitval;\
		}
		switch (c) {
			case 'H' :
					if (0 == sscanf(optarg, "%d", &heuristic)) prexit("Invalid Heuristic Option (-H): %s\n", optarg);					
					if ( heuristic >= 2 && ntask > 1) prexit("2-degree Heuristic is allowed in single-gpu configuration (-H): %s\n", optarg);
					break;
					//heuristic selection
			case 'o' :
					ofile = strdup(optarg);
					break;
			case 'p':
					strncpy(strmesh,optarg,10);
					p = strtok(optarg, "x");
					if (!p) prexit("Invalid proc mesh field.\n");
					if (0 == sscanf(p, "%d", &R)) prexit("Invalid number of rows for proc mesh (-p): %s\n", p);
					p = strtok(NULL, "x");
					if (!p) prexit("Invalid proc mesh field.\n");
					if (0 == sscanf(p, "%d", &C)) prexit("Invalid number of columns for proc mesh (-p): %s\n", p);
					break;
			case 'd':
					// Dump RMAT Generated graph
					dump = 1;
					break;
			case 'R':
					// Random Root
					sscanf(optarg, "%d", &random);
					if (random != 1 && random != 2 && random != 3)
						prexit("Invalid random option (-R): %s\n", optarg);
					break;
			case 'D':
					// DEBUG
					debug = 1;
					break;
			case 'm':
					// Mono-Multi GPU
					mono = 0;
					break;
			case 'f':
					CHECKRTYPE(0, 'f')
					gfile = strdup(optarg);
					break;
			case 'n':
					CHECKRTYPE(0, 'n')
					if (0 == sscanf(optarg, "%"PRIu64, &N)) prexit("Invalid number of vertices (-n): %s\n", optarg);
					break;
			case 'r':
					//CHECKRTYPE(0, 'r')
					if (0 == sscanf(optarg, "%"PRIu64, &startv)) prexit("Invalid root vertex (-r): %s\n", optarg);
					rootset = 1;
					break;
			case 'S':
					CHECKRTYPE(1, 'S')
					if (0 == sscanf(optarg, "%d", &scale)) prexit("Invalid scale (-S): %s\n", optarg);
					N = ((uint64_t) 1) << scale;
					break;
			case 'E':
					CHECKRTYPE(1, 'E')
					if (0 == sscanf(optarg, "%d", &edgef)) prexit("Invalid edge factor (-S): %s\n", optarg);
					break;
			case 'N':
					if (0 == sscanf(optarg, "%ld", &nbfs)) prexit("Invalid number of bfs (-N): %s\n", optarg);
					break;
			case 'U':
					// Undirected
					undirected = 0;
					break;
			case 'a':
					// Degree analysis
					analyze_degree = 1;
					break;
			case 'h':
			case '?':
			default:
					usage(argv[0]);
					exit(EXIT_FAILURE);
		}
#undef CHECKRTYPE
	}

	if (gread) {
		if (!gfile || !N)
			prexit("Graph file (-f) and number of vertices (-n)"
				   " must be specified for file based bfs.\n");
	}

	if (0 >= R || MAX_PROC_I < R || 0 >= C || MAX_PROC_J < C)
		prexit("R and C must be in range [1,%d] and [1,%d], respectively.\n", MAX_PROC_I, MAX_PROC_J);

	if (0 != N%(R*C))
		prexit("N must be multiple of both R and C.\n");

	ntask=R*C;
	cntask = gntask/ntask;
	if(gntask%ntask) {
		fprintf(stderr,
			  "Invalid configuration: total number of task is %d, cluster size is %d\n",
			  gntask,ntask);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	if ( heuristic >= 2 && ntask > 1){
		 prexit("\n\n2-degree Heuristic is allowed in single-gpu configuration (-H): %d\n", heuristic);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

#ifndef _LARGE_LVERTS_NUM
	if ((N/R) > UINT32_MAX) {
		prexit("Number of vertics per processor too big (%"LOCPRI"), please"
			   "define _LARGE_LVERTS_NUM macro in %s.\n", (N/R), __FILE__);
	}
#endif

	if (startv >= N)
		prexit("Invalid start vertex: %"PRIu64".\n", startv);

	int color = gmyid/ntask;
	MPI_Comm_split(MPI_COMM_WORLD, color, gmyid, &MPI_COMM_CLUSTER);
	MPI_Comm_rank( MPI_COMM_CLUSTER, &myid);
	MPI_Comm_size( MPI_COMM_CLUSTER, &ntask);

	myrow = myid/C;
	mycol = myid%C;

	MPI_Comm_split(MPI_COMM_WORLD,(myrow*C)+mycol,gmyid,&MPI_COMM_COL);

	row_bl = N/(R*C); /* adjacency matrix rows per block:    N/(RC) */
	col_bl = N/C;     /* adjacency matrix columns per block: N/C */
	row_pp = N/R;     /* adjacency matrix rows per proc:     N/(RC)*C = N/R */

	if ((gmyid==0) && (debug==1)) {
	  char     fname[MAX_LINE];
	  snprintf(fname, MAX_LINE, "%s_%d.log", "debug", gmyid);
	  outdebug = Fopen(fname,"w");
	}

	char *resname = NULL;

	if (ntask > 1) mono = 0;
	// Disable random when a starting node is provided
	if (rootset > 0) random = 0;

	if (gmyid == 0) {
		fprintf(stdout,"\n\n****** DEVEL VERSION ******\n\n\n\n***************************\n\n");
		fprintf(stdout, "Total number of vertices (N): %"PRIu64"\n", N);
		fprintf(stdout, "Processor mesh rows (R): %d\n", R);
		fprintf(stdout, "Processor mesh columns (C): %d\n", C);
//              fprintf(stdout, "Number of rows per block (N/(R*C)): %"LOCPRI"\n", row_bl);
//              fprintf(stdout, "Number of columns per block (N/C): %"LOCPRI"\n", col_bl);
//              fprintf(stdout, "Total rows per processor (N/R): %"LOCPRI"\n", row_pp);
		if (gread) {
			fprintf(stdout, "Reading graph from file: %s\n", gfile);
		} else {
			fprintf(stdout, "RMAT graph scale: %d\n", scale);
			fprintf(stdout, "RMAT graph edge factor: %d\n", edgef);
			fprintf(stdout, "Number of bc rounds: %ld\n", nbfs);
			if (random) fprintf(stdout, "Random mode\n");
			else fprintf(stdout, "First node: %d\n", startv);
		}
                fprintf(stdout,"\n\n");
                if (heuristic == 0){
                      fprintf(stdout, "HEURISTICs: OFF: %d\n", heuristic);

                }
                else if (heuristic == 1){
			fprintf(stdout, "HEURISTICs: 1-degree reduction ON: %d\n", heuristic);
		}
		else if (heuristic == 2){
			fprintf(stdout, "HEURISTICs: 2-degree reduction ON: %d\n", heuristic);
		}
		else if (heuristic == 3){
			fprintf(stdout, "HEURISTICs: 1-degree and 2-degree reduction ON: %d\n", heuristic);
		}
#ifdef THRUST
                fprintf(stdout,"PREFIX SCAN library: THRUST\n");
#else
                fprintf(stdout,"PREFIX SCAN library: CUB\n");
#endif

#ifdef OVERLAP
		fprintf(stdout,"OVERLAP: ON\n");
#else
		fprintf(stdout,"OVERLAP: OFF\n");
#endif
#ifdef ONEPREFIX
      	        fprintf(stdout,"PREFIX SCAN optimization: ON\n");
#else
      	        fprintf(stdout,"PREFIX SCAN optimization: OFF\n");
#endif

                fprintf(stdout,"\n");
	}

	if (NULL != ofile) {
		fprintf(stdout, "Result written to file: %s\n", ofile);
		resname = (char*) malloc((sizeof(ofile)+MAX_LINE)*sizeof(*resname));
		sprintf(resname, "%s_%dX%d_%d.log",ofile, R,C, gmyid);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	/* fill processor mesh */
	memset(pmesh, -1, sizeof(pmesh));
	for(i = 0; i < R; i++)
		for(j = 0; j < C; j++)
			pmesh[i][j] = i*C + j;

	if (myid == 0) fprintf(stdout, "%s graph...\n", gread?"Reading":"Generating");
	TIMER_START(0);
	if (gread) ned = read_graph(myid, ntask, gfile, &edge); // Read from file
	else       ned = gen_graph(scale, edgef, &edge);        // Generate RMAT
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, " done in %f secs\n", TIMER_ELAPSED(0)/1.0E+6);
	prstat(ned, gread?"Edges read from file:":"Edges generated:", 1);

	if (heuristic != 0){
		l = norm_graph(edge, ned, NULL);
		prstat(ned-l, "First Multi-edges removed:", 1);
		ned = l;
	}

	if (dump > 0 && gread == 0 && ntask == 1){
		fprintf(stdout, "Dump file...\n");
		dump_rmat(edge, ned, scale, edgef);
	}

	// 1 DEGREE PREPROCESSING TIMING ON
	if (heuristic == 1 || heuristic == 3){
		if (gmyid == 0) fprintf(stdout, "Degree reduction graph (%d)...\n", heuristic);
		TIMER_START(0);
		// DEGREE REDUCTION - Edge Based
		ned = degree_reduction(myid, ntask, &edge, ned, &rem_edge, &rem_ed);
		TIMER_STOP(0);
		degree_reduction_time = TIMER_ELAPSED(0);
	}
    // 2-D PARTITIONING
	if (gmyid == 0) fprintf(stdout, "Partitioning graph... ");
	TIMER_START(0);
	if (ntask > 1) ned = part_graph(myid, ntask, &edge, ned, 2); // 2-D graph partitioning
	if (ntask > 1) rem_ed = part_graph(myid, ntask, &rem_edge, rem_ed,2);
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);
	prstat(ned, "Edges assigned after partitioning:", 1);
	//dump_edges(edge, ned,"Edges 2-D");
#ifndef _LARGE_LVERTS_NUM
	if (ned > UINT32_MAX) {
		fprintf(stderr,"Too many vertices assigned to me. Change LOCINT\n");
		exit(EXIT_FAILURE);
	}
#endif
	if (myid == 0) fprintf(stdout, "task %d: Verifying partitioning...", gmyid);
	TIMER_START(0);
	for(n = 0; n < ned; n++) {
		if (EDGE2PROC(edge[2*n], edge[2*n+1]) != myid) {
			fprintf(stdout, "[%d] error, received edge (%"PRIu64", %"PRIu64"), should have been sent to %d\n",
					myid, edge[2*n],edge[2*n+1], EDGE2PROC(edge[2*n], edge[2*n+1]));
			break;
		}
	}
	s = (n != ned);
	MPI_Allreduce(&s, &t, 1, MPI_INT, MPI_LOR, MPI_COMM_CLUSTER);
	TIMER_STOP(0);
	if (t)  prexit("Error in 2D decomposition.\n");
	else if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);

	/* Graph has been partitioned correctly */
	if (myid == 0) fprintf(stdout, "task %d Removing multi-edges...",gmyid);
	deg = (LOCINT *)Malloc(row_pp*sizeof(*deg));

	TIMER_START(0);
	// Normalize graph: remove loops and duplicates edges cappi o loops?
	// THIS ALSO CALCULATES DEGREES
	l = norm_graph(edge, ned, deg);
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);
	prstat(ned-l, "Multi-edges removed:", 1);
	ned = l;

	// check whether uint64 edges can fit in 32bit CSC
	if (4 == sizeof(LOCINT)) {
		if (!verify_32bit_fit(edge, ned))
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	if (myid == 0) fprintf(stdout, "task %d, Creating CSC...", gmyid);
	TIMER_START(0);

	build_csc(edge, ned, &col, &row);  // Make the CSC Structure
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);
	freeMem(edge);
	freeMem(deg);
	n = initcuda(ned, col, row);
	prstat(n>>20, "Device memory allocated (MB):", 1);

	MPI_Comm_split(MPI_COMM_CLUSTER, myrow, mycol, &Row_comm);
	MPI_Comm_split(MPI_COMM_CLUSTER, mycol, myrow, &Col_comm);

	// Allocate The frontier
	frt = (LOCINT *)CudaMallocHostSet(row_bl*sizeof(*frt),0);
	frt_all = (LOCINT *)CudaMallocHostSet(MAX(col_bl, row_pp)*sizeof(*frt), 0);
	// Allocate The BFS level
	lvl = (int *)CudaMallocHostSet(row_pp*sizeof(*lvl),0);
	//	cudaHostRegister(lvl, row_pp*sizeof(*lvl), 0);
	// Allocate the frontier offset for each level
	hFnum = (int*)CudaMallocHostSet(MAX(row_pp, col_bl)*sizeof(*hFnum),0);
	// Allocate Degree array
	degree = (LOCINT *)CudaMallocHostSet(col_bl*sizeof(*degree),0);
	// Allocate sigma
	sigma = (LOCINT *)CudaMallocHostSet(row_pp*sizeof(*sigma),0);
	//	cudaHostRegister(sigma, row_pp*sizeof(*sigma), 0);
	// Allocate Frontier Sigma
	frt_sigma = (LOCINT *)CudaMallocHostSet(row_bl*sizeof(*frt_sigma),0);
	// Allocate delta
	delta = (float*)CudaMallocHostSet(row_bl*sizeof(*delta),0);
	// Allocate BC
	bc_val = (float*)CudaMallocHostSet(row_pp*sizeof(*bc_val),0);
	reach = (LOCINT*)CudaMallocHostSet(row_pp*sizeof(*reach),0);
	if (heuristic == 1 || heuristic == 3 ){
		//if(myid==0) printf("task %d edges removed %d ...\n",gmyid,rem_ed);
		init_bc_1degree(rem_edge, rem_ed, N, reach);
		MPI_Allreduce(MPI_IN_PLACE, &rem_ed, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
		MPI_Allreduce(MPI_IN_PLACE, reach, row_pp, MPI_INT, MPI_SUM, Row_comm);
		if(myid==0) printf("task %d Total edges removed %d\n",gmyid, rem_ed);
	}
	//used for HEURISTIC 2 + 3 only
	if (heuristic==2 || heuristic == 3 ){
		prd = (LOCINT *)CudaMallocHostSet(N*sizeof(*prd),0);
		bc_order =(LOCINT *)CudaMallocHostSet(N*sizeof(*bc_order),0);
		frt_v1 = (LOCINT *)CudaMallocHostSet(row_bl*sizeof(*frt),0);
		frt_all_v1 = (LOCINT *)CudaMallocHostSet(MAX(col_bl, row_pp)*sizeof(*frt), 0);
		hFnum_v1 = (int*)CudaMallocHostSet(MAX(row_pp, col_bl)*sizeof(*hFnum),0);
	}
	
    	get_deg(degree);
	// Calculate degree
	MPI_Allreduce(MPI_IN_PLACE, degree, col_bl, MPI_INT, MPI_SUM, Col_comm);

	init_bc_1degree_device(reach);

	if (analyze_degree == 1)
		analyze_deg(degree, col_bl);

	// Allocate BitMask to store visited unique vertices ???
	msk = (LOCINT *)Malloc(((row_pp+BITS(msk)-1)/BITS(msk))*sizeof(*msk));

	if (!mono){
		vRbuf = (LOCINT *)CudaMallocHostSet(2*col_bl*sizeof(*vRbuf), 0);  // We need to double the size for sigma
		vRnum = (int *)CudaMallocHostSet(R*sizeof(*vRnum), 0);
		hSbuf = (LOCINT *)CudaMallocHostSet(2*row_pp*sizeof(*hSbuf), 0);  // We need to double the size for sigma
		hSnum = (int *)CudaMallocHostSet(C*sizeof(*hSnum), 0);
		hRbuf = (LOCINT *)CudaMallocHostSet(2*row_pp*sizeof(*hRbuf), 0);  // We need to double the size for sigma
		hRnum = (int *)CudaMallocHostSet(C*sizeof(*hRnum), 0);
		hSFbuf = (float*)CudaMallocHostSet(MAX(col_bl, row_pp)*sizeof(*hSFbuf), 0);
		hRFbuf = (float*)CudaMallocHostSet(MAX(col_bl, row_pp)*sizeof(*hRFbuf), 0);

		status  =  (MPI_Status *) Malloc(MAX(C,R)*sizeof(*status));
		vrequest = (MPI_Request *)Malloc(MAX(C,R)*sizeof(*vrequest));
		hrequest = (MPI_Request *)Malloc(MAX(C,R)*sizeof(*hrequest));

		// exchange for mpi warm-up
		exchange_vert4x2(frt, frt_sigma, row_bl, vRbuf, row_bl, vRnum, vrequest, status, 1);
		for(i = 0; i < C; i++) hSnum[i] = row_bl;
		exchange_horiz4x2(hSbuf, row_bl, hSnum, hRbuf, row_bl, hRnum, hrequest, status, 1);
	}

	nbfs = MIN(N, nbfs);

#ifdef _FINE_TIMINGS
    // Allocate for statistical data
    mystats = (STATDATA*)Malloc(N*sizeof(STATDATA));
    memset(mystats, 0, N*sizeof(STATDATA));
#endif

#ifdef ONEPREFIX
//      	fprintf(stdout,"PREFIX SCAN optimization: ON\n");

	tlvl = (LOCINT*)Malloc(MAX_LVL*sizeof(*tlvl));
	if (heuristic == 2 || heuristic ==3){
		tlvl_v1 = (LOCINT*)Malloc(MAX_LVL*sizeof(*tlvl_v1));

	}
#endif

	LOCINT skip, reach_v0, nrounds=0, skipped=0;
	LOCINT two_dg_c = 0;
	LOCINT two_dg_n = 0;
	uint64_t all_time=0, bc_time=0, min_time=UINT_MAX, max_time=0;

	// commu_all_time = overall time spent in communication
	// compu_all_time = overall time spent in computation

	uint64_t commu_all_time=0, commu_time=0, compu_all_time=0, compu_time=0;
	if (myid == 0) fprintf(stdout, "task %d: BC computation is running...\n", gmyid);
	if (heuristic == 2 || heuristic == 3 ){

        	LOCINT reach_v1 = 0, reach_v2;
		v0 = v1 = v2 = 0;
		//Identify 2-degree nodes. Let b to be a 2-degree nodes: solution sorting
		//identify child-a and child-b
		//Perform mono_bc (modified ) from a
		//Store in GPU sigma_a and lvl_a
		//each 2_degree steps  must be profiled by using two_degree_reduction_time
		for (i = 0 ; i < N ; i++){
			bc_order [i] = i; //naive iniit 
		}
		TIMER_START(0);
		sort_by_degree(degree,  bc_order);
		TIMER_STOP(0);
		two_degree_reduction_time = TIMER_ELAPSED(0);
		ui = color;
		while (nrounds < nbfs && (ui < N)) //for (ui = color; ui < nbfs; ui++)
		{
			skip = 0;
			v0 = bc_order[ui];
			// bc_time = 0;
			reach_v0 = 0;
			if (prd[v0] != 0){
				ui++;
				continue; // bc scores already computed for these verticies
			}

#ifdef _FINE_TIMINGS
			setStats(v0, degree[GJ2LOCJ(v0)]);
#endif
			if (degree[ui] == 0){
				skip = 1;
				prd[v0] = 1;
				skipped++;
				ui++;
				continue;
			}
			else if (degree[ui] == 2) {
				two_dg_n++;
				v1 = row[col[v0]];
				v2 = row[col[v0]+1];
				reach_v0 = reach[GI2LOCI(v0)];
				if (prd[v1] != 0 ||  prd[v2] != 0){
						setcuda(ned, col, row, reach_v0);
						bc_func_mono(row, col,  frt_all, frt,  hFnum, msk,   lvl,   degree,  sigma, frt_sigma, delta, rem_ed, v0, vRbuf,  vRnum, hSbuf, hSnum, hRbuf, hRnum, hSFbuf, hRFbuf, reach, vrequest, hrequest, status, &bc_time, 0);
						all_time += bc_time;
						prd[v0] = 1;
						nrounds++;
				}
				else{
						two_dg_c++;
						reach_v1 = reach[GI2LOCI(v1)];
						reach_v2 = reach[GI2LOCI(v2)];
						//setcuda per v0 v1 v2
						//perform mono_bc on v1
						//perform mono_bc on v2  and compute v0
						//prd[v0] = v0; prd[v1] = v0; prd[v2] = v0; to skip
						setcuda_2degree(reach_v0, reach_v1, reach_v2);
						bc_func_mono_2degree(row, col,  frt_all, frt,  hFnum, frt_all_v1, frt_v1, hFnum_v1 ,msk,   lvl, degree,  sigma, frt_sigma, delta, rem_ed, v0, v1, v2, reach, &bc_time, 0);
						prd[v0] = prd[v1]  = prd[v2] = 1;
						all_time += bc_time;
						nrounds+=3;
				}
			}
			else{	
				reach_v0 = reach[GI2LOCI(v0)];
				setcuda(ned, col, row, reach_v0);
                		prd[v0]= 1;
				bc_func_mono(row, col,  frt_all, frt,  hFnum, msk,   lvl,   degree,  sigma, frt_sigma, delta, rem_ed, v0, vRbuf,  vRnum, hSbuf, hSnum, hRbuf, hRnum, hSFbuf, hRFbuf, reach, vrequest, hrequest, status, &bc_time, 0);
				all_time += bc_time;
				nrounds++;
			}
			if (bc_time > max_time ) max_time = bc_time;
			if (bc_time < min_time && bc_time != 0 ) min_time= bc_time;
			ui++;
		}
	}
	else{// NO H 2 o H 3
		for(ui = color; ui < nbfs; ui += cntask) {

			switch (random) {
				case 0: v0 = startv + ui; break;
				case 1: v0 = select_root1(col); break;
				case 2: v0 = select_root2(col); break;
				case 3: v0 = select_root3(col); break;
				default: v0 = startv + ui; break;
			}

			skip = 0;
            //v0 = startv + ui;
			bc_time = 0;
			reach_v0 = 0;
			if (VERT2PROC(v0) == myid) {
			//fprintf(stdout,"Root = %lu; TaskId=%d; LocalId=%d\n", v0, myid, GJ2LOCJ(v0));

#ifdef _FINE_TIMINGS
			setStats(v0, degree[GJ2LOCJ(v0)]);
#endif
				// Check v0 degree
				if (degree[GJ2LOCJ(v0)]==0) {
					skip=1;
				}
				reach_v0 = reach[GI2LOCI(v0)];
			}
			if (ntask > 1)
				MPI_Allreduce(MPI_IN_PLACE, &skip, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
			// Root vertex with degree 0
			if (skip) {
				//teps[ui] = 0;
					skipped++;
					continue;
			}

			if (ntask > 1)
				MPI_Allreduce(MPI_IN_PLACE, &reach_v0, 1, LOCINT_MPI, MPI_SUM, MPI_COMM_CLUSTER);
				// fprintf(outdebug,"Root = %lu; Reach = %d;\n", v0, reach_v0);

			setcuda(ned, col, row, reach_v0);

			if (mono == 0) {
                //teps[ui]=
				     bc_func(row, col,  frt_all, frt,  hFnum, msk,   lvl, degree,  sigma, frt_sigma, delta, rem_ed,
							 v0, vRbuf,  vRnum, hSbuf, hSnum, hRbuf, hRnum, hSFbuf, hRFbuf, reach,
								 vrequest, hrequest, status, &bc_time, &compu_time, &commu_time, 0);
			} else {
                //teps[ui]=
				    bc_func_mono(row, col,  frt_all, frt,  hFnum, msk,   lvl, degree,  sigma, frt_sigma, delta, rem_ed,
								      v0, vRbuf,  vRnum, hSbuf, hSnum, hRbuf, hRnum, hSFbuf, hRFbuf, reach,
									  vrequest, hrequest, status, &bc_time, 0);
			}
			nrounds++;
			all_time += bc_time;
			commu_all_time += commu_time;
			compu_all_time += compu_time;

			if (bc_time > max_time ) max_time = bc_time;
			if (bc_time < min_time && bc_time != 0 ) min_time= bc_time;
		}
	}

    TIMER_START(0);
	if (mycol == 0) {
		get_bc(bc_val);
		if(color==0) {
			MPI_Reduce(MPI_IN_PLACE,bc_val,row_pp,MPI_FLOAT,MPI_SUM,0,MPI_COMM_COL);
		} else {
			MPI_Reduce(bc_val,NULL,row_pp,MPI_FLOAT,MPI_SUM,0,MPI_COMM_COL);
		}
	}
	TIMER_STOP(0);
	uint64_t bcred_time = TIMER_ELAPSED(0);

//        if(gmyid==0) {
//        MPI_Allreduce(MPI_IN_PLACE,bc_val,row_pp,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
//        } else {
//          MPI_AllReduce(bc_val,NULL,row_pp,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
//        }
	if (mycol == 0 && color == 0 && resname != NULL) {
		FILE *resout = fopen(resname,"w");

		fprintf(resout,"BC RESULTS\n");
		fprintf(resout,"ROWPP %u\n", row_pp);
		fprintf(resout,"NodeId\tBC_VAL\n");
		LOCINT k;
		for (k=0;k<row_pp;k++) {
			fprintf(resout,"%d\t%.2f\n", LOCI2GI(k) ,bc_val[k]/2.0);
		}
		fclose(resout);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	// REDUCE MPI compu_all_time/1.0E+6
	// commu_all_time/1.0E+6

	uint64_t comp_avg_time = 0;
	uint64_t  comm_avg_time= 0;
	MPI_Reduce(&compu_all_time,&comp_avg_time,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_CLUSTER);
	MPI_Reduce(&commu_all_time,&comm_avg_time,1,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_CLUSTER);
	comp_avg_time =  comp_avg_time/ntask;	
	comm_avg_time = comm_avg_time/ntask;

	MPI_Barrier(MPI_COMM_WORLD);
	if (myid == 0) {

		fprintf(stdout, "\n------------- RESULTS ---------------\n");

		fprintf(stdout,"ClusterId\tSkip\tRounds\tExecTime\tRoundsTime\tCUDATime(avg)\tMPITime(avg)\t\tMax\tMin\tMean\t1-dReduTime\n");
		fprintf(stdout,"%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
				 	 	 	 	 	 	 	 color, skipped, nrounds,
				                             (all_time+bcred_time)/1.0E+6,
											 all_time/1.0E+6,
											 comp_avg_time/1.0E+6,
											 comm_avg_time/1.0E+6,
											 max_time/1.0E+6,
											 min_time/1.0E+6,
											 all_time/1.0E+6/nrounds,
											 degree_reduction_time/1.0E+6);

		fprintf(stdout,"task %d BC skipped: %d \n", gmyid, skipped);
		fprintf(stdout,"task %d BC rounds: %d  \n", gmyid, nrounds);
		fprintf(stdout,"task %d BC execution total time: %lf sec\n",gmyid,(all_time+bcred_time)/1.0E+6);
		fprintf(stdout,"task %d BC rounds total time: %lf sec\n",gmyid,all_time/1.0E+6);
		fprintf(stdout,"task %d BC rounds Computation CUDA avg time: %lf sec over %d procs\n",gmyid,  comp_avg_time/1.0E+6, ntask);
		fprintf(stdout,"task %d BC rounds Communication MPI avg time: %lf sec over %d procs\n",gmyid, comm_avg_time/1.0E+6, ntask);
		fprintf(stdout,"task %d BC Max time: %lf sec\n",gmyid, max_time/1.0E+6);
		fprintf(stdout,"task %d BC Min time: %lf sec\n",gmyid, min_time/1.0E+6);
		fprintf(stdout,"task %d BC Mean time: %lf sec\n",gmyid, all_time/1.0E+6/nrounds);
		if (nbfs < N){
			unsigned int vskipped = 0;
			vskipped=(N-nbfs)*skipped/nbfs;
			double avg = all_time/1.0E+6/nrounds; 
			fprintf(stdout,"task %d BC simulated time: %lf sec (virtual skipped %d)\n",gmyid, avg*(N-skipped-vskipped)/cntask, vskipped);
			fprintf(stdout,"task %d BC simulated time-2: %lf sec\n",gmyid, ((((all_time+bcred_time)/1.0E+6)/nbfs)*N)/cntask);

		}
		if ( heuristic == 1 || heuristic == 3 ){
			fprintf(stdout,"task %d 1-Degree reduction: %lf sec\n",gmyid, degree_reduction_time/1.0E+6);
		}
		if ( heuristic == 2 || heuristic == 3 ){
			fprintf(stdout,"task %d 2-Degree reduction: %lf sec.(2-degree nodes computed by 2-degree fun %d/%d)\n",gmyid, two_degree_reduction_time/1.0E+6, two_dg_c, two_dg_n);
                }                
		//fprintf(stdout,"task %d Sigma-lvl time: %lf sec\n",gmyid, overlap_time/1.0E+6);
		fprintf(stdout, "\n");
	}

#ifdef _FINE_TIMINGS
#ifdef PRINTSTATS
	writeStats();
#endif
#endif
	MPI_Barrier(MPI_COMM_WORLD);
	if (gmyid == 0){
		fprintf(stdout,"System summary:\n Total(gntask) GPUs %d - Total(fd) ntask %d - Total(fr)  cntask %d\n", gntask, ntask, cntask);

	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (outdebug!=NULL) fclose(outdebug);
		fprintf(stdout, "WNODE Global-ID %d - Cluster-ID %d -  Local-ID %d ... closing\n",gmyid,color,  myid);

	MPI_Barrier(MPI_COMM_WORLD);
	freeMem(col);
	freeMem(row);
	freeMem(mystats);
//	cudaHostUnregister(lvl);
//	freeMem(lvl);
	freeMem(msk);
	//freeMem(deg);
	//freeMem(status);
	//freeMem(vrequest);
	//freeMem(hrequest);
	freeMem(gfile);
	freeMem(teps);
//	cudaHostUnregister(sigma);
//	freeMem(sigma);

	fincuda();
	CudaFreeHost(lvl);
	CudaFreeHost(sigma);
	CudaFreeHost(frt);
	CudaFreeHost(frt_all);
	CudaFreeHost(degree);
	CudaFreeHost(frt_sigma);
	CudaFreeHost(hFnum);
	CudaFreeHost(delta);
	CudaFreeHost(bc_val);
	CudaFreeHost(reach);
	CudaFreeHost(prd);
	CudaFreeHost(vRbuf);
	CudaFreeHost(vRnum);
	CudaFreeHost(hSbuf);
	CudaFreeHost(hSnum);
	CudaFreeHost(hRbuf);
	CudaFreeHost(hRnum);
	CudaFreeHost(hSFbuf);
	CudaFreeHost(hRFbuf);
//2-degree
        CudaFreeHost(frt_v1);
        CudaFreeHost(frt_all_v1);
       	CudaFreeHost(hFnum_v1);
	
//ONEPREFIX
	freeMem(tlvl);
	freeMem(tlvl_v1);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Barrier(Row_comm);
	MPI_Barrier(Col_comm);
	MPI_Comm_free(&Row_comm);
	MPI_Comm_free(&Col_comm);
	MPI_Comm_free(&MPI_COMM_CLUSTER);
	MPI_Finalize();
	exit(EXIT_SUCCESS);
}

