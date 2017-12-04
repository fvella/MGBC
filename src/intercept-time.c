/*
Comple like this
gcc -fPIC -Wall -shared -o intercept-time.so intercept-time.c -ldl
Use Like this
export LD_PRELOAD=$(pwd)/intercept-time.so
*/
#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>
#include <time.h>
int (*_original_clock_gettime)(clockid_t clk_id, struct timespec *tp);
void init(void) __attribute__((constructor));
void init(void)
{
	_original_clock_gettime = (int (*)(clockid_t, struct timespec *))
	dlsym(RTLD_NEXT, "clock_gettime");
}
static double starting_timer;
int clock_gettime(clockid_t clk_id, struct timespec *tp);

int clock_gettime(clockid_t clk_id, struct timespec *tp) {
	if(clk_id == CLOCK_REALTIME) {
	  int ret_val = _original_clock_gettime(4, tp);
	  starting_timer = (double)(*tp).tv_sec + (double)(*tp).tv_nsec*1.0e-9;
	return ret_val;
	}else{
	  int ret_val = _original_clock_gettime(4, tp);
	  double dtime = ((double)(*tp).tv_sec + (double)(*tp).tv_nsec*1e-9) - starting_timer;
	  (*tp).tv_sec = (int)dtime;
	  (*tp).tv_nsec = (int)((dtime-(int)dtime)*1.0e9);
	return ret_val;
	}
}
