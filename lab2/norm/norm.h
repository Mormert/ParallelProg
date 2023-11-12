#ifndef NORM_H_
#define NORM_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#ifndef LENGTH
#define LENGTH 100000000
#endif

#ifndef SEED
#define SEED 0
#endif

#ifndef SCALE
#define SCALE 10
#endif

#ifndef NB_THREADS
#define NB_THREADS 32
#endif

#if NB_THREADS > 1

void *dot(void *ptr);

#endif

double timediff(struct timespec *begin, struct timespec *end);

int main(int argc, char* argv[]);

#endif

