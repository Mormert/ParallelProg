#include <algorithm>
#include <execution>
#include "norm.h"
#include "scope_profile.h"

//Global variables
double v[LENGTH];
double norm = 0.;

pthread_barrier_t barrier;

#if NB_THREADS > 1

double partial_sums[NB_THREADS] = {0};

void *dot(void *ptr) {
    int id = *(int *) ptr;
    int start = id * (LENGTH / NB_THREADS);
    int end = (id + 1) * (LENGTH / NB_THREADS);

    for (int i = start; i < end; i++) {
        partial_sums[id] += v[i] * v[i];
    }

#ifndef USE_JOIN
    pthread_barrier_wait(&barrier);
#endif

    return nullptr;
}

#endif

int main(int argc, char *argv[]) {

    //Check validity of constant parameters.
    if (sizeof(LENGTH) != sizeof(int) || LENGTH < 1) {
        printf("Invalid vector length.\n");
        return -1;
    }

    if (sizeof(SEED) != sizeof(int)) {
        printf("Invalid seed for random vectors.\n");
        return -1;
    }

    if (sizeof(SCALE) != sizeof(int) || SCALE < 1) {
        printf("Invalid scaling factor.\n");
        return -1;
    }

    if (sizeof(NB_THREADS) != sizeof(int)) {
        printf("Invalid number of threads.\n");
        return -1;
    }

    printf("Number of threads: %d.\n", NB_THREADS);

    //Initialize structures.
    srand(SEED);

    for (int i = 0; i < LENGTH; i++) v[i] = SCALE * rand() / (float) RAND_MAX;

    {
        SCOPED_PROFILE_LOG("Calculate norm execution time")


#if NB_THREADS < 2
        double x = 0;
        for (const double &i: v) {
            x = x + i * i;
        }
        norm = std::sqrt(x);
#else

        pthread_t threads[NB_THREADS];
        int thread_ids[NB_THREADS];

#ifndef USE_JOIN
        pthread_barrier_init(&barrier, NULL, NB_THREADS + 1);
#endif

        // Spawn threads
        for (int i = 0; i < NB_THREADS; i++) {
            thread_ids[i] = i;
            if (pthread_create(&threads[i], NULL, dot, &thread_ids[i])) {
                fprintf(stderr, "Error creating thread\n");
                return 1;
            }
        }

#ifdef USE_JOIN
        // Wait for threads to finish and add their results
        for(int i = 0; i < NB_THREADS; i++) {
            if(pthread_join(threads[i], NULL)) {
                fprintf(stderr, "Error joining thread\n");
                return 2;
            }
            norm += partial_sums[i];
        }
#else // Use barrier

        // Wait for threads to finish and add their results
        pthread_barrier_wait(&barrier);

        // We dont need to join here, technically, since we are synced already!
       // for (int i=0; i < NB_THREADS; i++) {
       //     pthread_join(threads[i], NULL);
       // }

        pthread_barrier_destroy(&barrier);
        for (double partial_sum : partial_sums) {
            norm += partial_sum;
        }
#endif
        norm = std::sqrt(norm);
#endif
    }

    printf("The vector has norm: %f\n", norm);

    return 0;

}