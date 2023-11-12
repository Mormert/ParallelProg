#include <algorithm>
#include <execution>
#include "norm.h"
#include "scope_profile.h"

//Global variables
double v[LENGTH];
double norm = 0.;

#if NB_THREADS > 1

double partial_sums[NB_THREADS] = {0};

void *dot(void *ptr) {
    int id = *(int *) ptr;
    int start = id * (LENGTH / NB_THREADS);
    int end = (id + 1) * (LENGTH / NB_THREADS);

    for (int i = start; i < end; i++) {
        partial_sums[id] += v[i] * v[i];
    }

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


        norm = 0.0f;
#pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < LENGTH; i++) {
            norm += v[i] * v[i];
        }


        norm = std::sqrt(norm);
#endif
    }

    printf("The vector has norm: %f\n", norm);

    return 0;

}