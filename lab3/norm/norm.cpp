#include <algorithm>
#include <execution>
#include "norm.h"
#include "scope_profile.h"

#include <omp.h>

//Global variables
std::vector<double> v;
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

double computeNorm(const std::vector<double> &vec, int start, int end) {
    int diff = end - start;
    if (diff <= 0) {
        return 0;
    } else if (diff == 1) {
        return vec[start] * vec[start];
    } else {
        int mid = start + (end - start) / 2;
        return computeNorm(vec, start, mid) + computeNorm(vec, mid, end);
    }
}

double normOfVector(std::vector<double> &vec) {
    return std::sqrt(computeNorm(vec, 0, vec.size()));
}

double computeNormParallel(const std::vector<double> *vec, int start, int end) {
    int diff = end - start;
    constexpr int threshold = 10000; // Seems to be a good threshold.

    if (diff <= 0) {
        return 0;
    } else if (diff == 1) {

        return vec->at(start) * vec->at(start);
    }
    if (diff <= threshold) {
        // Sequential to avoid too fine-grained task creation
        double sum = 0.;
        for (int i = start; i < end; ++i) {
            sum += vec->at(i) * vec->at(i);
        }
        return sum;
    } else {
        int mid = start + diff / 2;
        double sum1, sum2;

#pragma omp task shared(sum1)
        sum1 = computeNormParallel(vec, start, mid);

#pragma omp task shared(sum2)
        sum2 = computeNormParallel(vec, mid, end);

#pragma omp taskwait
        return sum1 + sum2;
    }
}


double normOfVectorParallel(std::vector<double> &vec) {
    return std::sqrt(computeNormParallel(&vec, 0, vec.size()));
}


int main(int argc, char *argv[]) {

    omp_set_num_threads(NB_THREADS);

    // Check validity of constant parameters.
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

    v.resize(LENGTH);
    for (int i = 0; i < LENGTH; i++) v[i] = SCALE * rand() / (float) RAND_MAX;

    {
        SCOPED_PROFILE_LOG("Calculate norm execution time sequential recursive")

        norm = normOfVector(v);
    }

    {
        SCOPED_PROFILE_LOG("Calculate norm execution time parallel recursive")

        norm = normOfVectorParallel(v);
    }

    printf("The vector has norm: %f\n", norm);

    return 0;

}